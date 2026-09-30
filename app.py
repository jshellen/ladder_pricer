from __future__ import annotations

import math
from datetime import datetime, timedelta
from copy import deepcopy
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, State, dcc, html, no_update

from trinity.defaults import default_config
from trinity.service import ENGINES
from trinity.ui_config import build_config, checked, parse_numbers

APP_TITLE = "Trinity 2.0"
SESSION_HORIZON = 570.0
BASE_CCY = "EUR"
QUOTE_CCY = "SEK"

SESSION_CLOCK_ORIGIN = datetime(2000, 1, 1, 7, 45)
SESSION_TIME_TICK_MINUTES = [0.0, 120.0, 240.0, 360.0, 480.0, SESSION_HORIZON]


def session_timestamp(elapsed_minutes: float) -> datetime:
    """Map elapsed session minutes to a clock timestamp for plotting."""
    return SESSION_CLOCK_ORIGIN + timedelta(minutes=float(elapsed_minutes))


def session_timestamps(elapsed_minutes) -> list[datetime]:
    return [session_timestamp(x) for x in elapsed_minutes]


def session_clock_label(elapsed_minutes: float, seconds: bool = True) -> str:
    fmt = "%H:%M:%S" if seconds else "%H:%M"
    return session_timestamp(elapsed_minutes).strftime(fmt)


def apply_session_clock_axis(fig: go.Figure, title: str = "Session time") -> go.Figure:
    tickvals = session_timestamps(SESSION_TIME_TICK_MINUTES)
    ticktext = [session_clock_label(x, seconds=False) for x in SESSION_TIME_TICK_MINUTES]
    fig.update_xaxes(
        type="date",
        title_text=title,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        hoverformat="%H:%M:%S",
    )
    return fig

# -----------------------------------------------------------------------------
# Unified Trinity / Streamlit-style dark Plotly theme
# -----------------------------------------------------------------------------
PLOT_BG = "#0e1117"
PAPER_BG = "#111821"
PLOT_TEXT = "#fafafa"
PLOT_MUTED = "#a3a8b3"
PLOT_GRID = "#2b3542"
PLOT_ZERO = "#485568"
PLOT_ACCENT = "#58a6ff"
BID_COLOR = "#58a6ff"
ASK_COLOR = "#ff5c5c"
SIZE_DASHES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

_trinity_dark = go.layout.Template(pio.templates["plotly_dark"])
_trinity_dark.layout.update(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=PLOT_TEXT, family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size=12),
    title=dict(font=dict(color=PLOT_TEXT, size=16), x=0.02, xanchor="left"),
    colorway=[
        "#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f778ba",
        "#39c5cf", "#ff7b72", "#79c0ff", "#7ee787", "#e3b341",
    ],
    hoverlabel=dict(bgcolor="#262730", bordercolor="#3b4656", font=dict(color=PLOT_TEXT)),
    legend=dict(bgcolor="rgba(17,24,33,0.82)", bordercolor="#2b3542", borderwidth=1, font=dict(color=PLOT_TEXT)),
    margin=dict(l=58, r=24, t=58, b=52),
    autosize=True,
)

_axis = dict(
    color=PLOT_MUTED,
    gridcolor=PLOT_GRID,
    zerolinecolor=PLOT_ZERO,
    linecolor="#344150",
    tickcolor="#344150",
    title_font=dict(color="#d8dee9"),
    tickfont=dict(color=PLOT_MUTED),
    automargin=True,
)
_trinity_dark.layout.xaxis.update(_axis)
_trinity_dark.layout.yaxis.update(_axis)
_trinity_dark.layout.scene.update(
    bgcolor=PLOT_BG,
    xaxis=dict(backgroundcolor=PLOT_BG, gridcolor=PLOT_GRID, zerolinecolor=PLOT_ZERO, color=PLOT_MUTED),
    yaxis=dict(backgroundcolor=PLOT_BG, gridcolor=PLOT_GRID, zerolinecolor=PLOT_ZERO, color=PLOT_MUTED),
    zaxis=dict(backgroundcolor=PLOT_BG, gridcolor=PLOT_GRID, zerolinecolor=PLOT_ZERO, color=PLOT_MUTED),
)
pio.templates["trinity_dark"] = _trinity_dark
pio.templates.default = "trinity_dark"

app = Dash(__name__, title=APP_TITLE, suppress_callback_exceptions=True)
server = app.server


def number_input(label: str, component_id: str, value: float, step: float | str = "any"):
    return html.Label([
        html.Span(label, className="field-label"),
        dcc.Input(id=component_id, type="number", value=value, step=step, className="field-input"),
    ], className="field")


def text_input(label: str, component_id: str, value: str):
    return html.Label([
        html.Span(label, className="field-label"),
        dcc.Input(id=component_id, type="text", value=value, className="field-input"),
    ], className="field")


def checkbox(label: str, component_id: str, checked: bool):
    return dcc.Checklist(
        id=component_id,
        options=[{"label": label, "value": "on"}],
        value=["on"] if checked else [],
        className="checklist",
    )


def select(label: str, component_id: str, options, value):
    return html.Label([
        html.Span(label, className="field-label"),
        dcc.Dropdown(id=component_id, options=options, value=value, clearable=False, className="field-dropdown"),
    ], className="field")


GRAPH_CONFIG = {"displaylogo": False, "responsive": True}


def graph(component_id: str, height: int = 420):
    """Responsive Plotly graph with a stable card height.

    Plot styling is controlled by the global Plotly template.  Keeping the
    container sizing here avoids CSS rules that reach into Plotly's internal
    SVG layers.
    """
    return dcc.Graph(
        id=component_id,
        config=GRAPH_CONFIG,
        responsive=True,
        style={"height": f"{int(height)}px", "width": "100%"},
    )


def panel(title: str, children, open_: bool = False):
    return html.Details([html.Summary(title), html.Div(children, className="panel-body")], open=open_, className="panel")


def diagnostic_expander(title: str, children, open_: bool = False):
    """Streamlit-style collapsible diagnostics section used in Policy/Tier diagnostics."""
    if not isinstance(children, (list, tuple)):
        children = [children]
    return html.Details(
        [html.Summary(title), html.Div(list(children), className="diagnostic-expander-body")],
        open=open_,
        className="diagnostic-expander",
    )


def tier_panel(index: int, tier: dict[str, Any], open_: bool = False):
    p = f"t{index}"
    return panel(tier["name"], [
        checkbox("Enabled", f"{p}-enabled", tier["enabled"]),
        text_input("Name", f"{p}-name", tier["name"]),
        text_input("Sizes [EUR M]", f"{p}-sizes", ", ".join(str(x) for x in tier["sizes"])),
        html.Div("Flow", className="subhead"),
        html.Div([
            number_input("A0 / min", f"{p}-A0", tier["flow"]["A0"], 0.001),
            number_input("Theta", f"{p}-theta", tier["flow"]["theta"], 0.001),
            number_input("Beta", f"{p}-beta", tier["flow"]["beta"], 0.001),
            number_input("Steepness", f"{p}-steep", tier["flow"]["steepness"], 0.01),
            number_input("Shift", f"{p}-shift", tier["flow"]["shift"], 0.01),
            number_input("Volume shift", f"{p}-vshift", tier["flow"]["volumeShift"], 0.001),
        ], className="grid-2"),
        html.Div("Markout", className="subhead"),
        checkbox("Enabled", f"{p}-markout-enabled", tier["useMarkout"]),
        html.Div([
            number_input("Scale [pips]", f"{p}-impact", tier["markout"]["impactScalePips"], 0.1),
            number_input("Size exponent", f"{p}-impact-beta", tier["markout"]["sizeExponent"], 0.05),
            number_input("Tau [min]", f"{p}-impact-tau", tier["markout"]["tauMinutes"], 0.05),
            number_input("Delta min", f"{p}-dmin", tier["deltaMin"], 0.5),
            number_input("Delta max", f"{p}-dmax", tier["deltaMax"], 0.5),
        ], className="grid-2"),
    ], open_)


def card(title: str, value_id: str, subtitle: str = ""):
    return html.Div([
        html.Div(title, className="metric-title"),
        html.Div("—", id=value_id, className="metric-value"),
        html.Div(subtitle, className="metric-subtitle"),
    ], className="metric-card")


def format_ccy(x: float) -> str:
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:,.2f}m"
    if abs(x) >= 1_000:
        return f"{x / 1_000:,.1f}k"
    return f"{x:,.0f}"


def _active_tier_configs(cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not cfg:
        return []
    return [tier for tier in cfg.get("tiers", []) if tier.get("enabled", False)]


def _tier_cfg(cfg: dict[str, Any], tier_index: int) -> dict[str, Any]:
    tiers = _active_tier_configs(cfg)
    if not 0 <= int(tier_index) < len(tiers):
        raise IndexError("tier index out of range")
    return tiers[int(tier_index)]


def _logistic(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    expx = np.exp(arr[~pos])
    out[~pos] = expx / (1.0 + expx)
    return float(out) if out.ndim == 0 else out


def _activity(tier: dict[str, Any], z: float) -> float:
    flow = tier["flow"]
    return float(flow["A0"]) * float(z) ** (-float(flow["theta"]) - float(flow["beta"]) * float(z))


def _delta50(tier: dict[str, Any], z: float) -> float:
    flow = tier["flow"]
    return float(flow["shift"]) - float(flow["volumeShift"]) * (float(z) - 1.0)


def _hit_ratio(tier: dict[str, Any], delta, z: float):
    kappa = float(tier["flow"]["steepness"])
    return _logistic(kappa * (np.asarray(delta, dtype=float) - _delta50(tier, z)))


def _arrival_rate(tier: dict[str, Any], delta, z: float):
    return _activity(tier, z) * _hit_ratio(tier, delta, z)


def _markout_pips(tier: dict[str, Any], z: float, t_minutes):
    if not tier.get("useMarkout", True):
        return np.zeros_like(np.asarray(t_minutes, dtype=float))
    spec = tier["markout"]
    tau = max(float(spec["tauMinutes"]), 1e-15)
    asymptotic = float(spec["impactScalePips"]) * float(z) ** float(spec["sizeExponent"])
    return asymptotic * (1.0 - np.exp(-np.asarray(t_minutes, dtype=float) / tau))


def _quote_vs_mid_pips(delta, side: str, spread_pips: float):
    delta = np.asarray(delta, dtype=float)
    if side == "bid":
        return float(spread_pips) * (delta - 0.5)
    return float(spread_pips) * (0.5 - delta)


def _volume_premium_pips(delta_ref, delta_cur, spread_pips: float):
    return float(spread_pips) * (np.asarray(delta_ref, dtype=float) - np.asarray(delta_cur, dtype=float))


def _interp_policy(q_grid, values, q):
    qg = np.asarray(q_grid, dtype=float)
    vals = np.asarray(values, dtype=float)
    return float(np.interp(float(q), qg, vals))


def _simple_table(columns: list[str], rows: list[list[Any]]) -> html.Table:
    header = html.Thead(html.Tr([html.Th(c) for c in columns]))
    body = html.Tbody([html.Tr([html.Td(v) for v in row]) for row in rows])
    return html.Table([header, body], className="data-table")


def flow_curve_figure(tier: dict[str, Any]) -> go.Figure:
    grid = np.linspace(-1.0, 1.0, 160)
    fig = go.Figure()
    for z in tier["sizes"]:
        fig.add_trace(go.Scatter(x=grid, y=_arrival_rate(tier, grid, float(z)), mode="lines", name=f"{float(z):g}M"))
    fig.update_layout(title=f"Flow curves λ(δ,z) · {tier['name']}", xaxis_title="δ", yaxis_title="Fill intensity [1/min]", template="trinity_dark")
    return fig


def hit_ratio_figure(tier: dict[str, Any]) -> go.Figure:
    grid = np.linspace(-1.0, 1.0, 160)
    fig = go.Figure()
    for z in tier["sizes"]:
        fig.add_trace(go.Scatter(x=grid, y=_hit_ratio(tier, grid, float(z)), mode="lines", name=f"{float(z):g}M"))
    fig.update_layout(title=f"Hit ratios HR(δ,z) · {tier['name']}", xaxis_title="δ", yaxis_title="Hit ratio", yaxis_range=[0,1], template="trinity_dark")
    return fig


def implied_hit_ratio_figure(solution: dict[str, Any], cfg: dict[str, Any], tier_index: int) -> go.Figure:
    tier_sol = solution["tiers"][int(tier_index)]
    tier = _tier_cfg(cfg, int(tier_index))
    q = np.asarray(solution["qGrid"], dtype=float)
    fig = go.Figure()
    for j, z in enumerate(tier_sol["sizes"]):
        dash = SIZE_DASHES[j % len(SIZE_DASHES)]
        bid = np.asarray([row[j] for row in tier_sol["bid"]], dtype=float)
        ask = np.asarray([row[j] for row in tier_sol["ask"]], dtype=float)
        fig.add_trace(go.Scatter(
            x=q, y=_hit_ratio(tier, bid, z), mode="lines", name=f"{z:g}M bid",
            line=dict(color=BID_COLOR, dash=dash), legendgroup=f"size-{j}",
        ))
        fig.add_trace(go.Scatter(
            x=q, y=_hit_ratio(tier, ask, z), mode="lines", name=f"{z:g}M ask",
            line=dict(color=ASK_COLOR, dash=dash), legendgroup=f"size-{j}",
        ))
    fig.update_layout(title=f"Implied hit ratios vs inventory · {tier['name']}", xaxis_title="Inventory q [EUR M]", yaxis_title="Hit ratio", yaxis_range=[0,1], template="trinity_dark")
    return fig


def quote_inventory_figure(solution: dict[str, Any], cfg: dict[str, Any], tier_index: int) -> go.Figure:
    tier_sol = solution["tiers"][int(tier_index)]
    tier = _tier_cfg(cfg, int(tier_index))
    spread_pips = float(cfg["spreadPips"])
    q = np.asarray(solution["qGrid"], dtype=float)
    fig = go.Figure()
    for j, z in enumerate(tier_sol["sizes"]):
        dash = SIZE_DASHES[j % len(SIZE_DASHES)]
        bid = np.asarray([row[j] for row in tier_sol["bid"]], dtype=float)
        ask = np.asarray([row[j] for row in tier_sol["ask"]], dtype=float)
        fig.add_trace(go.Scatter(
            x=q, y=_quote_vs_mid_pips(bid, "bid", spread_pips), mode="lines", name=f"{z:g}M bid",
            line=dict(color=BID_COLOR, dash=dash), legendgroup=f"size-{j}",
        ))
        fig.add_trace(go.Scatter(
            x=q, y=_quote_vs_mid_pips(ask, "ask", spread_pips), mode="lines", name=f"{z:g}M ask",
            line=dict(color=ASK_COLOR, dash=dash), legendgroup=f"size-{j}",
        ))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title=f"Quotes vs inventory · {tier['name']}", xaxis_title="Inventory q [EUR M]", yaxis_title="Quote relative to mid [pips]", template="trinity_dark")
    return fig


def quote_surface_figure(solution: dict[str, Any], cfg: dict[str, Any], tier_index: int, side: str) -> go.Figure:
    tier_sol = solution["tiers"][int(tier_index)]
    tier = _tier_cfg(cfg, int(tier_index))
    spread_pips = float(cfg["spreadPips"])
    matrix = tier_sol[side]
    surface = []
    for j, _z in enumerate(tier_sol["sizes"]):
        deltas = np.asarray([row[j] for row in matrix], dtype=float)
        surface.append(_quote_vs_mid_pips(deltas, side, spread_pips))
    fig = go.Figure(go.Surface(
        x=solution["qGrid"], y=tier_sol["sizes"], z=np.asarray(surface),
        colorscale="Blues" if side == "bid" else "Reds", showscale=False, opacity=0.9,
    ))
    fig.update_layout(
        title=f"{side.capitalize()} quote surface · {tier['name']}", template="trinity_dark",
        scene=dict(xaxis_title="Inventory q", yaxis_title="Rung size [M]", zaxis_title="Quote vs mid [pips]"),
        margin=dict(l=0,r=0,t=45,b=0),
    )
    return fig


def ladder_figure(solution: dict[str, Any], cfg: dict[str, Any], tier_index: int, q_value: float) -> go.Figure:
    tier_sol = solution["tiers"][int(tier_index)]
    spread_pips = float(cfg["spreadPips"])
    qg = solution["qGrid"]
    sizes = tier_sol["sizes"]
    bid_d = [_interp_policy(qg, [row[j] for row in tier_sol["bid"]], q_value) for j in range(len(sizes))]
    ask_d = [_interp_policy(qg, [row[j] for row in tier_sol["ask"]], q_value) for j in range(len(sizes))]
    bid_ref, ask_ref = bid_d[0], ask_d[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=_volume_premium_pips(bid_ref, bid_d, spread_pips), mode="lines+markers",
        name=f"Bid q={q_value:g}", line=dict(color=BID_COLOR), marker=dict(color=BID_COLOR),
    ))
    fig.add_trace(go.Scatter(
        x=sizes, y=_volume_premium_pips(ask_ref, ask_d, spread_pips), mode="lines+markers",
        name=f"Ask q={q_value:g}", line=dict(color=ASK_COLOR), marker=dict(color=ASK_COLOR),
    ))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title=f"Volume premium · {tier_sol['name']} at q={q_value:g}M", xaxis_title="Trade size [EUR M]", yaxis_title="Premium vs smallest rung [pips]", template="trinity_dark")
    return fig


def flow_parameter_table(tier: dict[str, Any]) -> html.Table:
    rows=[]
    for z in tier["sizes"]:
        rows.append([
            f"{float(z):g}", f"{_activity(tier,float(z)):.6g}", f"{_delta50(tier,float(z)):.4f}",
            f"{float(tier['flow']['steepness']):.4g}", f"{float(_markout_pips(tier,float(z),1.0)):.3f}",
        ])
    return _simple_table(["Size [M]","A(z) [1/min]","δ50(z)","Steepness","Markout @1m [pips]"], rows)


def ladder_table(solution: dict[str, Any], cfg: dict[str, Any], tier_index: int, q_value: float) -> html.Table:
    tier_sol = solution["tiers"][int(tier_index)]
    spread_pips=float(cfg["spreadPips"])
    qg=solution["qGrid"]
    rows=[]
    for j,z in enumerate(tier_sol["sizes"]):
        bd=_interp_policy(qg,[row[j] for row in tier_sol["bid"]],q_value)
        ad=_interp_policy(qg,[row[j] for row in tier_sol["ask"]],q_value)
        b0=_interp_policy(qg,[row[0] for row in tier_sol["bid"]],q_value)
        a0=_interp_policy(qg,[row[0] for row in tier_sol["ask"]],q_value)
        rows.append([
            f"{z:g}", f"{bd:.5f}", f"{ad:.5f}",
            f"{float(_quote_vs_mid_pips(bd,'bid',spread_pips)):.3f}",
            f"{float(_quote_vs_mid_pips(ad,'ask',spread_pips)):.3f}",
            f"{float(_volume_premium_pips(b0,bd,spread_pips)):.3f}",
            f"{float(_volume_premium_pips(a0,ad,spread_pips)):.3f}",
        ])
    return _simple_table(["Size [M]","Bid δ","Ask δ","Bid vs mid [pips]","Ask vs mid [pips]","Bid vol prem [pips]","Ask vol prem [pips]"],rows)


def internalization_figure(cfg: dict[str, Any]) -> go.Figure:
    q=np.linspace(0,float(cfg["grid"]["maxAbs"]),300)
    p=cfg["internalization"]
    t=float(p["tau0"])+float(p["tau1"])*q+float(p["tau2"])*q*q
    fig=go.Figure(go.Scatter(x=q,y=t,mode="lines",name="t(|q|)"))
    fig.update_layout(title="Internalization time",xaxis_title="|q| [EUR M]",yaxis_title="Minutes",template="trinity_dark")
    return fig


def markout_figure(tier: dict[str, Any]) -> go.Figure:
    t=np.linspace(0,15,300)
    fig=go.Figure()
    for z in tier["sizes"]:
        fig.add_trace(go.Scatter(x=t,y=_markout_pips(tier,float(z),t),mode="lines",name=f"{float(z):g}M"))
    fig.update_layout(title=f"Saturating markout · {tier['name']}",xaxis_title="Time since trade [min]",yaxis_title="Adverse markout [pips]",template="trinity_dark")
    return fig


def convergence_figure(values, title: str, ytitle: str) -> go.Figure:
    fig=go.Figure()
    if values:
        y=np.maximum(np.asarray(values,dtype=float),1e-300)
        fig.add_trace(go.Scatter(x=np.arange(1,len(y)+1),y=y,mode="lines+markers"))
    fig.update_layout(title=title,xaxis_title="Howard iteration",yaxis_title=ytitle,yaxis_type="log",template="trinity_dark")
    return fig


def dark_pool_arrival_figure(cfg: dict[str, Any]) -> go.Figure:
    dp=cfg["darkPool"]
    posted=sorted(int(round(float(u))) for u in dp["postedSizes"])
    max_size=max(posted) if posted else 1
    k=np.arange(1,max_size+1)
    fig=go.Figure()
    if dp["distribution"]=="geometric":
        for label,p in [("bid",float(dp["pBid"])),("ask",float(dp["pAsk"]))]:
            y=p*(1-p)**(k-1)
            fig.add_trace(go.Bar(x=k,y=y,name=label,opacity=.72, marker_color=BID_COLOR if label == "bid" else ASK_COLOR))
    else:
        for label,mu,p0 in [("bid",float(dp["muBid"]),float(dp["p0Bid"])),("ask",float(dp["muAsk"]),float(dp["p0Ask"]))]:
            raw=np.asarray([math.exp(-mu+int(x)*math.log(mu)-math.lgamma(int(x)+1)) if mu>0 else 0 for x in k],dtype=float)
            z=raw.sum()
            y=(1-p0)*raw/z if z>1e-15 else np.zeros_like(raw)
            fig.add_trace(go.Bar(x=k,y=y,name=label,opacity=.72, marker_color=BID_COLOR if label == "bid" else ASK_COLOR))
    fig.update_layout(title="Dark-pool fill-size density",xaxis_title="Fill size",yaxis_title="Probability",barmode="group",template="trinity_dark")
    return fig


def dark_pool_full_fill_figure(cfg: dict[str, Any]) -> go.Figure:
    dp=cfg["darkPool"]
    posted=sorted(int(round(float(u))) for u in dp["postedSizes"])
    fig=go.Figure()
    if dp["distribution"]=="geometric":
        for label,p in [("bid",float(dp["pBid"])),("ask",float(dp["pAsk"]))]:
            fig.add_trace(go.Bar(x=posted,y=[(1-p)**(u-1) for u in posted],name=label,opacity=.72, marker_color=BID_COLOR if label == "bid" else ASK_COLOR))
    else:
        for label,mu,p0 in [("bid",float(dp["muBid"]),float(dp["p0Bid"])),("ask",float(dp["muAsk"]),float(dp["p0Ask"]))]:
            vals=[]
            for u in posted:
                ks=np.arange(1,u+1)
                raw=np.asarray([math.exp(-mu+int(x)*math.log(mu)-math.lgamma(int(x)+1)) if mu>0 else 0 for x in ks],dtype=float)
                vals.append((1-p0)*raw[-1]/raw.sum() if raw.sum()>1e-15 else 0.0)
            fig.add_trace(go.Bar(x=posted,y=vals,name=label,opacity=.72, marker_color=BID_COLOR if label == "bid" else ASK_COLOR))
    fig.update_layout(title="Dark-pool P(full fill)",xaxis_title="Posted size",yaxis_title="Probability",barmode="group",template="trinity_dark")
    return fig


def dark_pool_policy_figure(solution: dict[str, Any]) -> go.Figure:
    dp=solution.get("darkPool") if solution else None
    fig=go.Figure()
    if not dp:
        fig.add_annotation(text="Dark pool disabled",showarrow=False)
    else:
        q=dp["qGrid"]
        bid=[float(x) if bool(a) else 0.0 for x,a in zip(dp["bidSize"],dp["bidActive"])]
        ask=[float(x) if bool(a) else 0.0 for x,a in zip(dp["askSize"],dp["askActive"])]
        fig.add_trace(go.Bar(x=q,y=bid,name="Posted bid size",opacity=.72, marker_color=BID_COLOR))
        fig.add_trace(go.Bar(x=q,y=ask,name="Posted ask size",opacity=.72, marker_color=ASK_COLOR))
    fig.update_layout(title="Dark-pool optimal posted size vs inventory",xaxis_title="Inventory q [EUR M]",yaxis_title="Posted size [M]",barmode="overlay",template="trinity_dark")
    return fig


def dark_pool_table(solution: dict[str, Any]) -> html.Table | html.Div:
    dp=solution.get("darkPool") if solution else None
    if not dp:
        return html.Div("Dark pool disabled",className="muted-note")
    rows=[]
    for q,bs,ba,asz,aa in zip(dp["qGrid"],dp["bidSize"],dp["bidActive"],dp["askSize"],dp["askActive"]):
        rows.append([f"{q:g}", f"{bs:g}" if ba else "—", f"{asz:g}" if aa else "—"])
    return _simple_table(["Inventory","Bid posted size","Ask posted size"],rows)


DEFAULT = default_config()

sidebar = html.Aside([
    html.Div([
        html.Div("T2", className="brand-mark"),
        html.Div([html.Strong("Trinity 2.0"), html.Span("C++ Howard / Python / Dash")], className="brand-copy"),
    ], className="brand"),
    panel("Global", [
        select("Inventory grid", "grid-mode", [
            {"label": "Piecewise", "value": "piecewise"},
            {"label": "Uniform", "value": "uniform"},
        ], DEFAULT["grid"]["mode"]),
        html.Div([
            number_input("Max |q|", "qmax", DEFAULT["grid"]["maxAbs"], 0.5),
            number_input("Uniform step", "qstep", DEFAULT["grid"]["step"], 0.25),
            number_input("Fine half-width", "qfinehalf", DEFAULT["grid"]["fineHalfWidth"], 0.5),
            number_input("Fine step", "qfinestep", DEFAULT["grid"]["fineStep"], 0.05),
            number_input("Coarse step", "qcoarsestep", DEFAULT["grid"]["coarseStep"], 0.25),
        ], className="grid-2"),
    ], True),
    panel("Spot & risk", [
        html.Div([
            number_input("EURSEK spot", "spot", DEFAULT["spot"], 0.1),
            number_input("Spread [pips]", "spread", DEFAULT["spreadPips"], 1.0),
            number_input("Drift / min", "drift", DEFAULT["spotDrift"], 0.0001),
            number_input("Vol [pips / √min]", "sigma", DEFAULT["sigmaPips"], 1.0),
            number_input("Gamma / φ", "gamma", DEFAULT["gamma"], 0.01),
        ], className="grid-2"),
    ], True),
    panel("Internalization time", [
        html.Div([
            number_input("Tau 0 [min]", "tau0", DEFAULT["internalization"]["tau0"], 0.1),
            number_input("Tau 1", "tau1", DEFAULT["internalization"]["tau1"], 0.01),
            number_input("Tau 2", "tau2", DEFAULT["internalization"]["tau2"], 0.0005),
        ], className="grid-2"),
    ]),
    html.Div("Pricing tiers", className="section-title"),
    *[tier_panel(i, t, open_=(i == 0)) for i, t in enumerate(DEFAULT["tiers"])],
    panel("Dark pool", [
        checkbox("Enabled", "dp-enabled", DEFAULT["darkPool"]["enabled"]),
        select("Distribution", "dp-dist", [
            {"label": "Geometric", "value": "geometric"},
            {"label": "Zero-inflated Poisson", "value": "zip"},
        ], DEFAULT["darkPool"]["distribution"]),
        html.Div([
            number_input("Bid λ / min", "dp-lb", DEFAULT["darkPool"]["lambdaBid"], 0.1),
            number_input("Ask λ / min", "dp-la", DEFAULT["darkPool"]["lambdaAsk"], 0.1),
            number_input("Bid p", "dp-pb", DEFAULT["darkPool"]["pBid"], 0.05),
            number_input("Ask p", "dp-pa", DEFAULT["darkPool"]["pAsk"], 0.05),
            number_input("Bid μ", "dp-mub", DEFAULT["darkPool"]["muBid"], 0.1),
            number_input("Ask μ", "dp-mua", DEFAULT["darkPool"]["muAsk"], 0.1),
            number_input("Bid p0", "dp-p0b", DEFAULT["darkPool"]["p0Bid"], 0.05),
            number_input("Ask p0", "dp-p0a", DEFAULT["darkPool"]["p0Ask"], 0.05),
            number_input("Bid fee", "dp-fb", DEFAULT["darkPool"]["feeBid"], 0.1),
            number_input("Ask fee", "dp-fa", DEFAULT["darkPool"]["feeAsk"], 0.1),
        ], className="grid-2"),
        text_input("Posted sizes", "dp-sizes", ", ".join(str(x) for x in DEFAULT["darkPool"]["postedSizes"])),
        checkbox("Both sides simultaneously", "dp-both", DEFAULT["darkPool"]["allowBothSides"]),
    ]),
    html.Button("Solve Howard", id="solve", className="primary-button"),
], className="sidebar")

policy_tab = html.Div([
    html.Div([
        card("Converged", "m-converged"),
        card("Howard iterations", "m-iterations"),
        card("Average reward", "m-rho"),
        card("Bellman residual", "m-residual"),
    ], className="metrics-row"),
    dcc.Tabs(id="policy-subtabs", value="overview", children=[
        dcc.Tab(label="Overview", value="overview", children=html.Div([
            diagnostic_expander("Continuation value h(q)", graph("value-chart"), open_=True),
            diagnostic_expander("Internalization time", graph("internalization-chart")),
            diagnostic_expander("Bellman residual", graph("residual-chart")),
            diagnostic_expander("Value-function change", graph("value-change-chart")),
        ], className="subtab-inner diagnostics-stack"), className="dash-tab", selected_className="dash-tab dash-tab-selected"),
        dcc.Tab(label="Tier diagnostics", value="tiers", children=html.Div([
            html.Div([
                select("Tier", "tier-select", [], None),
            ], className="toolbar toolbar-wide"),
            diagnostic_expander("Tier parameters", [
                html.Div(id="flow-parameter-table", className="table-wrap"),
            ]),
            diagnostic_expander("Flow curves", graph("flow-chart")),
            diagnostic_expander("Hit ratios", graph("hit-ratio-chart")),
            diagnostic_expander("Implied hit ratios vs inventory", graph("implied-hit-chart")),
            diagnostic_expander("Markouts", graph("markout-chart")),
            diagnostic_expander("Quotes vs inventory", graph("quote-inventory-chart")),
            diagnostic_expander("Bid quote surface (3D)", graph("bid-surface-chart", height=500)),
            diagnostic_expander("Ask quote surface (3D)", graph("ask-surface-chart", height=500)),
            diagnostic_expander("Volume premium", [
                html.Div([select("Inventory level for ladder", "ladder-q", [], None)], className="toolbar"),
                graph("ladder-chart"),
                html.Div("Ladder at selected inventory", className="card-title table-section-title"),
                html.Div(id="ladder-table", className="table-wrap"),
            ]),
        ], className="subtab-inner tier-diagnostics-stack"), className="dash-tab", selected_className="dash-tab dash-tab-selected"),
        dcc.Tab(label="Dark pool", value="dark", children=html.Div([
            diagnostic_expander("Arrival size density", graph("dp-arrival-chart")),
            diagnostic_expander("P(full fill) by posted size", graph("dp-full-fill-chart")),
            diagnostic_expander("Dark pool policy", [
                graph("dp-policy-chart"),
                html.Div("Dark-pool policy", className="card-title table-section-title"),
                html.Div(id="dp-policy-table", className="table-wrap tall-table"),
            ]),
        ], className="subtab-inner diagnostics-stack"), className="dash-tab", selected_className="dash-tab dash-tab-selected"),
    ], className="tabs policy-tabs"),
], className="tab-inner")

mc_tab = html.Div([
    html.Div([
        number_input("Paths", "mc-paths", 1000, 500),
        number_input("Initial inventory [M]", "mc-q0", 0.0, 0.25),
        number_input("Random seed", "mc-seed", 12345, 1),
        html.Button("Run Monte Carlo", id="run-mc", className="secondary-button"),
    ], className="toolbar toolbar-wide"),
    html.Div([
        card("MC mean PnL [EUR]", "mc-mean"),
        card("Closed-form mean [EUR]", "cf-mean"),
        card("MC stdev [EUR]", "mc-std"),
        card("Closed-form stdev [EUR]", "cf-std"),
        card("5% PnL [EUR]", "mc-p5"),
        card("P(PnL < 0)", "mc-loss"),
    ], className="metrics-row metrics-six"),

    html.Div([
        diagnostic_expander("Trading-session PnL distribution", graph("pnl-chart", 500), open_=True),
        diagnostic_expander("Inventory paths · median and 95% interval", graph("inventory-chart", 440)),

        html.Div([
            select("Retained path", "path-select", [], None),
            select("Quote size", "path-size-select", [], None),
        ], className="toolbar toolbar-wide mc-path-toolbar"),

        diagnostic_expander("Inventory path · exact fill times", graph("path-chart", 400)),
        diagnostic_expander("Spot, quotes and fills", graph("spot-path-chart", 520)),
        diagnostic_expander("Fills by tier and side", [
            graph("fill-counts-chart", 360),
            html.Div("Fill tape", className="card-title table-section-title"),
            html.Div(id="fill-tape", className="table-wrap tall-table"),
        ]),
    ], className="diagnostics-stack mc-diagnostics-stack"),
], className="tab-inner")

frontier_tab = html.Div([
    html.Div([
        number_input("φ min", "frontier-min", 0.01, 0.01),
        number_input("φ max", "frontier-max", 0.5, 0.05),
        number_input("Points", "frontier-points", 9, 1),
        checkbox("Log-spaced φ", "frontier-log", True),
        html.Button("Compute frontier", id="run-frontier", className="secondary-button"),
    ], className="toolbar toolbar-wide"),
    html.Div([
        diagnostic_expander("Efficient frontier", graph("frontier-chart", 500), open_=True),
        diagnostic_expander("Risk-adjusted PnL vs φ", graph("ratio-chart", 440)),
        diagnostic_expander("Frontier table", html.Div(id="frontier-table", className="table-wrap tall-table")),
    ], className="diagnostics-stack frontier-diagnostics-stack"),
], className="tab-inner")

app.layout = html.Div([
    dcc.Store(id="config-store"),
    dcc.Store(id="solution-store"),
    dcc.Store(id="mc-store"),
    dcc.Store(id="stats-store"),
    dcc.Store(id="frontier-store"),
    sidebar,
    html.Main([
        html.Header([
            html.Div([
                html.H1("Trinity 2.0"),
                html.P("Inventory-aware FX ladder pricing · native C++ Howard engine"),
            ]),
            html.Div("Ready", id="status", className="status ready"),
        ], className="topbar"),
        dcc.Tabs(id="tabs", value="policy", children=[
            dcc.Tab(label="Policy", value="policy", children=policy_tab, className="dash-tab", selected_className="dash-tab dash-tab-selected"),
            dcc.Tab(label="Monte Carlo", value="mc", children=mc_tab, className="dash-tab", selected_className="dash-tab dash-tab-selected"),
            dcc.Tab(label="Efficient frontier", value="frontier", children=frontier_tab, className="dash-tab", selected_className="dash-tab dash-tab-selected"),
        ], className="tabs"),
    ], className="main"),
], className="app-shell")


CONFIG_FIELDS = [
    ("grid-mode", State("grid-mode", "value")), ("qmax", State("qmax", "value")), ("qstep", State("qstep", "value")),
    ("qfinehalf", State("qfinehalf", "value")), ("qfinestep", State("qfinestep", "value")), ("qcoarsestep", State("qcoarsestep", "value")),
    ("spot", State("spot", "value")), ("spread", State("spread", "value")), ("drift", State("drift", "value")),
    ("sigma", State("sigma", "value")), ("gamma", State("gamma", "value")), ("tau0", State("tau0", "value")),
    ("tau1", State("tau1", "value")), ("tau2", State("tau2", "value")),
]
for i in range(3):
    p = f"t{i}"
    for suffix in (
        "enabled", "name", "sizes", "A0", "theta", "beta", "steep", "shift", "vshift",
        "markout-enabled", "impact", "impact-beta", "impact-tau", "dmin", "dmax",
    ):
        CONFIG_FIELDS.append((f"{p}-{suffix}", State(f"{p}-{suffix}", "value")))
for component_id in (
    "dp-enabled", "dp-dist", "dp-lb", "dp-la", "dp-pb", "dp-pa", "dp-mub", "dp-mua",
    "dp-p0b", "dp-p0a", "dp-fb", "dp-fa", "dp-sizes", "dp-both",
):
    CONFIG_FIELDS.append((component_id, State(component_id, "value")))

CONFIG_KEYS = [key for key, _ in CONFIG_FIELDS]
CONFIG_STATES = [state for _, state in CONFIG_FIELDS]


@app.callback(
    Output("config-store", "data"), Output("solution-store", "data"), Output("status", "children"), Output("status", "className"),
    Input("solve", "n_clicks"), *CONFIG_STATES, prevent_initial_call=True,
)
def solve_model(_clicks, *values):
    try:
        cfg = build_config(dict(zip(CONFIG_KEYS, values, strict=True)))
        solution = ENGINES.get(cfg).solve()
        return cfg, solution, f"Solved · {solution['iterations']} Howard iterations", "status ready"
    except Exception as exc:
        return no_update, no_update, f"Solve failed · {exc}", "status error"


@app.callback(
    Output("m-converged", "children"), Output("m-iterations", "children"), Output("m-rho", "children"), Output("m-residual", "children"),
    Output("value-chart", "figure"), Output("residual-chart", "figure"), Output("value-change-chart", "figure"), Output("internalization-chart", "figure"),
    Output("tier-select", "options"), Output("tier-select", "value"), Output("ladder-q", "options"), Output("ladder-q", "value"),
    Input("solution-store", "data"), Input("config-store", "data"),
)
def render_solution(solution, cfg):
    if not solution or not cfg:
        return "—", "—", "—", "—", go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], None, [], None
    fig = go.Figure(go.Scatter(x=solution["qGrid"], y=solution["value"], mode="lines+markers", name="h(q)"))
    fig.update_layout(title="Continuation value h(q)", xaxis_title="Inventory q [EUR M]", yaxis_title="h(q)", template="trinity_dark")
    residual_fig = convergence_figure(solution.get("bellmanResidual", []), "Bellman residual", "max |R(q) - ρ|")
    value_change_fig = convergence_figure(solution.get("valueChange", []), "Value-function change", "max |Δh|")
    internal_fig = internalization_figure(cfg)
    opts = [{"label": t["name"], "value": i} for i, t in enumerate(solution["tiers"])]
    qopts = [{"label": f"{float(q):g}M", "value": float(q)} for q in solution["qGrid"]]
    q0 = min((float(q) for q in solution["qGrid"]), key=abs) if qopts else None
    return (
        "Yes" if solution["converged"] else "No", str(solution["iterations"]),
        f"{solution['averageReward']:.6g}", f"{solution['residual']:.3g}", fig, residual_fig, value_change_fig, internal_fig,
        opts, (0 if opts else None), qopts, q0,
    )


@app.callback(
    Output("flow-chart", "figure"), Output("hit-ratio-chart", "figure"), Output("implied-hit-chart", "figure"),
    Output("markout-chart", "figure"), Output("quote-inventory-chart", "figure"),
    Output("bid-surface-chart", "figure"), Output("ask-surface-chart", "figure"), Output("ladder-chart", "figure"),
    Output("flow-parameter-table", "children"), Output("ladder-table", "children"),
    Input("solution-store", "data"), Input("config-store", "data"), Input("tier-select", "value"), Input("ladder-q", "value"),
)
def render_tier_diagnostics(solution, cfg, tier_index, q_value):
    empty = (go.Figure(),) * 8 + (html.Div("Solve the model first"), html.Div("Solve the model first"))
    if not solution or not cfg or tier_index is None:
        return empty
    try:
        ti = int(tier_index)
        tier = _tier_cfg(cfg, ti)
        qv = float(q_value if q_value is not None else 0.0)
        return (
            flow_curve_figure(tier),
            hit_ratio_figure(tier),
            implied_hit_ratio_figure(solution, cfg, ti),
            markout_figure(tier),
            quote_inventory_figure(solution, cfg, ti),
            quote_surface_figure(solution, cfg, ti, "bid"),
            quote_surface_figure(solution, cfg, ti, "ask"),
            ladder_figure(solution, cfg, ti, qv),
            flow_parameter_table(tier),
            ladder_table(solution, cfg, ti, qv),
        )
    except Exception as exc:
        err = html.Div(f"Tier diagnostic failed: {exc}", className="error-note")
        return (go.Figure(),) * 8 + (err, err)


@app.callback(
    Output("dp-arrival-chart", "figure"), Output("dp-full-fill-chart", "figure"),
    Output("dp-policy-chart", "figure"), Output("dp-policy-table", "children"),
    Input("solution-store", "data"), Input("config-store", "data"),
)
def render_dark_pool_diagnostics(solution, cfg):
    if not solution or not cfg:
        return go.Figure(), go.Figure(), go.Figure(), html.Div("Solve the model first")
    if not cfg.get("darkPool", {}).get("enabled", False):
        disabled = go.Figure()
        disabled.add_annotation(text="Dark pool disabled", showarrow=False)
        disabled.update_layout(template="trinity_dark")
        return disabled, disabled, disabled, html.Div("Dark pool disabled", className="muted-note")
    return (
        dark_pool_arrival_figure(cfg), dark_pool_full_fill_figure(cfg),
        dark_pool_policy_figure(solution), dark_pool_table(solution),
    )


@app.callback(
    Output("path-size-select", "options"), Output("path-size-select", "value"),
    Input("solution-store", "data"),
)
def update_path_quote_sizes(solution):
    if not solution:
        return [], None
    sizes = sorted({float(z) for tier in solution.get("tiers", []) for z in tier.get("sizes", [])})
    opts = [{"label": f"{z:g}M", "value": z} for z in sizes]
    return opts, (sizes[0] if sizes else None)


@app.callback(
    Output("mc-store", "data"), Output("stats-store", "data"), Output("status", "children", allow_duplicate=True), Output("status", "className", allow_duplicate=True),
    Input("run-mc", "n_clicks"), State("config-store", "data"), State("mc-paths", "value"), State("mc-q0", "value"), State("mc-seed", "value"), prevent_initial_call=True,
)
def run_mc(_clicks, cfg, paths, q0, seed):
    if not cfg:
        return no_update, no_update, "Solve the model first", "status error"
    try:
        engine = ENGINES.get(cfg)
        mc = engine.simulate(SESSION_HORIZON, int(paths), float(q0), int(seed), retained_paths=8, sample_points=381)
        stats = engine.statistics(SESSION_HORIZON, float(q0))
        return mc, stats, "Monte Carlo complete", "status ready"
    except Exception as exc:
        return no_update, no_update, f"Monte Carlo failed · {exc}", "status error"


@app.callback(
    Output("mc-mean", "children"), Output("cf-mean", "children"), Output("mc-std", "children"), Output("cf-std", "children"),
    Output("mc-p5", "children"), Output("mc-loss", "children"), Output("pnl-chart", "figure"), Output("inventory-chart", "figure"),
    Output("path-select", "options"), Output("path-select", "value"),
    Input("mc-store", "data"), Input("stats-store", "data"),
)
def render_mc(mc, stats):
    if not mc or not stats:
        return "—", "—", "—", "—", "—", "—", go.Figure(), go.Figure(), [], None
    pnl = np.asarray(mc["pnlBase"], dtype=float)
    mc_mean = float(np.mean(pnl)); mc_std = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    p5 = float(np.percentile(pnl, 5)); loss = 100.0 * float(np.mean(pnl < 0.0))

    hist = go.Figure(go.Histogram(x=pnl, nbinsx=max(20, min(80, int(math.sqrt(len(pnl))) * 2)), name="PnL"))
    hist.add_vline(x=mc_mean, line_dash="dash", annotation_text="MC mean")
    hist.add_vline(x=float(stats["meanBase"]), line_dash="dot", annotation_text="Closed form")
    hist.update_layout(title="Trading-session PnL distribution", xaxis_title="PnL [EUR]", yaxis_title="Path count", template="trinity_dark")

    inv = go.Figure()
    times = session_timestamps(mc["times"])
    inv.add_trace(go.Scatter(x=times, y=mc["inventoryLower"], mode="lines", line=dict(width=0), showlegend=False))
    inv.add_trace(go.Scatter(x=times, y=mc["inventoryUpper"], mode="lines", line=dict(width=0), fill="tonexty", name="95% interval"))
    inv.add_trace(go.Scatter(x=times, y=mc["inventoryMedian"], mode="lines", name="Median inventory"))
    inv.add_hline(y=0, line_dash="dot")
    inv.update_layout(title="Inventory paths · median and 95% interval", yaxis_title="Inventory [EUR M]", template="trinity_dark")
    apply_session_clock_axis(inv)
    options = [{"label": f"Path {i + 1}", "value": i} for i in range(len(mc.get("samplePaths", [])))]
    return format_ccy(mc_mean), format_ccy(stats["meanBase"]), format_ccy(mc_std), format_ccy(stats["stdBase"]), format_ccy(p5), f"{loss:.1f}%", hist, inv, options, (0 if options else None)


@app.callback(
    Output("path-chart", "figure"), Output("spot-path-chart", "figure"),
    Output("fill-counts-chart", "figure"), Output("fill-tape", "children"),
    Input("mc-store", "data"), Input("path-select", "value"), Input("path-size-select", "value"),
    Input("solution-store", "data"), Input("config-store", "data"),
)
def render_path(mc, path_index, quote_size, solution, cfg):
    if not mc or path_index is None or not mc.get("samplePaths"):
        return go.Figure(), go.Figure(), go.Figure(), "No retained path"
    path = mc["samplePaths"][int(path_index)]

    # Exact event-time inventory path reconstructed from the fill tape.
    q0 = float(path["inventories"][0]) if path.get("inventories") else 0.0
    event_times = [0.0]
    event_inventory = [q0]
    for fill in path.get("fills", []):
        event_times.append(float(fill["time"]))
        event_inventory.append(float(fill["inventoryAfter"]))
    event_times.append(float(SESSION_HORIZON))
    event_inventory.append(event_inventory[-1])
    event_clock = session_timestamps(event_times)
    fig = go.Figure(go.Scatter(x=event_clock, y=event_inventory, mode="lines", line_shape="hv", name="Inventory"))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="Inventory path · exact fill times", yaxis_title="Inventory [EUR M]", template="trinity_dark")
    apply_session_clock_axis(fig)

    path_clock = session_timestamps(path["times"])
    spot_fig = go.Figure(go.Scatter(x=path_clock, y=path["spots"], mode="lines", name="Spot", line=dict(width=2.5), line_shape="hv"))
    if solution and cfg and quote_size is not None:
        spread = float(cfg["spreadPips"]) / 10_000.0
        inventories = np.asarray(path["inventories"], dtype=float)
        spots = np.asarray(path["spots"], dtype=float)
        q_grid = solution["qGrid"]
        for tier in solution.get("tiers", []):
            sizes = [float(z) for z in tier.get("sizes", [])]
            matches = [j for j,z in enumerate(sizes) if math.isclose(z, float(quote_size), rel_tol=0.0, abs_tol=1e-12)]
            if not matches:
                continue
            j = matches[0]
            bid_col = [row[j] for row in tier["bid"]]
            ask_col = [row[j] for row in tier["ask"]]
            bid_delta = np.asarray([_interp_policy(q_grid, bid_col, q) for q in inventories])
            ask_delta = np.asarray([_interp_policy(q_grid, ask_col, q) for q in inventories])
            bid_px = spots - spread * (0.5 - bid_delta)
            ask_px = spots + spread * (0.5 - ask_delta)
            spot_fig.add_trace(go.Scatter(x=path_clock, y=bid_px, mode="lines", name=f"{tier['name']} bid {float(quote_size):g}M", opacity=.72, line=dict(color=BID_COLOR), line_shape="hv"))
            spot_fig.add_trace(go.Scatter(x=path_clock, y=ask_px, mode="lines", line=dict(color=ASK_COLOR, dash="dash"), line_shape="hv", name=f"{tier['name']} ask {float(quote_size):g}M", opacity=.72))

    for side, symbol in (("bid", "triangle-up"), ("ask", "triangle-down")):
        fills_side = [f for f in path["fills"] if str(f["side"]).lower() == side]
        if fills_side:
            spot_fig.add_trace(go.Scatter(
                x=[session_timestamp(f["time"]) for f in fills_side], y=[f["price"] for f in fills_side], mode="markers",
                marker=dict(symbol=symbol, size=8, color=BID_COLOR if side == "bid" else ASK_COLOR), name=f"{side.capitalize()} fills",
                customdata=[[f["tier"], f["size"]] for f in fills_side],
                hovertemplate="%{x|%H:%M:%S}<br>px=%{y:.6f}<br>%{customdata[0]} · %{customdata[1]}M<extra></extra>",
            ))
    title_size = "" if quote_size is None else f" · {float(quote_size):g}M quotes"
    spot_fig.update_layout(title=f"Spot, quotes and fills{title_size}", yaxis_title="EURSEK", template="trinity_dark")
    apply_session_clock_axis(spot_fig)

    counts: dict[tuple[str,str], int] = {}
    tier_order=[]
    for f in path.get("fills", []):
        tier=str(f["tier"]); side=str(f["side"]).lower()
        if tier not in tier_order: tier_order.append(tier)
        counts[(tier,side)] = counts.get((tier,side),0)+1
    fill_counts = go.Figure()
    for side in ("bid","ask"):
        fill_counts.add_trace(go.Bar(x=tier_order,y=[counts.get((tier,side),0) for tier in tier_order],name=side.capitalize(), marker_color=BID_COLOR if side == "bid" else ASK_COLOR))
    fill_counts.update_layout(title="Fills by tier and side",xaxis_title="Tier",yaxis_title="Fill count",barmode="group",template="trinity_dark")

    header = html.Thead(html.Tr([html.Th(x) for x in ["Time", "Tier", "Side", "Size", "Price", "q before", "q after"]]))
    rows = []
    for f in path["fills"][:250]:
        rows.append(html.Tr([
            html.Td(session_clock_label(f["time"])), html.Td(f["tier"]), html.Td(f["side"]), html.Td(f"{f['size']:g}"),
            html.Td(f"{f['price']:.6f}"), html.Td(f"{f['inventoryBefore']:.2f}"), html.Td(f"{f['inventoryAfter']:.2f}"),
        ]))
    table = html.Table([header, html.Tbody(rows or [html.Tr(html.Td("No fills", colSpan=7))])], className="data-table")
    return fig, spot_fig, fill_counts, table


@app.callback(
    Output("frontier-store", "data"), Output("status", "children", allow_duplicate=True), Output("status", "className", allow_duplicate=True),
    Input("run-frontier", "n_clicks"), State("config-store", "data"), State("frontier-min", "value"), State("frontier-max", "value"),
    State("frontier-points", "value"), State("frontier-log", "value"), State("mc-q0", "value"), prevent_initial_call=True,
)
def run_frontier(_clicks, cfg, lo, hi, points, log_values, q0):
    if not cfg:
        return no_update, "Solve the model first", "status error"
    try:
        lo, hi, n = float(lo), float(hi), int(points)
        if not 0 < lo < hi or n < 3:
            raise ValueError("Require 0 < φ min < φ max and at least 3 points")
        gammas = np.geomspace(lo, hi, n) if checked(log_values) else np.linspace(lo, hi, n)
        current_gamma = float(cfg["gamma"])
        if lo <= current_gamma <= hi:
            gammas = np.append(gammas, current_gamma)
        gammas = np.unique(gammas).tolist()
        frontier = ENGINES.get(cfg).frontier(gammas, SESSION_HORIZON, float(q0))
        return frontier, "Efficient frontier complete", "status ready"
    except Exception as exc:
        return no_update, f"Frontier failed · {exc}", "status error"


@app.callback(Output("frontier-chart", "figure"), Output("ratio-chart", "figure"), Output("frontier-table", "children"), Input("frontier-store", "data"), State("config-store", "data"))
def render_frontier(frontier, cfg):
    if not frontier:
        return go.Figure(), go.Figure(), "Compute a frontier to populate this view."
    frontier = sorted(frontier, key=lambda x: x["std"])
    fig = go.Figure(go.Scatter(x=[x["std"] for x in frontier], y=[x["mean"] for x in frontier], mode="lines+markers", name="Frontier", customdata=[x["gamma"] for x in frontier], hovertemplate="φ=%{customdata:.4g}<br>std=%{x:,.0f}<br>mean=%{y:,.0f}<extra></extra>"))
    if cfg:
        current = min(frontier, key=lambda x: abs(x["gamma"] - cfg["gamma"]))
        fig.add_trace(go.Scatter(x=[current["std"]], y=[current["mean"]], mode="markers", marker=dict(symbol="star", size=16), name=f"Current φ={current['gamma']:.4g}"))
    fig.update_layout(title="Efficient frontier", xaxis_title="Closed-form PnL stdev [EUR]", yaxis_title="Expected PnL [EUR]", template="trinity_dark")

    ordered = sorted(frontier, key=lambda x: x["gamma"])
    ratio = go.Figure(go.Scatter(x=[x["gamma"] for x in ordered], y=[x["ratio"] for x in ordered], mode="lines+markers", name="Mean / stdev"))
    ratio.update_layout(title="Risk-adjusted PnL vs φ", xaxis_title="φ (risk aversion)", yaxis_title="Expected PnL / stdev", template="trinity_dark")

    header = html.Thead(html.Tr([html.Th(x) for x in ["φ", "Expected PnL", "Stdev", "Mean / stdev", "Howard iters"]]))
    rows = [html.Tr([html.Td(f"{x['gamma']:.5g}"), html.Td(format_ccy(x["mean"])), html.Td(format_ccy(x["std"])), html.Td(f"{x['ratio']:.4f}"), html.Td(str(x["iterations"]))]) for x in ordered]
    return fig, ratio, html.Table([header, html.Tbody(rows)], className="data-table")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
