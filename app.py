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

APP_TITLE = "FX Ladder Pricer"
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
    """Use a real datetime axis whose labels remain useful while zooming.

    A fixed ``tickvals`` array looks tidy at the full-session level but leaves
    the axis with no labels after zooming between those preselected ticks.
    Let Plotly choose ticks dynamically and only control how those timestamps
    are formatted at different zoom levels.
    """
    fig.update_xaxes(
        type="date",
        title_text=title,
        tickmode="auto",
        nticks=12,
        tickformatstops=[
            # Very tight inspection: include milliseconds.
            dict(dtickrange=[None, 1_000], value="%H:%M:%S.%L"),
            # Intraday quote/fill inspection: always show seconds.
            dict(dtickrange=[1_000, 3_600_000], value="%H:%M:%S"),
            # Full-session / long-horizon view: minutes are sufficient.
            dict(dtickrange=[3_600_000, None], value="%H:%M"),
        ],
        hoverformat="%H:%M:%S.%L",
        automargin=True,
    )
    return fig

# -----------------------------------------------------------------------------
# Unified dark Plotly theme
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


def graph(component_id: str, height: int = 420, config: dict[str, Any] | None = None):
    """Responsive Plotly graph with a stable card height.

    Plot styling is controlled by the global Plotly template.  Keeping the
    container sizing here avoids CSS rules that reach into Plotly's internal
    SVG layers.  ``config`` can override the shared Plotly interaction
    settings for graphs that need specialized navigation behavior.
    """
    graph_config = dict(GRAPH_CONFIG)
    if config:
        graph_config.update(config)
    return dcc.Graph(
        id=component_id,
        config=graph_config,
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
            number_input("A0 / min", f"{p}-A0", tier["flow"]["A0"], "any"),
            number_input("Theta", f"{p}-theta", tier["flow"]["theta"], "any"),
            number_input("Beta", f"{p}-beta", tier["flow"]["beta"], "any"),
            number_input("Steepness", f"{p}-steep", tier["flow"]["steepness"], "any"),
            number_input("Shift", f"{p}-shift", tier["flow"]["shift"], "any"),
            number_input("Volume shift", f"{p}-vshift", tier["flow"]["volumeShift"], "any"),
        ], className="grid-2"),
        html.Div("Trading economics", className="subhead"),
        html.Div([
            number_input("Fee [pips]", f"{p}-fee", tier.get("feePips", 0.0), "any"),
        ], className="grid-2"),
        html.Div("Markout", className="subhead"),
        checkbox("Enabled", f"{p}-markout-enabled", tier["useMarkout"]),
        html.Div([
            number_input("Scale [pips]", f"{p}-impact", tier["markout"]["impactScalePips"], "any"),
            number_input("Size exponent", f"{p}-impact-beta", tier["markout"]["sizeExponent"], "any"),
            number_input("Tau [min]", f"{p}-impact-tau", tier["markout"]["tauMinutes"], "any"),
            number_input("Delta min", f"{p}-dmin", tier["deltaMin"], "any"),
            number_input("Delta max", f"{p}-dmax", tier["deltaMax"], "any"),
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
    fig.update_layout(title=f"Won-trade intensity λ_RFQ(z) × p_win(δ,z) · {tier['name']}", xaxis_title="δ", yaxis_title="Won trades [1/min]", template="trinity_dark")
    return fig


def exogenous_arrival_figure(tier: dict[str, Any]) -> go.Figure:
    sizes = np.asarray(tier["sizes"], dtype=float)
    rates = np.asarray([_activity(tier, float(z)) for z in sizes], dtype=float)
    fig = go.Figure(go.Scatter(x=sizes, y=rates, mode="lines+markers", name="RFQ arrivals"))
    fig.update_layout(
        title=f"Exogenous RFQ arrival intensity λ_RFQ(z) · {tier['name']}",
        xaxis_title="RFQ size [EUR M]",
        yaxis_title="RFQs [1/min]",
        template="trinity_dark",
    )
    return fig


def hit_ratio_figure(tier: dict[str, Any]) -> go.Figure:
    grid = np.linspace(-1.0, 1.0, 160)
    fig = go.Figure()
    for z in tier["sizes"]:
        fig.add_trace(go.Scatter(x=grid, y=_hit_ratio(tier, grid, float(z)), mode="lines", name=f"{float(z):g}M"))
    fig.update_layout(title=f"Win probabilities p_win(δ,z) · {tier['name']}", xaxis_title="δ", yaxis_title="Win probability", yaxis_range=[0,1], template="trinity_dark")
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
    fig.update_layout(title=f"Implied win probabilities vs inventory · {tier['name']}", xaxis_title="Inventory q [EUR M]", yaxis_title="Win probability", yaxis_range=[0,1], template="trinity_dark")
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
            f"{float(tier['flow']['steepness']):.4g}", f"{float(tier.get('feePips', 0.0)):.3f}",
            f"{float(_markout_pips(tier,float(z),1.0)):.3f}",
        ])
    return _simple_table(["Size [M]","λ_RFQ(z) [1/min]","δ50(z)","Steepness","Fee [pips]","Markout @1m [pips]"], rows)


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
    fig.update_layout(title=f"RFQ markout · {tier['name']}",xaxis_title="Time since trade [min]",yaxis_title="Adverse markout [pips]",template="trinity_dark")
    return fig


def convergence_figure(values, title: str, ytitle: str) -> go.Figure:
    fig=go.Figure()
    if values:
        y=np.maximum(np.asarray(values,dtype=float),1e-300)
        fig.add_trace(go.Scatter(x=np.arange(1,len(y)+1),y=y,mode="lines+markers"))
    fig.update_layout(title=title,xaxis_title="Howard iteration",yaxis_title=ytitle,yaxis_type="log",template="trinity_dark")
    return fig


def _zip_positive_pmf(mu: float, max_k: int) -> np.ndarray:
    """Zero-truncated Poisson probabilities for incoming positive sizes 1..max_k."""
    ks=np.arange(1,max_k+1)
    raw=np.asarray([math.exp(-mu+int(x)*math.log(mu)-math.lgamma(int(x)+1)) if mu>0 else 0.0 for x in ks],dtype=float)
    norm=1.0-math.exp(-mu) if mu>0 else 0.0
    return raw/norm if norm>1e-15 else np.zeros_like(raw)


def dark_pool_arrival_figure(cfg: dict[str, Any]) -> go.Figure:
    dp=cfg["darkPool"]
    posted=sorted(int(round(float(u))) for u in dp["postedSizes"])
    max_size=max((posted[-1] if posted else 1)+5,10)
    k=np.arange(0,max_size+1)
    fig=go.Figure()
    mu=float(dp["mu"]); p0=float(dp["p0"])
    y=np.zeros_like(k,dtype=float)
    y[0]=p0
    y[1:]=(1.0-p0)*_zip_positive_pmf(mu,max_size)
    fig.add_trace(go.Bar(x=k,y=y,name="bid / ask",opacity=.72))
    fig.update_layout(title="Dark-pool incoming-order size distribution",xaxis_title="Incoming order size",yaxis_title="Probability",barmode="group",template="trinity_dark")
    return fig


def dark_pool_full_fill_figure(cfg: dict[str, Any]) -> go.Figure:
    dp=cfg["darkPool"]
    posted=sorted(int(round(float(u))) for u in dp["postedSizes"])
    fig=go.Figure()
    mu=float(dp["mu"]); p0=float(dp["p0"]); vals=[]
    positive_norm=1.0-math.exp(-mu) if mu>0 else 0.0
    for u in posted:
        if positive_norm<=1e-15:
            vals.append(0.0)
            continue
        below=sum(math.exp(-mu+k*math.log(mu)-math.lgamma(k+1)) for k in range(1,u))
        tail=max(0.0,1.0-below/positive_norm)
        vals.append((1.0-p0)*tail)
    fig.add_trace(go.Bar(x=posted,y=vals,name="bid / ask",opacity=.72))
    fig.update_layout(title="Dark-pool P(full fill = incoming size ≥ posted size)",xaxis_title="Posted size",yaxis_title="Probability per arrival",barmode="group",template="trinity_dark")
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


def _ecn_arrival_rate(cfg: dict[str, Any], delta_pips):
    flow = cfg.get("passiveEcn", {}).get("flow", {})
    A = float(flow.get("A", 0.0))
    k = float(flow.get("k", 1.0))
    return A * np.exp(np.clip(-k * np.asarray(delta_pips, dtype=float), -745.0, 709.0))


def _ecn_step_line(q: np.ndarray, values: np.ndarray, active: np.ndarray, side: str) -> tuple[list[float | None], list[float | None], np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    values = np.asarray(values, dtype=float)
    active = np.asarray(active, dtype=bool)
    mask = active & np.isfinite(values)
    x_mark = q[mask]
    y_mark = values[mask]
    if x_mark.size == 0:
        return [], [], x_mark, y_mark

    if x_mark.size == 1:
        step = 1.0
    else:
        step = float(np.median(np.diff(x_mark)))
        if not np.isfinite(step) or step <= 0.0:
            step = 1.0
    half = 0.5 * step

    left_edges = x_mark - half
    right_edges = x_mark + half

    x_line: list[float | None] = [float(left_edges[0]), float(right_edges[0])]
    y_line: list[float | None] = [float(y_mark[0]), float(y_mark[0])]

    for i in range(1, x_mark.size):
        li = float(left_edges[i])
        ri = float(right_edges[i])
        yi_prev = float(y_mark[i - 1])
        yi = float(y_mark[i])
        prev_right = float(right_edges[i - 1])
        if abs(li - prev_right) <= 1e-12:
            x_line.extend([li, ri])
            y_line.extend([yi, yi])
        else:
            x_line.extend([None, li, ri])
            y_line.extend([None, yi, yi])

    return x_line, y_line, x_mark, y_mark


def _add_ecn_inventory_trace(fig: go.Figure, q: np.ndarray, values: np.ndarray, active: np.ndarray,
                             name: str, color: str, side: str) -> None:
    x_line, y_line, x_mark, y_mark = _ecn_step_line(q, values, active, side)
    if len(x_line) == 0:
        return
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines", name=name,
        line=dict(color=color), legendgroup=name,
    ))
    fig.add_trace(go.Scatter(
        x=x_mark, y=y_mark, mode="markers", name=name,
        marker=dict(color=color), legendgroup=name, showlegend=False,
        hovertemplate="Inventory %{x:g}<br>Value %{y:g}<extra></extra>",
    ))


def passive_ecn_parameter_table(cfg: dict[str, Any]) -> html.Table:
    e = cfg.get("passiveEcn", {})
    flow = e.get("flow", {})
    z = float(e.get("quoteSize", 1.0))
    rows = [[
        f"{z:g}",
        f"{float(flow.get('A', 0.0)):.6g}",
        f"{float(flow.get('k', 0.0)):.4g}",
        f"{float(e.get('makerFeePips', 0.0)):.4f}",
        f"{float(e.get('minDistancePips', 0.0)):g}",
        f"{float(e.get('maxDistancePips', 0.0)):g}",
        "0.5",
    ]]
    return _simple_table(
        ["Quote size [M]", "A = ECN arrivals/side [trades/min]", "k [1/pip]", "Maker fee [pips]",
         "Min distance [pips]", "Max distance [pips]", "Grid step [pips]"],
        rows,
    )


def passive_ecn_fill_figure(cfg: dict[str, Any]) -> go.Figure:
    e = cfg.get("passiveEcn", {})
    z = float(e.get("quoteSize", 1.0))
    lo = float(e.get("minDistancePips", 0.0))
    hi = float(e.get("maxDistancePips", 20.0))
    grid = np.linspace(lo, hi, 240)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=_ecn_arrival_rate(cfg, grid), mode="lines", name=f"{z:g}M ECN"))
    deltas = np.asarray(e.get("deltas", []), dtype=float)
    if deltas.size:
        fig.add_trace(go.Scatter(x=deltas, y=_ecn_arrival_rate(cfg, deltas), mode="markers", name="0.5-pip grid"))
    fig.update_layout(title="Implied ECN fill intensity λ_fill(δ)=A exp(−kδ)",
                      xaxis_title="Quote distance from mid δ [pips]", yaxis_title="Fill intensity [trades/min]",
                      template="trinity_dark")
    fig.update_xaxes(range=[lo, hi])
    return fig


def passive_ecn_hit_ratio_figure(cfg: dict[str, Any]) -> go.Figure:
    e = cfg.get("passiveEcn", {})
    flow = e.get("flow", {})
    k = float(flow.get("k", 1.0))
    z = float(e.get("quoteSize", 1.0))
    lo = float(e.get("minDistancePips", 0.0))
    hi = float(e.get("maxDistancePips", 20.0))
    grid = np.linspace(lo, hi, 240)
    relative = np.exp(np.clip(-k * grid, -745.0, 709.0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=relative, mode="lines", name=f"{z:g}M ECN"))
    deltas = np.asarray(e.get("deltas", []), dtype=float)
    if deltas.size:
        fig.add_trace(go.Scatter(x=deltas, y=np.exp(np.clip(-k * deltas, -745.0, 709.0)), mode="markers", name="0.5-pip grid"))
    fig.update_layout(title="ECN reach probability P(D ≥ δ)=exp(−kδ)", xaxis_title="Quote distance from mid δ [pips]",
                      yaxis_title="Reach probability", template="trinity_dark")
    fig.update_xaxes(range=[lo, hi])
    return fig


def passive_ecn_implied_hit_ratio_figure(solution: dict[str, Any], cfg: dict[str, Any]) -> go.Figure:
    e = solution.get("passiveEcn") if solution else None
    fig = go.Figure()
    if not e:
        fig.add_annotation(text="Passive ECN disabled", showarrow=False)
    else:
        q = np.asarray(e["qGrid"], dtype=float)
        bid_delta = np.asarray(e["bidDelta"], dtype=float)
        ask_delta = np.asarray(e["askDelta"], dtype=float)
        bid_active = np.asarray(e["bidActive"], dtype=bool)
        ask_active = np.asarray(e["askActive"], dtype=bool)
        bid_lam = np.where(bid_active, _ecn_arrival_rate(cfg, bid_delta), np.nan)
        ask_lam = np.where(ask_active, _ecn_arrival_rate(cfg, ask_delta), np.nan)
        _add_ecn_inventory_trace(fig, q, bid_lam, bid_active, "Bid hedge", BID_COLOR, "bid")
        _add_ecn_inventory_trace(fig, q, ask_lam, ask_active, "Ask hedge", ASK_COLOR, "ask")
    fig.update_layout(title="Fill intensity vs inventory · Passive ECN", xaxis_title="Inventory q [EUR M]", yaxis_title="Trade intensity [trades/min]", template="trinity_dark")
    return fig


def passive_ecn_quote_inventory_figure(solution: dict[str, Any], cfg: dict[str, Any]) -> go.Figure:
    e = solution.get("passiveEcn") if solution else None
    fig = go.Figure()
    if not e:
        fig.add_annotation(text="Passive ECN disabled", showarrow=False)
    else:
        q = np.asarray(e["qGrid"], dtype=float)
        bid_delta = np.asarray(e["bidDelta"], dtype=float)
        ask_delta = np.asarray(e["askDelta"], dtype=float)
        bid_active = np.asarray(e["bidActive"], dtype=bool)
        ask_active = np.asarray(e["askActive"], dtype=bool)
        bid_px = np.where(bid_active, -bid_delta, np.nan)
        ask_px = np.where(ask_active, ask_delta, np.nan)
        _add_ecn_inventory_trace(fig, q, bid_px, bid_active, "Bid hedge", BID_COLOR, "bid")
        _add_ecn_inventory_trace(fig, q, ask_px, ask_active, "Ask hedge", ASK_COLOR, "ask")
        fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="Quotes vs inventory · Passive ECN", xaxis_title="Inventory q [EUR M]", yaxis_title="Quote relative to mid [pips]", template="trinity_dark")
    return fig


def passive_ecn_policy_figure(solution: dict[str, Any]) -> go.Figure:
    e=solution.get("passiveEcn") if solution else None
    fig=go.Figure()
    if not e:
        fig.add_annotation(text="Passive ECN disabled", showarrow=False)
    else:
        q = np.asarray(e["qGrid"], dtype=float)
        bid = np.asarray([float(x) if bool(a) else np.nan for x, a in zip(e["bidDelta"], e["bidActive"])], dtype=float)
        ask = np.asarray([float(x) if bool(a) else np.nan for x, a in zip(e["askDelta"], e["askActive"])], dtype=float)
        bid_active = np.asarray(e["bidActive"], dtype=bool)
        ask_active = np.asarray(e["askActive"], dtype=bool)
        _add_ecn_inventory_trace(fig, q, bid, bid_active, "Bid hedge", BID_COLOR, "bid")
        _add_ecn_inventory_trace(fig, q, ask, ask_active, "Ask hedge", ASK_COLOR, "ask")
    fig.update_layout(title="Optimal passive ECN distance vs inventory",xaxis_title="Inventory q [EUR M]",yaxis_title="Distance from mid [pips]",template="trinity_dark")
    return fig

def passive_ecn_table(solution: dict[str, Any]) -> html.Table | html.Div:
    e=solution.get("passiveEcn") if solution else None
    if not e:
        return html.Div("Passive ECN disabled",className="muted-note")
    rows=[]
    for q,bd,ba,ad,aa in zip(e["qGrid"],e["bidDelta"],e["bidActive"],e["askDelta"],e["askActive"]):
        rows.append([f"{q:g}", f"{bd:g}" if ba else "OFF", f"{ad:g}" if aa else "OFF"])
    return _simple_table(["Inventory", "Bid distance [pips]", "Ask distance [pips]"], rows)


DEFAULT = default_config()

sidebar = html.Aside([
    html.Div([
        html.Div("T2", className="brand-mark"),
        html.Div([html.Strong("FX Ladder Pricer")], className="brand-copy"),
    ], className="brand"),
    panel("Global", [
        html.Div([
            number_input("Max |q| [M]", "qmax", DEFAULT["grid"]["maxAbs"], 1.0),
        ], className="grid-2"),
        html.Div("Inventory grid is fixed to uniform 1M spacing.", className="muted-note"),
    ], True),
    panel("Spot & risk", [
        html.Div([
            number_input("EURSEK spot", "spot", DEFAULT["spot"], "any"),
            number_input("Spread [pips]", "spread", DEFAULT["spreadPips"], "any"),
            number_input("Drift / min", "drift", DEFAULT["spotDrift"], "any"),
            number_input("Vol [pips / √min]", "sigma", DEFAULT["sigmaPips"], "any"),
            number_input("Gamma / φ", "gamma", DEFAULT["gamma"], "any"),
        ], className="grid-2"),
    ], True),
    panel("Internalization time", [
        html.Div([
            number_input("Tau 0 [min]", "tau0", DEFAULT["internalization"]["tau0"], "any"),
            number_input("Tau 1", "tau1", DEFAULT["internalization"]["tau1"], "any"),
            number_input("Tau 2", "tau2", DEFAULT["internalization"]["tau2"], "any"),
        ], className="grid-2"),
    ]),
    html.Div("Pricing tiers", className="section-title"),
    *[tier_panel(i, t, open_=(i == 0)) for i, t in enumerate(DEFAULT["tiers"])],
    panel("Dark pool", [
        checkbox("Enabled", "dp-enabled", DEFAULT["darkPool"]["enabled"]),
        html.Div("Zero-inflated Poisson fill-size model", className="muted-note"),
        html.Div([
            number_input("λ / side / min", "dp-lambda", DEFAULT["darkPool"]["lambda"], "any"),
            number_input("μ", "dp-mu", DEFAULT["darkPool"]["mu"], "any"),
            number_input("p0", "dp-p0", DEFAULT["darkPool"]["p0"], "any"),
            number_input("Broker fee [pips]", "dp-fee", DEFAULT["darkPool"]["feePips"], "any"),
        ], className="grid-2"),
        html.Div("Hedge-only: only the inventory-reducing side is posted and fills may not cross through flat.", className="muted-note"),
        text_input("Posted sizes", "dp-sizes", ", ".join(str(x) for x in DEFAULT["darkPool"]["postedSizes"])),
    ]),
    panel("ECN", [
        checkbox("Enabled", "ecn-enabled", DEFAULT["passiveEcn"]["enabled"]),
        html.Div([
            number_input("A arrivals / side [trades/min]", "ecn-A", DEFAULT["passiveEcn"]["flow"]["A"], "any"),
            number_input("k [1/pip]", "ecn-k", DEFAULT["passiveEcn"]["flow"]["k"], "any"),
        ], className="grid-2"),
        html.Div([
            number_input("Min distance from mid [pips]", "ecn-dmin", DEFAULT["passiveEcn"]["minDistancePips"], 0.5),
            number_input("Max distance from mid [pips]", "ecn-dmax", DEFAULT["passiveEcn"]["maxDistancePips"], 0.5),
        ], className="grid-2"),
        html.Div([
            number_input("Quote size [M]", "ecn-size", DEFAULT["passiveEcn"]["quoteSize"], "any"),
            number_input("Maker fee [pips]", "ecn-fee", DEFAULT["passiveEcn"]["makerFeePips"], "any"),
        ], className="grid-2"),
    ]),
    html.Button("Calibrate", id="solve", className="primary-button"),
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
            diagnostic_expander("Exogenous RFQ arrival intensity", graph("rfq-arrival-chart")),
            diagnostic_expander("Won-trade intensity", graph("flow-chart")),
            diagnostic_expander("Win probabilities", graph("hit-ratio-chart")),
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
        dcc.Tab(label="Passive ECN", value="ecn", children=html.Div([
            diagnostic_expander("ECN parameters", [
                html.Div(id="ecn-parameter-table", className="table-wrap"),
            ]),
            diagnostic_expander("Flow curves", graph("ecn-fill-chart")),
            diagnostic_expander("Relative intensity", graph("ecn-hit-ratio-chart")),
            diagnostic_expander("Fill intensity vs inventory", graph("ecn-implied-hit-chart")),
            diagnostic_expander("Quotes vs inventory", graph("ecn-quote-inventory-chart")),
            diagnostic_expander("Optimal quote delta", [
                graph("ecn-policy-chart"),
                html.Div("Passive ECN policy", className="card-title table-section-title"),
                html.Div(id="ecn-policy-table", className="table-wrap tall-table"),
            ]),
        ], className="subtab-inner tier-diagnostics-stack"), className="dash-tab", selected_className="dash-tab dash-tab-selected"),
    ], className="tabs policy-tabs"),
], className="tab-inner")

mc_tab = html.Div([
    html.Div([
        number_input("Paths", "mc-paths", 1000, 1),
        number_input("Initial inventory [M]", "mc-q0", 0.0, "any"),
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
        diagnostic_expander("Internalization time to zero", graph("mc-internalization-time-chart", 440)),

        html.Div([
            select("Retained path", "path-select", [], None),
            select("Quote size", "path-size-select", [], None),
        ], className="toolbar toolbar-wide mc-path-toolbar"),

        diagnostic_expander("Mark-to-market PnL", graph("path-pnl-chart", 400)),
        diagnostic_expander("Inventory path · exact fill times", graph("path-chart", 400)),
        diagnostic_expander(
            "Spot, quotes and fills",
            graph(
                "spot-path-chart",
                520,
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "doubleClick": "reset+autosize",
                },
            ),
        ),
        diagnostic_expander("Fills by tier and side", [
            graph("fill-counts-chart", 360),
            html.Div("Fill tape", className="card-title table-section-title"),
            html.Div(id="fill-tape", className="table-wrap tall-table"),
        ]),
    ], className="diagnostics-stack mc-diagnostics-stack"),
], className="tab-inner")

frontier_tab = html.Div([
    html.Div([
        number_input("φ min", "frontier-min", 0.01, "any"),
        number_input("φ max", "frontier-max", 0.5, "any"),
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
                html.H1("FX Ladder Pricer"),
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
    ("qmax", State("qmax", "value")),
    ("spot", State("spot", "value")), ("spread", State("spread", "value")), ("drift", State("drift", "value")),
    ("sigma", State("sigma", "value")), ("gamma", State("gamma", "value")), ("tau0", State("tau0", "value")),
    ("tau1", State("tau1", "value")), ("tau2", State("tau2", "value")),
]
for i in range(3):
    p = f"t{i}"
    for suffix in (
        "enabled", "name", "sizes", "A0", "theta", "beta", "steep", "shift", "vshift",
        "fee", "markout-enabled", "impact", "impact-beta", "impact-tau", "dmin", "dmax",
    ):
        CONFIG_FIELDS.append((f"{p}-{suffix}", State(f"{p}-{suffix}", "value")))
for component_id in (
    "dp-enabled", "dp-lambda", "dp-mu", "dp-p0", "dp-fee", "dp-sizes",
):
    CONFIG_FIELDS.append((component_id, State(component_id, "value")))
for component_id in (
    "ecn-enabled", "ecn-A", "ecn-k", "ecn-dmin", "ecn-dmax", "ecn-size", "ecn-fee",
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
    Output("rfq-arrival-chart", "figure"), Output("flow-chart", "figure"), Output("hit-ratio-chart", "figure"), Output("implied-hit-chart", "figure"),
    Output("markout-chart", "figure"), Output("quote-inventory-chart", "figure"),
    Output("bid-surface-chart", "figure"), Output("ask-surface-chart", "figure"), Output("ladder-chart", "figure"),
    Output("flow-parameter-table", "children"), Output("ladder-table", "children"),
    Input("solution-store", "data"), Input("config-store", "data"), Input("tier-select", "value"), Input("ladder-q", "value"),
)
def render_tier_diagnostics(solution, cfg, tier_index, q_value):
    empty = (go.Figure(),) * 9 + (html.Div("Solve the model first"), html.Div("Solve the model first"))
    if not solution or not cfg or tier_index is None:
        return empty
    try:
        ti = int(tier_index)
        tier = _tier_cfg(cfg, ti)
        qv = float(q_value if q_value is not None else 0.0)
        return (
            exogenous_arrival_figure(tier),
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
        return (go.Figure(),) * 9 + (err, err)


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


def expected_internalization_times(solution: dict[str, Any] | None, cfg: dict[str, Any] | None) -> dict[str, list[float]]:
    """Expected first-passage time to q=0 under the solved fixed policy.

    This builds the inventory CTMC from exactly the same policy-dependent event
    intensities used by the Monte Carlo engine, then solves -Q_T tau = 1 with
    q=0 absorbing.  The production UI uses a uniform 1M inventory grid, so the
    configured trade sizes must land back on that grid for this exact curve.
    """
    if not solution or not cfg:
        return {"qGrid": [], "minutes": []}
    q_grid = np.asarray(solution.get("qGrid", []), dtype=float)
    if q_grid.size == 0:
        return {"qGrid": [], "minutes": []}

    def state_index(x: float) -> int:
        j = int(np.argmin(np.abs(q_grid - x)))
        if abs(float(q_grid[j]) - float(x)) > 1e-8:
            raise ValueError(
                "Internalization-time curve requires fill sizes aligned with the 1M inventory grid"
            )
        return j

    zero = state_index(0.0)
    n = len(q_grid)
    Q = np.zeros((n, n), dtype=float)

    enabled_tiers = {t["name"]: t for t in cfg.get("tiers", []) if t.get("enabled", False)}

    def logistic_rate(flow: dict[str, Any], delta: float, size: float) -> float:
        scale = float(flow["A0"]) * size ** (-float(flow["theta"]) - float(flow["beta"]) * size)
        center = float(flow["shift"]) - float(flow["volumeShift"]) * (size - 1.0)
        y = float(flow["steepness"]) * (delta - center)
        if y >= 0.0:
            e = math.exp(-y) if y < 745.0 else 0.0
            hit = 1.0 / (1.0 + e)
        else:
            e = math.exp(y) if y > -745.0 else 0.0
            hit = e / (1.0 + e)
        return scale * hit

    def add_transition(i: int, q2: float, rate: float) -> None:
        if rate <= 0.0:
            return
        j = state_index(q2)
        if j == i:
            return
        Q[i, j] += rate
        Q[i, i] -= rate

    # Customer tiers.
    for tier_policy in solution.get("tiers", []):
        tier_cfg = enabled_tiers.get(tier_policy.get("name"))
        if not tier_cfg:
            continue
        sizes = [float(z) for z in tier_policy.get("sizes", [])]
        flow = tier_cfg["flow"]
        for i, q in enumerate(q_grid):
            for side, direction in (("bid", 1.0), ("ask", -1.0)):
                row = tier_policy[side][i]
                for j, z in enumerate(sizes):
                    q2 = float(q) + direction * z
                    if q2 < q_grid[0] - 1e-8 or q2 > q_grid[-1] + 1e-8:
                        continue
                    delta = float(row[j])
                    add_transition(i, q2, logistic_rate(flow, delta, z))

    # Dark-pool fills.
    dark_policy = solution.get("darkPool")
    dark_cfg = cfg.get("darkPool", {})
    if dark_policy and dark_cfg.get("enabled", False):

        def dark_fill_rates(side: str, posted: int) -> list[tuple[int, float]]:
            intensity = float(dark_cfg["lambda"])
            mu = float(dark_cfg["mu"])
            p0 = float(dark_cfg["p0"])
            positive_norm = 1.0 - math.exp(-mu)
            if posted <= 0 or intensity <= 0.0 or positive_norm <= 1e-15:
                return []
            positive_mass = 1.0 - p0
            out=[]
            cumulative=0.0
            for k in range(1, posted):
                pk=math.exp(-mu+k*math.log(mu)-math.lgamma(k+1))/positive_norm
                cumulative += pk
                out.append((k, intensity*positive_mass*pk))
            out.append((posted, intensity*positive_mass*max(0.0,1.0-cumulative)))
            return out

        for i, q in enumerate(q_grid):
            for side, direction in (("bid", 1.0), ("ask", -1.0)):
                active = bool(dark_policy[f"{side}Active"][i])
                if not active:
                    continue
                posted = int(round(float(dark_policy[f"{side}Size"][i])))
                if posted <= 0:
                    continue
                for fill, rate in dark_fill_rates(side, posted):
                    q2 = float(q) + direction * float(fill)
                    if q_grid[0] - 1e-8 <= q2 <= q_grid[-1] + 1e-8:
                        add_transition(i, q2, rate)

    # Passive ECN hedge.
    ecn_policy = solution.get("passiveEcn")
    ecn_cfg = cfg.get("passiveEcn", {})
    if ecn_policy and ecn_cfg.get("enabled", False):
        A = float(ecn_cfg["flow"]["A"])
        k = float(ecn_cfg["flow"]["k"])
        z = float(ecn_cfg["quoteSize"])
        for i, q in enumerate(q_grid):
            for side, direction in (("bid", 1.0), ("ask", -1.0)):
                if not bool(ecn_policy[f"{side}Active"][i]):
                    continue
                delta = float(ecn_policy[f"{side}Delta"][i])
                rate = A * math.exp(-k * delta)
                q2 = float(q) + direction * z
                if q_grid[0] - 1e-8 <= q2 <= q_grid[-1] + 1e-8:
                    add_transition(i, q2, rate)

    transient = [i for i in range(n) if i != zero]
    minus_q = -Q[np.ix_(transient, transient)]
    rhs = np.ones(len(transient), dtype=float)
    try:
        tau_transient = np.linalg.solve(minus_q, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Internalization time is not finite for the current policy") from exc

    tau = np.zeros(n, dtype=float)
    tau[transient] = np.maximum(0.0, tau_transient)
    qmax = float(cfg.get("grid", {}).get("maxAbs", np.max(np.abs(q_grid))))
    operational = np.abs(q_grid) <= qmax + 1e-8
    return {
        "qGrid": q_grid[operational].tolist(),
        "minutes": tau[operational].tolist(),
    }


@app.callback(
    Output("ecn-fill-chart", "figure"), Output("ecn-hit-ratio-chart", "figure"),
    Output("ecn-implied-hit-chart", "figure"), Output("ecn-quote-inventory-chart", "figure"),
    Output("ecn-policy-chart", "figure"), Output("ecn-parameter-table", "children"),
    Output("ecn-policy-table", "children"),
    Input("solution-store", "data"), Input("config-store", "data"),
)
def render_passive_ecn_diagnostics(solution, cfg):
    if not solution or not cfg:
        return (go.Figure(),) * 5 + (html.Div("Solve the model first"), html.Div("Solve the model first"))
    if not cfg.get("passiveEcn", {}).get("enabled", False):
        disabled = go.Figure()
        disabled.add_annotation(text="Passive ECN disabled", showarrow=False)
        disabled.update_layout(template="trinity_dark")
        note = html.Div("Passive ECN disabled", className="muted-note")
        return disabled, disabled, disabled, disabled, disabled, note, note
    return (
        passive_ecn_fill_figure(cfg),
        passive_ecn_hit_ratio_figure(cfg),
        passive_ecn_implied_hit_ratio_figure(solution, cfg),
        passive_ecn_quote_inventory_figure(solution, cfg),
        passive_ecn_policy_figure(solution),
        passive_ecn_parameter_table(cfg),
        passive_ecn_table(solution),
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
    Input("run-mc", "n_clicks"), State("config-store", "data"), State("solution-store", "data"), State("mc-paths", "value"), State("mc-q0", "value"), State("mc-seed", "value"), prevent_initial_call=True,
)
def run_mc(_clicks, cfg, solution, paths, q0, seed):
    if not cfg:
        return no_update, no_update, "Solve the model first", "status error"
    try:
        engine = ENGINES.get(cfg)
        mc = engine.simulate(SESSION_HORIZON, int(paths), float(q0), int(seed), retained_paths=8, sample_points=381)
        stats = engine.statistics(SESSION_HORIZON, float(q0))
        if solution:
            stats["internalizationTimes"] = expected_internalization_times(solution, cfg)
        return mc, stats, "Monte Carlo complete", "status ready"
    except Exception as exc:
        return no_update, no_update, f"Monte Carlo failed · {exc}", "status error"


@app.callback(
    Output("mc-mean", "children"), Output("cf-mean", "children"), Output("mc-std", "children"), Output("cf-std", "children"),
    Output("mc-p5", "children"), Output("mc-loss", "children"), Output("pnl-chart", "figure"), Output("inventory-chart", "figure"),
    Output("mc-internalization-time-chart", "figure"), Output("path-select", "options"), Output("path-select", "value"),
    Input("mc-store", "data"), Input("stats-store", "data"),
)
def render_mc(mc, stats):
    if not mc or not stats:
        return "—", "—", "—", "—", "—", "—", go.Figure(), go.Figure(), go.Figure(), [], None
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
    internal = go.Figure()
    internal_data = stats.get("internalizationTimes") or {}
    q_grid = np.asarray(internal_data.get("qGrid", []), dtype=float)
    tau = np.asarray(internal_data.get("minutes", []), dtype=float)
    if len(q_grid) == len(tau) and len(q_grid):
        positive = q_grid >= -1e-12
        negative = q_grid <= 1e-12
        # Plot inventory *size* on the horizontal axis.  Long and short sides are
        # separate traces so any asymmetry in the optimized policy remains visible.
        internal.add_trace(go.Scatter(
            x=q_grid[positive], y=tau[positive], mode="lines+markers",
            name="Long inventory", hovertemplate="Inventory %{x:.0f}M<br>Time to zero %{y:.2f} min<extra></extra>"
        ))
        internal.add_trace(go.Scatter(
            x=np.abs(q_grid[negative][::-1]), y=tau[negative][::-1], mode="lines+markers",
            name="Short inventory", hovertemplate="Inventory %{x:.0f}M<br>Time to zero %{y:.2f} min<extra></extra>"
        ))
    internal.update_layout(
        title="Expected internalization time to zero",
        xaxis_title="Inventory size [EUR M]",
        yaxis_title="Expected time to zero [min]",
        template="trinity_dark",
        hovermode="x unified",
    )
    internal.update_xaxes(rangemode="tozero")
    internal.update_yaxes(rangemode="tozero")

    options = [{"label": f"Path {i + 1}", "value": i} for i in range(len(mc.get("samplePaths", [])))]
    return format_ccy(mc_mean), format_ccy(stats["meanBase"]), format_ccy(mc_std), format_ccy(stats["stdBase"]), format_ccy(p5), f"{loss:.1f}%", hist, inv, internal, options, (0 if options else None)


@app.callback(
    Output("path-pnl-chart", "figure"), Output("path-chart", "figure"), Output("spot-path-chart", "figure"),
    Output("fill-counts-chart", "figure"), Output("fill-tape", "children"),
    Input("mc-store", "data"), Input("path-select", "value"), Input("path-size-select", "value"),
    Input("solution-store", "data"), Input("config-store", "data"),
)
def render_path(mc, path_index, quote_size, solution, cfg):
    if not mc or path_index is None or not mc.get("samplePaths"):
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), "No retained path"
    path = mc["samplePaths"][int(path_index)]

    # Exact retained-path mark-to-market PnL. Cash is stored by the native
    # simulator at the same timestamps as spot/inventory, so all execution
    # prices and venue/tier fees are included without reconstruction.
    pnl_fig = go.Figure()
    cashes = np.asarray(path.get("cashes", []), dtype=float)
    spots_path = np.asarray(path.get("spots", []), dtype=float)
    inventories_path = np.asarray(path.get("inventories", []), dtype=float)
    times_path = np.asarray(path.get("times", []), dtype=float)
    if len(cashes) == len(spots_path) == len(inventories_path) == len(times_path) and len(cashes) > 0:
        q0_pnl = float(inventories_path[0])
        s0_pnl = float(spots_path[0])
        pnl_quote = (cashes + inventories_path * spots_path - q0_pnl * s0_pnl) * 1_000_000.0
        pnl_base = np.divide(pnl_quote, spots_path, out=np.full_like(pnl_quote, np.nan), where=np.abs(spots_path) > 1e-15)
        pnl_fig.add_trace(go.Scatter(
            x=session_timestamps(times_path.tolist()), y=pnl_base, mode="lines",
            name="MtM PnL", line=dict(width=2.5), line_shape="hv",
            hovertemplate="%{x|%H:%M:%S}<br>MtM PnL=%{y:,.0f} EUR<extra></extra>",
        ))
        pnl_fig.add_hline(y=0, line_dash="dot")
    else:
        pnl_fig.add_annotation(text="Rebuild the native extension to record path cash/PnL", showarrow=False)
    pnl_fig.update_layout(title="Mark-to-market PnL", yaxis_title="PnL [EUR]", template="trinity_dark")
    apply_session_clock_axis(pnl_fig)

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

        # Passive ECN is a separate optimized venue, so it is not present in
        # solution["tiers"]. Reconstruct the ECN quote state explicitly using
        # the same left-bracket inventory lookup as the C++ Monte Carlo engine.
        ecn = solution.get("passiveEcn")
        if ecn:
            ecn_q = np.asarray(ecn["qGrid"], dtype=float)
            bid_delta_grid = np.asarray(ecn["bidDelta"], dtype=float)
            ask_delta_grid = np.asarray(ecn["askDelta"], dtype=float)
            bid_active_grid = np.asarray(ecn["bidActive"], dtype=bool)
            ask_active_grid = np.asarray(ecn["askActive"], dtype=bool)

            def ecn_state_indices(q_values):
                idx = np.searchsorted(ecn_q, q_values, side="left")
                idx = np.clip(idx, 0, len(ecn_q) - 1)
                exact = np.isclose(ecn_q[idx], q_values, rtol=0.0, atol=1e-10)
                idx = np.where((idx > 0) & ~exact, idx - 1, idx)
                return idx.astype(int)

            ecn_idx = ecn_state_indices(inventories)
            ecn_bid_delta = bid_delta_grid[ecn_idx]
            ecn_ask_delta = ask_delta_grid[ecn_idx]
            ecn_bid_active = bid_active_grid[ecn_idx]
            ecn_ask_active = ask_active_grid[ecn_idx]
            ecn_bid_px = np.where(ecn_bid_active, spots - ecn_bid_delta / 10_000.0, np.nan)
            ecn_ask_px = np.where(ecn_ask_active, spots + ecn_ask_delta / 10_000.0, np.nan)
            ecn_size = float(ecn["quoteSize"])

            spot_fig.add_trace(go.Scatter(
                x=path_clock, y=ecn_bid_px, mode="lines",
                name=f"Passive ECN bid {ecn_size:g}M",
                opacity=.95, line=dict(color=BID_COLOR, width=3, dash="dot"),
                line_shape="hv", connectgaps=False,
            ))
            spot_fig.add_trace(go.Scatter(
                x=path_clock, y=ecn_ask_px, mode="lines",
                name=f"Passive ECN ask {ecn_size:g}M",
                opacity=.95, line=dict(color=ASK_COLOR, width=3, dash="dot"),
                line_shape="hv", connectgaps=False,
            ))

    ecn_arrivals = list(path.get("ecnArrivals", []))
    if ecn_arrivals:
        def _ecn_plot_price(event):
            side = str(event.get("side", "")).lower()
            ref = float(event.get("referencePrice", np.nan))
            quote_depth = float(event.get("quoteDepthPips", event.get("depthPips", np.nan)))
            trade_px = float(event.get("tradePrice", ref))
            if bool(event.get("won", False)) and np.isfinite(ref) and np.isfinite(quote_depth):
                signed = quote_depth / 10_000.0
                return ref - signed if side == "bid" else ref + signed
            return trade_px

        def _ecn_trace(events, side, won, name, color, fill_status):
            chosen = [e for e in events if str(e.get("side", "")).lower() == side and bool(e.get("won", False)) == won]
            if not chosen:
                return None
            return go.Scatter(
                x=[session_timestamp(e["time"]) for e in chosen],
                y=[_ecn_plot_price(e) for e in chosen],
                mode="markers",
                name=name,
                marker=dict(
                    symbol="circle",
                    size=8,
                    color=color if won else "rgba(0,0,0,0)",
                    line=dict(color=color, width=2),
                ),
                customdata=[[
                    side.capitalize(),
                    fill_status,
                    float(e.get("tradeDistancePips", np.nan)),
                    float(e.get("quoteDepthPips", e.get("depthPips", np.nan))),
                    float(e.get("tradePrice", np.nan)),
                    bool(e.get("quoteActive", False)),
                ] for e in chosen],
                hovertemplate=(
                    "%{x|%H:%M:%S}<br>ECN %{customdata[0]} trade · %{customdata[1]}"
                    "<br>plot px=%{y:.6f}<br>trade px=%{customdata[4]:.6f}"
                    "<br>trade distance=%{customdata[2]:.2f} pips"
                    "<br>quote depth=%{customdata[3]:.2f} pips"
                    "<br>quote active=%{customdata[5]}<extra></extra>"
                ),
            )

        for args in (
            ("ask", True,  "ECN ask fills",     ASK_COLOR, "filled"),
            ("ask", False, "ECN ask not filled", ASK_COLOR, "not filled"),
            ("bid", True,  "ECN bid fills",     BID_COLOR, "filled"),
            ("bid", False, "ECN bid not filled", BID_COLOR, "not filled"),
        ):
            trace = _ecn_trace(ecn_arrivals, *args)
            if trace is not None:
                spot_fig.add_trace(trace)

    for side, symbol in (("bid", "triangle-up"), ("ask", "triangle-down")):
        fills_side = [
            f for f in path["fills"]
            if str(f["side"]).lower() == side and str(f.get("tier", "")) != "Passive ECN"
        ]
        if fills_side:
            spot_fig.add_trace(go.Scatter(
                x=[session_timestamp(f["time"]) for f in fills_side], y=[f["price"] for f in fills_side], mode="markers",
                marker=dict(symbol=symbol, size=8, color=BID_COLOR if side == "bid" else ASK_COLOR), name=f"{side.capitalize()} fills",
                customdata=[[f["tier"], f["size"]] for f in fills_side],
                hovertemplate="%{x|%H:%M:%S}<br>px=%{y:.6f}<br>%{customdata[0]} · %{customdata[1]}M<extra></extra>",
            ))
    title_size = "" if quote_size is None else f" · {float(quote_size):g}M quotes"
    spot_fig.update_layout(
        title=f"Spot, quotes and fills{title_size}",
        yaxis_title="EURSEK",
        template="trinity_dark",
        dragmode="pan",
        uirevision="simulation-spot-quotes-fills",
    )
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
    return pnl_fig, fig, spot_fig, fill_counts, table


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
        gammas = np.unique(np.append(gammas, float(cfg["gamma"]))).tolist()
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
