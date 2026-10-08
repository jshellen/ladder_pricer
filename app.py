from __future__ import annotations

import concurrent.futures
import json
import math
import threading
import time
import uuid
from datetime import datetime, timedelta
from copy import deepcopy
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from trinity.defaults import (
    ECN_TRADE_SIZE_PROBABILITIES,
    default_config,
    default_ecn_trade_size_probabilities,
    default_fx_mid,
)
from trinity.service import ENGINES
from trinity.ui_config import build_config, checked, parse_numbers

APP_TITLE = "FX Ladder Pricer"
SESSION_HORIZON = 570.0
BASE_CCY = "EUR"
QUOTE_CCY = "SEK"

FX_PAIRS = [
    "EURSEK", "USDSEK", "GBPSEK", "NOKSEK", "DKKSEK",
    "EURNOK", "USDNOK", "GBPNOK", "SEKNOK", "DKKNOK",
    "EURDKK", "USDDKK", "GBPDKK", "SEKDKK", "NOKDKK",
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
    "EURGBP", "EURCHF", "EURJPY", "GBPCHF", "GBPJPY", "AUDJPY", "NZDJPY",
]
FX_PAIR_OPTIONS = [{"label": pair, "value": pair} for pair in FX_PAIRS]

def _pair_legs(pair: str | None) -> tuple[str, str] | None:
    pair = str(pair or "").upper().strip()
    if len(pair) != 6 or not pair.isalpha():
        return None
    return pair[:3], pair[3:]

def _source_pair_options(target_pair: str | None) -> list[dict[str, str]]:
    legs = _pair_legs(target_pair)
    if not legs:
        return FX_PAIR_OPTIONS
    _, quote = legs
    pairs = [pair for pair in FX_PAIRS if _pair_legs(pair) and _pair_legs(pair)[1] == quote and pair != target_pair]
    return [{"label": pair, "value": pair} for pair in pairs]

def _derive_cross_pair(target_pair: str | None, source_pair: str | None) -> str | None:
    target = _pair_legs(target_pair)
    source = _pair_legs(source_pair)
    if not target or not source or target[1] != source[1] or target[0] == source[0]:
        return None
    # The backend expects X = target-base/source-base so that
    # source_price = target_price / X. Keep this orientation even when the
    # market convention normally quotes the inverse cross.
    return target[0] + source[0]

def _cross_pair_options(target_pair: str | None, source_pair: str | None) -> list[dict[str, str]]:
    derived = _derive_cross_pair(target_pair, source_pair)
    pairs = list(FX_PAIRS)
    if derived and derived not in pairs:
        pairs.insert(0, derived)
    return [{"label": pair, "value": pair} for pair in pairs]

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
SIZE_COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#d2a8ff", "#f778ba", "#a5d6ff", "#e3b341", "#7ee787"]

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


# Monte-Carlo jobs run in a worker thread so the Dash request that starts a
# large simulation returns immediately.  The native simulator releases the GIL
# and reports completed-path progress back through a sparse callback, while a
# lightweight dcc.Interval polls this in-process registry for the modal UI.
# This keeps the existing in-process Engine cache (and its solved policy) rather
# than re-solving the model in a separate background process.
_MC_JOBS: dict[str, dict[str, Any]] = {}
_MC_JOBS_LOCK = threading.RLock()
_MC_JOB_RETENTION_SECONDS = 300.0

# Large Monte Carlo payloads (especially 1-second retained paths) stay on the
# server. Sending them through dcc.Store forces expensive JSON serialization
# and can block progress polling while Python holds the GIL.
_MC_RESULTS: dict[str, dict[str, Any]] = {}
_MC_RESULTS_LOCK = threading.RLock()
_MC_RESULTS_MAX_ENTRIES = 6


def _store_mc_result(result_id: str, mc: dict[str, Any]) -> None:
    with _MC_RESULTS_LOCK:
        if result_id not in _MC_RESULTS and len(_MC_RESULTS) >= _MC_RESULTS_MAX_ENTRIES:
            oldest = next(iter(_MC_RESULTS))
            _MC_RESULTS.pop(oldest, None)
        _MC_RESULTS[result_id] = mc


def _resolve_mc_store(payload: Any) -> dict[str, Any] | None:
    """Resolve a legacy inline MC dict or the new lightweight result token."""
    if not payload:
        return None
    if isinstance(payload, dict) and "pnlBase" in payload:
        return payload
    if isinstance(payload, dict) and payload.get("resultId"):
        result_id = str(payload["resultId"])
        with _MC_RESULTS_LOCK:
            return _MC_RESULTS.get(result_id)
    return None

# Closed-form PnL statistics and internalization times are deterministic for a
# solved economic configuration / initial inventory.  Cache them separately
# from stochastic MC output so repeated runs with a different seed/path count
# do not pay the analytical cost again.
_MC_ANALYTICS_CACHE: dict[str, dict[str, Any]] = {}
_MC_ANALYTICS_CACHE_LOCK = threading.RLock()
_MC_ANALYTICS_CACHE_MAX_ENTRIES = 16


def _prune_finished_mc_jobs_locked(now: float | None = None) -> None:
    """Drop only old completed jobs.

    Completed/error jobs deliberately remain in the registry for a short grace
    period.  dcc.Interval can have another poll request already in flight when
    the first completion response is being serialized.  Keeping the result
    makes those late polls idempotent instead of turning them into a spurious
    "state unavailable" error.
    """
    now = time.time() if now is None else float(now)
    stale = [
        job_id
        for job_id, job in _MC_JOBS.items()
        if str(job.get("state")) in {"done", "error"}
        and now - float(job.get("finishedAt", now)) > _MC_JOB_RETENTION_SECONDS
    ]
    for job_id in stale:
        _MC_JOBS.pop(job_id, None)


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



def _default_cross_flow_source(index: int = 0, target_pair: str | None = None) -> dict[str, Any]:
    return {
        "name": "Select FX pair",
        "pair": None,
        "crossPair": None,
        "mapping": {"type": "crossed", "crossMid": None},
        "flow": {"A0": 0.01, "theta": 0.144, "beta": 0.0857, "steepness": 8.42, "shift": 0.52, "volumeShift": 0.026},
    }


def _direct_flow_source(flow: dict[str, Any], target_pair: str) -> dict[str, Any]:
    return {
        "name": str(target_pair),
        "pair": str(target_pair),
        "mapping": {"type": "identity"},
        "flow": deepcopy(flow),
    }


def _legacy_ecn_size_probabilities(mean_trade_size: float = 1.0) -> list[float]:
    """Map the former exponential size model onto the five fixed pillars."""
    mu = float(mean_trade_size)
    if mu <= 0.0:
        raise ValueError("ECN legacy mean trade size must be positive")
    s0175 = math.exp(-0.175 / mu)
    s0375 = math.exp(-0.375 / mu)
    s0625 = math.exp(-0.625 / mu)
    s0875 = math.exp(-0.875 / mu)
    return [s0875, s0625 - s0875, s0375 - s0625, s0175 - s0375, 1.0 - s0175]


def _ecn_size_probabilities(source: dict[str, Any] | None) -> list[float]:
    source = source or {}
    raw = source.get("tradeSizeProbabilities")
    if raw is None:
        return _legacy_ecn_size_probabilities(float(source.get("meanTradeSize", 1.0)))
    probs = [float(x) for x in raw]
    if len(probs) != 5:
        raise ValueError("ECN trade-size probabilities must contain five entries")
    if any((not math.isfinite(x) or x < 0.0 or x > 1.0) for x in probs):
        raise ValueError("ECN trade-size probabilities must lie between 0 and 1")
    if abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError("ECN trade-size probabilities must sum to 1")
    return probs


def _ecn_size_probabilities_from_four(p1m, p750k, p500k, p250k) -> list[float]:
    values = [p1m, p750k, p500k, p250k]
    if any(x is None for x in values):
        raise ValueError("Enter probabilities for 1M, 750k, 500k and 250k")
    probs4 = [float(x) for x in values]
    if any((not math.isfinite(x) or x < 0.0 or x > 1.0) for x in probs4):
        raise ValueError("ECN trade-size probabilities must lie between 0 and 1")
    residual = 1.0 - sum(probs4)
    if residual < -1e-12:
        raise ValueError("ECN 1M + 750k + 500k + 250k probabilities may not exceed 1")
    return probs4 + [max(0.0, residual)]


def _default_cross_ecn_flow_source(index: int = 0, target_pair: str | None = None) -> dict[str, Any]:
    return {
        "name": "Select FX pair",
        "pair": None,
        "crossPair": None,
        "mapping": {"type": "crossed", "crossMid": None},
        "flow": {"A": 0.10, "k": 8.4},
        "tradeSizeProbabilities": list(ECN_TRADE_SIZE_PROBABILITIES),
    }


def _direct_ecn_flow_source(flow: dict[str, Any], target_pair: str, trade_size_probabilities=None) -> dict[str, Any]:
    probs = list(default_ecn_trade_size_probabilities(target_pair) if trade_size_probabilities is None else trade_size_probabilities)
    return {
        "name": str(target_pair),
        "pair": str(target_pair),
        "mapping": {"type": "identity"},
        "flow": deepcopy(flow),
        "tradeSizeProbabilities": probs,
    }


def flow_pair_card(key: str, label: str, selected: bool, removable: bool = False):
    classes = "flow-pair-card" + (" selected" if selected else "")
    children = [
        html.Button(
            label,
            id={"type": "flow-pair-select", "key": key},
            className=classes,
            n_clicks=0,
        )
    ]
    children.append(html.Span("crossed" if removable else "direct", className="flow-pair-badge"))
    return html.Div(children, className="flow-pair-card-wrap")


def _dropdown_field(label: str, component_id: str, value: str | None, options: list[dict[str, str]], placeholder: str = "Select FX pair"):
    return html.Label([
        html.Span(label),
        dcc.Dropdown(id=component_id, value=value, options=options, placeholder=placeholder, clearable=False, className="field-dropdown", persistence=True, persistence_type="memory"),
    ], className="field")


def flow_source_settings_editor(source_key: str, source: dict[str, Any], target_pair: str):
    flow = source.get("flow", {})
    mapping = source.get("mapping", {})
    direct = source_key == "direct"

    header = html.Div([
        html.Div([
            html.Div(str(source.get("pair") or target_pair), className="flow-source-title"),
            html.Div("Direct target-pair flow" if direct else "Crossed source", className="muted-note source-kind-note"),
        ]),
        None if direct else html.Button(
            "Remove FX pair",
            id="flow-source-remove-selected",
            className="ghost-button danger-ghost-button",
            n_clicks=0,
        ),
    ], className="flow-source-row-head")

    if direct:
        source_fields = [html.Div([
            dcc.Dropdown(id="flow-edit-pair", value=target_pair, options=FX_PAIR_OPTIONS),
            dcc.Dropdown(id="flow-edit-cross-pair", value=None, options=FX_PAIR_OPTIONS),
            dcc.Input(id="flow-edit-cross-mid", type="number", value=1.0),
            html.Button("Remove FX pair", id="flow-source-remove-selected", n_clicks=0),
        ], style={"display": "none"})]
    else:
        pair = source.get("pair")
        cross_pair = source.get("crossPair") or _derive_cross_pair(target_pair, pair)
        configured_mid = mapping.get("crossMid")
        cross_mid = configured_mid if configured_mid is not None else default_fx_mid(cross_pair)
        source_fields = [
            html.Div([
                _dropdown_field("Source pair", "flow-edit-pair", pair, _source_pair_options(target_pair)),
                _dropdown_field("Cross / hedge pair", "flow-edit-cross-pair", cross_pair, _cross_pair_options(target_pair, pair), "Select hedge pair"),
                number_input("Cross mid", "flow-edit-cross-mid", cross_mid, "any"),
            ], className="grid-2"),
            html.Div("Flow calibration", className="subhead"),
        ]

    return html.Div([
        header,
        *source_fields,
        html.Div(
            "The exogenous RFQ-arrival curve λ(z)=A0·z^(-θ-βz) defines both the size-specific arrival intensities and the RFQ-size distribution. Monte Carlo samples from the full RFQ-size grid between the smallest and largest pricing knots; prices between knots are linearly interpolated.",
            className="muted-note modal-note",
        ),
        html.Div([
            number_input("A0 · exogenous intensity scale", "flow-edit-A0", float(flow.get("A0", 0.01)), "any"),
            number_input("Size θ", "flow-edit-theta", float(flow.get("theta", 0.144)), "any"),
            number_input("Size β", "flow-edit-beta", float(flow.get("beta", 0.0857)), "any"),
            number_input("Steepness", "flow-edit-steep", float(flow.get("steepness", 8.42)), "any"),
            number_input("Shift", "flow-edit-shift", float(flow.get("shift", 0.52)), "any"),
            number_input("Volume shift", "flow-edit-vshift", float(flow.get("volumeShift", 0.026)), "any"),
        ], className="grid-2 flow-calibration-grid"),
    ], className="flow-source-settings")

def ecn_flow_pair_card(key: str, label: str, selected: bool, removable: bool = False):
    classes = "flow-pair-card" + (" selected" if selected else "")
    return html.Div([
        html.Button(
            label,
            id={"type": "ecn-flow-pair-select", "key": key},
            className=classes,
            n_clicks=0,
        ),
        html.Span("crossed" if removable else "direct", className="flow-pair-badge"),
    ], className="flow-pair-card-wrap")


def _ecn_dropdown_field(label: str, component_id: str, value: str | None, options: list[dict[str, str]], placeholder: str = "Select FX pair"):
    return html.Label([
        html.Span(label),
        dcc.Dropdown(id=component_id, value=value, options=options, placeholder=placeholder, clearable=False, className="field-dropdown", persistence=True, persistence_type="memory"),
    ], className="field")


def ecn_flow_source_settings_editor(source_key: str, source: dict[str, Any], target_pair: str):
    flow = source.get("flow", {})
    mapping = source.get("mapping", {})
    direct = source_key == "direct"

    header = html.Div([
        html.Div([
            html.Div(str(source.get("pair") or target_pair), className="flow-source-title"),
            html.Div("Direct target-pair ECN flow" if direct else "Crossed ECN source", className="muted-note source-kind-note"),
        ]),
        None if direct else html.Button(
            "Remove FX pair", id="ecn-flow-source-remove-selected",
            className="ghost-button danger-ghost-button", n_clicks=0,
        ),
    ], className="flow-source-row-head")

    if direct:
        source_fields = [html.Div([
            dcc.Dropdown(id="ecn-flow-edit-pair", value=target_pair, options=FX_PAIR_OPTIONS),
            dcc.Dropdown(id="ecn-flow-edit-cross-pair", value=None, options=FX_PAIR_OPTIONS),
            dcc.Input(id="ecn-flow-edit-cross-mid", type="number", value=1.0),
            html.Button("Remove FX pair", id="ecn-flow-source-remove-selected", n_clicks=0),
        ], style={"display": "none"})]
    else:
        pair = source.get("pair")
        cross_pair = source.get("crossPair") or _derive_cross_pair(target_pair, pair)
        configured_mid = mapping.get("crossMid")
        cross_mid = configured_mid if configured_mid is not None else default_fx_mid(cross_pair)
        source_fields = [
            html.Div([
                _ecn_dropdown_field("Source pair", "ecn-flow-edit-pair", pair, _source_pair_options(target_pair)),
                _ecn_dropdown_field("Cross / hedge pair", "ecn-flow-edit-cross-pair", cross_pair, _cross_pair_options(target_pair, pair), "Select hedge pair"),
                number_input("Cross mid", "ecn-flow-edit-cross-mid", cross_mid, "any"),
            ], className="grid-2"),
            html.Div("ECN flow calibration", className="subhead"),
        ]

    return html.Div([
        header,
        *source_fields,
        html.Div([
            number_input("A at mid / side [trades/min]", "ecn-flow-edit-A", float(flow.get("A", 0.10)), "any"),
            number_input("k", "ecn-flow-edit-k", float(flow.get("k", 8.4)), "any"),
        ], className="grid-2 flow-calibration-grid"),
        html.Div("Discrete ECN parent trade sizes · probabilities in source base currency", className="subhead"),
        html.Div([
            number_input("P(1M)", "ecn-flow-edit-p1m", _ecn_size_probabilities(source)[0], "any"),
            number_input("P(750k)", "ecn-flow-edit-p750k", _ecn_size_probabilities(source)[1], "any"),
            number_input("P(500k)", "ecn-flow-edit-p500k", _ecn_size_probabilities(source)[2], "any"),
            number_input("P(250k)", "ecn-flow-edit-p250k", _ecn_size_probabilities(source)[3], "any"),
        ], className="grid-2 flow-calibration-grid"),
        html.Div(id="ecn-flow-edit-p100k-note", className="muted-note"),
    ], className="flow-source-settings")


def tier_panel(index: int, tier: dict[str, Any], open_: bool = False):
    p = f"t{index}"
    return panel(tier["name"], [
        checkbox("Enabled", f"{p}-enabled", tier["enabled"]),
        text_input("Name", f"{p}-name", tier["name"]),
        text_input("Pricing sizes [EUR M]", f"{p}-sizes", ", ".join(str(x) for x in tier["sizes"])),
        number_input("RFQ size step [EUR M]", f"{p}-rfq-size-step", float(tier.get("rfqSizeStep", 1.0)), "any"),
        html.Div([
            html.Button("Configure flow sources", id={"type": "open-flow-sources", "tier": index}, className="secondary-button compact-button", n_clicks=0),
        ], className="flow-source-config-line"),
        # Keep the original direct-flow inputs mounted for backward-compatible config
        # construction.  Their values are synchronized from the modal editor.
        html.Div([
            number_input("A0", f"{p}-A0", tier["flow"].get("A0", 0.0), "any"),
            number_input("Theta", f"{p}-theta", tier["flow"]["theta"], "any"),
            number_input("Beta", f"{p}-beta", tier["flow"]["beta"], "any"),
            number_input("Steepness", f"{p}-steep", tier["flow"]["steepness"], "any"),
            number_input("Shift", f"{p}-shift", tier["flow"]["shift"], "any"),
            number_input("Volume shift", f"{p}-vshift", tier["flow"]["volumeShift"], "any"),
        ], style={"display": "none"}),
        html.Div("Trading economics", className="subhead"),
        html.Div([
            number_input("Fee [EUR / EURm]", f"{p}-fee", tier.get("feeEurPerEurM", 0.0), "any"),
        ], className="grid-2"),
        html.Div("Markout", className="subhead"),
        checkbox("Enabled", f"{p}-markout-enabled", tier["useMarkout"]),
        html.Div([
            number_input("Asymptotic markout [EUR / EURm traded]", f"{p}-markout-level", tier["markout"]["asymptoticEurPerEurM"], "any"),
            number_input("Tau [min]", f"{p}-markout-tau", tier["markout"]["tauMinutes"], "any"),
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


def _source_size_scale(source: dict[str, Any]) -> float:
    mapping = source.get("mapping", {}) or {}
    mapping_type = str(mapping.get("type", "identity"))
    if mapping_type == "crossed":
        return float(mapping.get("crossMid", 1.0))
    if mapping_type == "affine":
        return float(mapping.get("sourceSizePerTarget", 1.0))
    return 1.0


def _tier_flow_sources(tier: dict[str, Any]) -> list[dict[str, Any]]:
    sources = tier.get("flowSources")
    if sources:
        return list(sources)
    return [{"name": tier.get("name", "direct"), "flow": tier["flow"], "mapping": {"type": "identity"}}]


def _rfq_size_grid(tier: dict[str, Any]) -> np.ndarray:
    """Customer RFQ sizes, independent of the pricing-control knots."""
    pricing = np.asarray(tier["sizes"], dtype=float)
    if pricing.size == 0 or np.any(pricing <= 0.0):
        raise ValueError("Pricing sizes must be positive")
    step = float(tier.get("rfqSizeStep", 1.0))
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("RFQ size step must be positive")
    lo, hi = float(pricing[0]), float(pricing[-1])
    regular = np.arange(lo, hi + 0.5 * step, step, dtype=float)
    regular = regular[regular <= hi + 1e-12]
    values = np.unique(np.round(np.concatenate([regular, pricing, [hi]]), 12))
    return values[(values >= lo - 1e-12) & (values <= hi + 1e-12)]


def _source_curve_at_sizes(tier: dict[str, Any], source: dict[str, Any], target_sizes) -> np.ndarray:
    """Unnormalised RFQ-size density implied by the calibrated arrival curve."""
    sizes = np.asarray(target_sizes, dtype=float) * _source_size_scale(source)
    flow = source["flow"]
    exponent = -float(flow["theta"]) - float(flow["beta"]) * sizes
    weights = np.power(sizes, exponent)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("RFQ arrival-intensity curve must be finite and nonnegative")
    return weights


def _source_curve_shape(tier: dict[str, Any], source: dict[str, Any]) -> np.ndarray:
    weights = _source_curve_at_sizes(tier, source, _rfq_size_grid(tier))
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("RFQ arrival-intensity curve must have positive mass")
    return weights


def _source_a0(tier: dict[str, Any], source: dict[str, Any]) -> float:
    flow = source["flow"]
    if "A0" in flow:
        return float(flow["A0"])
    if "totalRfqRate" in flow:
        # Compatibility with the short-lived total-rate config format.  Recover
        # the original one-sided curve scale on the *full* RFQ size support.
        shape_mass = float(np.sum(_source_curve_shape(tier, source)))
        return float(flow["totalRfqRate"]) / (2.0 * shape_mass)
    raise ValueError("RFQ flow requires A0")


def _source_size_distribution(tier: dict[str, Any], source: dict[str, Any]) -> np.ndarray:
    # There is no independent RFQ-size calibration: the exogenous arrival
    # intensity curve itself is the density.  Normalize it on the full customer
    # RFQ grid, not only on the pricing-control knots.
    weights = _source_curve_shape(tier, source)
    return weights / float(np.sum(weights))


def _source_rfq_rates_per_side(tier: dict[str, Any], source: dict[str, Any]) -> np.ndarray:
    return _source_a0(tier, source) * _source_curve_shape(tier, source)


def _source_rfq_rate_at(tier: dict[str, Any], source: dict[str, Any], z: float) -> float:
    shape = float(_source_curve_at_sizes(tier, source, [float(z)])[0])
    return _source_a0(tier, source) * shape


def _source_total_rfq_rate(tier: dict[str, Any], source: dict[str, Any]) -> float:
    # One pooled source clock across bid and ask customer RFQs.
    return 2.0 * float(np.sum(_source_rfq_rates_per_side(tier, source)))


def _activity(tier: dict[str, Any], z: float) -> float:
    """One-sided exogenous RFQ arrival intensity at target size z."""
    return sum(_source_rfq_rate_at(tier, source, float(z))
               for source in _tier_flow_sources(tier))


def _delta50(tier: dict[str, Any], z: float) -> float:
    flow = tier["flow"]
    return float(flow["shift"]) - float(flow["volumeShift"]) * (float(z) - 1.0)


def _source_delta_scale(source: dict[str, Any], target_spread_pips: float) -> float:
    mapping = source.get("mapping", {}) or {}
    mapping_type = str(mapping.get("type", "identity"))
    if mapping_type == "crossed":
        cross_mid = float(mapping.get("crossMid", 0.0))
        if cross_mid <= 0.0:
            raise ValueError("Crossed RFQ source requires positive cross mid")
        # The source pair is priced by crossing the target-pair quote itself.
        # Mid and spread therefore scale by the same FX cross, so the
        # normalized spread-improvement delta is invariant.
        return 1.0
    if mapping_type == "affine":
        return float(mapping.get("deltaScale", 1.0))
    if mapping_type != "identity":
        raise ValueError(f"Unsupported RFQ flow-source mapping {mapping_type}")
    return 1.0


def _source_hit_ratio(source: dict[str, Any], target_delta, target_size: float, target_spread_pips: float):
    flow = source["flow"]
    delta = np.asarray(target_delta, dtype=float)
    source_delta = 0.5 + _source_delta_scale(source, target_spread_pips) * (delta - 0.5)
    source_size = float(target_size) * _source_size_scale(source)
    center = float(flow["shift"]) - float(flow["volumeShift"]) * (source_size - 1.0)
    return _logistic(float(flow["steepness"]) * (source_delta - center))


def _hit_ratio(tier: dict[str, Any], delta, z: float, target_spread_pips: float):
    numerator = np.zeros_like(np.asarray(delta, dtype=float), dtype=float)
    denominator = 0.0
    for source in _tier_flow_sources(tier):
        rate = _source_rfq_rate_at(tier, source, float(z))
        numerator = numerator + rate * np.asarray(
            _source_hit_ratio(source, delta, z, target_spread_pips), dtype=float
        )
        denominator += rate
    result = numerator / denominator if denominator > 0.0 else np.zeros_like(numerator)
    return float(result) if result.ndim == 0 else result


def _arrival_rate(tier: dict[str, Any], delta, z: float, target_spread_pips: float):
    return _activity(tier, z) * _hit_ratio(tier, delta, z, target_spread_pips)


def _markout_eur_per_eurm(tier: dict[str, Any], t_minutes):
    """Normalized RFQ markout curve in EUR per EURm traded.

    The empirical markout is a single size-independent curve per tier:
        m(t) = M_inf * (1 - exp(-t / tau)).
    Trade size therefore does not scale the per-EURm price move.
    """
    if not tier.get("useMarkout", True):
        return np.zeros_like(np.asarray(t_minutes, dtype=float))
    spec = tier["markout"]
    tau = max(float(spec["tauMinutes"]), 1e-15)
    asymptotic = float(spec["asymptoticEurPerEurM"])
    values = asymptotic * (1.0 - np.exp(-np.asarray(t_minutes, dtype=float) / tau))
    return float(values) if np.asarray(values).ndim == 0 else values

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


def flow_curve_figure(tier: dict[str, Any], target_spread_pips: float) -> go.Figure:
    grid = np.linspace(-1.0, 1.0, 160)
    fig = go.Figure()
    for z in tier["sizes"]:
        fig.add_trace(go.Scatter(x=grid, y=_arrival_rate(tier, grid, float(z), target_spread_pips), mode="lines", name=f"{float(z):g}M"))
    fig.update_layout(title=f"Won-trade intensity per side · {tier['name']}", xaxis_title="δ", yaxis_title="Won trades [1/min]", template="trinity_dark")
    return fig


def exogenous_arrival_figure(tier: dict[str, Any]) -> go.Figure:
    sizes = _rfq_size_grid(tier)
    rates = np.asarray([_activity(tier, float(z)) for z in sizes], dtype=float)
    fig = go.Figure(go.Scatter(x=sizes, y=rates, mode="lines+markers", name="RFQ arrivals"))
    fig.update_layout(
        title=f"Exogenous RFQ arrival intensity λ_RFQ(z) · {tier['name']}",
        xaxis_title="RFQ size [EUR M]",
        yaxis_title="RFQs / side [1/min]",
        template="trinity_dark",
    )
    return fig


def hit_ratio_figure(tier: dict[str, Any], target_spread_pips: float) -> go.Figure:
    grid = np.linspace(-1.0, 1.0, 160)
    fig = go.Figure()
    for z in tier["sizes"]:
        fig.add_trace(go.Scatter(x=grid, y=_hit_ratio(tier, grid, float(z), target_spread_pips), mode="lines", name=f"{float(z):g}M"))
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
            x=q, y=_hit_ratio(tier, bid, z, float(cfg["spreadPips"])), mode="lines", name=f"{z:g}M bid",
            line=dict(color=BID_COLOR, dash=dash), legendgroup=f"size-{j}",
        ))
        fig.add_trace(go.Scatter(
            x=q, y=_hit_ratio(tier, ask, z, float(cfg["spreadPips"])), mode="lines", name=f"{z:g}M ask",
            line=dict(color=ASK_COLOR, dash=dash), legendgroup=f"size-{j}",
        ))
    fig.update_layout(title=f"Implied win probabilities vs inventory · {tier['name']}", xaxis_title="Inventory q [EUR M]", yaxis_title="Win probability", yaxis_range=[0,1], template="trinity_dark")
    return fig


def mc_rfq_validation_figures(
    mc: dict[str, Any], solution: dict[str, Any], cfg: dict[str, Any], tier_index: int,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Compare all-path Monte Carlo RFQ samples with the model inputs used by MC."""
    ti = int(tier_index)
    tier = _tier_cfg(cfg, ti)
    tier_sol = solution["tiers"][ti]
    tier_name = str(tier["name"])
    spread_pips = float(cfg["spreadPips"])
    q_grid = np.asarray(solution["qGrid"], dtype=float)
    pricing_sizes = [float(z) for z in tier_sol["sizes"]]

    # --- 1) Exogenous RFQ arrival intensity by size -----------------------
    # RFQ requests pool bid + ask.  The model curve is one-sided, therefore
    # divide the observed request count by 2 * path-minutes.
    arrival_fig = go.Figure()
    size_rows = [
        row for row in mc.get("rfqTierSizeAggregates", [])
        if str(row.get("tier", "")) == tier_name
    ]
    requests_by_size = {
        float(row.get("size", 0.0)): int(row.get("requests", 0))
        for row in size_rows
    }
    rfq_sizes = np.asarray(_rfq_size_grid(tier), dtype=float)
    model_arrivals = np.asarray([_activity(tier, float(z)) for z in rfq_sizes], dtype=float)
    n_paths = len(mc.get("pnlBase", []))
    exposure = 2.0 * float(n_paths) * SESSION_HORIZON
    mc_arrivals = np.asarray([
        requests_by_size.get(float(z), 0) / exposure if exposure > 0.0 else 0.0
        for z in rfq_sizes
    ], dtype=float)
    arrival_fig.add_trace(go.Scatter(
        x=rfq_sizes, y=model_arrivals, mode="lines+markers", name="Model input λ_RFQ(z)",
        line=dict(width=2.5),
    ))
    arrival_fig.add_trace(go.Scatter(
        x=rfq_sizes, y=mc_arrivals, mode="markers", name="MC estimate",
        marker=dict(size=9, symbol="circle-open"),
        customdata=np.asarray([requests_by_size.get(float(z), 0) for z in rfq_sizes]),
        hovertemplate=(
            "RFQ size=%{x:g}M<br>MC λ=%{y:.6f} /min/side"
            "<br>Requests=%{customdata:,.0f}<extra></extra>"
        ),
    ))
    arrival_fig.update_layout(
        title=f"Exogenous RFQ arrival intensity · {tier_name}",
        xaxis_title="RFQ size [EUR M]", yaxis_title="RFQs / side [1/min]",
        template="trinity_dark", hovermode="x unified",
    )

    # --- 2) Win probability as a function of quoted delta ----------------
    # Only admissible RFQs enter the empirical denominator here.  Otherwise
    # inventory-boundary clipping (where MC deliberately uses p=0) would be
    # mixed into the calibration curve p_win(delta, z).
    win_fig = go.Figure()
    delta_rows = [
        row for row in mc.get("rfqDeltaAggregates", [])
        if str(row.get("tier", "")) == tier_name
    ]
    for j, z in enumerate(pricing_sizes):
        color = SIZE_COLORS[j % len(SIZE_COLORS)]
        rows = sorted(
            [row for row in delta_rows if abs(float(row.get("size", 0.0)) - z) <= 1e-9],
            key=lambda row: float(row.get("delta", 0.0)),
        )

        # Plot the model only over the quote-delta domain actually reachable by
        # this rung's HJB policy (plus a small visual margin).
        bid_policy = np.asarray([row[j] for row in tier_sol["bid"]], dtype=float)
        ask_policy = np.asarray([row[j] for row in tier_sol["ask"]], dtype=float)
        policy_domain = np.concatenate([bid_policy, ask_policy])
        lo = float(np.nanmin(policy_domain)) if policy_domain.size else -1.0
        hi = float(np.nanmax(policy_domain)) if policy_domain.size else 1.0
        if rows:
            lo = min(lo, min(float(row.get("delta", lo)) for row in rows))
            hi = max(hi, max(float(row.get("delta", hi)) for row in rows))
        pad = max(0.02, 0.05 * max(hi - lo, 0.1))
        grid = np.linspace(lo - pad, hi + pad, 180)
        win_fig.add_trace(go.Scatter(
            x=grid, y=_hit_ratio(tier, grid, z, spread_pips), mode="lines",
            name=f"{z:g}M · model", legendgroup=f"mc-win-{j}",
            line=dict(color=color, width=2),
        ))

        empirical = [row for row in rows if int(row.get("admissibleRequests", 0)) > 0]
        if empirical:
            x = [float(row.get("delta", 0.0)) for row in empirical]
            n = [int(row.get("admissibleRequests", 0)) for row in empirical]
            wins = [int(row.get("wins", 0)) for row in empirical]
            y = [w / count for w, count in zip(wins, n)]
            win_fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=f"{z:g}M · MC",
                legendgroup=f"mc-win-{j}",
                marker=dict(color=color, size=8, symbol="circle-open"),
                customdata=np.column_stack([wins, n]),
                hovertemplate=(
                    f"{z:g}M RFQ<br>δ=%{{x:.4f}}<br>MC win probability=%{{y:.1%}}"
                    "<br>Wins=%{customdata[0]:.0f} / admissible RFQs=%{customdata[1]:.0f}<extra></extra>"
                ),
            ))
    win_fig.update_layout(
        title=f"Win probability p_win(δ,z) · {tier_name}",
        xaxis_title="Quoted δ", yaxis_title="Win probability", yaxis_range=[0, 1],
        template="trinity_dark",
    )
    win_fig.update_yaxes(tickformat=".0%")

    # --- 3) Effective / implied win probability by inventory -------------
    # The model line uses the exact tier policy on the HJB q-grid and applies
    # the same hard-inventory admissibility check as the native MC engine.
    inventory_fig = go.Figure()
    inventory_rows = [
        row for row in mc.get("rfqInventoryAggregates", [])
        if str(row.get("tier", "")) == tier_name
    ]
    q_min = float(np.min(q_grid)) if q_grid.size else 0.0
    q_max = float(np.max(q_grid)) if q_grid.size else 0.0
    for j, z in enumerate(pricing_sizes):
        dash = SIZE_DASHES[j % len(SIZE_DASHES)]
        for side, color, direction in (("bid", BID_COLOR, 1.0), ("ask", ASK_COLOR, -1.0)):
            policy = np.asarray([row[j] for row in tier_sol[side]], dtype=float)
            model_p = np.asarray(_hit_ratio(tier, policy, z, spread_pips), dtype=float)
            admissible = (q_grid + direction * z >= q_min - 1e-10) & (q_grid + direction * z <= q_max + 1e-10)
            model_p = np.where(admissible, model_p, 0.0)
            label_side = side.capitalize()
            inventory_fig.add_trace(go.Scatter(
                x=q_grid, y=model_p, mode="lines",
                name=f"{z:g}M {label_side} · model", legendgroup=f"mc-inv-{side}-{j}",
                line=dict(color=color, dash=dash, width=2),
            ))

            empirical = sorted(
                [
                    row for row in inventory_rows
                    if str(row.get("side", "")).lower() == side
                    and abs(float(row.get("size", 0.0)) - z) <= 1e-9
                    and int(row.get("requests", 0)) > 0
                ],
                key=lambda row: float(row.get("inventory", 0.0)),
            )
            if empirical:
                x = [float(row.get("inventory", 0.0)) for row in empirical]
                requests = [int(row.get("requests", 0)) for row in empirical]
                wins = [int(row.get("wins", 0)) for row in empirical]
                y = [w / n for w, n in zip(wins, requests)]
                inventory_fig.add_trace(go.Scatter(
                    x=x, y=y, mode="markers",
                    name=f"{z:g}M {label_side} · MC", legendgroup=f"mc-inv-{side}-{j}",
                    marker=dict(color=color, size=7, symbol="circle-open"),
                    customdata=np.column_stack([wins, requests]),
                    hovertemplate=(
                        f"{z:g}M {label_side}<br>Inventory=%{{x:.3f}}M"
                        "<br>MC hit ratio=%{y:.1%}<br>Wins=%{customdata[0]:.0f} / RFQs=%{customdata[1]:.0f}<extra></extra>"
                    ),
                ))
    inventory_fig.update_layout(
        title=f"Implied hit ratios by inventory · {tier_name}",
        xaxis_title="Inventory q [EUR M]", yaxis_title="Win probability", yaxis_range=[0, 1],
        template="trinity_dark",
    )
    inventory_fig.update_yaxes(tickformat=".0%")
    return arrival_fig, win_fig, inventory_fig


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
    one_side_total = sum(0.5 * _source_total_rfq_rate(tier, source) for source in _tier_flow_sources(tier))
    pricing = np.asarray(tier["sizes"], dtype=float)
    for z in _rfq_size_grid(tier):
        rate = _activity(tier, float(z))
        size_share = rate / one_side_total if one_side_total > 0.0 else 0.0
        is_knot = bool(np.any(np.isclose(pricing, float(z), rtol=0.0, atol=1e-12)))
        rows.append([
            f"{float(z):g}" + (" · price knot" if is_knot else ""),
            f"{100.0 * size_share:.2f}%", f"{rate:.6g}", f"{_delta50(tier,float(z)):.4f}",
            f"{float(tier['flow']['steepness']):.4g}", f"{float(tier.get('feeEurPerEurM', 0.0)):.3f}",
        ])
    return _simple_table(["Size [M]","RFQ size share","λ_RFQ(z) [1/min]","δ50(z)","Steepness","Fee [EUR / EURm]"], rows)


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
    fig=go.Figure(go.Scatter(x=q,y=t,mode="lines",name="legacy t(|q|)"))
    fig.update_layout(title="Legacy internalization-time benchmark (not used by HJB)",xaxis_title="|q| [EUR M]",yaxis_title="Minutes",template="trinity_dark")
    return fig


def markout_exposure_figure(solution: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    curves = solution.get("markoutExposure", []) if solution else []
    for curve in curves:
        q = np.asarray(curve.get("qGrid", []), dtype=float)
        r = np.asarray(curve.get("effectiveInventory", []), dtype=float)
        if q.size and r.size == q.size:
            tau = float(curve.get("tauMinutes", 0.0))
            fig.add_trace(go.Scatter(x=q, y=r, mode="lines+markers", name=f"tau={tau:g} min"))
    if curves:
        q = np.asarray(curves[0].get("qGrid", []), dtype=float)
        if q.size:
            fig.add_trace(go.Scatter(x=q, y=q, mode="lines", name="No internalization: r(q)=q", line={"dash":"dot"}))
    fig.update_layout(
        title="Policy-implied exponentially weighted inventory exposure",
        xaxis_title="Starting inventory q [EUR M]",
        yaxis_title="Effective inventory exposed to markout [EUR M]",
        template="trinity_dark",
    )
    return fig


def markout_figure(tier: dict[str, Any]) -> go.Figure:
    t = np.linspace(0.0, 15.0, 300)
    values = _markout_eur_per_eurm(tier, t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=values, mode="lines", name="Markout"))
    fig.update_layout(
        title=f"RFQ markout · {tier['name']}",
        xaxis_title="Time since trade [min]",
        yaxis_title="Adverse markout [EUR / EURm traded]",
        template="trinity_dark",
        uirevision="tier-markout-eur-per-eurm",
    )
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


def _ecn_flow_sources(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    e = cfg.get("passiveEcn", {})
    configured = e.get("flowSources")
    if not configured:
        configured = [{
            "name": str(cfg.get("targetPair", "direct")),
            "flow": e.get("flow", {}),
            "tradeSizeProbabilities": e.get("tradeSizeProbabilities", ECN_TRADE_SIZE_PROBABILITIES),
            "mapping": {"type": "identity"},
        }]

    target_spread = float(cfg.get("spreadPips", 0.0))
    out = []
    for raw in configured:
        flow = raw.get("flow", {})
        mapping = raw.get("mapping", {}) or {}
        mapping_type = str(mapping.get("type", "identity"))
        alpha = 1.0
        size_scale = 1.0
        if mapping_type == "crossed":
            cross_mid = float(mapping.get("crossMid", 0.0))
            if cross_mid <= 0.0:
                raise ValueError("Crossed ECN flow source requires positive cross mid")
            # Crossing the target quote scales mid and spread identically, so
            # d_source == d_target.  Cross mid is still required for size conversion.
            alpha = 1.0
            size_scale = cross_mid
        elif mapping_type == "affine":
            alpha = float(mapping.get("deltaScale", 1.0))
            size_scale = float(mapping.get("sourceSizePerTarget", 1.0))
        elif mapping_type != "identity":
            raise ValueError(f"Unsupported ECN flow-source mapping {mapping_type}")
        probs = _ecn_size_probabilities(raw)
        source_sizes = [1.0, 0.75, 0.50, 0.25, 0.10]
        mean_trade_size = sum(z * p for z, p in zip(source_sizes, probs))
        out.append({
            "name": str(raw.get("name") or raw.get("pair") or "source"),
            "A": float(flow.get("A", 0.0)),
            "k": float(flow.get("k", 1.0)),
            "tradeSizeProbabilities": probs,
            "meanTradeSize": mean_trade_size,
            "meanTargetTradeSize": mean_trade_size / size_scale,
            "alpha": alpha,
            "sourceSizePerTarget": size_scale,
            "mapping": mapping,
        })
    return out


def _ecn_source_fill_distribution(source: dict[str, Any], target_posted_size: float) -> list[tuple[float, float]]:
    """Exact target-inventory fill atoms for one discrete ECN source."""
    posted = float(target_posted_size)
    if posted <= 0.0:
        return []
    scale = float(source.get("sourceSizePerTarget", 1.0))
    if scale <= 0.0:
        return []
    probs = _ecn_size_probabilities(source)
    source_sizes = [1.0, 0.75, 0.50, 0.25, 0.10]
    source_posted = posted * scale
    merged: dict[float, float] = {}
    for size, probability in zip(source_sizes, probs):
        fill = min(size, source_posted) / scale
        # Stable key because the support is tiny and deterministic.
        key = round(float(fill), 12)
        merged[key] = merged.get(key, 0.0) + float(probability)
    return sorted((fill, probability) for fill, probability in merged.items() if probability > 0.0)


def _ecn_source_full_fill_probability(source: dict[str, Any], target_posted_size: float) -> float:
    posted = float(target_posted_size)
    return sum(p for fill, p in _ecn_source_fill_distribution(source, posted)
               if abs(fill - posted) <= 1e-10)


def _ecn_source_expected_fill(source: dict[str, Any], target_posted_size: float) -> float:
    return sum(fill * p for fill, p in _ecn_source_fill_distribution(source, target_posted_size))


def _ecn_source_arrival_rate(source: dict[str, Any], delta):
    d = np.asarray(delta, dtype=float)
    exponent = -float(source["k"]) * float(source["alpha"]) * (0.5 - d)
    return float(source["A"]) * np.exp(np.clip(exponent, -745.0, 709.0))


def _ecn_arrival_rate(cfg: dict[str, Any], delta):
    d = np.asarray(delta, dtype=float)
    total = np.zeros_like(d, dtype=float)
    for source in _ecn_flow_sources(cfg):
        total = total + _ecn_source_arrival_rate(source, d)
    return float(total) if total.ndim == 0 else total


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
    z = float(e.get("quoteSize", 1.0))
    fee = float(e.get("makerFeeEurPerEurM", 0.0))
    rows = []
    for source in _ecn_flow_sources(cfg):
        mapping = source.get("mapping", {})
        mapping_type = str(mapping.get("type", "identity"))
        if mapping_type == "crossed":
            mapping_text = (
                f"delta 1:1; cross={float(mapping.get('crossMid', 0.0)):.6g}; "
                f"size scale={float(source['sourceSizePerTarget']):.6g}"
            )
        else:
            mapping_text = "direct"
        mean_target = float(source["meanTargetTradeSize"])
        p_full = _ecn_source_full_fill_probability(source, z)
        expected_fill = _ecn_source_expected_fill(source, z)
        probs = source["tradeSizeProbabilities"]
        prob_text = " / ".join(f"{100.0 * float(p):.1f}%" for p in probs)
        rows.append([
            str(source["name"]),
            f"{float(source['A']):.6g}",
            f"{float(source['k']):.4g}",
            prob_text,
            f"{float(source['meanTradeSize']):.4g}",
            f"{mean_target:.4g}",
            f"{expected_fill:.4g}",
            f"{100.0 * p_full:.1f}%",
            mapping_text,
            f"{float(source['sourceSizePerTarget']):.6g}",
            f"{z:g}",
            f"{fee:.4f}",
        ])
    return _simple_table(
        ["ECN source", "A = mid intensity/side", "k",
         "Size probs 1M / 750k / 500k / 250k / 100k", "Mean trade [source M]",
         "Mean trade [target M]", "E[fill] [target M]", "P(full fill)", "Mapping",
         "Source size / target size", "Target-equivalent quote [M]", "Maker fee [EUR / EURm]"],
        rows,
    )


def passive_ecn_fill_figure(cfg: dict[str, Any]) -> go.Figure:
    e = cfg.get("passiveEcn", {})
    lo = float(e.get("minDelta", 0.0))
    hi = float(e.get("maxDelta", 0.5))
    grid = np.linspace(lo, hi, 240)
    fig = go.Figure()
    sources = _ecn_flow_sources(cfg)
    for source in sources:
        fig.add_trace(go.Scatter(
            x=grid, y=_ecn_source_arrival_rate(source, grid), mode="lines",
            name=str(source["name"]), opacity=.72,
        ))
    if len(sources) > 1:
        fig.add_trace(go.Scatter(
            x=grid, y=_ecn_arrival_rate(cfg, grid), mode="lines",
            name="Total ECN", line=dict(width=4),
        ))
    deltas = np.asarray(e.get("deltas", []), dtype=float)
    if deltas.size:
        fig.add_trace(go.Scatter(
            x=deltas, y=_ecn_arrival_rate(cfg, deltas), mode="markers",
            name="Total · 1%-point grid", marker=dict(size=5),
        ))
    fig.update_layout(
        title="ECN fill intensity · source curves mapped from master target-pair delta",
        xaxis_title="Master price-improvement delta d (0 = touch, 0.5 = mid)",
        yaxis_title="Fill intensity [trades/min]", template="trinity_dark",
    )
    fig.update_xaxes(range=[lo, hi])
    return fig


def passive_ecn_hit_ratio_figure(cfg: dict[str, Any]) -> go.Figure:
    e = cfg.get("passiveEcn", {})
    lo = float(e.get("minDelta", 0.0))
    hi = float(e.get("maxDelta", 0.5))
    grid = np.linspace(lo, hi, 240)
    sources = _ecn_flow_sources(cfg)
    fig = go.Figure()
    for source in sources:
        A = float(source["A"])
        relative = _ecn_source_arrival_rate(source, grid) / A if A > 0.0 else np.zeros_like(grid)
        fig.add_trace(go.Scatter(x=grid, y=relative, mode="lines", name=str(source["name"])))
    total_mid = sum(float(s["A"]) for s in sources)
    if len(sources) > 1 and total_mid > 0.0:
        fig.add_trace(go.Scatter(
            x=grid, y=_ecn_arrival_rate(cfg, grid) / total_mid, mode="lines",
            name="Total / total mid intensity", line=dict(width=4),
        ))
    fig.update_layout(
        title="ECN relative intensity by flow source",
        xaxis_title="Master price-improvement delta d (0 = touch, 0.5 = mid)",
        yaxis_title="Intensity relative to source mid", template="trinity_dark",
    )
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
        spread_pips = float(cfg.get("spreadPips", 0.0))
        bid_px = np.where(bid_active, -spread_pips * (0.5 - bid_delta), np.nan)
        ask_px = np.where(ask_active, spread_pips * (0.5 - ask_delta), np.nan)
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
    fig.update_layout(title="Optimal passive ECN delta vs inventory",xaxis_title="Inventory q [EUR M]",yaxis_title="Price-improvement delta d",template="trinity_dark")
    return fig

def passive_ecn_table(solution: dict[str, Any]) -> html.Table | html.Div:
    e=solution.get("passiveEcn") if solution else None
    if not e:
        return html.Div("Passive ECN disabled",className="muted-note")
    rows=[]
    for q,bd,ba,ad,aa in zip(e["qGrid"],e["bidDelta"],e["bidActive"],e["askDelta"],e["askActive"]):
        rows.append([f"{q:g}", f"{bd:g}" if ba else "OFF", f"{ad:g}" if aa else "OFF"])
    return _simple_table(["Inventory", "Bid delta", "Ask delta"], rows)


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
    # Legacy tau(q) parameters are no longer user controls.  Keep the Dash
    # components mounted invisibly because the config/callback plumbing still
    # carries the legacy benchmark fields for backwards compatibility.
    html.Div([
        number_input("Tau 0 [min]", "tau0", DEFAULT["internalization"]["tau0"], "any"),
        number_input("Tau 1", "tau1", DEFAULT["internalization"]["tau1"], "any"),
        number_input("Tau 2", "tau2", DEFAULT["internalization"]["tau2"], "any"),
    ], style={"display": "none"}),
    html.Div("Pricing tiers", className="section-title"),
    *[tier_panel(i, t, open_=(i == 0)) for i, t in enumerate(DEFAULT["tiers"])],
    panel("Dark pool", [
        checkbox("Enabled", "dp-enabled", DEFAULT["darkPool"]["enabled"]),
        html.Div("Zero-inflated Poisson fill-size model", className="muted-note"),
        html.Div([
            number_input("λ / side / min", "dp-lambda", DEFAULT["darkPool"]["lambda"], "any"),
            number_input("μ", "dp-mu", DEFAULT["darkPool"]["mu"], "any"),
            number_input("p0", "dp-p0", DEFAULT["darkPool"]["p0"], "any"),
            number_input("Broker fee [EUR / EURm]", "dp-fee", DEFAULT["darkPool"]["feeEurPerEurM"], "any"),
        ], className="grid-2"),
        html.Div("Hedge-only: only the inventory-reducing side is posted and fills may not cross through flat.", className="muted-note"),
        text_input("Posted sizes", "dp-sizes", ", ".join(str(x) for x in DEFAULT["darkPool"]["postedSizes"])),
    ]),
    panel("ECN", [
        checkbox("Enabled", "ecn-enabled", DEFAULT["passiveEcn"]["enabled"]),
        html.Div([
            html.Button("Configure flow sources", id="open-ecn-flow-sources", className="secondary-button compact-button", n_clicks=0),
        ], className="flow-source-config-line"),
        # Direct ECN calibration remains mounted for config building.
        # The modal editor synchronizes A/k and the direct size-probability Store.
        html.Div([
            number_input("A at mid / side [trades/min]", "ecn-A", DEFAULT["passiveEcn"]["flow"]["A"], "any"),
            number_input("k", "ecn-k", DEFAULT["passiveEcn"]["flow"]["k"], "any"),
        ], style={"display": "none"}),
        html.Div([
            number_input("Min delta", "ecn-dmin", DEFAULT["passiveEcn"]["minDelta"], 0.01),
            number_input("Max delta", "ecn-dmax", DEFAULT["passiveEcn"]["maxDelta"], 0.01),
        ], className="grid-2"),
        html.Div([
            number_input("Quote size [M]", "ecn-size", DEFAULT["passiveEcn"]["quoteSize"], "any"),
            number_input("Maker fee [EUR / EURm]", "ecn-fee", DEFAULT["passiveEcn"]["makerFeeEurPerEurM"], "any"),
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
            diagnostic_expander("Policy-implied markout exposure (resolvent)", graph("internalization-chart")),
            diagnostic_expander("Bellman residual", graph("residual-chart")),
            diagnostic_expander("Value-function change", graph("value-change-chart")),
            diagnostic_expander("Markout resolvent change", graph("markout-resolvent-change-chart")),
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
            html.Div([
                select(
                    "Measure",
                    "fill-tier-metric",
                    [
                        {"label": "Number of trades", "value": "count"},
                        {"label": "Total volume", "value": "volume"},
                    ],
                    "count",
                ),
            ], className="toolbar toolbar-wide fill-tier-toolbar"),
            graph("fill-counts-chart", 360),
            html.Div("Fill tape", className="card-title table-section-title"),
            html.Div(id="fill-tape", className="table-wrap tall-table"),
        ]),
        diagnostic_expander("RFQ hit ratios", graph("rfq-realized-hit-chart", 360)),
        diagnostic_expander("RFQ Monte Carlo validation", [
            html.Div([
                select("Tier", "mc-rfq-tier-select", [], None),
            ], className="toolbar toolbar-wide"),
            html.Div(
                "Markers are empirical estimates from all simulated paths; lines are the corresponding model curves used by the simulator. "
                "The arrival curve is reported per side. Win-probability calibration excludes RFQs blocked only by the hard inventory limit, while the inventory chart includes that boundary effect.",
                className="muted-note",
            ),
            graph("mc-rfq-arrival-validation-chart", 380),
            graph("mc-rfq-win-validation-chart", 440),
            graph("mc-rfq-inventory-validation-chart", 500),
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
    dcc.Store(id="mc-job-store"),
    dcc.Interval(id="mc-progress-poll", interval=250, n_intervals=0, disabled=True),
    dcc.Store(id="extra-flow-sources-store", data={"0": [], "1": [], "2": []}),
    dcc.Store(id="flow-source-tier-store", data=None),
    dcc.Store(id="flow-source-selected-store", data="direct"),
    dcc.Store(id="flow-source-modal-open-store", data=False),
    dcc.Store(id="ecn-extra-flow-sources-store", data=[]),
    dcc.Store(id="ecn-direct-size-probs-store", data=list(DEFAULT["passiveEcn"]["tradeSizeProbabilities"])),
    dcc.Store(id="ecn-flow-source-selected-store", data="direct"),
    dcc.Store(id="ecn-flow-source-modal-open-store", data=False),
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Configure flow sources", className="modal-title"),
                    html.Div("Choose the target FX pair, then add the FX pairs whose customer flow should be mapped into that target risk.", className="muted-note modal-note"),
                ]),
                html.Button("×", id="flow-source-close", className="modal-close", n_clicks=0),
            ], className="modal-head"),
            html.Div(id="flow-source-tier-label", className="flow-source-target-label"),
            html.Div([
                html.Label([
                    html.Span("Target / priced FX pair"),
                    dcc.Dropdown(id="flow-target-pair", options=FX_PAIR_OPTIONS, value=DEFAULT.get("targetPair", "EURSEK"), placeholder="Select target FX pair", clearable=False, className="field-dropdown", persistence=True, persistence_type="memory"),
                ], className="field"),
            ], className="flow-target-picker"),
            html.Div([
                html.Div(id="flow-source-cards", className="flow-pair-cards"),
                html.Div(id="flow-source-settings"),
            ], id="flow-source-editor", className="flow-source-editor"),
            html.Div([
                html.Button("+ Add FX pair", id="flow-source-add", className="secondary-button", n_clicks=0),
                html.Div([
                    html.Button("Close", id="flow-source-cancel", className="ghost-button", n_clicks=0),
                    html.Button("Apply", id="flow-source-apply", className="primary-button", n_clicks=0),
                ], className="modal-actions-right"),
            ], className="modal-actions"),
        ], className="flow-source-modal-card"),
    ], id="flow-source-modal", className="flow-source-modal", style={"display": "none"}),
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Configure ECN flow sources", className="modal-title"),
                    html.Div("The ECN optimizer chooses one master delta on the target pair. Crossed ECN pairs inherit that quote through the same FX transformation used by Tier flow sources, while keeping their own A, k and discrete 1M/750k/500k/250k/100k trade-size probabilities.", className="muted-note modal-note"),
                ]),
                html.Button("×", id="ecn-flow-source-close", className="modal-close", n_clicks=0),
            ], className="modal-head"),
            html.Div(id="ecn-flow-source-target-label", className="flow-source-target-label"),
            html.Div([
                html.Div(id="ecn-flow-source-cards", className="flow-pair-cards"),
                html.Div(id="ecn-flow-source-settings"),
            ], className="flow-source-editor"),
            html.Div([
                html.Button("+ Add FX pair", id="ecn-flow-source-add", className="secondary-button", n_clicks=0),
                html.Div([
                    html.Button("Close", id="ecn-flow-source-cancel", className="ghost-button", n_clicks=0),
                    html.Button("Apply", id="ecn-flow-source-apply", className="primary-button", n_clicks=0),
                ], className="modal-actions-right"),
            ], className="modal-actions"),
        ], className="flow-source-modal-card"),
    ], id="ecn-flow-source-modal", className="flow-source-modal", style={"display": "none"}),
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Running Monte Carlo", className="modal-title"),
                    html.Div(
                        "The progress bar reflects completed native simulation paths.",
                        className="muted-note modal-note",
                    ),
                ]),
            ], className="modal-head"),
            html.Div([
                html.Div(id="mc-progress-message", children="Starting simulation…", className="mc-progress-message"),
                html.Progress(id="mc-progress-bar", value="0", max="100", className="mc-progress-bar"),
                html.Div([
                    html.Span(id="mc-progress-count", children="0 / 0 paths"),
                    html.Span(id="mc-progress-percent", children="0%"),
                ], className="mc-progress-meta"),
            ], className="mc-progress-body"),
        ], className="mc-progress-modal-card"),
    ], id="mc-progress-modal", className="flow-source-modal mc-progress-modal", style={"display": "none"}),
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
        "enabled", "name", "sizes", "rfq-size-step", "A0", "theta", "beta", "steep", "shift", "vshift",
        "fee", "markout-enabled", "markout-level", "markout-tau", "dmin", "dmax",
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
CONFIG_FIELDS.append(("ecn-size-probs", State("ecn-direct-size-probs-store", "data")))

CONFIG_KEYS = [key for key, _ in CONFIG_FIELDS]
CONFIG_STATES = [state for _, state in CONFIG_FIELDS]


@app.callback(
    Output("flow-source-add", "disabled"),
    Input("flow-target-pair", "value"),
)
def disable_add_flow_source_without_target(target_pair):
    return not bool(target_pair)


@app.callback(
    Output("flow-source-modal-open-store", "data"),
    Output("flow-source-tier-store", "data"),
    Input({"type": "open-flow-sources", "tier": ALL}, "n_clicks"),
    Input("flow-source-close", "n_clicks"),
    Input("flow-source-cancel", "n_clicks"),
    Input("flow-source-apply", "n_clicks"),
    State("flow-source-modal-open-store", "data"),
    State("flow-source-tier-store", "data"),
    prevent_initial_call=True,
)
def set_flow_source_modal_open(_open_clicks, _close, _cancel, _apply, is_open, selected_tier):
    """Own modal visibility independently from editor state.

    Source-pair edits rebuild the editor subtree; they must never participate in
    opening or closing the modal.  Keeping visibility in a dedicated Store also
    prevents a component remount from resetting the popup.
    """
    trigger = ctx.triggered_id
    if isinstance(trigger, dict) and trigger.get("type") == "open-flow-sources":
        return True, int(trigger["tier"])
    if trigger in {"flow-source-close", "flow-source-cancel", "flow-source-apply"}:
        return False, selected_tier
    return bool(is_open), selected_tier


@app.callback(
    Output("flow-source-modal", "style"),
    Input("flow-source-modal-open-store", "data"),
)
def render_flow_source_modal_visibility(is_open):
    return {"display": "flex"} if is_open else {"display": "none"}


@app.callback(
    Output("flow-source-selected-store", "data"),
    Input("flow-source-tier-store", "data"),
    Input("flow-target-pair", "value"),
    Input({"type": "flow-pair-select", "key": ALL}, "n_clicks"),
    State("flow-source-selected-store", "data"),
)
def select_flow_source(_tier_index, _target_pair, _clicks, selected):
    trigger = ctx.triggered_id
    if trigger in ("flow-source-tier-store", "flow-target-pair"):
        return "direct"
    if isinstance(trigger, dict) and trigger.get("type") == "flow-pair-select":
        # Pattern-matching Inputs are also triggered when a newly-rendered card
        # enters the layout.  That is not a user click: its n_clicks value is 0.
        # Ignoring those mount events prevents a card refresh from changing the
        # active source while the editor is being created.
        triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
        if not triggered_value:
            return no_update
        return str(trigger.get("key", "direct"))
    return selected or "direct"


@app.callback(
    Output("flow-source-cards", "children"),
    Output("flow-source-tier-label", "children"),
    Input("flow-source-tier-store", "data"),
    Input("flow-target-pair", "value"),
    Input("flow-source-selected-store", "data"),
    Input("extra-flow-sources-store", "data"),
)
def render_flow_source_cards(tier_index, target_pair, selected, data):
    """Render only the source selector cards.

    The cards may react to edits in ``extra-flow-sources-store`` (for example
    changing a card label from "Select FX pair" to "USDSEK").  The active
    settings form is deliberately rendered by a separate callback so editing a
    dropdown never destroys and remounts that dropdown mid-selection.
    """
    if tier_index is None:
        return [], ""
    ti = int(tier_index)
    if not target_pair:
        return [], f"Tier {ti + 1}"

    extras = (data or {}).get(str(ti), [])
    selected = selected or "direct"
    cards = [flow_pair_card("direct", str(target_pair), selected == "direct")]
    for i, source in enumerate(extras):
        key = f"extra-{i}"
        cards.append(flow_pair_card(key, str(source.get("pair") or "Select FX pair"), selected == key, removable=True))
    return cards, f"Tier {ti + 1} · target risk {target_pair}"


@app.callback(
    Output("flow-source-settings", "children"),
    Input("flow-source-tier-store", "data"),
    Input("flow-target-pair", "value"),
    Input("flow-source-selected-store", "data"),
    State("extra-flow-sources-store", "data"),
    *[State(f"t{i}-{suffix}", "value") for i in range(3) for suffix in ("A0", "theta", "beta", "steep", "shift", "vshift")],
)
def render_flow_source_settings(tier_index, target_pair, selected, data, *direct_values):
    """Render the settings form only when its identity changes.

    ``extra-flow-sources-store`` and the direct-flow inputs are *State*, not
    Input.  Consequently ordinary edits persist into the stores without
    remounting the form that the user is currently interacting with.
    """
    if tier_index is None:
        return []
    ti = int(tier_index)
    if not target_pair:
        return html.Div(
            "Select the target / priced FX pair above to configure its direct flow and add crossed flow sources.",
            className="muted-note flow-source-empty",
        )

    extras = (data or {}).get(str(ti), [])
    selected = selected or "direct"
    offset = ti * 6
    vals = direct_values[offset:offset + 6]
    direct_flow = {
        "A0": vals[0], "theta": vals[1], "beta": vals[2],
        "steepness": vals[3], "shift": vals[4], "volumeShift": vals[5],
    }

    if selected == "direct":
        source = _direct_flow_source(direct_flow, str(target_pair))
    else:
        try:
            idx = int(str(selected).split("-", 1)[1])
            source = extras[idx]
        except (ValueError, IndexError):
            source = _direct_flow_source(direct_flow, str(target_pair))
            selected = "direct"

    return flow_source_settings_editor(selected, source, str(target_pair))


@app.callback(
    Output("flow-edit-cross-pair", "value"),
    Input("flow-edit-pair", "value"),
    State("flow-target-pair", "value"),
    prevent_initial_call=True,
)
def derive_flow_source_cross_pair(source_pair, target_pair):
    """Populate the economically implied hedge cross without rebuilding the form."""
    if not source_pair or not target_pair:
        return no_update
    return _derive_cross_pair(target_pair, source_pair) or no_update


@app.callback(
    Output("flow-edit-cross-mid", "value"),
    Input("flow-edit-cross-pair", "value"),
    prevent_initial_call=True,
)
def populate_flow_source_cross_mid(cross_pair):
    """Populate the temporary default reference mid in-place."""
    mid = default_fx_mid(cross_pair)
    return mid if mid is not None else no_update


@app.callback(
    Output("extra-flow-sources-store", "data"),
    Output("flow-source-selected-store", "data", allow_duplicate=True),
    Output("t0-A0", "value"), Output("t0-theta", "value"), Output("t0-beta", "value"), Output("t0-steep", "value"), Output("t0-shift", "value"), Output("t0-vshift", "value"),
    Output("t1-A0", "value"), Output("t1-theta", "value"), Output("t1-beta", "value"), Output("t1-steep", "value"), Output("t1-shift", "value"), Output("t1-vshift", "value"),
    Output("t2-A0", "value"), Output("t2-theta", "value"), Output("t2-beta", "value"), Output("t2-steep", "value"), Output("t2-shift", "value"), Output("t2-vshift", "value"),
    Input("flow-source-add", "n_clicks"),
    Input("flow-source-remove-selected", "n_clicks"),
    Input("flow-edit-pair", "value"), Input("flow-edit-cross-pair", "value"),
    Input("flow-edit-cross-mid", "value"),
    Input("flow-edit-A0", "value"), Input("flow-edit-theta", "value"), Input("flow-edit-beta", "value"),
    Input("flow-edit-steep", "value"), Input("flow-edit-shift", "value"), Input("flow-edit-vshift", "value"),
    State("flow-source-tier-store", "data"), State("flow-target-pair", "value"), State("flow-source-selected-store", "data"), State("extra-flow-sources-store", "data"),
    *[State(f"t{i}-{suffix}", "value") for i in range(3) for suffix in ("A0", "theta", "beta", "steep", "shift", "vshift")],
    prevent_initial_call=True,
)
def update_flow_source_editor(_add, _remove, pair, cross_pair, cross_mid, A0, theta, beta, steep, shift, vshift,
                              tier_index, target_pair, selected, data, *direct_values):
    if tier_index is None:
        return (no_update,) * 20
    ti = int(tier_index)
    key = str(ti)
    out = deepcopy(data or {"0": [], "1": [], "2": []})
    direct = list(direct_values)
    selected = selected or "direct"
    trigger = ctx.triggered_id

    if trigger == "flow-source-add":
        # n_clicks=0 can be observed during component initialisation; only a
        # positive click count represents an actual Add action.
        if not _add:
            return (no_update,) * 20
        if not target_pair:
            return out, selected, *direct
        out.setdefault(key, []).append(_default_cross_flow_source(len(out.get(key, [])), target_pair))
        new_selected = f"extra-{len(out[key]) - 1}"
        return out, new_selected, *direct

    if trigger == "flow-source-remove-selected":
        # The Remove button is dynamically mounted when a crossed source is
        # selected. Dash can fire the callback once on that mount with
        # n_clicks=0. Treat only a positive click count as a removal.
        if not _remove:
            return (no_update,) * 20
        if selected.startswith("extra-"):
            idx = int(selected.split("-", 1)[1])
            if 0 <= idx < len(out.get(key, [])):
                out[key].pop(idx)
        return out, "direct", *direct

    if selected == "direct":
        if all(x is not None for x in (A0, theta, beta, steep, shift, vshift)):
            offset = ti * 6
            direct[offset:offset + 6] = [float(A0), float(theta), float(beta), float(steep), float(shift), float(vshift)]
        return out, no_update, *direct

    if selected.startswith("extra-"):
        idx = int(selected.split("-", 1)[1])
        if 0 <= idx < len(out.get(key, [])) and all(x is not None for x in (A0, theta, beta, steep, shift, vshift)):
            src = out[key][idx]
            chosen_pair = str(pair) if pair else None
            derived_cross = _derive_cross_pair(target_pair, chosen_pair)
            chosen_cross = derived_cross if trigger == "flow-edit-pair" and derived_cross else (str(cross_pair) if cross_pair else derived_cross)

            # Pair selection is market-data driven: use the temporary default
            # mid for the newly selected cross.  A manually edited Cross mid is
            # preserved for all other triggers.
            chosen_mid = None if cross_mid is None else float(cross_mid)
            if trigger in ("flow-edit-pair", "flow-edit-cross-pair"):
                looked_up_mid = default_fx_mid(chosen_cross)
                if looked_up_mid is not None:
                    chosen_mid = looked_up_mid
            if chosen_pair and chosen_mid is None:
                raise ValueError(f"No default mid available for cross {chosen_cross}")
            if chosen_mid is not None and chosen_mid <= 0.0:
                raise ValueError("Cross mid must be positive")
            src.update({"name": chosen_pair or "Select FX pair", "pair": chosen_pair, "crossPair": chosen_cross})
            src["mapping"] = {"type": "crossed", "crossMid": chosen_mid}
            src["flow"] = {"A0": float(A0), "theta": float(theta), "beta": float(beta), "steepness": float(steep), "shift": float(shift), "volumeShift": float(vshift)}
        return out, no_update, *direct

    return out, no_update, *direct


@app.callback(
    Output("ecn-flow-source-modal-open-store", "data"),
    Input("open-ecn-flow-sources", "n_clicks"),
    Input("ecn-flow-source-close", "n_clicks"),
    Input("ecn-flow-source-cancel", "n_clicks"),
    Input("ecn-flow-source-apply", "n_clicks"),
    State("ecn-flow-source-modal-open-store", "data"),
    prevent_initial_call=True,
)
def set_ecn_flow_source_modal_open(_open, _close, _cancel, _apply, is_open):
    trigger = ctx.triggered_id
    if trigger == "open-ecn-flow-sources" and _open:
        return True
    if trigger in {"ecn-flow-source-close", "ecn-flow-source-cancel", "ecn-flow-source-apply"}:
        return False
    return bool(is_open)


@app.callback(
    Output("ecn-flow-source-modal", "style"),
    Input("ecn-flow-source-modal-open-store", "data"),
)
def render_ecn_flow_source_modal_visibility(is_open):
    return {"display": "flex"} if is_open else {"display": "none"}


@app.callback(
    Output("ecn-flow-source-add", "disabled"),
    Input("flow-target-pair", "value"),
)
def disable_add_ecn_flow_source_without_target(target_pair):
    return not bool(target_pair)


@app.callback(
    Output("ecn-flow-source-selected-store", "data"),
    Input("flow-target-pair", "value"),
    Input({"type": "ecn-flow-pair-select", "key": ALL}, "n_clicks"),
    State("ecn-flow-source-selected-store", "data"),
)
def select_ecn_flow_source(_target_pair, _clicks, selected):
    trigger = ctx.triggered_id
    if trigger == "flow-target-pair":
        return "direct"
    if isinstance(trigger, dict) and trigger.get("type") == "ecn-flow-pair-select":
        triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
        if not triggered_value:
            return no_update
        return str(trigger.get("key", "direct"))
    return selected or "direct"


@app.callback(
    Output("ecn-flow-source-cards", "children"),
    Output("ecn-flow-source-target-label", "children"),
    Input("flow-target-pair", "value"),
    Input("ecn-flow-source-selected-store", "data"),
    Input("ecn-extra-flow-sources-store", "data"),
)
def render_ecn_flow_source_cards(target_pair, selected, data):
    if not target_pair:
        return [], "Select the target / priced FX pair in a Tier flow-source dialog first"
    extras = data or []
    selected = selected or "direct"
    cards = [ecn_flow_pair_card("direct", str(target_pair), selected == "direct")]
    for i, source in enumerate(extras):
        key = f"extra-{i}"
        cards.append(ecn_flow_pair_card(key, str(source.get("pair") or "Select FX pair"), selected == key, removable=True))
    return cards, f"ECN · target risk {target_pair}"


@app.callback(
    Output("ecn-flow-source-settings", "children"),
    Input("flow-target-pair", "value"),
    Input("ecn-flow-source-selected-store", "data"),
    State("ecn-extra-flow-sources-store", "data"),
    State("ecn-A", "value"), State("ecn-k", "value"), State("ecn-direct-size-probs-store", "data"),
)
def render_ecn_flow_source_settings(target_pair, selected, data, direct_A, direct_k, direct_size_probs):
    if not target_pair:
        return html.Div(
            "Select the target / priced FX pair in a Tier flow-source dialog first.",
            className="muted-note flow-source-empty",
        )
    extras = data or []
    selected = selected or "direct"
    if selected == "direct":
        source = _direct_ecn_flow_source({"A": direct_A, "k": direct_k}, str(target_pair), direct_size_probs)
    else:
        try:
            idx = int(str(selected).split("-", 1)[1])
            source = extras[idx]
        except (ValueError, IndexError):
            source = _direct_ecn_flow_source({"A": direct_A, "k": direct_k}, str(target_pair), direct_size_probs)
            selected = "direct"
    return ecn_flow_source_settings_editor(selected, source, str(target_pair))


@app.callback(
    Output("ecn-flow-edit-A", "value"),
    Output("ecn-flow-edit-p1m", "value"),
    Output("ecn-flow-edit-p750k", "value"),
    Output("ecn-flow-edit-p500k", "value"),
    Output("ecn-flow-edit-p250k", "value"),
    Input("ecn-flow-edit-pair", "value"),
    State("ecn-flow-source-selected-store", "data"),
    prevent_initial_call=True,
)
def populate_ecn_trade_size_defaults(source_pair, selected):
    # Only crossed sources have a user-editable source-pair dropdown.  When a
    # calibrated pair is selected, seed its empirical size distribution.
    # Unknown pairs preserve the values already shown in the editor.
    if not source_pair or str(selected or "direct") == "direct":
        return no_update, no_update, no_update, no_update, no_update
    key = str(source_pair).upper().strip()
    if key not in {"EURSEK", "USDSEK"}:
        return no_update, no_update, no_update, no_update, no_update
    probs = default_ecn_trade_size_probabilities(key)
    source_A = 3.0 if key == "USDSEK" else no_update
    return source_A, probs[0], probs[1], probs[2], probs[3]


@app.callback(
    Output("ecn-flow-edit-cross-pair", "value"),
    Input("ecn-flow-edit-pair", "value"),
    State("flow-target-pair", "value"),
    prevent_initial_call=True,
)
def derive_ecn_flow_source_cross_pair(source_pair, target_pair):
    if not source_pair or not target_pair:
        return no_update
    return _derive_cross_pair(target_pair, source_pair) or no_update


@app.callback(
    Output("ecn-flow-edit-cross-mid", "value"),
    Input("ecn-flow-edit-cross-pair", "value"),
    prevent_initial_call=True,
)
def populate_ecn_flow_source_cross_mid(cross_pair):
    mid = default_fx_mid(cross_pair)
    return mid if mid is not None else no_update


@app.callback(
    Output("ecn-flow-edit-p100k-note", "children"),
    Input("ecn-flow-edit-p1m", "value"), Input("ecn-flow-edit-p750k", "value"),
    Input("ecn-flow-edit-p500k", "value"), Input("ecn-flow-edit-p250k", "value"),
)
def render_ecn_100k_residual(p1m, p750k, p500k, p250k):
    try:
        residual = _ecn_size_probabilities_from_four(p1m, p750k, p500k, p250k)[4]
        return f"P(100k) = 1 − others = {residual:.6f}"
    except Exception as exc:
        return f"Invalid size probabilities · {exc}"


@app.callback(
    Output("ecn-extra-flow-sources-store", "data"),
    Output("ecn-flow-source-selected-store", "data", allow_duplicate=True),
    Output("ecn-A", "value"), Output("ecn-k", "value"), Output("ecn-direct-size-probs-store", "data"),
    Input("ecn-flow-source-add", "n_clicks"),
    Input("ecn-flow-source-remove-selected", "n_clicks"),
    Input("ecn-flow-edit-pair", "value"), Input("ecn-flow-edit-cross-pair", "value"),
    Input("ecn-flow-edit-cross-mid", "value"),
    Input("ecn-flow-edit-A", "value"), Input("ecn-flow-edit-k", "value"),
    Input("ecn-flow-edit-p1m", "value"), Input("ecn-flow-edit-p750k", "value"),
    Input("ecn-flow-edit-p500k", "value"), Input("ecn-flow-edit-p250k", "value"),
    State("flow-target-pair", "value"), State("ecn-flow-source-selected-store", "data"),
    State("ecn-extra-flow-sources-store", "data"), State("ecn-A", "value"), State("ecn-k", "value"),
    State("ecn-direct-size-probs-store", "data"),
    prevent_initial_call=True,
)
def update_ecn_flow_source_editor(_add, _remove, pair, cross_pair, cross_mid, A, k,
                                  p1m, p750k, p500k, p250k, target_pair, selected, data,
                                  direct_A, direct_k, direct_size_probs):
    out = deepcopy(data or [])
    selected = selected or "direct"
    trigger = ctx.triggered_id

    if trigger == "ecn-flow-source-add":
        if not _add:
            return (no_update,) * 5
        if not target_pair:
            return out, selected, direct_A, direct_k, direct_size_probs
        out.append(_default_cross_ecn_flow_source(len(out), target_pair))
        return out, f"extra-{len(out) - 1}", direct_A, direct_k, direct_size_probs

    if trigger == "ecn-flow-source-remove-selected":
        if not _remove:
            return (no_update,) * 5
        if selected.startswith("extra-"):
            idx = int(selected.split("-", 1)[1])
            if 0 <= idx < len(out):
                out.pop(idx)
        return out, "direct", direct_A, direct_k, direct_size_probs

    probs = _ecn_size_probabilities_from_four(p1m, p750k, p500k, p250k)

    if selected == "direct":
        if A is not None and k is not None:
            if float(A) < 0.0:
                raise ValueError("ECN A must be nonnegative")
            if float(k) <= 0.0:
                raise ValueError("ECN k must be positive")
            return out, no_update, float(A), float(k), probs
        return out, no_update, direct_A, direct_k, direct_size_probs

    if selected.startswith("extra-"):
        idx = int(selected.split("-", 1)[1])
        if 0 <= idx < len(out) and all(x is not None for x in (A, k)):
            src = out[idx]
            chosen_pair = str(pair) if pair else None
            derived_cross = _derive_cross_pair(target_pair, chosen_pair)
            chosen_cross = derived_cross if trigger == "ecn-flow-edit-pair" and derived_cross else (str(cross_pair) if cross_pair else derived_cross)

            chosen_mid = None if cross_mid is None else float(cross_mid)
            if trigger in ("ecn-flow-edit-pair", "ecn-flow-edit-cross-pair"):
                looked_up_mid = default_fx_mid(chosen_cross)
                if looked_up_mid is not None:
                    chosen_mid = looked_up_mid
            if chosen_pair and chosen_mid is None:
                raise ValueError(f"No default mid available for cross {chosen_cross}")
            if chosen_mid is not None and chosen_mid <= 0.0:
                raise ValueError("Cross mid must be positive")
            if float(A) < 0.0:
                raise ValueError("ECN A must be nonnegative")
            if float(k) <= 0.0:
                raise ValueError("ECN k must be positive")

            src.update({"name": chosen_pair or "Select FX pair", "pair": chosen_pair, "crossPair": chosen_cross})
            src["mapping"] = {"type": "crossed", "crossMid": chosen_mid}
            src["flow"] = {"A": float(A), "k": float(k)}
            src["tradeSizeProbabilities"] = probs
            src.pop("meanTradeSize", None)
        return out, no_update, direct_A, direct_k, direct_size_probs

    return out, no_update, direct_A, direct_k, direct_size_probs


@app.callback(
    Output("config-store", "data"), Output("solution-store", "data"), Output("status", "children"), Output("status", "className"),
    Input("solve", "n_clicks"), *CONFIG_STATES,
    State("extra-flow-sources-store", "data"), State("ecn-extra-flow-sources-store", "data"),
    State("flow-target-pair", "value"), prevent_initial_call=True,
)
def solve_model(_clicks, *values):
    try:
        *field_values, extra_sources, ecn_extra_sources, target_pair = values
        cfg = build_config(dict(zip(CONFIG_KEYS, field_values, strict=True)))
        for i, tier in enumerate(cfg.get("tiers", [])):
            extras = (extra_sources or {}).get(str(i), [])
            if extras:
                if not target_pair:
                    raise ValueError("Select the target / priced FX pair before adding crossed flow sources")
                target_legs = _pair_legs(target_pair)
                seen_pairs = set()
                for source in extras:
                    source_pair = source.get("pair")
                    if not source_pair:
                        raise ValueError("Select an FX pair for every flow source")
                    source_legs = _pair_legs(source_pair)
                    if not source_legs or not target_legs or source_legs[1] != target_legs[1]:
                        raise ValueError(f"Flow source {source_pair} must share the quote currency of target pair {target_pair}")
                    if source_pair == target_pair:
                        raise ValueError(f"{source_pair} is already the direct target-pair source")
                    if source_pair in seen_pairs:
                        raise ValueError(f"Flow source {source_pair} is configured more than once")
                    seen_pairs.add(source_pair)
                direct = {"name": str(target_pair), "pair": str(target_pair), "flow": deepcopy(tier["flow"]), "mapping": {"type": "identity"}}
                tier["flowSources"] = [direct] + deepcopy(extras)

        ecn_extras = list(ecn_extra_sources or [])
        if ecn_extras:
            if not target_pair:
                raise ValueError("Select the target / priced FX pair before adding crossed ECN flow sources")
            target_legs = _pair_legs(target_pair)
            seen_pairs = set()
            for source in ecn_extras:
                source_pair = source.get("pair")
                if not source_pair:
                    raise ValueError("Select an FX pair for every ECN flow source")
                source_legs = _pair_legs(source_pair)
                if not source_legs or not target_legs or source_legs[1] != target_legs[1]:
                    raise ValueError(f"ECN flow source {source_pair} must share the quote currency of target pair {target_pair}")
                if source_pair == target_pair:
                    raise ValueError(f"{source_pair} is already the direct ECN target-pair source")
                if source_pair in seen_pairs:
                    raise ValueError(f"ECN flow source {source_pair} is configured more than once")
                seen_pairs.add(source_pair)
            ecn = cfg.get("passiveEcn", {})
            direct = {
                "name": str(target_pair), "pair": str(target_pair),
                "flow": deepcopy(ecn.get("flow", {})), "mapping": {"type": "identity"},
                "tradeSizeProbabilities": list(ecn.get("tradeSizeProbabilities", ECN_TRADE_SIZE_PROBABILITIES)),
            }
            ecn["flowSources"] = [direct] + deepcopy(ecn_extras)

        if target_pair:
            cfg["targetPair"] = str(target_pair)
        solution = ENGINES.get(cfg).solve()
        return cfg, solution, f"Solved · {solution['iterations']} Howard iterations", "status ready"
    except Exception as exc:
        return no_update, no_update, f"Solve failed · {exc}", "status error"


@app.callback(
    Output("m-converged", "children"), Output("m-iterations", "children"), Output("m-rho", "children"), Output("m-residual", "children"),
    Output("value-chart", "figure"), Output("residual-chart", "figure"), Output("value-change-chart", "figure"), Output("markout-resolvent-change-chart", "figure"), Output("internalization-chart", "figure"),
    Output("tier-select", "options"), Output("tier-select", "value"), Output("ladder-q", "options"), Output("ladder-q", "value"),
    Input("solution-store", "data"), Input("config-store", "data"),
)
def render_solution(solution, cfg):
    if not solution or not cfg:
        return "—", "—", "—", "—", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], None, [], None
    fig = go.Figure(go.Scatter(x=solution["qGrid"], y=solution["value"], mode="lines+markers", name="h(q)"))
    fig.update_layout(title="Continuation value h(q)", xaxis_title="Inventory q [EUR M]", yaxis_title="h(q)", template="trinity_dark")
    residual_fig = convergence_figure(solution.get("bellmanResidual", []), "Bellman residual", "max |R(q) - ρ|")
    value_change_fig = convergence_figure(solution.get("valueChange", []), "Value-function change", "max |Δh|")
    markout_change_fig = convergence_figure(solution.get("markoutResolventChange", []), "Markout resolvent change", "max |Δr_tau(q)|")
    internal_fig = markout_exposure_figure(solution)
    opts = [{"label": t["name"], "value": i} for i, t in enumerate(solution["tiers"])]
    qopts = [{"label": f"{float(q):g}M", "value": float(q)} for q in solution["qGrid"]]
    q0 = min((float(q) for q in solution["qGrid"]), key=abs) if qopts else None
    return (
        "Yes" if solution["converged"] else "No", str(solution["iterations"]),
        f"{solution['averageReward']:.6g}", f"{solution['residual']:.3g}", fig, residual_fig, value_change_fig, markout_change_fig, internal_fig,
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
            flow_curve_figure(tier, float(cfg["spreadPips"])),
            hit_ratio_figure(tier, float(cfg["spreadPips"])),
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
    intensities used by the native model, then solves -Q_T tau = 1 with q=0
    absorbing. Off-grid fills use the same linear inventory-state interpolation
    as the C++ generator.
    """
    if not solution or not cfg:
        return {"qGrid": [], "minutes": []}
    q_grid = np.asarray(solution.get("qGrid", []), dtype=float)
    if q_grid.size == 0:
        return {"qGrid": [], "minutes": []}

    def state_index(x: float) -> int:
        j = int(np.argmin(np.abs(q_grid - x)))
        if abs(float(q_grid[j]) - float(x)) > 1e-8:
            raise ValueError(f"Inventory grid does not contain required state {x:g}")
        return j

    zero = state_index(0.0)
    n = len(q_grid)
    Q = np.zeros((n, n), dtype=float)

    enabled_tiers = {t["name"]: t for t in cfg.get("tiers", []) if t.get("enabled", False)}

    def source_won_rate(tier_cfg: dict[str, Any], source: dict[str, Any],
                        delta: float, size: float, size_index: int) -> float:
        flow = source["flow"]
        mapping = source.get("mapping", {}) or {}
        mapping_type = str(mapping.get("type", "identity"))
        size_scale = _source_size_scale(source)
        delta_scale = 1.0
        if mapping_type == "crossed":
            cross_mid = float(mapping.get("crossMid", 0.0))
            if cross_mid <= 0.0:
                raise ValueError("Crossed RFQ source requires positive cross mid")
            delta_scale = 1.0
        elif mapping_type == "affine":
            delta_scale = float(mapping.get("deltaScale", 1.0))
        elif mapping_type != "identity":
            raise ValueError(f"Unsupported RFQ flow-source mapping {mapping_type}")

        source_delta = 0.5 + delta_scale * (delta - 0.5)
        source_size = size * size_scale
        center = float(flow["shift"]) - float(flow["volumeShift"]) * (source_size - 1.0)
        y = float(flow["steepness"]) * (source_delta - center)
        if y >= 0.0:
            e = math.exp(-y) if y < 745.0 else 0.0
            hit = 1.0 / (1.0 + e)
        else:
            e = math.exp(y) if y > -745.0 else 0.0
            hit = e / (1.0 + e)

        exogenous_rate = float(_source_rfq_rates_per_side(tier_cfg, source)[size_index])
        return exogenous_rate * hit

    def add_transition(i: int, q2: float, rate: float) -> None:
        if rate <= 0.0:
            return
        q2 = float(np.clip(q2, q_grid[0], q_grid[-1]))
        right = int(np.searchsorted(q_grid, q2, side="left"))
        if right <= 0:
            weights = [(0, 1.0)]
        elif right >= n:
            weights = [(n - 1, 1.0)]
        elif abs(float(q_grid[right]) - q2) <= 1e-10:
            weights = [(right, 1.0)]
        else:
            left = right - 1
            width = float(q_grid[right] - q_grid[left])
            wr = (q2 - float(q_grid[left])) / width
            weights = [(left, 1.0 - wr), (right, wr)]
        for j, weight in weights:
            Q[i, j] += rate * weight
        Q[i, i] -= rate

    def ecn_fill_components(source: dict[str, Any], i: int, direction: float, posted: float) -> list[tuple[float, float]]:
        # The native HJB now uses the exact five-point ECN size distribution.
        # No continuous interval approximation / breakpoint integration is needed.
        del i, direction
        return _ecn_source_fill_distribution(source, posted)


    # Customer tiers.
    for tier_policy in solution.get("tiers", []):
        tier_cfg = enabled_tiers.get(tier_policy.get("name"))
        if not tier_cfg:
            continue
        sizes = [float(z) for z in tier_policy.get("sizes", [])]
        sources = _tier_flow_sources(tier_cfg)
        for i, q in enumerate(q_grid):
            for side, direction in (("bid", 1.0), ("ask", -1.0)):
                row = tier_policy[side][i]
                for j, z in enumerate(sizes):
                    q2 = float(q) + direction * z
                    if q2 < q_grid[0] - 1e-8 or q2 > q_grid[-1] + 1e-8:
                        continue
                    delta = float(row[j])
                    rate = sum(source_won_rate(tier_cfg, source, delta, z, j) for source in sources)
                    add_transition(i, q2, rate)

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
        z = float(ecn_cfg["quoteSize"])
        sources = _ecn_flow_sources(cfg)
        for i, q in enumerate(q_grid):
            for side, direction in (("bid", 1.0), ("ask", -1.0)):
                if not bool(ecn_policy[f"{side}Active"][i]):
                    continue
                delta = float(ecn_policy[f"{side}Delta"][i])
                for source in sources:
                    hit_rate = float(_ecn_source_arrival_rate(source, delta))
                    for fill, probability in ecn_fill_components(source, i, direction, z):
                        add_transition(i, float(q) + direction * fill, hit_rate * probability)

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


def _mc_analytics_cache_key(cfg: dict[str, Any], q0: float) -> str:
    return json.dumps(
        {"cfg": cfg, "horizon": SESSION_HORIZON, "q0": float(q0)},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _compute_mc_analytics(engine, cfg: dict[str, Any], solution: dict[str, Any] | None,
                          q0: float) -> dict[str, Any]:
    """Return deterministic analytical benchmarks, reusing them across MC runs."""
    key = _mc_analytics_cache_key(cfg, q0)
    with _MC_ANALYTICS_CACHE_LOCK:
        cached = _MC_ANALYTICS_CACHE.get(key)
        if cached is not None:
            return deepcopy(cached)

    stats = engine.statistics(SESSION_HORIZON, q0)
    if solution:
        stats["internalizationTimes"] = expected_internalization_times(solution, cfg)

    with _MC_ANALYTICS_CACHE_LOCK:
        # Dict insertion order is sufficient for this tiny FIFO cache.  The
        # config-keyed Engine cache already has similar bounded semantics.
        if key not in _MC_ANALYTICS_CACHE and len(_MC_ANALYTICS_CACHE) >= _MC_ANALYTICS_CACHE_MAX_ENTRIES:
            oldest = next(iter(_MC_ANALYTICS_CACHE))
            _MC_ANALYTICS_CACHE.pop(oldest, None)
        _MC_ANALYTICS_CACHE[key] = deepcopy(stats)
    return stats


def _run_mc_job(job_id: str, cfg: dict[str, Any], solution: dict[str, Any] | None,
                paths: int, q0: float, seed: int) -> None:
    """Execute MC and deterministic analytics concurrently, publishing progress."""
    try:
        engine = ENGINES.get(cfg)

        def report_progress(completed: int, total: int) -> None:
            # The callback fires immediately after each reported path batch and
            # once more on the final path, before native cross-path statistics
            # are finalized.  Expose that final phase explicitly instead of
            # leaving the modal apparently frozen at its last path percentage.
            finished_paths = completed >= total and total > 0
            percent = 98 if finished_paths else int(round(97.0 * completed / max(1, total)))
            message = (
                "Finalizing Monte Carlo statistics…"
                if finished_paths
                else f"Simulating path {completed:,} of {total:,}…"
            )
            with _MC_JOBS_LOCK:
                job = _MC_JOBS.get(job_id)
                if job is None:
                    return
                job.update(
                    completed=int(completed),
                    total=int(total),
                    percent=max(1, min(98, percent)),
                    message=message,
                )

        # statistics() and simulate() are read-only once the policy is solved.
        # Run the deterministic benchmark on another core instead of starting
        # it only after the last MC path has finished.  For normal production
        # path counts it is therefore already ready by the time MC completes.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mc-analytics-{job_id[:8]}"
        ) as analytics_pool:
            analytics_future = analytics_pool.submit(
                _compute_mc_analytics, engine, cfg, solution, q0
            )

            mc = engine.simulate(
                SESSION_HORIZON, paths, q0, seed,
                retained_paths=8, sample_points=381,
                progress_callback=report_progress,
                compact_result=True,
            )

            with _MC_JOBS_LOCK:
                job = _MC_JOBS.get(job_id)
                if job is not None:
                    if analytics_future.done():
                        job.update(percent=99, completed=paths, total=paths,
                                   message="Finalizing Monte Carlo output…")
                    else:
                        job.update(percent=98, completed=paths, total=paths,
                                   message="Waiting for analytical benchmark…")

            stats = analytics_future.result()

        _store_mc_result(job_id, mc)
        with _MC_JOBS_LOCK:
            job = _MC_JOBS.get(job_id)
            if job is not None:
                job.update(
                    state="done",
                    percent=100,
                    completed=paths,
                    total=paths,
                    message="Monte Carlo complete",
                    resultId=job_id,
                    stats=stats,
                    finishedAt=time.time(),
                )
    except Exception as exc:
        with _MC_JOBS_LOCK:
            job = _MC_JOBS.get(job_id)
            if job is not None:
                job.update(
                    state="error",
                    message="Monte Carlo failed",
                    error=str(exc),
                    finishedAt=time.time(),
                )


@app.callback(
    Output("mc-job-store", "data"),
    Output("mc-progress-modal", "style"),
    Output("mc-progress-poll", "disabled"),
    Output("run-mc", "disabled"),
    Output("mc-progress-bar", "value"),
    Output("mc-progress-message", "children"),
    Output("mc-progress-count", "children"),
    Output("mc-progress-percent", "children"),
    Output("status", "children", allow_duplicate=True),
    Output("status", "className", allow_duplicate=True),
    Input("run-mc", "n_clicks"),
    State("config-store", "data"), State("solution-store", "data"),
    State("mc-paths", "value"), State("mc-q0", "value"), State("mc-seed", "value"),
    prevent_initial_call=True,
)
def start_mc(_clicks, cfg, solution, paths, q0, seed):
    if not cfg:
        return (
            no_update, {"display": "none"}, True, False, "0",
            "Simulation not started", "0 / 0 paths", "0%",
            "Solve the model first", "status error",
        )
    try:
        paths = int(paths)
        q0 = float(q0)
        seed = int(seed)
        if paths <= 0:
            raise ValueError("Paths must be positive")
    except Exception as exc:
        return (
            no_update, {"display": "none"}, True, False, "0",
            "Simulation not started", "0 / 0 paths", "0%",
            f"Monte Carlo failed · {exc}", "status error",
        )

    job_id = uuid.uuid4().hex
    # Copy JSON-like callback state before handing it to a worker thread.
    cfg_copy = deepcopy(cfg)
    solution_copy = deepcopy(solution) if solution else None
    with _MC_JOBS_LOCK:
        _prune_finished_mc_jobs_locked()
        _MC_JOBS[job_id] = {
            "state": "running",
            "completed": 0,
            "total": paths,
            "percent": 0,
            "message": "Starting native simulation…",
            "error": None,
        }

    worker = threading.Thread(
        target=_run_mc_job,
        args=(job_id, cfg_copy, solution_copy, paths, q0, seed),
        name=f"mc-{job_id[:8]}",
        daemon=True,
    )
    worker.start()
    return (
        job_id, {"display": "flex"}, False, True, "0",
        "Starting native simulation…", f"0 / {paths:,} paths", "0%",
        "Monte Carlo running…", "status running",
    )


@app.callback(
    Output("mc-job-store", "data", allow_duplicate=True),
    Output("mc-store", "data"), Output("stats-store", "data"),
    Output("mc-progress-modal", "style", allow_duplicate=True),
    Output("mc-progress-poll", "disabled", allow_duplicate=True),
    Output("run-mc", "disabled", allow_duplicate=True),
    Output("mc-progress-bar", "value", allow_duplicate=True),
    Output("mc-progress-message", "children", allow_duplicate=True),
    Output("mc-progress-count", "children", allow_duplicate=True),
    Output("mc-progress-percent", "children", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("status", "className", allow_duplicate=True),
    Input("mc-progress-poll", "n_intervals"),
    State("mc-job-store", "data"),
    prevent_initial_call=True,
)
def poll_mc_progress(_ticks, job_id):
    if not job_id:
        return (
            no_update, no_update, no_update, {"display": "none"}, True, False,
            "0", "Simulation idle", "0 / 0 paths", "0%", no_update, no_update,
        )

    with _MC_JOBS_LOCK:
        job = _MC_JOBS.get(str(job_id))
        if job is None:
            return (
                None, no_update, no_update, {"display": "none"}, True, False,
                "0", "Simulation state unavailable", "0 / 0 paths", "0%",
                "Monte Carlo state unavailable", "status error",
            )

        state = str(job.get("state", "running"))
        completed = int(job.get("completed", 0))
        total = int(job.get("total", 0))
        percent = int(job.get("percent", 0))
        message = str(job.get("message", "Running Monte Carlo…"))

        if state == "done":
            result_id = str(job.get("resultId") or job_id)
            stats = job.get("stats")
            # Return only a lightweight browser token; callbacks resolve the
            # large Monte Carlo payload from the in-process result cache.
            # Returning the same completed payload is intentionally idempotent.
            return (
                None, {"resultId": result_id}, stats, {"display": "none"}, True, False,
                "100", "Monte Carlo complete", f"{total:,} / {total:,} paths", "100%",
                "Monte Carlo complete", "status ready",
            )

        if state == "error":
            error = str(job.get("error") or "Unknown error")
            # Keep terminal errors during the same grace period for the same
            # reason: late polls should see a stable terminal state.
            return (
                None, no_update, no_update, {"display": "none"}, True, False,
                str(percent), "Monte Carlo failed", f"{completed:,} / {total:,} paths", f"{percent}%",
                f"Monte Carlo failed · {error}", "status error",
            )

    return (
        no_update, no_update, no_update, {"display": "flex"}, False, True,
        str(percent), message, f"{completed:,} / {total:,} paths", f"{percent}%",
        f"Monte Carlo running · {percent}%", "status running",
    )


@app.callback(
    Output("mc-mean", "children"), Output("cf-mean", "children"), Output("mc-std", "children"), Output("cf-std", "children"),
    Output("mc-p5", "children"), Output("mc-loss", "children"), Output("pnl-chart", "figure"), Output("inventory-chart", "figure"),
    Output("mc-internalization-time-chart", "figure"), Output("path-select", "options"), Output("path-select", "value"),
    Input("mc-store", "data"), Input("stats-store", "data"),
)
def render_mc(mc, stats):
    mc = _resolve_mc_store(mc)
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
    Output("mc-rfq-tier-select", "options"), Output("mc-rfq-tier-select", "value"),
    Input("solution-store", "data"), Input("config-store", "data"),
)
def update_mc_rfq_validation_tiers(solution, cfg):
    if not solution or not cfg:
        return [], None
    options = [
        {"label": str(tier.get("name", f"Tier {i + 1}")), "value": i}
        for i, tier in enumerate(solution.get("tiers", []))
    ]
    return options, (0 if options else None)


@app.callback(
    Output("mc-rfq-arrival-validation-chart", "figure"),
    Output("mc-rfq-win-validation-chart", "figure"),
    Output("mc-rfq-inventory-validation-chart", "figure"),
    Input("mc-store", "data"), Input("solution-store", "data"), Input("config-store", "data"),
    Input("mc-rfq-tier-select", "value"),
)
def render_mc_rfq_validation(mc, solution, cfg, tier_index):
    mc = _resolve_mc_store(mc)
    if not mc or not solution or not cfg or tier_index is None:
        return go.Figure(), go.Figure(), go.Figure()
    try:
        return mc_rfq_validation_figures(mc, solution, cfg, int(tier_index))
    except Exception as exc:
        empty = go.Figure()
        empty.add_annotation(text=f"RFQ validation unavailable · {exc}", showarrow=False)
        empty.update_layout(template="trinity_dark")
        return empty, empty, empty


@app.callback(
    Output("path-pnl-chart", "figure"), Output("path-chart", "figure"), Output("spot-path-chart", "figure"),
    Output("fill-counts-chart", "figure"), Output("rfq-realized-hit-chart", "figure"), Output("fill-tape", "children"),
    Input("mc-store", "data"), Input("path-select", "value"), Input("path-size-select", "value"),
    Input("fill-tier-metric", "value"), Input("solution-store", "data"), Input("config-store", "data"),
)
def render_path(mc, path_index, quote_size, fill_tier_metric, solution, cfg):
    mc = _resolve_mc_store(mc)
    if not mc or path_index is None or not mc.get("samplePaths"):
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), "No retained path"
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
            ecn_bid_px = np.where(ecn_bid_active, spots - spread * (0.5 - ecn_bid_delta), np.nan)
            ecn_ask_px = np.where(ecn_ask_active, spots + spread * (0.5 - ecn_ask_delta), np.nan)
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
                    str(e.get("source", "ECN")),
                    side.capitalize(),
                    fill_status,
                    float(e.get("tradeDistancePips", np.nan)),
                    float(e.get("quoteDepthPips", e.get("depthPips", np.nan))),
                    float(e.get("tradePrice", np.nan)),
                    bool(e.get("quoteActive", False)),
                    float(e.get("sourceTradeSize", np.nan)),
                    float(e.get("targetFillSize", 0.0)),
                ] for e in chosen],
                hovertemplate=(
                    "%{x|%H:%M:%S}<br>%{customdata[0]} ECN %{customdata[1]} trade · %{customdata[2]}"
                    "<br>plot px=%{y:.6f}<br>trade px=%{customdata[5]:.6f}"
                    "<br>target-equivalent trade distance=%{customdata[3]:.2f} pips"
                    "<br>master quote depth=%{customdata[4]:.2f} pips"
                    "<br>source parent size=%{customdata[7]:.3f}M"
                    "<br>target fill size=%{customdata[8]:.3f}M"
                    "<br>quote active=%{customdata[6]}<extra></extra>"
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
            if str(f["side"]).lower() == side and not str(f.get("tier", "")).startswith("Passive ECN")
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

    # Long-run fill composition comes from compact population aggregates built
    # over every Monte Carlo path.  Retained paths are only for the interactive
    # time-series diagnostics above and must not drive model-validation shares.
    fill_metric = "volume" if fill_tier_metric == "volume" else "count"
    fill_aggregates = list(mc.get("fillAggregates", []))
    fills_by_tier_side: dict[tuple[str, str], float] = {}
    tier_order = []
    for aggregate in fill_aggregates:
        tier = str(aggregate.get("tier", ""))
        side = str(aggregate.get("side", "")).lower()
        if tier and tier not in tier_order:
            tier_order.append(tier)
        raw_value = (
            float(aggregate.get("volume", 0.0))
            if fill_metric == "volume"
            else float(aggregate.get("tradeCount", 0.0))
        )
        fills_by_tier_side[(tier, side)] = raw_value

    fill_total = sum(fills_by_tier_side.values())

    fill_counts = go.Figure()
    for side in ("bid", "ask"):
        raw_values = [fills_by_tier_side.get((tier, side), 0.0) for tier in tier_order]
        shares = [value / fill_total if fill_total > 0.0 else 0.0 for value in raw_values]
        fill_counts.add_trace(go.Bar(
            x=tier_order,
            y=shares,
            name=side.capitalize(),
            marker_color=BID_COLOR if side == "bid" else ASK_COLOR,
            customdata=np.asarray(raw_values, dtype=float),
            hovertemplate=(
                "%{x}<br>Side=" + side.capitalize()
                + "<br>Share=%{y:.1%}<br>Executed volume=%{customdata:,.3f}M<extra></extra>"
                if fill_metric == "volume"
                else "%{x}<br>Side=" + side.capitalize()
                + "<br>Share=%{y:.1%}<br>Trades=%{customdata:,.0f}<extra></extra>"
            ),
        ))
    fill_counts.update_layout(
        title="Fills by tier and side · all simulated paths",
        xaxis_title="Tier",
        yaxis_title="Share of executed volume" if fill_metric == "volume" else "Share of trades",
        barmode="group",
        template="trinity_dark",
    )
    fill_counts.update_yaxes(tickformat=".0%", rangemode="tozero")
    if not fill_aggregates:
        fill_counts.add_annotation(
            text="No population fill aggregates · rebuild the native extension",
            showarrow=False,
        )

    # Realized customer RFQ hit ratio over the complete Monte Carlo population.
    # Tier and side remain pooled, matching the business question: what fraction
    # of all simulated RFQs of this size did we actually win?
    rfq_aggregates = sorted(
        list(mc.get("rfqAggregates", [])), key=lambda row: float(row.get("size", 0.0))
    )
    rfq_hit = go.Figure()
    if rfq_aggregates:
        rfq_sizes = [float(row.get("size", 0.0)) for row in rfq_aggregates]
        wins = [int(row.get("wins", 0)) for row in rfq_aggregates]
        requests = [int(row.get("requests", 0)) for row in rfq_aggregates]
        hit_ratios = [w / n if n else 0.0 for w, n in zip(wins, requests)]
        rfq_hit.add_trace(go.Bar(
            x=rfq_sizes,
            y=hit_ratios,
            name="Realized hit ratio",
            customdata=np.column_stack([wins, requests]),
            hovertemplate=(
                "RFQ size=%{x:g}M<br>Hit ratio=%{y:.1%}"
                "<br>Won=%{customdata[0]:.0f} / %{customdata[1]:.0f}<extra></extra>"
            ),
        ))
    else:
        rfq_hit.add_annotation(
            text="No population RFQ aggregates · rebuild the native extension",
            showarrow=False,
        )
    rfq_hit.update_layout(
        title="Realized RFQ hit ratio by size · all simulated paths",
        xaxis_title="RFQ size [M]",
        yaxis_title="Won RFQs / total RFQs",
        yaxis_range=[0, 1],
        template="trinity_dark",
    )
    rfq_hit.update_yaxes(tickformat=".0%")

    header = html.Thead(html.Tr([html.Th(x) for x in ["Time", "Tier", "Side", "Size", "Price", "q before", "q after"]]))
    rows = []
    for f in path["fills"][:250]:
        rows.append(html.Tr([
            html.Td(session_clock_label(f["time"])), html.Td(f["tier"]), html.Td(f["side"]), html.Td(f"{f['size']:g}"),
            html.Td(f"{f['price']:.6f}"), html.Td(f"{f['inventoryBefore']:.2f}"), html.Td(f"{f['inventoryAfter']:.2f}"),
        ]))
    table = html.Table([header, html.Tbody(rows or [html.Tr(html.Td("No fills", colSpan=7))])], className="data-table")
    return pnl_fig, fig, spot_fig, fill_counts, rfq_hit, table


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
