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
from plotly.subplots import make_subplots
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from trinity.defaults import (
    ECN_TRADE_SIZE_PROBABILITIES,
    default_config,
    default_ecn_trade_size_probabilities,
    default_fx_mid,
    default_fx_pair_market_data, default_fx_pair_volatility,
)
from trinity.markout import markout_eur_per_million
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


def _editor_cross_mid(cross_pair: str | None, entered_mid: float | None,
                      pair_changed: bool) -> float | None:
    """Resolve a crossed-source mid without depending on Dash callback order.

    The JSON defaults are used on pair changes or while the input is blank.
    Manual values are otherwise preserved.  An incomplete selection is
    allowed while editing; final model validation happens on Solve.
    """
    reference_mid = default_fx_mid(cross_pair)
    if reference_mid is not None and (pair_changed or entered_mid is None):
        return reference_mid
    return None if entered_mid is None else float(entered_mid)

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


def checkbox(label: str, component_id: str, checked: bool, *, channel_enable: bool = False):
    """Render channel enable text explicitly white on the dark sidebar.

    A checklist's plain-string option label can inherit a dark colour from Dash's
    component styling. Using a real Span with an inline foreground colour keeps
    the Tier/Dark pool/ECN labels readable across Dash versions and themes.
    """
    visible_label = (
        html.Span(
            label,
            className="channel-enable-label",
            style={"color": "#ffffff", "WebkitTextFillColor": "#ffffff"},
        )
        if channel_enable else label
    )
    return dcc.Checklist(
        id=component_id,
        options=[{"label": visible_label, "value": "on"}],
        value=["on"] if checked else [],
        className="checklist channel-enable-checkbox" if channel_enable else "checklist",
        labelStyle={"color": "#ffffff"} if channel_enable else {},
    )


def select(label: str, component_id: str, options, value):
    # Keep white-background policy dropdowns readable in both Dash 4 and
    # legacy React-Select. The ladder inventory picker uses the same light
    # dropdown appearance and therefore needs this class too.
    policy_selector = component_id.endswith(("-regime", "-rfq-size")) or component_id in {
        "tier-select", "mc-rfq-tier-select", "ladder-q",
    }
    dropdown_class = "field-dropdown policy-selector-dropdown" if policy_selector else "field-dropdown"
    placeholder = "Solve to select a regime" if component_id.endswith("-regime") else (
        "Select tier" if component_id in {"tier-select", "mc-rfq-tier-select"} else (
        "Select RFQ size" if component_id.endswith("-rfq-size") else None)
    )
    return html.Label([
        html.Span(label, className="field-label"),
        dcc.Dropdown(
            id=component_id, options=options, value=value,
            clearable=False, className=dropdown_class, placeholder=placeholder,
        ),
    ], className="field policy-selector-field" if policy_selector else "field")


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


# Each regime-dependent plot has its own local choice.  These choices affect
# presentation only: they select an existing (q, sigma) slice of the Solve output.
POLICY_REGIME_SELECTOR_IDS = (
    "overview-value-regime",
    "tier-implied-hit-regime",
    "tier-quote-inventory-regime",
    "tier-ladder-regime",
    "dark-policy-regime",
    "ecn-implied-hit-regime",
    "ecn-quote-inventory-regime",
    "ecn-policy-regime",
)


def regime_selector(component_id: str, *, wrapper_id: str | None = None):
    """Regime choice directly inside a regime-dependent diagnostics expander."""
    attrs = {"className": "toolbar regime-local-selector"}
    if wrapper_id is not None:
        attrs["id"] = wrapper_id
    return html.Div([select("Volatility regime", component_id, [], None)], **attrs)


def ecn_heatmap_toggle(component_id: str):
    """Per-expander toggle: the heatmap shows *all* volatility regimes."""
    return dcc.Checklist(
        id=component_id,
        options=[{"label": html.Span("Heatmap", className="ecn-heatmap-toggle-label"),
                  "value": "heatmap"}],
        value=[],
        className="ecn-heatmap-toggle",
        labelStyle={"display": "flex", "alignItems": "center", "cursor": "pointer"},
        inputStyle={"marginRight": "12px"},
    )


RFQ_HEATMAP_PREFIXES = (
    "tier-implied-hit", "tier-quote-inventory",
)


def rfq_heatmap_controls(prefix: str):
    """An independent view toggle, single-regime selector and RFQ-knot selector."""
    return [
        ecn_heatmap_toggle(f"{prefix}-heatmap"),
        regime_selector(f"{prefix}-regime", wrapper_id=f"{prefix}-regime-wrapper"),
        html.Div(
            [select("RFQ size [EUR M]", f"{prefix}-rfq-size", [], None)],
            id=f"{prefix}-rfq-size-wrapper",
            className="toolbar regime-local-selector",
            style={"display": "none"},
        ),
    ]


def _ecn_heatmap_enabled(value) -> bool:
    return "heatmap" in (value or [])



def _default_cross_flow_source(index: int = 0, target_pair: str | None = None) -> dict[str, Any]:
    return {
        "name": "Select FX pair",
        "pair": None,
        "crossPair": None,
        "mapping": {"type": "crossed", "crossMid": None, "sourceSpreadPips": 20.0},
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
        "mapping": {"type": "crossed", "crossMid": None, "sourceSpreadPips": 20.0},
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
        # Crossed source choices are scoped to the selected source, so never
        # restore a previous source's selection when a fresh card is added.
        dcc.Dropdown(id=component_id, value=value, options=options, placeholder=placeholder,
                     clearable=False, className="field-dropdown",
                     persistence=(component_id == "global-target-pair"), persistence_type="memory"),
    ], className="field")


def volatility_generator_matrix(sigma, rates):
    """Full Markov-chain generator: off-diagonal arrival rates, diagonal = -row sum."""
    matrix = np.asarray(rates, dtype=float).copy()
    n=len(sigma)
    if n not in (2, 3) or matrix.shape != (n, n):
        raise ValueError("Expected two or three volatility regimes")
    for i in range(n):
        matrix[i, i] = -sum(matrix[i, j] for j in range(n) if i != j)
    return matrix


def readonly_volatility_matrix(volatility: dict[str, Any]):
    """Render immutable JSON regime levels and the generator Q as numbers."""
    sigma = volatility["sigmaPips"]
    rates = volatility["ratesPerMinute"]
    matrix = volatility_generator_matrix(sigma, rates)
    names = ("Low", "High") if len(sigma) == 2 else ("Low", "Normal", "High")
    sigma_table = html.Table([
        html.Thead(html.Tr([html.Th("Regime"), html.Th("σ [pips/√min]")])),
        html.Tbody([html.Tr([html.Td(name), html.Td(f"{value:g}")])
                    for name, value in zip(names, sigma)]),
    ], className="data-table volatility-readonly-table")
    generator_table = html.Table([
        html.Thead(html.Tr([html.Th("From / To"), *[html.Th(n) for n in names]])),
        html.Tbody([html.Tr([html.Th(label), *[html.Td(f"{float(v):.6f}") for v in row]])
                    for label, row in zip(names, matrix)]),
    ], className="data-table volatility-readonly-table")
    return html.Div([
        html.Div("Volatility levels", className="subhead"),
        sigma_table,
        html.Div("Generator Q [1/min] · rows: from regime, columns: to regime", className="subhead"),
        generator_table,
    ], className="volatility-readonly-matrix")


def flow_markout_figure(enabled, asymptote, tau_seconds, pair):
    t = np.linspace(0.0, 300.0, 301)
    curve = markout_eur_per_million(t, float(asymptote), float(tau_seconds)) if enabled and float(tau_seconds) > 0.0 else np.zeros_like(t)
    fig = go.Figure(go.Scatter(x=t, y=curve, mode="lines", name=str(pair), line={"color": ASK_COLOR, "width": 3}))
    fig.add_hline(y=0, line_dash="dot", opacity=0.3)
    fig.update_layout(xaxis_title="seconds", yaxis_title="EUR / EUR m",
                      template="trinity_dark",
                      margin={"l": 65, "r": 20, "t": 25, "b": 45})
    return fig


def flow_source_settings_editor(source_key: str, source: dict[str, Any], target_pair: str,
                                markout=None):
    flow = source.get("flow", {})
    mapping = source.get("mapping", {})
    direct = source_key == "direct"
    markout = markout or {"enabled": True, "asymptote": -50.8, "tau": 36.7}

    # The Direct card already identifies the target pair, and the locked
    # Target pair dropdown immediately below it does so again. Do not render
    # a redundant title above the selector: the Direct and Crossed inputs
    # must start at the same vertical position.
    if direct:
        source_fields = [html.Div([
            dcc.Dropdown(id="flow-edit-pair", value=target_pair, options=FX_PAIR_OPTIONS),
            dcc.Dropdown(id="flow-edit-cross-pair", value=None, options=FX_PAIR_OPTIONS),
            dcc.Input(id="flow-edit-cross-mid", type="number", value=1.0),
            dcc.Input(id="flow-edit-spread", type="number", value=1.0),
        ], style={"display": "none"})]
    else:
        pair = source.get("pair")
        cross_pair = source.get("crossPair")
        configured_mid = mapping.get("crossMid")
        cross_mid = configured_mid if configured_mid is not None else default_fx_mid(cross_pair)
        source_fields = [html.Div([
            _dropdown_field("Crossed pair", "flow-edit-pair", pair, _source_pair_options(target_pair)),
            _dropdown_field("Crossing pair", "flow-edit-cross-pair", cross_pair, _cross_pair_options(target_pair, pair)),
        ], className="flow-pair-selectors")]

    # Crossed-source market data belongs with the other collapsible source
    # settings, not next to the FX-pair selectors. Keep the existing input IDs
    # so the pair defaults, Apply action and solver configuration are unchanged.
    crossed_market_data = diagnostic_expander("Market Data", [
        html.Div([
            number_input("Crossing Mid-Rate", "flow-edit-cross-mid", cross_mid, "any"),
            number_input("Crossing Spread", "flow-edit-spread", float(mapping.get("sourceSpreadPips", 20.0)), "any"),
        ], className="grid-2"),
    ]) if not direct else None

    volatility_panel = diagnostic_expander("Volatility", [
        readonly_volatility_matrix(default_fx_pair_volatility(target_pair)),
    ]) if direct else None

    # Return header and expanders as separate groups.  The modal places the
    # persistent Market Data/Pricing Rungs fields between these groups, in the
    # *same* source card.  Splitting the groups avoids remounting the inputs
    # whenever a source, Tier or target pair changes.
    return html.Div([
        html.Div(source_fields, className="flow-source-header-fields"),
        html.Div([
            crossed_market_data,
            diagnostic_expander("RFQ flow intensities", [
                html.Div([
                    number_input("A0 · exogenous intensity scale", "flow-edit-A0", float(flow.get("A0", 0.01)), "any"),
                    number_input("Size θ", "flow-edit-theta", float(flow.get("theta", 0.144)), "any"),
                    number_input("Size β", "flow-edit-beta", float(flow.get("beta", 0.0857)), "any"),
                    number_input("Steepness", "flow-edit-steep", float(flow.get("steepness", 8.42)), "any"),
                    number_input("Shift", "flow-edit-shift", float(flow.get("shift", 0.52)), "any"),
                    number_input("Volume shift", "flow-edit-vshift", float(flow.get("volumeShift", 0.026)), "any"),
                ], className="grid-2 flow-calibration-grid"),
            ]),
            volatility_panel,
            diagnostic_expander("Markouts", [
                html.Div("Markout is a Tier-level curve shared by the direct pair and its crossed flow sources; the C++ HJB does not support per-source markout.", className="muted-note"),
                checkbox("Enabled", "flow-edit-markout-enabled", bool(markout["enabled"]), channel_enable=True),
                html.Div([
                    number_input("Asymptote [EUR / EUR 1M]", "flow-edit-markout-asymptote", markout["asymptote"], "any"),
                    number_input("Tau [seconds]", "flow-edit-markout-tau", markout["tau"], "any"),
                ], className="grid-2"),
                dcc.Graph(id="flow-markout-preview", figure=flow_markout_figure(markout["enabled"], markout["asymptote"], markout["tau"], source.get("pair") or target_pair),
                          responsive=True, style={"height": "330px", "width": "100%"}),
            ]),
        ], className="flow-source-dynamic-expanders"),
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


def ecn_flow_source_settings_editor(source_key: str, source: dict[str, Any], target_pair: str):
    """Use the same Direct/Crossed pair selectors as the RFQ source editor.

    ECN-specific intensity and trade-size calibration remains unchanged.
    """
    flow = source.get("flow", {})
    mapping = source.get("mapping", {})
    direct = source_key == "direct"

    # The selected Direct/Crossed tile already displays the pair name.
    # No second header here: align the form for both source types.

    if direct:
        # All callback IDs must remain mounted even when editing a direct
        # source, just as they do in the Tier editor.
        source_fields = [html.Div([
            dcc.Dropdown(id="ecn-flow-edit-pair", value=target_pair, options=FX_PAIR_OPTIONS),
            dcc.Dropdown(id="ecn-flow-edit-cross-pair", value=None, options=FX_PAIR_OPTIONS),
            dcc.Input(id="ecn-flow-edit-cross-mid", type="number", value=1.0),
            dcc.Input(id="ecn-flow-edit-spread", type="number", value=1.0),
        ], style={"display": "none"})]
    else:
        pair = source.get("pair")
        cross_pair = source.get("crossPair")
        configured_mid = mapping.get("crossMid")
        cross_mid = configured_mid if configured_mid is not None else default_fx_mid(cross_pair)
        source_fields = [
            html.Div([
                _dropdown_field("Crossed pair", "ecn-flow-edit-pair", pair,
                                _source_pair_options(target_pair)),
                _dropdown_field("Crossing pair", "ecn-flow-edit-cross-pair", cross_pair,
                                _cross_pair_options(target_pair, pair)),
            ], className="flow-pair-selectors"),
            html.Div([
                number_input("Cross mid", "ecn-flow-edit-cross-mid", cross_mid, "any"),
                number_input("Source spread [pips]", "ecn-flow-edit-spread", float(mapping.get("sourceSpreadPips", 20.0)), "any"),
            ], className="grid-2"),
        ]

    return html.Div([
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
        html.Div(f"P(100k) = 1 − others = {_ecn_size_probabilities(source)[4]:.6f}", id="ecn-flow-edit-p100k-note", className="muted-note"),
    ], className="flow-source-settings")

def tier_panel(index: int, tier: dict[str, Any], open_: bool = False):
    p = f"t{index}"
    return panel(tier["name"], [
        checkbox("Enabled", f"{p}-enabled", tier["enabled"], channel_enable=True),
        text_input("Name", f"{p}-name", tier["name"]),
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
            number_input("Fee [pips]", f"{p}-fee", tier.get("feePips", 0.0), "any"),
        ], className="grid-2"),
        html.Div([
            checkbox("Enabled", f"{p}-markout-enabled", tier["useMarkout"]),
            number_input("Asymptote", f"{p}-markout-asymptote", tier["markout"]["asymptoticEurPerMillion"], "any"),
            number_input("Tau", f"{p}-markout-tau-seconds", tier["markout"]["tauSeconds"], "any"),
        ], style={"display": "none"}),
        html.Div([
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


def _rfq_sizes(tier: dict[str, Any]) -> np.ndarray:
    """The pricing knots are exactly the discrete customer RFQ sizes."""
    sizes = np.asarray(tier["sizes"], dtype=float)
    if sizes.size == 0 or np.any(~np.isfinite(sizes)) or np.any(sizes <= 0.0) or np.any(np.diff(sizes) <= 0.0):
        raise ValueError("RFQ pricing sizes must be finite, positive and strictly increasing")
    return sizes

def _source_curve_at_sizes(tier: dict[str, Any], source: dict[str, Any], target_sizes) -> np.ndarray:
    """Unnormalised intensity weights on the configured discrete RFQ sizes."""
    sizes = np.asarray(target_sizes, dtype=float) * _source_size_scale(source)
    flow = source["flow"]
    exponent = -float(flow["theta"]) - float(flow["beta"]) * sizes
    weights = np.power(sizes, exponent)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("RFQ arrival-intensity curve must be finite and nonnegative")
    return weights


def _source_curve_shape(tier: dict[str, Any], source: dict[str, Any]) -> np.ndarray:
    weights = _source_curve_at_sizes(tier, source, _rfq_sizes(tier))
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("RFQ arrival-intensity curve must have positive mass")
    return weights


def _source_a0(tier: dict[str, Any], source: dict[str, Any]) -> float:
    flow = source["flow"]
    if "A0" in flow:
        return float(flow["A0"])
    if "totalRfqRate" in flow:
        # Compatibility with the short-lived total-rate config format.  Recover
        # the original one-sided curve scale on the pricing knots.
        shape_mass = float(np.sum(_source_curve_shape(tier, source)))
        return float(flow["totalRfqRate"]) / (2.0 * shape_mass)
    raise ValueError("RFQ flow requires A0")


def _source_size_distribution(tier: dict[str, Any], source: dict[str, Any]) -> np.ndarray:
    # There is no independent RFQ-size calibration: the exogenous arrival
    # intensity curve itself is the density. Normalize it on pricing knots.
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
        source_spread = float(mapping.get("sourceSpreadPips", 0.0))
        if cross_mid <= 0.0 or source_spread <= 0.0:
            raise ValueError("Crossed RFQ source requires positive cross mid and source spread")
        return float(target_spread_pips) / (cross_mid * source_spread)
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


def _markout_eur_per_million(tier: dict[str, Any], cfg: dict[str, Any], t_seconds):
    """Tier-specific signed markout; identical per EUR 1M for each RFQ size."""
    t = np.asarray(t_seconds, dtype=float)
    if not tier.get("useMarkout", True):
        return np.zeros_like(t)
    spec = tier["markout"]
    return markout_eur_per_million(
        t, float(spec["asymptoticEurPerMillion"]), float(spec["tauSeconds"])
    )


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
    sizes = _rfq_sizes(tier)
    rates = np.asarray([_activity(tier, float(z)) for z in sizes], dtype=float)
    fig = go.Figure(go.Bar(x=[f"{z:g}" for z in sizes], y=rates, name="RFQ arrivals"))
    fig.update_layout(
        title=f"Exogenous RFQ arrival intensity λ_RFQ(z) · {tier['name']}",
        xaxis_title="RFQ pricing knot [EUR M]",
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
    rfq_sizes = np.asarray(_rfq_sizes(tier), dtype=float)
    model_arrivals = np.asarray([_activity(tier, float(z)) for z in rfq_sizes], dtype=float)
    n_paths = len(mc.get("pnlBase", []))
    exposure = 2.0 * float(n_paths) * SESSION_HORIZON
    mc_arrivals = np.asarray([
        requests_by_size.get(float(z), 0) / exposure if exposure > 0.0 else 0.0
        for z in rfq_sizes
    ], dtype=float)
    rfq_labels = [f"{z:g}" for z in rfq_sizes]
    arrival_fig.add_trace(go.Bar(
        x=rfq_labels, y=model_arrivals, name="Model input λ_RFQ(z)",
        opacity=0.7,
    ))
    arrival_fig.add_trace(go.Scatter(
        x=rfq_labels, y=mc_arrivals, mode="markers", name="MC estimate",
        marker=dict(size=9, symbol="circle-open"),
        customdata=np.asarray([requests_by_size.get(float(z), 0) for z in rfq_sizes]),
        hovertemplate=(
            "RFQ size=%{x}M<br>MC λ=%{y:.6f} /min/side"
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


def rfq_regime_heatmap_figure(
    solution: dict[str, Any], cfg: dict[str, Any], tier_index: int,
    rfq_size: float | None, *, metric: str = "quotes",
) -> go.Figure:
    """Both RFQ quote sides at one pricing knot, across inventory and volatility regimes.

    The size must be an actual RFQ/pricing knot; it is never interpolated.
    """
    if metric not in ("quotes", "win_probability"):
        raise ValueError(f"Unsupported RFQ heatmap metric: {metric}")
    tiers = (solution or {}).get("tiers") or []
    regimes = (solution or {}).get("regimes") or []
    ti = int(tier_index)
    if not regimes or ti >= len(tiers):
        fig = go.Figure()
        fig.add_annotation(text="Solve to view RFQ policies across volatility regimes", showarrow=False)
        fig.update_layout(template="trinity_dark")
        return fig
    sizes = np.asarray(tiers[ti]["sizes"], dtype=float)
    if sizes.size == 0:
        raise ValueError("Tier has no pricing knots")
    if rfq_size is None:
        j = 0
    else:
        matching = np.flatnonzero(np.isclose(sizes, float(rfq_size), rtol=0, atol=1e-9))
        if not matching.size:
            raise ValueError(f"RFQ size {rfq_size:g}M is not a pricing knot in this tier")
        j = int(matching[0])
    z = float(sizes[j])
    q = np.asarray(solution["qGrid"], dtype=float)
    sigma_names = ("Low", "High") if len(regimes) == 2 else (
        ("Low", "Normal", "High") if len(regimes) == 3 else
        tuple(f"Regime {i + 1}" for i in range(len(regimes)))
    )
    y_labels = [f"{name} · σ={float(reg['sigmaPips']):g}" for name, reg in zip(sigma_names, regimes)]
    tier_cfg = _tier_cfg(cfg, ti)
    spread = float(cfg["spreadPips"])
    sides = ("bid", "ask")
    matrices: dict[str, np.ndarray] = {}
    for direction in sides:
        rows = []
        for regime in regimes:
            r_tier = regime["tiers"][ti]
            r_sizes = np.asarray(r_tier["sizes"], dtype=float)
            if r_sizes.shape != sizes.shape or not np.allclose(r_sizes, sizes, rtol=0, atol=1e-9):
                raise ValueError("RFQ pricing knots differ between volatility regimes")
            policy = np.asarray(r_tier[direction], dtype=float)
            if policy.shape != (q.size, sizes.size):
                raise ValueError("RFQ policy matrix does not match the inventory/knot grid")
            delta = policy[:, j]
            if metric == "quotes":
                values = _quote_vs_mid_pips(delta, direction, spread)
            else:
                values = np.asarray(_hit_ratio(tier_cfg, delta, z, spread), dtype=float)
            rows.append(values)
        matrices[direction] = np.vstack(rows)

    label = "Quote vs mid [pips]" if metric == "quotes" else "Win probability"
    headline = "RFQ quotes" if metric == "quotes" else "Implied RFQ win probabilities"
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.18,
        subplot_titles=("BID", "ASK"),
    )
    if metric == "quotes":
        bound = max((float(np.max(np.abs(array))) for array in matrices.values()), default=0.0)
        bound = max(bound, 1e-6)
        cmin, cmax, scale = -bound, bound, "RdBu_r"
    else:
        cmin, cmax, scale = 0.0, 1.0, "Viridis"

    for idx, direction in enumerate(sides, start=1):
        trace = go.Heatmap(
            x=q, y=y_labels, z=matrices[direction],
            coloraxis="coloraxis", xgap=0.75, ygap=2,
            hovertemplate=(
                f"{direction.upper()} · {z:g}M RFQ<br>Inventory: %{{x:g}}M"
                f"<br>Regime: %{{y}} pips/√min"
                f"<br>{label}: %{{z:.4f}}<extra></extra>"
            ),
        )
        fig.add_trace(trace, row=idx, col=1)
        fig.update_yaxes(title_text="Volatility regime", type="category", row=idx, col=1)
    fig.update_xaxes(title_text="Inventory q [EUR M]", row=2, col=1)
    fig.update_layout(
        title=f"{headline} · {tier_cfg['name']} · {z:g}M",
        template="trinity_dark",
        height=610,
        margin=dict(l=115, r=85, t=82, b=65),
        coloraxis=dict(colorscale=scale, cmin=cmin, cmax=cmax,
                       colorbar=dict(title=label, thickness=17, len=0.82)),
    )
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


def flow_parameter_table(tier: dict[str, Any], cfg: dict[str, Any]) -> html.Table:
    rows=[]
    one_side_total = sum(0.5 * _source_total_rfq_rate(tier, source) for source in _tier_flow_sources(tier))
    for z in _rfq_sizes(tier):
        rate = _activity(tier, float(z))
        size_share = rate / one_side_total if one_side_total > 0.0 else 0.0
        rows.append([
            f"{float(z):g}",
            f"{100.0 * size_share:.2f}%", f"{rate:.6g}", f"{_delta50(tier,float(z)):.4f}",
            f"{float(tier['flow']['steepness']):.4g}", f"{float(tier.get('feePips', 0.0)):.3f}",
            f"{float(_markout_eur_per_million(tier,cfg,60.0)):.2f}",
        ])
    return _simple_table(["Size [M]","RFQ size share","λ_RFQ(z) [1/min]","δ50(z)","Steepness","Fee [pips]","Markout @60s [EUR/M]"], rows)


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


def markout_figure(tier: dict[str, Any], cfg: dict[str, Any]) -> go.Figure:
    t = np.linspace(0.0, 300.0, 301)
    fig = go.Figure()
    # Show the selected tier prominently and other active tiers for comparison.
    for other in cfg["tiers"]:
        if other is tier or not other.get("enabled", False):
            continue
        fig.add_trace(go.Scatter(
            x=t, y=_markout_eur_per_million(other, cfg, t), mode="lines",
            name=str(other["name"]), line=dict(dash="dash", width=1.5), opacity=0.65,
        ))
    fig.add_trace(go.Scatter(
        x=t, y=_markout_eur_per_million(tier, cfg, t), mode="lines",
        name=str(tier["name"]), line=dict(color=ASK_COLOR, width=3),
    ))
    fig.update_layout(
        title=f"RFQ markout · {tier['name']}",
        xaxis_title="Time since RFQ [seconds]",
        yaxis_title="MTM markout [EUR per traded EUR 1M]",
        template="trinity_dark",
    )
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
            source_spread = float(mapping.get("sourceSpreadPips", 0.0))
            if cross_mid <= 0.0 or source_spread <= 0.0:
                raise ValueError("Crossed ECN flow source requires positive cross mid and source spread")
            alpha = target_spread / (cross_mid * source_spread)
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
    fee = float(e.get("makerFeePips", 0.0))
    rows = []
    for source in _ecn_flow_sources(cfg):
        mapping = source.get("mapping", {})
        mapping_type = str(mapping.get("type", "identity"))
        if mapping_type == "crossed":
            mapping_text = (
                f"α={float(source['alpha']):.4g}; "
                f"cross={float(mapping.get('crossMid', 0.0)):.6g}; "
                f"spread={float(mapping.get('sourceSpreadPips', 0.0)):.4g}p"
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
         "Source size / target size", "Target-equivalent quote [M]", "Maker fee [pips]"],
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
            name="Total", legendgroup="total", line=dict(width=4),
        ))
    deltas = np.asarray(e.get("deltas", []), dtype=float)
    if deltas.size:
        fig.add_trace(go.Scatter(
            x=deltas, y=_ecn_arrival_rate(cfg, deltas), mode="markers",
            name="Total", legendgroup="total", showlegend=len(sources) <= 1,
            marker=dict(size=5),
        ))
    fig.update_layout(
        title="Trade intensity curves",
        xaxis_title="Target pair price improvement",
        yaxis_title="trades / min", template="trinity_dark",
    )
    fig.update_xaxes(range=[lo, hi])
    return fig


def _ecn_max_total_intensity(cfg: dict[str, Any]) -> float:
    """Maximum aggregate ECN rate on the configured quote-delta interval.

    Each source is a positive exponential in delta, so their sum is convex;
    its maximum over an interval is attained at one of the two endpoints.
    Both bid and ask use the same per-side aggregate rate (not twice it).
    """
    e = cfg.get("passiveEcn", {})
    lo = float(e.get("minDelta", 0.0))
    hi = float(e.get("maxDelta", 0.5))
    return max(float(_ecn_arrival_rate(cfg, lo)), float(_ecn_arrival_rate(cfg, hi)))


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
    fig.update_layout(
        title="Targeted trade intensity", xaxis_title="Inventory EURm",
        yaxis_title="trades / min", template="trinity_dark",
    )
    # Match the total ECN arrival-intensity ceiling in Trade intensity curves.
    max_total = _ecn_max_total_intensity(cfg)
    fig.update_yaxes(range=[0.0, max(max_total, 1e-9)], autorange=False)
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
    # Use the full 0..0.5 pricing-improvement scale on every volatility regime.
    fig.update_yaxes(range=[0.0, 0.5], autorange=False)
    return fig

def passive_ecn_regime_heatmap_figure(
    solution: dict[str, Any], cfg: dict[str, Any], metric: str,
) -> go.Figure:
    """Visualize both ECN sides on the (inventory, volatility-regime) grid.

    Disabled quotes are masked, not represented by zero delta or zero intensity.
    Every side and regime shares the same scale, so colour is comparable.
    """
    quantities = {
        "delta": ("Optimal ECN price-improvement delta", "Delta d"),
        "quotes": ("ECN quote relative to mid", "Pips"),
        "intensity": ("Targeted trade intensity", "trades / min"),
    }
    if metric not in quantities:
        raise ValueError(f"Unknown ECN heatmap metric: {metric}")
    heading, colorbar_title = quantities[metric]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.18,
        subplot_titles=("BID · hedge short inventory", "ASK · hedge long inventory"),
    )
    regimes = (solution or {}).get("regimes") or []
    if not regimes or not all(regime.get("passiveEcn") for regime in regimes):
        fig.add_annotation(text="Solve with passive ECN enabled to see the policy",
                           showarrow=False)
        fig.update_layout(title=heading, template="trinity_dark")
        return fig

    q = np.asarray(regimes[0]["passiveEcn"]["qGrid"], dtype=float)
    names = ("Low", "High") if len(regimes) == 2 else (
        ("Low", "Normal", "High") if len(regimes) == 3 else
        tuple(f"Regime {i+1}" for i in range(len(regimes)))
    )
    y_labels = [f"{name} · σ={float(r['sigmaPips']):g}" for name, r in zip(names, regimes)]
    spread_pips = float(cfg.get("spreadPips", 0.0))
    matrices = {}
    for side in ("bid", "ask"):
        rows = []
        for regime in regimes:
            policy = regime["passiveEcn"]
            q_regime = np.asarray(policy["qGrid"], dtype=float)
            if q_regime.shape != q.shape or not np.array_equal(q_regime, q):
                raise ValueError("All ECN volatility regimes must share the same inventory grid")
            delta = np.asarray(policy[f"{side}Delta"], dtype=float)
            active = np.asarray(policy[f"{side}Active"], dtype=bool)
            if delta.shape != q.shape or active.shape != q.shape:
                raise ValueError("ECN quote arrays must match the inventory grid")
            if metric == "delta":
                z = delta
            elif metric == "quotes":
                z = spread_pips * (0.5 - delta) * (-1.0 if side == "bid" else 1.0)
            else:
                z = np.asarray(_ecn_arrival_rate(cfg, delta), dtype=float)
            rows.append(np.where(active, z, np.nan))
        matrices[side] = np.vstack(rows)

    if metric == "delta":
        config = cfg.get("passiveEcn", {})
        cmin = float(config.get("minDelta", 0.0))
        cmax = float(config.get("maxDelta", 0.5))
        colorscale = "Viridis"
    elif metric == "quotes":
        bound = max(0.5 * abs(spread_pips), 1e-6)
        cmin, cmax, colorscale = -bound, bound, "RdBu_r"
    else:
        # Keep the intensity colour scale consistent with the line-chart axis.
        cmin = 0.0
        cmax = max(_ecn_max_total_intensity(cfg), 1e-9)
        colorscale = "Viridis"
    if cmax <= cmin:
        cmax = cmin + 1e-9

    for row, side in enumerate(("bid", "ask"), start=1):
        fig.add_trace(go.Heatmap(
            x=q, y=y_labels, z=matrices[side],
            coloraxis="coloraxis", xgap=0.8, ygap=2,
            hovertemplate=(
                f"{side.upper()}<br>Inventory: %{{x:g}}M"
                f"<br>Regime: %{{y}} pips/√min"
                f"<br>{colorbar_title}: %{{z:.4f}}<extra></extra>"
            ),
        ), row=row, col=1)
        fig.update_yaxes(title_text="Volatility regime", type="category", row=row, col=1)

    fig.update_xaxes(title_text="Inventory EURm" if metric == "intensity" else "Inventory q [EUR M]", row=2, col=1)
    fig.update_layout(
        title=f"{heading} · all volatility regimes",
        template="trinity_dark",
        plot_bgcolor="#202c38",  # slate indicates OFF/no resting quote
        height=610,
        margin=dict(l=115, r=80, t=82, b=80),
        coloraxis=dict(
            colorscale=colorscale, cmin=cmin, cmax=cmax,
            colorbar=dict(title=colorbar_title, thickness=16, len=0.85),
        ),
    )
    fig.add_annotation(
        text="Slate cells = no active ECN quote", x=0, y=-0.13,
        xref="paper", yref="paper", showarrow=False,
        font=dict(color=PLOT_MUTED, size=11), xanchor="left",
    )
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
        _dropdown_field("Target FX pair", "global-target-pair", DEFAULT["targetPair"], FX_PAIR_OPTIONS),
        html.Div([
            number_input("Max |q| [M]", "qmax", DEFAULT["grid"]["maxAbs"], 1.0),
            number_input("Gamma / φ", "gamma", DEFAULT["gamma"], "any"),
        ], className="grid-2"),
    ], True),
    html.Div("Pricing tiers", className="section-title"),
    *[tier_panel(i, t, open_=(i == 0)) for i, t in enumerate(DEFAULT["tiers"])],
    panel("Dark pool", [
        checkbox("Enabled", "dp-enabled", DEFAULT["darkPool"]["enabled"], channel_enable=True),
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
        checkbox("Enabled", "ecn-enabled", DEFAULT["passiveEcn"]["enabled"], channel_enable=True),
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
            number_input("Maker fee [pips]", "ecn-fee", DEFAULT["passiveEcn"]["makerFeePips"], "any"),
        ], className="grid-2"),
    ]),
    html.Button("Calibrate", id="solve", className="primary-button"),
], className="sidebar")

policy_tab = html.Div([
    html.Div([
        card("Converged", "m-converged"),
        card("Best-response iterations", "m-iterations"),
        card("Average reward", "m-rho"),
    ], className="metrics-row"),
    dcc.Tabs(id="policy-subtabs", value="overview", children=[
        dcc.Tab(label="Overview", value="overview", children=html.Div([
            diagnostic_expander("Continuation value h(q)", [
                regime_selector("overview-value-regime"),
                graph("value-chart"),
            ], open_=True),
        ], className="subtab-inner diagnostics-stack"), className="dash-tab", selected_className="dash-tab dash-tab-selected"),
        dcc.Tab(label="Tier diagnostics", value="tiers", children=html.Div([
            html.Div([select("Tier", "tier-select", [], None)], className="toolbar toolbar-wide"),
            diagnostic_expander("Tier parameters", [
                html.Div(id="flow-parameter-table", className="table-wrap"),
            ]),
            diagnostic_expander("Exogenous RFQ arrival intensity", graph("rfq-arrival-chart")),
            diagnostic_expander("Won-trade intensity", graph("flow-chart")),
            diagnostic_expander("Win probabilities", graph("hit-ratio-chart")),
            diagnostic_expander("Implied hit ratios vs inventory", [
                *rfq_heatmap_controls("tier-implied-hit"),
                graph("implied-hit-chart"),
            ]),
            diagnostic_expander("Markouts", graph("markout-chart")),
            diagnostic_expander("Quotes vs inventory", [
                *rfq_heatmap_controls("tier-quote-inventory"),
                graph("quote-inventory-chart"),
            ]),
            diagnostic_expander("Volume premium", [
                regime_selector("tier-ladder-regime"),
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
                regime_selector("dark-policy-regime"),
                graph("dp-policy-chart"),
                html.Div("Dark-pool policy", className="card-title table-section-title"),
                html.Div(id="dp-policy-table", className="table-wrap tall-table"),
            ]),
        ], className="subtab-inner diagnostics-stack"), className="dash-tab", selected_className="dash-tab dash-tab-selected"),
        dcc.Tab(label="Passive ECN", value="ecn", children=html.Div([
            diagnostic_expander("ECN parameters", [
                html.Div(id="ecn-parameter-table", className="table-wrap"),
            ]),
            diagnostic_expander("Trade intensity curves", graph("ecn-fill-chart")),
            diagnostic_expander("Targeted trade intensity", [
                ecn_heatmap_toggle("ecn-implied-hit-heatmap"),
                regime_selector("ecn-implied-hit-regime", wrapper_id="ecn-implied-hit-regime-wrapper"),
                graph("ecn-implied-hit-chart"),
            ]),
            diagnostic_expander("Quotes vs inventory", [
                ecn_heatmap_toggle("ecn-quote-inventory-heatmap"),
                regime_selector("ecn-quote-inventory-regime", wrapper_id="ecn-quote-inventory-regime-wrapper"),
                graph("ecn-quote-inventory-chart"),
            ]),
            diagnostic_expander("Optimal quote delta", [
                ecn_heatmap_toggle("ecn-policy-heatmap"),
                regime_selector("ecn-policy-regime", wrapper_id="ecn-policy-regime-wrapper"),
                graph("ecn-policy-chart"),
                html.Div([
                    html.Div("Passive ECN policy", className="card-title table-section-title"),
                    html.Div(id="ecn-policy-table", className="table-wrap tall-table"),
                ], id="ecn-policy-table-wrapper"),
            ]),
        ], className="subtab-inner tier-diagnostics-stack"), className="dash-tab", selected_className="dash-tab dash-tab-selected"),
    ], className="tabs policy-tabs"),
], className="tab-inner")

mc_tab = html.Div([
    html.Div("Monte Carlo simulates volatility switching and changes pricing policy after every regime transition.", className="muted-note"),
    html.Div([
        number_input("Paths", "mc-paths", 1000, 1),
        number_input("Initial inventory [M]", "mc-q0", 0.0, "any"),
        select("Initial volatility regime", "mc-initial-regime", [], None),
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
    html.Div("Regime-aware efficient frontier: each risk parameter re-solves the joint HJB and evaluates finite-horizon PnL moments.", className="muted-note"),
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
    dcc.Store(id="extra-flow-sources-store", data={"0": [], "1": []}),
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
                    html.Div("The target FX pair is set in Global. Configure its direct flow or add crossed FX pairs here.", className="muted-note modal-note"),
                ]),
                html.Button("×", id="flow-source-close", className="modal-close", n_clicks=0),
            ], className="modal-head"),
            html.Div([
                html.Div(id="flow-source-cards", className="flow-pair-cards"),
                # One visually unified source card: the header and dynamic
                # RFQ/volatility/markout controls surround these stable inputs.
                # Keep both Tier size fields mounted exactly once so switching
                # sources or Tiers cannot discard unsaved edits.
                html.Div([
                    html.Div(id="flow-source-heading"),
                    # Keep the locked target display mounted at all times: crossed
                    # source calculations and Solve read the same value.  The
                    # Direct card shows a disabled crossing selector in the
                    # same position where crossed sources expose the active one.
                    html.Div([
                        html.Label([
                            html.Span("Target pair"),
                            dcc.Dropdown(
                                id="flow-direct-target-display", value=DEFAULT["targetPair"],
                                options=FX_PAIR_OPTIONS, disabled=True, clearable=False,
                                className="field-dropdown",
                            ),
                        ], className="field"),
                        html.Label([
                            html.Span("Crossing pair"),
                            dcc.Dropdown(
                                id="flow-direct-crossing-placeholder",
                                options=FX_PAIR_OPTIONS,
                                value=None,
                                placeholder="Select FX pair",
                                disabled=True,
                                clearable=False,
                                className="field-dropdown",
                            ),
                        ], className="field"),
                    ], id="flow-target-picker-container",
                       className="flow-target-picker flow-pair-selectors"),
                    html.Div([
                        diagnostic_expander("Market Data", [
                            html.Div([
                                html.Div([
                                    html.Label([
                                        html.Span("Mid-rate", className="field-label"),
                                        dcc.Input(id="spot", type="number", value=DEFAULT["spot"],
                                                  step="any", className="field-input"),
                                    ], className="field"),
                                    number_input("Spread [pips]", "spread", DEFAULT["spreadPips"], "any"),
                                ], className="grid-2"),
                            ], className="flow-target-market-data"),
                        ]),
                        diagnostic_expander("Pricing Rungs", [
                            html.Div([
                                html.Div(
                                    text_input("Base M",
                                               f"t{i}-sizes", ", ".join(str(z) for z in tier["sizes"])),
                                    id=f"flow-target-tier-knots-{i}",
                                    className="flow-target-tier-knots",
                                    style={"display": "none"},
                                )
                                for i, tier in enumerate(DEFAULT["tiers"])
                            ], className="flow-target-pricing-knots"),
                        ]),
                    ], id="flow-target-direct-settings", className="flow-target-direct-settings",
                       style={"display": "none"}),
                    html.Div(id="flow-source-settings"),
                ], id="flow-source-active-card", className="flow-source-settings"),
            ], id="flow-source-editor", className="flow-source-editor",
               style={"flex": "1 1 auto", "minHeight": 0,
                      "overflowY": "auto", "overflowX": "hidden"}),
            html.Div([
                html.Div([
                    html.Button("+ Add FX pair", id="flow-source-add", className="secondary-button", n_clicks=0),
                    html.Button("Remove FX pair", id="flow-source-remove-selected",
                                className="modal-remove-button", n_clicks=0,
                                disabled=True, style={"display": "none"}),
                ], className="modal-actions-left"),
                html.Div([
                    html.Button("Close", id="flow-source-cancel", className="modal-action-button modal-action-danger", n_clicks=0),
                    html.Button("Apply", id="flow-source-apply", className="modal-action-button modal-action-apply", n_clicks=0),
                ], className="modal-actions-right"),
            ], className="modal-actions"),
        ], className="flow-source-modal-card",
           style={"height": "min(800px, calc(100vh - 56px))",
                  "minHeight": "min(800px, calc(100vh - 56px))",
                  "maxHeight": "min(800px, calc(100vh - 56px))",
                  "display": "flex", "flexDirection": "column",
                  "flexShrink": 0, "overflow": "hidden", "boxSizing": "border-box"}),
    ], id="flow-source-modal", className="flow-source-modal", style={"display": "none"}),
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Configure ECN flow sources", className="modal-title"),
                ]),
                html.Button("×", id="ecn-flow-source-close", className="modal-close", n_clicks=0),
            ], className="modal-head"),
            html.Div([
                html.Div(id="ecn-flow-source-cards", className="flow-pair-cards"),
                # Keep the read-only direct target display mounted while a crossed
                # source is selected. The same global target is shared with Tier flow.
                html.Div([
                    html.Label([
                        html.Span("Target pair"),
                        dcc.Dropdown(
                            id="ecn-direct-target-display", value=DEFAULT["targetPair"],
                            options=FX_PAIR_OPTIONS, disabled=True, clearable=False,
                            className="field-dropdown",
                        ),
                    ], className="field"),
                    html.Label([
                        html.Span("Crossing pair"),
                        dcc.Dropdown(
                            id="ecn-direct-crossing-placeholder",
                            options=FX_PAIR_OPTIONS,
                            value=None,
                            placeholder="Select FX pair",
                            disabled=True,
                            clearable=False,
                            className="field-dropdown",
                        ),
                    ], className="field"),
                ], id="ecn-target-picker-container",
                   className="flow-target-picker flow-pair-selectors"),
                html.Div(id="ecn-flow-source-settings"),
            ], id="ecn-flow-source-editor", className="flow-source-editor",
               style={"flex": "1 1 auto", "minHeight": 0,
                      "overflowY": "auto", "overflowX": "hidden"}),
            html.Div([
                html.Div([
                    html.Button("+ Add FX pair", id="ecn-flow-source-add", className="secondary-button", n_clicks=0),
                    html.Button("Remove FX pair", id="ecn-flow-source-remove-selected",
                                className="modal-remove-button", n_clicks=0,
                                disabled=True, style={"display": "none"}),
                ], className="modal-actions-left"),
                html.Div([
                    html.Button("Close", id="ecn-flow-source-cancel", className="modal-action-button modal-action-danger", n_clicks=0),
                    html.Button("Apply", id="ecn-flow-source-apply", className="modal-action-button modal-action-apply", n_clicks=0),
                ], className="modal-actions-right"),
            ], className="modal-actions"),
        ], className="flow-source-modal-card",
           style={"height": "min(800px, calc(100vh - 56px))",
                  "minHeight": "min(800px, calc(100vh - 56px))",
                  "maxHeight": "min(800px, calc(100vh - 56px))",
                  "display": "flex", "flexDirection": "column",
                  "flexShrink": 0, "overflow": "hidden", "boxSizing": "border-box"}),
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
    ("spot", State("spot", "value")), ("spread", State("spread", "value")),
    ("gamma", State("gamma", "value")),
]
for i in range(len(DEFAULT["tiers"])):
    p = f"t{i}"
    for suffix in (
        "enabled", "name", "sizes", "A0", "theta", "beta", "steep", "shift", "vshift",
        "fee", "markout-enabled", "markout-asymptote", "markout-tau-seconds", "dmin", "dmax",
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
    Output("spot", "value"),
    Output("spread", "value"),
    Input("global-target-pair", "value"),
)
def update_target_market_defaults(target_pair):
    """Load both target-pair market inputs from the packaged JSON defaults.

    Only changes in Global reset the editable inputs. Manual spot/spread
    edits do not trigger this callback or get overwritten when solving.
    """
    if not target_pair:
        return no_update, no_update
    market_data = default_fx_pair_market_data(target_pair)
    if market_data is None:
        raise ValueError(f"Missing default FX pair market data for {target_pair}")
    return market_data["spot"], market_data["spreadPips"]


@app.callback(
    Output("flow-source-add", "disabled"),
    Input("global-target-pair", "value"),
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
    return {"display": "flex", "alignItems": "flex-start"} if is_open else {"display": "none"}


@app.callback(
    Output("flow-source-selected-store", "data"),
    Input("flow-source-tier-store", "data"),
    Input("global-target-pair", "value"),
    Input({"type": "flow-pair-select", "key": ALL}, "n_clicks"),
    State("flow-source-selected-store", "data"),
)
def select_flow_source(_tier_index, _target_pair, _clicks, selected):
    trigger = ctx.triggered_id
    if trigger in ("flow-source-tier-store", "global-target-pair"):
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
    Output("flow-target-picker-container", "style"),
    Input("flow-source-selected-store", "data"),
)
def show_target_pair_picker(selected):
    """Show the read-only target pair only while editing the direct source.

    Crossed sources use the Global target pair,
    so their mapping and Solve configuration continue to use the same target.
    """
    return {"display": "none"} if selected not in (None, "direct") else {"display": "flex"}


@app.callback(
    Output("flow-source-cards", "children"),
    Input("flow-source-tier-store", "data"),
    Input("global-target-pair", "value"),
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
    if tier_index is None or not target_pair:
        return []
    ti = int(tier_index)

    extras = (data or {}).get(str(ti), [])
    selected = selected or "direct"
    cards = [flow_pair_card("direct", str(target_pair), selected == "direct")]
    for i, source in enumerate(extras):
        key = f"extra-{i}"
        cards.append(flow_pair_card(key, str(source.get("pair") or "Select FX pair"), selected == key, removable=True))
    return cards


@app.callback(
    Output("flow-target-direct-settings", "style"),
    *[Output(f"flow-target-tier-knots-{i}", "style") for i in range(len(DEFAULT["tiers"]))],
    Input("flow-source-tier-store", "data"),
    Input("flow-source-selected-store", "data"),
)
def show_target_pair_settings_for_tier(tier_index, selected):
    """Show only the active Tier's RFQ knots, without unmounting either field.

    Target-pair market data and pricing knots are configured on the direct
    source card.  Crossed sources have their own separate flow parameters.
    """
    direct = tier_index is not None and (selected or "direct") == "direct"
    tier = int(tier_index) if direct else -1
    return (
        {"display": "block"} if direct and 0 <= tier < len(DEFAULT["tiers"]) else {"display": "none"},
        *({"display": "block"} if direct and i == tier else {"display": "none"}
          for i in range(len(DEFAULT["tiers"]))),
    )


def _removable_crossed_source(selected, sources) -> bool:
    """Only a currently selected, existing crossed source may be removed."""
    try:
        if not isinstance(selected, str) or not selected.startswith("extra-"):
            return False
        index = int(selected.split("-", 1)[1])
        return 0 <= index < len(sources or [])
    except (ValueError, TypeError):
        return False


@app.callback(
    Output("flow-source-remove-selected", "style"),
    Output("flow-source-remove-selected", "disabled"),
    Input("flow-source-selected-store", "data"),
    Input("flow-source-tier-store", "data"),
    Input("extra-flow-sources-store", "data"),
)
def show_flow_source_remove_button(selected, tier_index, data):
    sources = (data or {}).get(str(tier_index), []) if tier_index is not None else []
    visible = _removable_crossed_source(selected, sources)
    return ({"display": "inline-flex"} if visible else {"display": "none"}), not visible


@app.callback(
    Output("flow-source-heading", "children"),
    Output("flow-source-settings", "children"),
    Input("flow-source-tier-store", "data"),
    Input("global-target-pair", "value"),
    Input("flow-source-selected-store", "data"),
    State("extra-flow-sources-store", "data"),
    *[State(f"t{i}-{suffix}", "value") for i in range(len(DEFAULT["tiers"])) for suffix in ("A0", "theta", "beta", "steep", "shift", "vshift")],
    *[State(f"t{i}-{suffix}", "value") for i in range(len(DEFAULT["tiers"])) for suffix in ("markout-enabled", "markout-asymptote", "markout-tau-seconds")],
)
def render_flow_source_settings(tier_index, target_pair, selected, data, *direct_values):
    """Render the settings form only when its identity changes.

    ``extra-flow-sources-store`` and the direct-flow inputs are *State*, not
    Input.  Consequently ordinary edits persist into the stores without
    remounting the form that the user is currently interacting with.
    """
    if tier_index is None:
        return [], []
    ti = int(tier_index)
    if not target_pair:
        return html.Div(
            "Choose the target FX pair in Global to configure flow sources.",
            className="muted-note flow-source-empty",
        ), []

    extras = (data or {}).get(str(ti), [])
    selected = selected or "direct"
    tier_count = len(DEFAULT["tiers"])
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

    mo_offset = 6 * tier_count + ti * 3
    markout_vals = direct_values[mo_offset:mo_offset + 3]
    markout = {"enabled": checked(markout_vals[0]), "asymptote": markout_vals[1], "tau": markout_vals[2]}
    editor = flow_source_settings_editor(selected, source, str(target_pair), markout)
    heading, expanders = editor.children
    return heading, expanders


@app.callback(
    Output("flow-edit-cross-pair", "value"),
    Input("flow-edit-pair", "value", allow_optional=True),
    State("flow-source-tier-store", "data"),
    State("flow-source-selected-store", "data"),
    State("extra-flow-sources-store", "data"),
    prevent_initial_call=True,
)
def clear_tier_crossing_on_source_change(pair, tier_index, selected, data):
    """A new crossed-pair selection requires an explicit crossing choice."""
    if tier_index is None or not isinstance(selected, str) or not selected.startswith("extra-"):
        return no_update
    try:
        source = (data or {}).get(str(tier_index), [])[int(selected.split("-", 1)[1])]
    except (IndexError, ValueError):
        return no_update
    # The source store update may arrive before or after this callback.
    # Either way, reset if the pair changed or the mapping was invalidated.
    return None if source.get("pair") != pair or not source.get("crossPair") else no_update


@app.callback(
    Output("flow-edit-cross-mid", "value"),
    Input("flow-edit-cross-pair", "value", allow_optional=True),
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
    Input("flow-source-add", "n_clicks"),
    Input("flow-source-remove-selected", "n_clicks"),
    Input("flow-edit-pair", "value", allow_optional=True), Input("flow-edit-cross-pair", "value", allow_optional=True),
    Input("flow-edit-cross-mid", "value", allow_optional=True), Input("flow-edit-spread", "value", allow_optional=True),
    Input("flow-edit-A0", "value", allow_optional=True), Input("flow-edit-theta", "value", allow_optional=True), Input("flow-edit-beta", "value", allow_optional=True),
    Input("flow-edit-steep", "value", allow_optional=True), Input("flow-edit-shift", "value", allow_optional=True), Input("flow-edit-vshift", "value", allow_optional=True),
    State("flow-source-tier-store", "data"), State("global-target-pair", "value"), State("flow-source-selected-store", "data"), State("extra-flow-sources-store", "data"),
    *[State(f"t{i}-{suffix}", "value") for i in range(len(DEFAULT["tiers"])) for suffix in ("A0", "theta", "beta", "steep", "shift", "vshift")],
    prevent_initial_call=True,
)
def update_flow_source_editor(_add, _remove, pair, cross_pair, cross_mid, spread, A0, theta, beta, steep, shift, vshift,
                              tier_index, target_pair, selected, data, *direct_values):
    if tier_index is None:
        return (no_update,) * 14
    ti = int(tier_index)
    key = str(ti)
    out = deepcopy(data or {"0": [], "1": []})
    direct = list(direct_values)
    selected = selected or "direct"
    trigger = ctx.triggered_id

    if trigger == "flow-source-add":
        # n_clicks=0 can be observed during component initialisation; only a
        # positive click count represents an actual Add action.
        if not _add:
            return (no_update,) * 14
        if not target_pair:
            return out, selected, *direct
        out.setdefault(key, []).append(_default_cross_flow_source(len(out.get(key, [])), target_pair))
        new_selected = f"extra-{len(out[key]) - 1}"
        return out, new_selected, *direct

    if trigger == "flow-source-remove-selected":
        # The persistent footer action is only enabled for a crossed source.
        # Ignore callback initialisation (n_clicks=0).
        if not _remove:
            return (no_update,) * 14
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
        if 0 <= idx < len(out.get(key, [])) and all(x is not None for x in (spread, A0, theta, beta, steep, shift, vshift)):
            src = out[key][idx]
            chosen_pair = str(pair) if pair else None
            # Do not implicitly select a crossing pair. Changing the crossed
            # source invalidates the previous mapping until explicitly chosen.
            pair_changed = trigger == "flow-edit-pair" and chosen_pair != src.get("pair")
            chosen_cross = None if pair_changed else (str(cross_pair) if cross_pair and chosen_pair else None)

            # Pair selection is market-data driven: use the temporary default
            # mid for the newly selected cross.  A manually edited Cross mid is
            # preserved for all other triggers.
            # Dash updates the source-pair, crossing-pair and crossing-mid
            # inputs in separate callbacks.  A flow/size edit may reach this
            # store before the crossing-mid callback has populated its field.
            # Always resolve a missing mid from the configured FX defaults,
            # regardless of which input triggered the update.  Changing the
            # pair deliberately refreshes the default; other edits preserve a
            # manually supplied mid.
            chosen_mid = None if not chosen_cross else _editor_cross_mid(
                chosen_cross, cross_mid, trigger in ("flow-edit-pair", "flow-edit-cross-pair"))
            if chosen_mid is not None and chosen_mid <= 0.0:
                raise ValueError("Cross mid must be positive")
            if float(spread) <= 0.0:
                raise ValueError("Source spread must be positive")

            src.update({"name": chosen_pair or "Select FX pair", "pair": chosen_pair, "crossPair": chosen_cross})
            src["mapping"] = {"type": "crossed", "crossMid": chosen_mid, "sourceSpreadPips": float(spread)}
            src["flow"] = {"A0": float(A0), "theta": float(theta), "beta": float(beta), "steepness": float(steep), "shift": float(shift), "volumeShift": float(vshift)}
        return out, no_update, *direct

    return out, no_update, *direct


def _crossed_sources_ready(sources, selected, pair, crossing_pair):
    """An Apply must never accept a newly-added, incomplete crossed source.

    Also check the visible editor, so Apply stays disabled while Dash's
    editor-to-store callback is still processing a pair selection.
    """
    sources = sources or []
    if any(not source.get("pair") or not source.get("crossPair") for source in sources):
        return False
    if isinstance(selected, str) and selected.startswith("extra-"):
        try:
            source = sources[int(selected.split("-", 1)[1])]
        except (ValueError, IndexError):
            return False
        return bool(pair and crossing_pair and
                    pair == source.get("pair") and crossing_pair == source.get("crossPair"))
    return True


@app.callback(
    Output("flow-source-apply", "disabled"),
    Input("flow-source-tier-store", "data"),
    Input("extra-flow-sources-store", "data"),
    Input("flow-source-selected-store", "data"),
    Input("flow-edit-pair", "value", allow_optional=True),
    Input("flow-edit-cross-pair", "value", allow_optional=True),
)
def disable_incomplete_tier_flow_apply(tier_index, data, selected, pair, crossing_pair):
    if tier_index is None:
        return False
    return not _crossed_sources_ready((data or {}).get(str(tier_index), []), selected, pair, crossing_pair)


@app.callback(
    Output("ecn-flow-source-apply", "disabled"),
    Input("ecn-extra-flow-sources-store", "data"),
    Input("ecn-flow-source-selected-store", "data"),
    Input("ecn-flow-edit-pair", "value", allow_optional=True),
    Input("ecn-flow-edit-cross-pair", "value", allow_optional=True),
)
def disable_incomplete_ecn_flow_apply(data, selected, pair, crossing_pair):
    return not _crossed_sources_ready(data, selected, pair, crossing_pair)


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
    return {"display": "flex", "alignItems": "flex-start"} if is_open else {"display": "none"}


@app.callback(
    Output("ecn-flow-source-add", "disabled"),
    Input("global-target-pair", "value"),
)
def disable_add_ecn_flow_source_without_target(target_pair):
    return not bool(target_pair)


@app.callback(
    Output("ecn-flow-source-selected-store", "data"),
    Input("global-target-pair", "value"),
    Input({"type": "ecn-flow-pair-select", "key": ALL}, "n_clicks"),
    State("ecn-flow-source-selected-store", "data"),
)
def select_ecn_flow_source(_target_pair, _clicks, selected):
    trigger = ctx.triggered_id
    if trigger == "global-target-pair":
        return "direct"
    if isinstance(trigger, dict) and trigger.get("type") == "ecn-flow-pair-select":
        triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
        if not triggered_value:
            return no_update
        return str(trigger.get("key", "direct"))
    return selected or "direct"


@app.callback(
    Output("flow-direct-target-display", "value"),
    Output("ecn-direct-target-display", "value"),
    Input("global-target-pair", "value"),
)
def show_global_target_in_flow_editors(target_pair):
    """Read-only reflections of the only editable target FX pair."""
    return target_pair, target_pair


@app.callback(
    Output("extra-flow-sources-store", "data", allow_duplicate=True),
    Output("ecn-extra-flow-sources-store", "data", allow_duplicate=True),
    Input("global-target-pair", "value"),
    prevent_initial_call=True,
)
def clear_crossed_sources_on_target_change(_target_pair):
    """Old crosses can be incompatible with the new target base/quote.

    Do not silently reinterpret an old crossMid or source flow under a new
    target FX pair: require explicitly re-adding crossed sources instead.
    """
    return {str(i): [] for i in range(len(DEFAULT["tiers"]))}, []


@app.callback(
    Output("ecn-target-picker-container", "style"),
    Input("ecn-flow-source-selected-store", "data"),
)
def show_ecn_target_pair_picker(selected):
    return {"display": "none"} if selected not in (None, "direct") else {"display": "flex"}


@app.callback(
    Output("ecn-flow-source-cards", "children"),
    Input("global-target-pair", "value"),
    Input("ecn-flow-source-selected-store", "data"),
    Input("ecn-extra-flow-sources-store", "data"),
)
def render_ecn_flow_source_cards(target_pair, selected, data):
    if not target_pair:
        return []
    extras = data or []
    selected = selected or "direct"
    cards = [ecn_flow_pair_card("direct", str(target_pair), selected == "direct")]
    for i, source in enumerate(extras):
        key = f"extra-{i}"
        cards.append(ecn_flow_pair_card(key, str(source.get("pair") or "Select FX pair"), selected == key, removable=True))
    return cards


@app.callback(
    Output("ecn-flow-source-remove-selected", "style"),
    Output("ecn-flow-source-remove-selected", "disabled"),
    Input("ecn-flow-source-selected-store", "data"),
    Input("ecn-extra-flow-sources-store", "data"),
)
def show_ecn_flow_source_remove_button(selected, data):
    visible = _removable_crossed_source(selected, data)
    return ({"display": "inline-flex"} if visible else {"display": "none"}), not visible


@app.callback(
    Output("ecn-flow-source-settings", "children"),
    Input("global-target-pair", "value"),
    Input("ecn-flow-source-selected-store", "data"),
    State("ecn-extra-flow-sources-store", "data"),
    State("ecn-A", "value"), State("ecn-k", "value"), State("ecn-direct-size-probs-store", "data"),
)
def render_ecn_flow_source_settings(target_pair, selected, data, direct_A, direct_k, direct_size_probs):
    if not target_pair:
        return html.Div(
            "Choose the target FX pair in Global to configure ECN flow sources.",
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
    Output("ecn-flow-edit-p1m", "value"),
    Output("ecn-flow-edit-p750k", "value"),
    Output("ecn-flow-edit-p500k", "value"),
    Output("ecn-flow-edit-p250k", "value"),
    Input("ecn-flow-edit-pair", "value", allow_optional=True),
    State("ecn-flow-source-selected-store", "data"),
    prevent_initial_call=True,
)
def populate_ecn_trade_size_defaults(source_pair, selected):
    # Only crossed sources have a user-editable source-pair dropdown.  When a
    # calibrated pair is selected, seed its empirical size distribution.
    # Unknown pairs preserve the values already shown in the editor.
    if not source_pair or str(selected or "direct") == "direct":
        return no_update, no_update, no_update, no_update
    key = str(source_pair).upper().strip()
    if key not in {"EURSEK", "USDSEK"}:
        return no_update, no_update, no_update, no_update
    probs = default_ecn_trade_size_probabilities(key)
    return probs[0], probs[1], probs[2], probs[3]


@app.callback(
    Output("ecn-flow-edit-cross-pair", "value"),
    Input("ecn-flow-edit-pair", "value", allow_optional=True),
    State("ecn-flow-source-selected-store", "data"),
    State("ecn-extra-flow-sources-store", "data"),
    prevent_initial_call=True,
)
def clear_ecn_crossing_on_source_change(pair, selected, data):
    if not isinstance(selected, str) or not selected.startswith("extra-"):
        return no_update
    try:
        source = (data or [])[int(selected.split("-", 1)[1])]
    except (IndexError, ValueError):
        return no_update
    return None if source.get("pair") != pair or not source.get("crossPair") else no_update


@app.callback(
    Output("ecn-flow-edit-cross-mid", "value"),
    Input("ecn-flow-edit-cross-pair", "value", allow_optional=True),
    prevent_initial_call=True,
)
def populate_ecn_flow_source_cross_mid(cross_pair):
    mid = default_fx_mid(cross_pair)
    return mid if mid is not None else no_update


@app.callback(
    Output("ecn-flow-edit-p100k-note", "children"),
    Input("ecn-flow-edit-p1m", "value", allow_optional=True), Input("ecn-flow-edit-p750k", "value", allow_optional=True),
    Input("ecn-flow-edit-p500k", "value", allow_optional=True), Input("ecn-flow-edit-p250k", "value", allow_optional=True),
    prevent_initial_call=True,
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
    Input("ecn-flow-edit-pair", "value", allow_optional=True), Input("ecn-flow-edit-cross-pair", "value", allow_optional=True),
    Input("ecn-flow-edit-cross-mid", "value", allow_optional=True), Input("ecn-flow-edit-spread", "value", allow_optional=True),
    Input("ecn-flow-edit-A", "value", allow_optional=True), Input("ecn-flow-edit-k", "value", allow_optional=True),
    Input("ecn-flow-edit-p1m", "value", allow_optional=True), Input("ecn-flow-edit-p750k", "value", allow_optional=True),
    Input("ecn-flow-edit-p500k", "value", allow_optional=True), Input("ecn-flow-edit-p250k", "value", allow_optional=True),
    State("global-target-pair", "value"), State("ecn-flow-source-selected-store", "data"),
    State("ecn-extra-flow-sources-store", "data"), State("ecn-A", "value"), State("ecn-k", "value"),
    State("ecn-direct-size-probs-store", "data"),
    prevent_initial_call=True,
)
def update_ecn_flow_source_editor(_add, _remove, pair, cross_pair, cross_mid, spread, A, k,
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

    # The ECN controls may not yet be mounted immediately after opening the
    # dashboard/modal; optional Dash Inputs are None in that transient state.
    if any(value is None for value in (p1m, p750k, p500k, p250k)):
        return (no_update,) * 5
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
        if 0 <= idx < len(out) and all(x is not None for x in (spread, A, k)):
            src = out[idx]
            chosen_pair = str(pair) if pair else None
            pair_changed = trigger == "ecn-flow-edit-pair" and chosen_pair != src.get("pair")
            chosen_cross = None if pair_changed else (str(cross_pair) if cross_pair and chosen_pair else None)

            # The pair and mid inputs update asynchronously.  Look up the
            # default even if an unrelated field triggered this callback
            # while the Crossing Mid-Rate input is still blank.
            chosen_mid = None if not chosen_cross else _editor_cross_mid(
                chosen_cross, cross_mid, trigger in ("ecn-flow-edit-pair", "ecn-flow-edit-cross-pair"))
            if chosen_mid is not None and chosen_mid <= 0.0:
                raise ValueError("Cross mid must be positive")
            if float(spread) <= 0.0:
                raise ValueError("Source spread must be positive")
            if float(A) < 0.0:
                raise ValueError("ECN A must be nonnegative")
            if float(k) <= 0.0:
                raise ValueError("ECN k must be positive")

            src.update({"name": chosen_pair or "Select FX pair", "pair": chosen_pair, "crossPair": chosen_cross})
            src["mapping"] = {"type": "crossed", "crossMid": chosen_mid, "sourceSpreadPips": float(spread)}
            src["flow"] = {"A": float(A), "k": float(k)}
            src["tradeSizeProbabilities"] = probs
            src.pop("meanTradeSize", None)
        return out, no_update, direct_A, direct_k, direct_size_probs

    return out, no_update, direct_A, direct_k, direct_size_probs


@app.callback(
    Output("config-store", "data"), Output("solution-store", "data"), Output("status", "children"), Output("status", "className"),
    Input("solve", "n_clicks"), *CONFIG_STATES,
    State("extra-flow-sources-store", "data"), State("ecn-extra-flow-sources-store", "data"),
    State("global-target-pair", "value"), prevent_initial_call=True,
)
def solve_model(_clicks, *values):
    try:
        *field_values, extra_sources, ecn_extra_sources, target_pair = values
        named_values = dict(zip(CONFIG_KEYS, field_values, strict=True))
        named_values["target-pair"] = target_pair
        cfg = build_config(named_values)
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
        method = "Production C++ (q, σ)"
        return cfg, solution, f"Solved · {method} · {solution['iterations']} iterations", "status ready"
    except Exception as exc:
        return no_update, no_update, f"Solve failed · {exc}", "status error"


@app.callback(
    Output("m-converged", "children"), Output("m-iterations", "children"), Output("m-rho", "children"),
    Output("tier-select", "options"), Output("tier-select", "value"),
    Output("ladder-q", "options"), Output("ladder-q", "value"),
    Input("solution-store", "data"),
)
def render_solution_summary(solution):
    if not solution:
        return "—", "—", "—", [], None, [], None
    opts = [{"label": t["name"], "value": i} for i, t in enumerate(solution["tiers"])]
    qopts = [{"label": f"{float(q):g}M", "value": float(q)} for q in solution["qGrid"]]
    q0 = min((float(q) for q in solution["qGrid"]), key=abs) if qopts else None
    return (
        "Yes" if solution["converged"] else "No", str(solution["iterations"]),
        f"{solution['averageReward']:.6g}", opts, (0 if opts else None), qopts, q0,
    )


@app.callback(
    Output("value-chart", "figure"),
    Input("solution-store", "data"), Input("overview-value-regime", "value"),
)
def render_continuation_value(solution, regime_index):
    if not solution:
        return go.Figure()
    current = select_policy_regime(solution, regime_index)
    fig = go.Figure(go.Scatter(x=current["qGrid"], y=current["value"], mode="lines+markers", name="h(q)"))
    fig.update_layout(
        title=f"Continuation value h(q, σ={current.get('selectedSigmaPips', '')} pips/√min)",
        xaxis_title="Inventory q [EUR M]", yaxis_title="h(q)", template="trinity_dark",
    )
    return fig


@app.callback(
    Output("rfq-arrival-chart", "figure"), Output("flow-chart", "figure"), Output("hit-ratio-chart", "figure"), Output("implied-hit-chart", "figure"),
    Output("markout-chart", "figure"), Output("quote-inventory-chart", "figure"),
    Output("ladder-chart", "figure"),
    Output("flow-parameter-table", "children"), Output("ladder-table", "children"),
    Input("solution-store", "data"), Input("config-store", "data"), Input("tier-select", "value"), Input("ladder-q", "value"),
    *[Input(component_id, "value") for component_id in POLICY_REGIME_SELECTOR_IDS[1:4]],
    *[Input(f"{prefix}-heatmap", "value") for prefix in RFQ_HEATMAP_PREFIXES],
    *[Input(f"{prefix}-rfq-size", "value") for prefix in RFQ_HEATMAP_PREFIXES],
)
def render_tier_diagnostics(solution, cfg, tier_index, q_value, implied_regime,
                            quotes_regime, ladder_regime,
                            implied_heatmap=None, quotes_heatmap=None,
                            implied_size=None, quotes_size=None):
    empty = (go.Figure(),) * 7 + (html.Div("Solve the model first"), html.Div("Solve the model first"))
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
            (rfq_regime_heatmap_figure(solution, cfg, ti, implied_size, metric="win_probability")
             if _ecn_heatmap_enabled(implied_heatmap) else
             implied_hit_ratio_figure(select_policy_regime(solution, implied_regime), cfg, ti)),
            markout_figure(tier, cfg),
            (rfq_regime_heatmap_figure(solution, cfg, ti, quotes_size)
             if _ecn_heatmap_enabled(quotes_heatmap) else
             quote_inventory_figure(select_policy_regime(solution, quotes_regime), cfg, ti)),
            ladder_figure(select_policy_regime(solution, ladder_regime), cfg, ti, qv),
            flow_parameter_table(tier, cfg),
            ladder_table(select_policy_regime(solution, ladder_regime), cfg, ti, qv),
        )
    except Exception as exc:
        err = html.Div(f"Tier diagnostic failed: {exc}", className="error-note")
        return (go.Figure(),) * 7 + (err, err)


# Every RFQ heatmap has its own display toggle and knot-size dropdown.
# The dropdowns share the selected Tier's available actual RFQ sizes.


@app.callback(
    *[Output(f"{prefix}-rfq-size", "options") for prefix in RFQ_HEATMAP_PREFIXES],
    *[Output(f"{prefix}-rfq-size", "value") for prefix in RFQ_HEATMAP_PREFIXES],
    Input("solution-store", "data"), Input("tier-select", "value"),
    *[State(f"{prefix}-rfq-size", "value") for prefix in RFQ_HEATMAP_PREFIXES],
)
def update_tier_heatmap_sizes(solution, tier_index, *current_sizes):
    count = len(RFQ_HEATMAP_PREFIXES)
    if not solution or tier_index is None:
        return (*([[]] * count), *([None] * count))
    tiers = solution.get("tiers") or []
    ti = int(tier_index)
    if not (0 <= ti < len(tiers)):
        return (*([[]] * count), *([None] * count))
    sizes = [float(size) for size in tiers[ti].get("sizes", [])]
    options = [{"label": f"{size:g}M", "value": size} for size in sizes]
    selected = [
        next((size for size in sizes if choice is not None and abs(size - float(choice)) < 1e-9),
             sizes[0] if sizes else None)
        for choice in current_sizes
    ]
    return (*([options] * count), *selected)


@app.callback(
    *[Output(f"{prefix}-regime-wrapper", "style") for prefix in RFQ_HEATMAP_PREFIXES],
    *[Output(f"{prefix}-rfq-size-wrapper", "style") for prefix in RFQ_HEATMAP_PREFIXES],
    *[Output(chart_id, "style") for chart_id in
      ("implied-hit-chart", "quote-inventory-chart")],
    *[Input(f"{prefix}-heatmap", "value") for prefix in RFQ_HEATMAP_PREFIXES],
)
def toggle_tier_regime_heatmaps(*toggles):
    enabled = [_ecn_heatmap_enabled(value) for value in toggles]
    regime_styles = [{"display": "none"} if active else {} for active in enabled]
    size_styles = [{} if active else {"display": "none"} for active in enabled]
    chart_styles = [
        {"height": f"{610 if active else 420}px", "width": "100%"}
        for active in enabled
    ]
    return (*regime_styles, *size_styles, *chart_styles)


@app.callback(
    Output("dp-arrival-chart", "figure"), Output("dp-full-fill-chart", "figure"),
    Output("dp-policy-chart", "figure"), Output("dp-policy-table", "children"),
    Input("solution-store", "data"), Input("config-store", "data"), Input("dark-policy-regime", "value"),
)
def render_dark_pool_diagnostics(solution, cfg, regime_index):
    if not solution or not cfg:
        return go.Figure(), go.Figure(), go.Figure(), html.Div("Solve the model first")
    if not cfg.get("darkPool", {}).get("enabled", False):
        disabled = go.Figure()
        disabled.add_annotation(text="Dark pool disabled", showarrow=False)
        disabled.update_layout(template="trinity_dark")
        return disabled, disabled, disabled, html.Div("Dark pool disabled", className="muted-note")
    selected = select_policy_regime(solution, regime_index)
    return (
        dark_pool_arrival_figure(cfg), dark_pool_full_fill_figure(cfg),
        dark_pool_policy_figure(selected), dark_pool_table(selected),
    )


@app.callback(
    Output("ecn-fill-chart", "figure"), Output("ecn-implied-hit-chart", "figure"), Output("ecn-quote-inventory-chart", "figure"),
    Output("ecn-policy-chart", "figure"), Output("ecn-parameter-table", "children"),
    Output("ecn-policy-table", "children"),
    Input("solution-store", "data"), Input("config-store", "data"),
    Input("ecn-implied-hit-regime", "value"), Input("ecn-quote-inventory-regime", "value"),
    Input("ecn-policy-regime", "value"),
    Input("ecn-implied-hit-heatmap", "value"),
    Input("ecn-quote-inventory-heatmap", "value"),
    Input("ecn-policy-heatmap", "value"),
)
def render_passive_ecn_diagnostics(solution, cfg, implied_regime, quotes_regime, policy_regime,
                                   implied_heatmap=None, quotes_heatmap=None, policy_heatmap=None):
    if not solution or not cfg:
        return (go.Figure(),) * 4 + (html.Div("Solve the model first"), html.Div("Solve the model first"))
    if not cfg.get("passiveEcn", {}).get("enabled", False):
        disabled = go.Figure()
        disabled.add_annotation(text="Passive ECN disabled", showarrow=False)
        disabled.update_layout(template="trinity_dark")
        note = html.Div("Passive ECN disabled", className="muted-note")
        return disabled, disabled, disabled, disabled, note, note
    return (
        passive_ecn_fill_figure(cfg),
        (passive_ecn_regime_heatmap_figure(solution, cfg, "intensity")
         if _ecn_heatmap_enabled(implied_heatmap) else
         passive_ecn_implied_hit_ratio_figure(select_policy_regime(solution, implied_regime), cfg)),
        (passive_ecn_regime_heatmap_figure(solution, cfg, "quotes")
         if _ecn_heatmap_enabled(quotes_heatmap) else
         passive_ecn_quote_inventory_figure(select_policy_regime(solution, quotes_regime), cfg)),
        (passive_ecn_regime_heatmap_figure(solution, cfg, "delta")
         if _ecn_heatmap_enabled(policy_heatmap) else
         passive_ecn_policy_figure(select_policy_regime(solution, policy_regime))),
        passive_ecn_parameter_table(cfg),
        passive_ecn_table(select_policy_regime(solution, policy_regime)),
    )


@app.callback(
    Output("ecn-implied-hit-regime-wrapper", "style"),
    Output("ecn-quote-inventory-regime-wrapper", "style"),
    Output("ecn-policy-regime-wrapper", "style"),
    Output("ecn-implied-hit-chart", "style"),
    Output("ecn-quote-inventory-chart", "style"),
    Output("ecn-policy-chart", "style"),
    Output("ecn-policy-table-wrapper", "style"),
    Input("ecn-implied-hit-heatmap", "value"),
    Input("ecn-quote-inventory-heatmap", "value"),
    Input("ecn-policy-heatmap", "value"),
)
def toggle_ecn_regime_heatmaps(implied, quotes, policy):
    """Hide a selector only for its heatmap, preserving its choice for line view."""
    enabled = [_ecn_heatmap_enabled(value) for value in (implied, quotes, policy)]
    selectors = [{"display": "none"} if active else {} for active in enabled]
    chart_styles = [{"height": "610px" if active else "420px", "width": "100%"}
                    for active in enabled]
    return (*selectors, *chart_styles, {"display": "none"} if enabled[2] else {})


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


def _mc_analytics_cache_key(cfg: dict[str, Any], q0: float, initial_regime: int = -1) -> str:
    return json.dumps(
        {"cfg": cfg, "horizon": SESSION_HORIZON, "q0": float(q0), "regime": int(initial_regime)},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _compute_mc_analytics(engine, cfg: dict[str, Any], solution: dict[str, Any] | None,
                          q0: float, initial_regime: int = -1) -> dict[str, Any]:
    """Return deterministic analytical benchmarks, reusing them across MC runs."""
    key = _mc_analytics_cache_key(cfg, q0, initial_regime)
    with _MC_ANALYTICS_CACHE_LOCK:
        cached = _MC_ANALYTICS_CACHE.get(key)
        if cached is not None:
            return deepcopy(cached)

    stats = engine.statistics(SESSION_HORIZON, q0, initial_regime=initial_regime)

    with _MC_ANALYTICS_CACHE_LOCK:
        # Dict insertion order is sufficient for this tiny FIFO cache.  The
        # config-keyed Engine cache already has similar bounded semantics.
        if key not in _MC_ANALYTICS_CACHE and len(_MC_ANALYTICS_CACHE) >= _MC_ANALYTICS_CACHE_MAX_ENTRIES:
            oldest = next(iter(_MC_ANALYTICS_CACHE))
            _MC_ANALYTICS_CACHE.pop(oldest, None)
        _MC_ANALYTICS_CACHE[key] = deepcopy(stats)
    return stats


def _run_mc_job(job_id: str, cfg: dict[str, Any], solution: dict[str, Any] | None,
                paths: int, q0: float, seed: int, initial_regime: int) -> None:
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
                _compute_mc_analytics, engine, cfg, solution, q0, initial_regime
            )

            mc = engine.simulate(
                SESSION_HORIZON, paths, q0, seed,
                retained_paths=8, sample_points=381,
                progress_callback=report_progress,
                compact_result=True,
                initial_regime=initial_regime,
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
    Output("mc-initial-regime", "options"), Output("mc-initial-regime", "value"),
    Input("solution-store", "data"),
)
def mc_volatility_regime_options(solution):
    if not solution or not solution.get("regimes"):
        return [], None
    sigmas=solution.get("sigmaPips", [])
    names=("Low", "High") if len(sigmas)==2 else ("Low", "Normal", "High")
    options=[{"label": f"{name} · σ={float(sigma):g} pips/√min", "value": i}
             for i,(name,sigma) in enumerate(zip(names,sigmas))]
    return options, len(options)//2


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
    State("mc-initial-regime", "value"),
    prevent_initial_call=True,
)
def start_mc(_clicks, cfg, solution, paths, q0, seed, initial_regime):
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
        args=(job_id, cfg_copy, solution_copy, paths, q0, seed,
              int(initial_regime) if initial_regime is not None else -1),
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
        sample_regimes = np.asarray(path.get("regimes", [len(solution.get("regimes", []))//2] * len(inventories)), dtype=int)
        if len(sample_regimes) != len(inventories):
            sample_regimes = np.full(len(inventories), len(solution.get("regimes", []))//2, dtype=int)
        for tier_idx, tier in enumerate(solution.get("tiers", [])):
            sizes = [float(z) for z in tier.get("sizes", [])]
            matches = [j for j,z in enumerate(sizes) if math.isclose(z, float(quote_size), rel_tol=0.0, abs_tol=1e-12)]
            if not matches:
                continue
            j = matches[0]
            bid_col = [row[j] for row in tier["bid"]]
            ask_col = [row[j] for row in tier["ask"]]
            if solution.get("regimes"):
                regime_tiers=[r["tiers"][tier_idx] for r in solution["regimes"]]
                bid_delta = np.asarray([_interp_policy(q_grid, [row[j] for row in regime_tiers[int(regime)]["bid"]], q)
                    for q, regime in zip(inventories, sample_regimes)])
                ask_delta = np.asarray([_interp_policy(q_grid, [row[j] for row in regime_tiers[int(regime)]["ask"]], q)
                    for q, regime in zip(inventories, sample_regimes)])
            else:
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
            if solution.get("regimes"):
                ecn_policies=[r.get("passiveEcn") for r in solution["regimes"]]
                ecn_bid_delta = np.asarray([float(ecn_policies[r]["bidDelta"][i]) for r,i in zip(sample_regimes, ecn_idx)])
                ecn_ask_delta = np.asarray([float(ecn_policies[r]["askDelta"][i]) for r,i in zip(sample_regimes, ecn_idx)])
                ecn_bid_active = np.asarray([bool(ecn_policies[r]["bidActive"][i]) for r,i in zip(sample_regimes, ecn_idx)])
                ecn_ask_active = np.asarray([bool(ecn_policies[r]["askActive"][i]) for r,i in zip(sample_regimes, ecn_idx)])
            else:
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
    State("frontier-points", "value"), State("frontier-log", "value"), State("mc-q0", "value"),
    State("mc-initial-regime", "value"), prevent_initial_call=True,
)
def run_frontier(_clicks, cfg, lo, hi, points, log_values, q0, initial_regime):
    if not cfg:
        return no_update, "Solve the model first", "status error"
    try:
        lo, hi, n = float(lo), float(hi), int(points)
        if not 0 < lo < hi or n < 3:
            raise ValueError("Require 0 < φ min < φ max and at least 3 points")
        gammas = np.geomspace(lo, hi, n) if checked(log_values) else np.linspace(lo, hi, n)
        gammas = np.unique(np.append(gammas, float(cfg["gamma"]))).tolist()
        frontier = ENGINES.get(cfg).frontier(gammas, SESSION_HORIZON, float(q0),
                                            initial_regime=int(initial_regime) if initial_regime is not None else -1)
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

    header = html.Thead(html.Tr([html.Th(x) for x in ["φ", "Expected PnL", "Stdev", "Mean / stdev", "Policy iterations"]]))
    rows = [html.Tr([html.Td(f"{x['gamma']:.5g}"), html.Td(format_ccy(x["mean"])), html.Td(format_ccy(x["std"])), html.Td(f"{x['ratio']:.4f}"), html.Td(str(x["iterations"]))]) for x in ordered]
    return fig, ratio, html.Table([header, html.Tbody(rows)], className="data-table")


@app.callback(
    Output("flow-markout-preview", "figure"),
    Input("flow-edit-markout-enabled", "value", allow_optional=True),
    Input("flow-edit-markout-asymptote", "value", allow_optional=True),
    Input("flow-edit-markout-tau", "value", allow_optional=True),
    State("global-target-pair", "value"),
    prevent_initial_call=True,
)
def preview_flow_markout(enabled, asymptote, tau, pair):
    return flow_markout_figure(checked(enabled), asymptote if asymptote is not None else 0.,
                               tau if tau is not None else 1., pair or "target")


@app.callback(
    *[Output(f"t{i}-{suffix}", "value") for i in range(len(DEFAULT["tiers"]))
      for suffix in ("markout-enabled", "markout-asymptote", "markout-tau-seconds")],
    Input("flow-edit-markout-enabled", "value", allow_optional=True),
    Input("flow-edit-markout-asymptote", "value", allow_optional=True),
    Input("flow-edit-markout-tau", "value", allow_optional=True),
    State("flow-source-tier-store", "data"),
    *[State(f"t{i}-{suffix}", "value") for i in range(len(DEFAULT["tiers"]))
      for suffix in ("markout-enabled", "markout-asymptote", "markout-tau-seconds")],
    prevent_initial_call=True,
)
def sync_markout_from_flow_modal(enabled, asymptote, tau, tier_index, *existing):
    if tier_index is None or any(v is None for v in (asymptote, tau)):
        return (no_update,) * (3 * len(DEFAULT["tiers"]))
    # Modal input values are not a new configuration when an editor is mounted.
    if not ctx.triggered or ctx.triggered_id not in {
        "flow-edit-markout-enabled", "flow-edit-markout-asymptote", "flow-edit-markout-tau"
    }:
        return (no_update,) * (3 * len(DEFAULT["tiers"]))
    values = list(existing)
    offset = 3 * int(tier_index)
    values[offset:offset + 3] = [enabled, asymptote, tau]
    return tuple(values)


@app.callback(
    *[Output(component_id, "options") for component_id in POLICY_REGIME_SELECTOR_IDS],
    *[Output(component_id, "value") for component_id in POLICY_REGIME_SELECTOR_IDS],
    Input("solution-store", "data"),
)
def regime_dropdown_options(solution):
    count = len(POLICY_REGIME_SELECTOR_IDS)
    if not solution or not solution.get("regimes"):
        return (*([[]] * count), *([None] * count))
    sigmas = [regime["sigmaPips"] for regime in solution["regimes"]]
    names=("Low", "High") if len(sigmas)==2 else ("Low", "Normal", "High")
    opts = [{"label": f"{name} · σ={float(sigma):g} pips/√min", "value": i}
            for i, (name, sigma) in enumerate(zip(names, sigmas))]
    normal = len(opts) // 2
    return (*([opts] * count), *([normal] * count))


def select_policy_regime(solution, index):
    """Expose a selected stochastic-volatility policy in legacy plot schema."""
    if not solution or not solution.get("regimes"):
        return solution
    regimes = solution["regimes"]
    i = len(regimes) // 2 if index is None else int(index)
    if not 0 <= i < len(regimes):
        i = len(regimes) // 2
    selected = dict(solution)
    selected.update({key: regimes[i][key]
                     for key in ("tiers", "passiveEcn", "darkPool", "expectedMtm")})
    selected["value"] = solution["valueByRegime"][i]
    selected["selectedSigmaPips"] = regimes[i]["sigmaPips"]
    return selected


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
