from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Toy ECN Optimizer", layout="wide")


# ============================================================
# Model helpers
# ============================================================

def fill_intensity(delta: np.ndarray, lam_max: float, lam_mid: float, lam_steepness: float) -> np.ndarray:
    """Logistic fill curve.

    Smaller delta = quote closer to mid = more aggressive = higher fill rate.
    """
    return lam_max / (1.0 + np.exp(lam_steepness * (delta - lam_mid)))


def markout_cost(delta: np.ndarray, m_base: float, m_amp: float, m_decay: float) -> np.ndarray:
    """Expected adverse-selection cost conditional on fill.

    As delta decreases (quote gets closer to mid), markout worsens.
    """
    return m_base + m_amp * np.exp(-m_decay * delta)


def h_value(q: np.ndarray | float, gamma: float) -> np.ndarray | float:
    """Toy continuation value with quadratic inventory penalty."""
    return -0.5 * gamma * np.square(q)


def inventory_relief(q: float, z: float, gamma: float) -> float:
    """Inventory-relief term for one-sided ECN use.

    q > 0: long inventory -> use ask side -> sell z
    q < 0: short inventory -> use bid side -> buy z
    q = 0: no ECN use
    """
    if q > 0.0:
        return float(h_value(q - z, gamma) - h_value(q, gamma))
    if q < 0.0:
        return float(h_value(q + z, gamma) - h_value(q, gamma))
    return 0.0


def solve_best_quote(
    q: float,
    delta_grid: np.ndarray,
    z: float,
    gamma: float,
    fee: float,
    lam_max: float,
    lam_mid: float,
    lam_steepness: float,
    m_base: float,
    m_amp: float,
    m_decay: float,
) -> tuple[float, float, np.ndarray]:
    """Return (best_delta, best_value, objective_on_grid).

    If ECN is optimally off, best_delta is np.nan and best_value is 0.
    """
    if q == 0.0:
        return np.nan, 0.0, np.zeros_like(delta_grid)

    lam = fill_intensity(delta_grid, lam_max, lam_mid, lam_steepness)
    m = markout_cost(delta_grid, m_base, m_amp, m_decay)
    relief = inventory_relief(q, z, gamma)

    per_fill_value = delta_grid - m - fee + relief
    objective = lam * per_fill_value

    idx = int(np.argmax(objective))
    best_value = float(objective[idx])
    if best_value <= 0.0:
        return np.nan, 0.0, objective
    return float(delta_grid[idx]), best_value, objective


def solve_policy_over_inventory(
    q_grid: np.ndarray,
    delta_grid: np.ndarray,
    z: float,
    gamma: float,
    fee: float,
    lam_max: float,
    lam_mid: float,
    lam_steepness: float,
    m_base: float,
    m_amp: float,
    m_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float] | None]:
    ask_curve = np.full_like(q_grid, np.nan, dtype=float)
    bid_curve = np.full_like(q_grid, np.nan, dtype=float)
    value_curve = np.zeros_like(q_grid, dtype=float)

    for i, q in enumerate(q_grid):
        d_star, v_star, _ = solve_best_quote(
            float(q),
            delta_grid,
            z,
            gamma,
            fee,
            lam_max,
            lam_mid,
            lam_steepness,
            m_base,
            m_amp,
            m_decay,
        )
        value_curve[i] = v_star
        if np.isfinite(d_star):
            if q > 0.0:
                ask_curve[i] = d_star
            elif q < 0.0:
                bid_curve[i] = -d_star

    active_pos = q_grid[(q_grid > 0.0) & (value_curve > 0.0)]
    active_neg = q_grid[(q_grid < 0.0) & (value_curve > 0.0)]

    if active_pos.size == 0 or active_neg.size == 0:
        band = None
    else:
        band = (float(active_neg.max()), float(active_pos.min()))

    return ask_curve, bid_curve, value_curve, band


# ============================================================
# Sidebar controls
# ============================================================

st.title("Toy ECN optimizer")
st.write(
    "This app solves a one-sided toy ECN quoting problem. ECN can be optimally off near flat inventory, "
    "and the optimal quote is chosen by balancing execution value, markout, fees, and inventory relief."
)

with st.sidebar:
    st.header("Inventory model")
    gamma = st.slider("Inventory curvature γ", min_value=0.01, max_value=0.60, value=0.20, step=0.01)
    z = st.slider("Trade size z", min_value=0.25, max_value=3.00, value=1.00, step=0.05)
    fee = st.slider("Per-fill fee / friction", min_value=0.00, max_value=0.60, value=0.12, step=0.01)

    st.header("Fill curve λ(δ)")
    lam_max = st.slider("Max fill intensity", min_value=1.0, max_value=60.0, value=20.0, step=1.0)
    lam_mid = st.slider("Fill midpoint δ₀", min_value=0.01, max_value=0.50, value=0.18, step=0.01)
    lam_steepness = st.slider("Fill steepness k", min_value=1.0, max_value=80.0, value=20.0, step=1.0)

    st.header("Markout m(δ)")
    m_base = st.slider("Base markout", min_value=0.00, max_value=0.60, value=0.20, step=0.01)
    m_amp = st.slider("Markout amplitude", min_value=0.00, max_value=0.80, value=0.20, step=0.01)
    m_decay = st.slider("Markout decay", min_value=0.5, max_value=20.0, value=5.0, step=0.5)

    st.header("Grids")
    q_abs_max = st.slider("Max |inventory|", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
    q_step = st.slider("Inventory step", min_value=0.01, max_value=0.20, value=0.01, step=0.01)
    delta_min = st.slider("Min quote distance", min_value=0.001, max_value=0.15, value=0.01, step=0.001, format="%.3f")
    delta_max = st.slider("Max quote distance", min_value=0.05, max_value=1.00, value=0.35, step=0.01)
    delta_points = st.slider("Quote grid points", min_value=100, max_value=2000, value=500, step=50)

    st.header("Objective inspection")
    q_probe = st.slider("Inventory for objective plot", min_value=-float(q_abs_max), max_value=float(q_abs_max), value=2.0, step=float(q_step))

if delta_max <= delta_min:
    st.error("delta_max must be larger than delta_min.")
    st.stop()

# Avoid exact zero duplication noise from np.arange around zero.
q_grid = np.arange(-q_abs_max, q_abs_max + 0.5 * q_step, q_step, dtype=float)
q_grid[np.isclose(q_grid, 0.0, atol=1e-12)] = 0.0

delta_grid = np.linspace(delta_min, delta_max, int(delta_points))

ask_curve, bid_curve, value_curve, band = solve_policy_over_inventory(
    q_grid,
    delta_grid,
    z,
    gamma,
    fee,
    lam_max,
    lam_mid,
    lam_steepness,
    m_base,
    m_amp,
    m_decay,
)

probe_delta, probe_value, probe_objective = solve_best_quote(
    float(q_probe),
    delta_grid,
    z,
    gamma,
    fee,
    lam_max,
    lam_mid,
    lam_steepness,
    m_base,
    m_amp,
    m_decay,
)

lam_curve = fill_intensity(delta_grid, lam_max, lam_mid, lam_steepness)
markout_curve = markout_cost(delta_grid, m_base, m_amp, m_decay)

# ============================================================
# Summary metrics
# ============================================================

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Probe best δ", "off" if not np.isfinite(probe_delta) else f"{probe_delta:.4f}")
with metric_cols[1]:
    st.metric("Probe ECN value", f"{probe_value:.4f}")
with metric_cols[2]:
    if band is None:
        st.metric("No-trade band", "no active ECN")
    else:
        st.metric("No-trade band", f"[{band[0]:.2f}, {band[1]:.2f}]")
with metric_cols[3]:
    active_frac = float(np.mean(value_curve > 0.0))
    st.metric("Active inventory share", f"{100.0 * active_frac:.1f}%")

# ============================================================
# Main plots
# ============================================================

tab_policy, tab_primitives, tab_probe = st.tabs(["Policy", "Primitives", "Objective at selected inventory"])

with tab_policy:
    fig_policy = go.Figure()
    fig_policy.add_trace(go.Scatter(x=q_grid, y=ask_curve, mode="lines", name="Optimized ask quote"))
    fig_policy.add_trace(go.Scatter(x=q_grid, y=bid_curve, mode="lines", name="Optimized bid quote"))
    fig_policy.add_hline(y=0.0)
    if band is not None:
        fig_policy.add_vline(x=band[0], line_dash="dash")
        fig_policy.add_vline(x=band[1], line_dash="dash")
    fig_policy.update_layout(
        title="Toy optimized ECN quote curve",
        xaxis_title="Inventory q",
        yaxis_title="Optimal quoted distance from mid (+ask, -bid)",
        height=450,
    )
    st.plotly_chart(fig_policy, use_container_width=True)

    fig_value = go.Figure()
    fig_value.add_trace(go.Scatter(x=q_grid, y=value_curve, mode="lines", name="Optimized ECN value"))
    fig_value.add_hline(y=0.0)
    if band is not None:
        fig_value.add_vline(x=band[0], line_dash="dash")
        fig_value.add_vline(x=band[1], line_dash="dash")
    fig_value.update_layout(
        title="Optimized ECN contribution vs inventory",
        xaxis_title="Inventory q",
        yaxis_title="Optimized ECN contribution",
        height=450,
        showlegend=False,
    )
    st.plotly_chart(fig_value, use_container_width=True)

with tab_primitives:
    col1, col2 = st.columns(2)

    with col1:
        fig_fill = go.Figure()
        fig_fill.add_trace(go.Scatter(x=delta_grid, y=lam_curve, mode="lines", name="λ(δ)"))
        fig_fill.update_layout(
            title="Fill intensity",
            xaxis_title="Quote distance δ",
            yaxis_title="λ(δ)",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_fill, use_container_width=True)

    with col2:
        fig_markout = go.Figure()
        fig_markout.add_trace(go.Scatter(x=delta_grid, y=markout_curve, mode="lines", name="m(δ)"))
        fig_markout.update_layout(
            title="Conditional markout cost",
            xaxis_title="Quote distance δ",
            yaxis_title="m(δ)",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_markout, use_container_width=True)

    relief = np.array([inventory_relief(float(q), z, gamma) for q in q_grid])
    fig_relief = go.Figure()
    fig_relief.add_trace(go.Scatter(x=q_grid, y=relief, mode="lines", name="Inventory relief"))
    fig_relief.add_hline(y=0.0)
    fig_relief.update_layout(
        title="Inventory-relief term h(q') - h(q)",
        xaxis_title="Inventory q",
        yaxis_title="Inventory relief",
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_relief, use_container_width=True)

with tab_probe:
    probe_lam = fill_intensity(delta_grid, lam_max, lam_mid, lam_steepness)
    probe_m = markout_cost(delta_grid, m_base, m_amp, m_decay)
    probe_relief = inventory_relief(float(q_probe), z, gamma)
    per_fill = delta_grid - probe_m - fee + probe_relief

    col1, col2 = st.columns(2)

    with col1:
        fig_obj = go.Figure()
        fig_obj.add_trace(go.Scatter(x=delta_grid, y=probe_objective, mode="lines", name="Objective"))
        fig_obj.add_hline(y=0.0)
        if np.isfinite(probe_delta):
            fig_obj.add_vline(x=probe_delta, line_dash="dash")
        fig_obj.update_layout(
            title=f"Objective over δ at q = {q_probe:.2f}",
            xaxis_title="Quote distance δ",
            yaxis_title="λ(δ) · [δ - m(δ) - fee + inventory relief]",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_obj, use_container_width=True)

    with col2:
        fig_parts = go.Figure()
        fig_parts.add_trace(go.Scatter(x=delta_grid, y=probe_lam, mode="lines", name="Fill intensity λ(δ)"))
        fig_parts.add_trace(go.Scatter(x=delta_grid, y=per_fill, mode="lines", name="Per-fill value"))
        if np.isfinite(probe_delta):
            fig_parts.add_vline(x=probe_delta, line_dash="dash")
        fig_parts.update_layout(
            title=f"Objective ingredients at q = {q_probe:.2f}",
            xaxis_title="Quote distance δ",
            yaxis_title="Value",
            height=420,
        )
        st.plotly_chart(fig_parts, use_container_width=True)

    st.write(
        f"At q = {q_probe:.2f}, inventory relief is **{probe_relief:.4f}**. "
        "If the best objective stays below zero, the toy ECN policy is off."
    )

st.markdown("---")
st.caption(
    "Toy objective: maximize λ(δ) · [δ - m(δ) - fee + h(q') - h(q)] with one-sided ECN use. "
    "This is a reduced-form example for experimentation, not a production model."
)
