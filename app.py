from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import ladder_pricer as lp
except ImportError as exc:
    raise ImportError(
        "Could not import 'ladder_pricer'. Make sure the package is built and installed "
        "into the same Python environment that runs Streamlit."
    ) from exc


# ============================================================
# Python-side UI spec
# ============================================================

@dataclass
class TierSpec:
    name: str
    sizes: list[float]
    flow_A0: float
    flow_theta: float
    flow_k: float
    flow_m0: float
    flow_m_alpha: float
    jump_base: float
    jump_coeff: float
    delta_min: float
    delta_max: float

    def build_cpp_tier(self) -> lp.PriceTier:
        flow = lp.LogisticFlowCurve(
            A0=float(self.flow_A0),
            theta=float(self.flow_theta),
            k=float(self.flow_k),
            m0=float(self.flow_m0),
            m_alpha=float(self.flow_m_alpha),
        )
        jump = lp.SqrtDriftJumpModel(
            base=float(self.jump_base),
            coeff=float(self.jump_coeff),
        )
        return lp.PriceTier(
            name=self.name,
            sizes=[float(z) for z in self.sizes],
            flow_curve=flow,
            jump_model=jump,
            delta_min=float(self.delta_min),
            delta_max=float(self.delta_max),
        )


# ============================================================
# Interop helpers for flat C++ cubes / enum side
# ============================================================

def _cube_shape(cube, ndim: int) -> tuple[int, ...]:
    if hasattr(cube, "shape"):
        shape = tuple(int(x) for x in cube.shape)
        if len(shape) == ndim:
            return shape

    if ndim == 3 and all(hasattr(cube, x) for x in ("nq", "ny", "nnu")):
        return (int(cube.nq), int(cube.ny), int(cube.nnu))

    if ndim == 4 and all(hasattr(cube, x) for x in ("nq", "ny", "nnu", "nz")):
        return (int(cube.nq), int(cube.ny), int(cube.nnu), int(cube.nz))

    raise TypeError(
        f"Could not infer shape for {type(cube).__name__}. "
        "Expose either .shape or the nq/ny/nnu(/nz) fields in pybind."
    )


def flatcube3d_to_numpy(cube) -> np.ndarray:
    if isinstance(cube, np.ndarray):
        arr = np.asarray(cube, dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got ndim={arr.ndim}.")
        return arr

    if hasattr(cube, "numpy") and callable(cube.numpy):
        arr = np.asarray(cube.numpy(), dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got ndim={arr.ndim}.")
        return arr

    if hasattr(cube, "data"):
        data = np.asarray(cube.data, dtype=float)
        shape = _cube_shape(cube, ndim=3)
        return data.reshape(shape)

    raise TypeError(
        f"Unsupported FlatCube3D conversion for object of type {type(cube).__name__}."
    )


def flatcube4d_to_numpy(cube) -> np.ndarray:
    if isinstance(cube, np.ndarray):
        arr = np.asarray(cube, dtype=float)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D array, got ndim={arr.ndim}.")
        return arr

    if hasattr(cube, "numpy") and callable(cube.numpy):
        arr = np.asarray(cube.numpy(), dtype=float)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D array, got ndim={arr.ndim}.")
        return arr

    if hasattr(cube, "data"):
        data = np.asarray(cube.data, dtype=float)
        shape = _cube_shape(cube, ndim=4)
        return data.reshape(shape)

    raise TypeError(
        f"Unsupported FlatCube4D conversion for object of type {type(cube).__name__}."
    )


def side_value(side_name: str):
    side_name = side_name.lower()
    if side_name not in {"bid", "ask"}:
        raise ValueError(f"Unknown side: {side_name}")

    if hasattr(lp, "Side"):
        return lp.Side.Bid if side_name == "bid" else lp.Side.Ask

    if hasattr(lp, "parse_side") and callable(lp.parse_side):
        return lp.parse_side(side_name)

    return side_name


LP_BID = side_value("bid")
LP_ASK = side_value("ask")


# ============================================================
# UI helpers
# ============================================================

def parse_float_list(raw: str, field_name: str) -> List[float]:
    try:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"Could not parse {field_name}. Use comma-separated numbers.") from exc

    if not vals:
        raise ValueError(f"{field_name} cannot be empty.")

    return vals


def default_tier_values(i: int) -> dict:
    presets = [
        {
            "name": "core_clients",
            "sizes": "1, 2, 5, 10",
            "flow_A0": 1.00,
            "flow_theta": 0.00,
            "flow_k": 2.00,
            "flow_m0": 0.15,
            "flow_m_alpha": 0.075,
            "jump_base": 0.000,
            "jump_coeff": 0.020,
            "delta_min": -0.5,
            "delta_max": 100.0,
        },
        {
            "name": "aggressive_clients",
            "sizes": "1, 2, 5, 10",
            "flow_A0": 1.30,
            "flow_theta": 0.20,
            "flow_k": 2.40,
            "flow_m0": 0.05,
            "flow_m_alpha": 0.070,
            "jump_base": 0.000,
            "jump_coeff": 0.025,
            "delta_min": -0.5,
            "delta_max": 100.0,
        },
        {
            "name": "sticky_clients",
            "sizes": "1, 2, 5, 10",
            "flow_A0": 0.85,
            "flow_theta": 0.15,
            "flow_k": 1.60,
            "flow_m0": 0.28,
            "flow_m_alpha": 0.090,
            "jump_base": 0.000,
            "jump_coeff": 0.015,
            "delta_min": -0.5,
            "delta_max": 100.0,
        },
    ]
    return presets[min(i, len(presets) - 1)]


def build_tier_spec_from_ui(i: int) -> TierSpec:
    defaults = default_tier_values(i)

    with st.sidebar.expander(f"Tier {i + 1}", expanded=(i == 0)):
        name = st.text_input(f"Tier name {i + 1}", value=defaults["name"], key=f"name_{i}")
        sizes_raw = st.text_input(f"Sizes {i + 1}", value=defaults["sizes"], key=f"sizes_{i}")

        st.markdown("**Flow curve parameters**")
        c1, c2 = st.columns(2)
        flow_A0 = c1.number_input(
            f"A0 {i + 1}",
            value=float(defaults["flow_A0"]),
            step=0.05,
            format="%.4f",
            key=f"flow_A0_{i}",
        )
        flow_theta = c2.number_input(
            f"theta {i + 1}",
            value=float(defaults["flow_theta"]),
            step=0.05,
            format="%.4f",
            key=f"flow_theta_{i}",
        )

        c3, c4 = st.columns(2)
        flow_k = c3.number_input(
            f"k {i + 1}",
            value=float(defaults["flow_k"]),
            step=0.05,
            format="%.4f",
            key=f"flow_k_{i}",
        )
        flow_m0 = c4.number_input(
            f"m0 {i + 1}",
            value=float(defaults["flow_m0"]),
            step=0.05,
            format="%.4f",
            key=f"flow_m0_{i}",
        )

        flow_m_alpha = st.number_input(
            f"m_alpha {i + 1}",
            value=float(defaults["flow_m_alpha"]),
            step=0.01,
            format="%.4f",
            key=f"flow_m_alpha_{i}",
        )

        st.markdown("**Drift-jump parameters**")
        c5, c6 = st.columns(2)
        jump_base = c5.number_input(
            f"Jump base {i + 1}",
            value=float(defaults["jump_base"]),
            step=0.001,
            format="%.5f",
            key=f"jump_base_{i}",
        )
        jump_coeff = c6.number_input(
            f"Jump coeff {i + 1}",
            value=float(defaults["jump_coeff"]),
            step=0.001,
            format="%.5f",
            key=f"jump_coeff_{i}",
        )

        st.markdown("**Tier quote settings**")
        c7, c8 = st.columns(2)
        tier_delta_min = c7.number_input(
            f"Tier delta_min {i + 1}",
            value=float(defaults["delta_min"]),
            step=0.1,
            format="%.4f",
            key=f"tier_delta_min_{i}",
        )
        tier_delta_max = c8.number_input(
            f"Tier delta_max {i + 1}",
            value=float(defaults["delta_max"]),
            step=0.1,
            format="%.4f",
            key=f"tier_delta_max_{i}",
        )

    sizes = parse_float_list(sizes_raw, f"sizes for tier {i + 1}")

    if flow_A0 < 0.0:
        raise ValueError(f"A0 for tier {i + 1} must be nonnegative.")
    if flow_k <= 0.0:
        raise ValueError(f"k for tier {i + 1} must be positive.")
    if tier_delta_max <= tier_delta_min:
        raise ValueError(f"Tier {i + 1}: delta_max must be greater than delta_min.")
    if any(z <= 0.0 for z in sizes):
        raise ValueError(f"Tier {i + 1}: all sizes must be positive.")
    if any(sizes[k] >= sizes[k + 1] for k in range(len(sizes) - 1)):
        raise ValueError(f"Tier {i + 1}: sizes must be strictly increasing.")

    return TierSpec(
        name=name,
        sizes=sizes,
        flow_A0=float(flow_A0),
        flow_theta=float(flow_theta),
        flow_k=float(flow_k),
        flow_m0=float(flow_m0),
        flow_m_alpha=float(flow_m_alpha),
        jump_base=float(jump_base),
        jump_coeff=float(jump_coeff),
        delta_min=float(tier_delta_min),
        delta_max=float(tier_delta_max),
    )


def build_centered_q_grid(q_min: float, q_max: float, q_step: float) -> np.ndarray:
    if q_step <= 0:
        raise ValueError("q_step must be positive.")

    half_width = max(abs(float(q_min)), abs(float(q_max)))
    half_steps = max(1, int(np.ceil(half_width / float(q_step))))
    grid = float(q_step) * np.arange(-half_steps, half_steps + 1, dtype=float)
    grid[half_steps] = 0.0
    return grid


def build_centered_linear_grid(
    x_min: float,
    x_max: float,
    n_points: int,
    center: float,
    field_name: str,
) -> np.ndarray:
    if n_points < 3:
        raise ValueError(f"{field_name} must have at least 3 points.")
    if n_points % 2 == 0:
        raise ValueError(f"{field_name} must have odd length.")

    half_width = max(abs(float(x_min) - float(center)), abs(float(x_max) - float(center)))
    if half_width <= 0.0:
        raise ValueError(f"{field_name} range must have positive width.")

    grid = np.linspace(
        float(center) - half_width,
        float(center) + half_width,
        int(n_points),
        dtype=float,
    )
    grid[int(n_points) // 2] = float(center)
    return np.round(grid, 12)


def build_centered_nu_grid(
    sigma_state_min: float,
    sigma_state_max: float,
    n_points: int,
    sigma_bar_state: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if sigma_state_min <= 0.0 or sigma_state_max <= 0.0 or sigma_bar_state <= 0.0:
        raise ValueError("Volatility state values must be strictly positive.")
    if n_points < 3:
        raise ValueError("Volatility grid must have at least 3 points.")
    if n_points % 2 == 0:
        raise ValueError("Volatility grid must have odd length.")

    nu_bar = float(np.log(float(sigma_bar_state)))
    nu_min_requested = float(np.log(float(sigma_state_min)))
    nu_max_requested = float(np.log(float(sigma_state_max)))

    half_width = max(abs(nu_min_requested - nu_bar), abs(nu_max_requested - nu_bar))
    if half_width <= 0.0:
        raise ValueError("Volatility state range must have positive width.")

    nu_grid = np.linspace(nu_bar - half_width, nu_bar + half_width, int(n_points), dtype=float)
    nu_grid[int(n_points) // 2] = nu_bar
    nu_grid = np.round(nu_grid, 12)

    sigma_state_grid = np.exp(nu_grid)
    return nu_grid, sigma_state_grid, nu_bar


def nearest_index(arr: np.ndarray, x: float) -> int:
    return int(np.argmin(np.abs(arr - x)))


def policy_arrays(
    cpp_tier: lp.PriceTier,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_grid = np.array(cpp_tier.policy.q_grid, dtype=float)
    y_grid = np.array(cpp_tier.policy.y_grid, dtype=float)
    nu_grid = np.array(cpp_tier.policy.nu_grid, dtype=float)
    bid = flatcube4d_to_numpy(cpp_tier.policy.bid)
    ask = flatcube4d_to_numpy(cpp_tier.policy.ask)
    return q_grid, y_grid, nu_grid, bid, ask


def sigma_state_grid_from_solution(solution: lp.HJBSolution) -> np.ndarray:
    nu_grid = np.array(solution.nu_grid, dtype=float)
    return np.exp(nu_grid)


def make_parameter_table(spec: TierSpec, cpp_tier: lp.PriceTier) -> pd.DataFrame:
    rows = []
    for z in spec.sizes:
        zf = float(z)
        rows.append(
            {
                "z": zf,
                "A(z)": spec.flow_A0 * zf ** (-spec.flow_theta),
                "m(z)": spec.flow_m0 + spec.flow_m_alpha * zf,
                "k": spec.flow_k,
                "jump_alpha(z)": cpp_tier.jump_size(zf),
            }
        )
    return pd.DataFrame(rows)


def make_h_slice_figure(solution: lp.HJBSolution, y_selected: float, sigma_state_selected: float) -> go.Figure:
    q_grid = np.array(solution.q_grid, dtype=float)
    y_grid = np.array(solution.y_grid, dtype=float)
    nu_grid = np.array(solution.nu_grid, dtype=float)
    sigma_state_grid = np.exp(nu_grid)
    h = flatcube3d_to_numpy(solution.h)

    iy = nearest_index(y_grid, y_selected)
    inu = nearest_index(sigma_state_grid, sigma_state_selected)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=q_grid,
            y=h[:, iy, inu],
            mode="lines+markers",
            name=f"h(q, y={y_grid[iy]:g}, σ={sigma_state_grid[inu]:g})",
        )
    )
    fig.update_layout(
        title=f"Value function slice at y = {y_grid[iy]:g}, σ = {sigma_state_grid[inu]:g}",
        xaxis_title="Inventory q",
        yaxis_title="h(q,y,ν)",
        height=420,
    )
    return fig


def make_flow_curve_figure(cpp_tier: lp.PriceTier, spec: TierSpec) -> go.Figure:
    grid = np.linspace(spec.delta_min, spec.delta_max, 200)
    fig = go.Figure()

    for z in spec.sizes:
        zf = float(z)
        vals = [cpp_tier.arrival_rate(float(d), zf) for d in grid]
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=vals,
                mode="lines",
                name=f"{zf:g}",
            )
        )

    fig.update_layout(
        title=f"Flow curves λ(δ, z) — {spec.name}",
        xaxis_title="delta",
        yaxis_title="arrival rate",
        height=420,
    )
    return fig


def make_quote_inventory_figure(
    cpp_tier: lp.PriceTier,
    spec: TierSpec,
    y_selected: float,
    sigma_state_selected: float,
) -> go.Figure:
    q_grid, y_grid, nu_grid, bid, ask = policy_arrays(cpp_tier)
    sigma_state_grid = np.exp(nu_grid)

    iy = nearest_index(y_grid, y_selected)
    inu = nearest_index(sigma_state_grid, sigma_state_selected)

    fig = go.Figure()
    for j, z in enumerate(spec.sizes):
        zf = float(z)
        fig.add_trace(
            go.Scatter(
                x=q_grid,
                y=-bid[:, iy, inu, j],
                mode="lines",
                name=f"{zf:g} bid",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=q_grid,
                y=ask[:, iy, inu, j],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{zf:g} ask",
            )
        )

    fig.add_hline(y=0.0)
    fig.update_layout(
        title=f"Quotes vs inventory at y = {y_grid[iy]:g}, σ = {sigma_state_grid[inu]:g} — {spec.name}",
        xaxis_title="Inventory q",
        yaxis_title="Quote around mid",
        yaxis=dict(range=[-10, 10]),
        height=520,
    )
    return fig


def make_quote_volatility_figure(
    cpp_tier: lp.PriceTier,
    spec: TierSpec,
    q_selected: float,
    y_selected: float,
) -> go.Figure:
    q_grid, y_grid, nu_grid, bid, ask = policy_arrays(cpp_tier)
    sigma_state_grid = np.exp(nu_grid)

    iq = nearest_index(q_grid, q_selected)
    iy = nearest_index(y_grid, y_selected)

    fig = go.Figure()
    for j, z in enumerate(spec.sizes):
        zf = float(z)
        fig.add_trace(
            go.Scatter(
                x=sigma_state_grid,
                y=-bid[iq, iy, :, j],
                mode="lines",
                name=f"{zf:g} bid",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sigma_state_grid,
                y=ask[iq, iy, :, j],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{zf:g} ask",
            )
        )

    fig.add_hline(y=0.0)
    fig.update_layout(
        title=f"Quotes vs volatility at q = {q_grid[iq]:g}, y = {y_grid[iy]:g} — {spec.name}",
        xaxis_title="Volatility state σ = exp(ν)",
        yaxis_title="Quote around mid",
        yaxis=dict(range=[-10, 10]),
        height=520,
    )
    return fig


def make_ask_policy_heatmap(
    cpp_tier: lp.PriceTier,
    spec: TierSpec,
    z_selected: float,
    sigma_state_selected: float,
) -> go.Figure:
    q_grid, y_grid, nu_grid, _, ask = policy_arrays(cpp_tier)
    sigma_state_grid = np.exp(nu_grid)
    z_arr = np.array(spec.sizes, dtype=float)

    iz = nearest_index(z_arr, z_selected)
    inu = nearest_index(sigma_state_grid, sigma_state_selected)

    values = ask[:, :, inu, iz]

    fig = go.Figure(
        data=go.Heatmap(
            x=y_grid,
            y=q_grid,
            z=values,
            colorbar_title="ask quote",
        )
    )
    fig.update_layout(
        title=f"Ask policy heatmap for z = {z_arr[iz]:g}, σ = {sigma_state_grid[inu]:g} — {spec.name}",
        xaxis_title="Drift state y",
        yaxis_title="Inventory q",
        height=520,
    )
    return fig


def make_ladder_figure(
    cpp_tier: lp.PriceTier,
    spec: TierSpec,
    q: float,
    y: float,
    sigma_state: float,
) -> go.Figure:
    nu = float(np.log(sigma_state))
    sizes = [float(z) for z in spec.sizes]
    bid_vals = [-cpp_tier.quote(float(q), float(y), nu, z, LP_BID) for z in sizes]
    ask_vals = [cpp_tier.quote(float(q), float(y), nu, z, LP_ASK) for z in sizes]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=bid_vals,
            mode="lines+markers",
            name=f"Bid q={q:g}, y={y:g}, σ={sigma_state:g}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=ask_vals,
            mode="lines+markers",
            line=dict(dash="dash"),
            name=f"Ask q={q:g}, y={y:g}, σ={sigma_state:g}",
        )
    )

    fig.add_hline(y=0.0)
    fig.update_layout(
        title=f"Ladders — {spec.name} at q = {q:g}, y = {y:g}, σ = {sigma_state:g}",
        xaxis_title="Trade size z",
        yaxis_title="Quote around mid",
        height=420,
    )
    return fig


def make_state_ladder_table(
    cpp_tier: lp.PriceTier,
    spec: TierSpec,
    q: float,
    y: float,
    sigma_state: float,
) -> pd.DataFrame:
    nu = float(np.log(sigma_state))
    rows = []
    for z in spec.sizes:
        zf = float(z)
        bid_delta = cpp_tier.quote(float(q), float(y), nu, zf, LP_BID)
        ask_delta = cpp_tier.quote(float(q), float(y), nu, zf, LP_ASK)
        rows.append(
            {
                "q": float(q),
                "y": float(y),
                "sigma_state": float(sigma_state),
                "nu": nu,
                "z": zf,
                "bid_quote_plot_value": -bid_delta,
                "ask_quote_plot_value": ask_delta,
                "raw_bid_delta": bid_delta,
                "raw_ask_delta": ask_delta,
                "jump_alpha(z)": cpp_tier.jump_size(zf),
            }
        )
    return pd.DataFrame(rows)


def make_convergence_figure(solution: lp.HJBSolution) -> go.Figure:
    fig = go.Figure()

    hist_h = list(solution.history_max_h_change)
    hist_rhs = list(solution.history_max_rhs)

    if hist_h:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(hist_h) + 1)),
                y=hist_h,
                mode="lines+markers",
                name="max |Δh|",
            )
        )

    if hist_rhs:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(hist_rhs) + 1)),
                y=hist_rhs,
                mode="lines+markers",
                name="max |rhs|",
            )
        )

    fig.update_layout(
        title="Convergence diagnostics",
        xaxis_title="Iteration",
        yaxis_title="Value",
        yaxis_type="log",
        height=420,
    )
    return fig


def build_signature(
    q_grid: np.ndarray,
    y_grid: np.ndarray,
    nu_grid: np.ndarray,
    dt: float,
    n_iter: int,
    kappa_y: float,
    kappa_nu: float,
    nu_bar: float,
    eta_nu: float,
    golden_tol: float,
    golden_max_iter: int,
    early_stop: bool,
    tol_h: float,
    tol_rhs: float,
    min_iter: int,
    consecutive_passes_required: int,
    risk_aversion: float,
    sigma_ref: float,
    tau0: float,
    cubic_coeff: float,
    quartic_coeff: float,
    tier_specs: List[TierSpec],
) -> Tuple:
    return (
        tuple(float(x) for x in q_grid),
        tuple(float(x) for x in y_grid),
        tuple(float(x) for x in nu_grid),
        float(dt),
        int(n_iter),
        float(kappa_y),
        float(kappa_nu),
        float(nu_bar),
        float(eta_nu),
        float(golden_tol),
        int(golden_max_iter),
        bool(early_stop),
        float(tol_h),
        float(tol_rhs),
        int(min_iter),
        int(consecutive_passes_required),
        float(risk_aversion),
        float(sigma_ref),
        float(tau0),
        float(cubic_coeff),
        float(quartic_coeff),
        tuple(
            (
                spec.name,
                tuple(float(z) for z in spec.sizes),
                spec.flow_A0,
                spec.flow_theta,
                spec.flow_k,
                spec.flow_m0,
                spec.flow_m_alpha,
                spec.jump_base,
                spec.jump_coeff,
                spec.delta_min,
                spec.delta_max,
            )
            for spec in tier_specs
        ),
    )


# ============================================================
# App
# ============================================================

st.set_page_config(page_title="HJB ladder with drift and volatility state", layout="wide")
st.title("HJB ladder playground with drift and volatility state")
st.markdown(
    "This version uses the `ladder_pricer` C++ package via pybind11. "
    "The state now includes inventory **q**, drift pressure **y**, and log-volatility **ν** "
    "(shown in the UI as **σ = exp(ν)**)."
)

with st.sidebar:
    st.header("Inventory grid")
    st.caption("The solver grid is forced to be symmetric around 0 with 0 exactly in the middle.")
    q_min = st.number_input("Requested q min", value=-20, step=1)
    q_max = st.number_input("Requested q max", value=20, step=1)
    q_step = st.number_input("q step", value=1, step=1, min_value=1)

    st.header("Drift-state grid")
    st.caption("The solver grid is forced to be symmetric around 0 with 0 exactly in the middle.")
    y_range = st.slider(
        "Requested drift-state range",
        min_value=-10.0,
        max_value=10.0,
        value=(-5.0, 5.0),
        step=0.1,
    )
    n_y_points = st.slider(
        "Number of drift points",
        min_value=5,
        max_value=41,
        value=11,
        step=2,
    )
    kappa_y = st.number_input("kappa_y (drift decay speed)", value=2.0, step=0.1, format="%.4f")

    st.header("Volatility-state grid")
    st.caption(
        "The solver grid is forced to be symmetric in log-volatility around ν̄, "
        "so σ = exp(ν) will generally be asymmetric in level."
    )
    sigma_state_range = st.slider(
        "Requested volatility state range σ = exp(ν)",
        min_value=0.10,
        max_value=3.00,
        value=(0.50, 1.50),
        step=0.05,
    )
    n_sigma_points = st.slider(
        "Number of volatility points",
        min_value=5,
        max_value=31,
        value=9,
        step=2,
    )
    sigma_bar_state = st.number_input(
        "Long-run volatility state σ̄ = exp(ν̄)",
        value=1.00,
        min_value=0.05,
        step=0.05,
        format="%.4f",
    )
    kappa_nu = st.number_input("kappa_nu (log-vol mean reversion)", value=1.0, step=0.1, format="%.4f")
    eta_nu = st.number_input("eta_nu (log-vol vol-of-vol)", value=0.2, step=0.05, format="%.4f")

    st.header("HJB iteration")
    dt = st.number_input("dt", value=0.001, step=0.001, format="%.4f")
    n_iter = st.number_input("n_iter", value=200, step=10, min_value=1)

    st.header("Golden-section optimizer")
    golden_tol = st.number_input("golden_tol", value=1e-4, min_value=1e-12, format="%.1e")
    golden_max_iter = st.number_input("golden_max_iter", value=32, step=1, min_value=1)

    st.header("Convergence / stopping")
    early_stop = st.checkbox("Enable early stopping", value=True)
    tol_h = st.number_input("tol_h", value=1e-5, format="%.1e")
    tol_rhs = st.number_input("tol_rhs", value=1e-4, format="%.1e")
    min_iter = st.number_input("min_iter", value=5, step=1, min_value=0)
    consecutive_passes_required = st.number_input(
        "consecutive passes required",
        value=3,
        step=1,
        min_value=1,
    )

    st.header("Inventory penalty")
    sigma_ref = st.number_input(
        "sigma_ref (base penalty scale)",
        value=0.25,
        step=0.01,
        format="%.4f",
    )
    risk_aversion = st.number_input("risk_aversion", value=2.0, step=0.1, format="%.4f")
    tau0 = st.number_input("tau0", value=2.0, step=0.1, format="%.4f")
    cubic_coeff = st.number_input("cubic coeff", value=0.1, step=0.01, format="%.4f")
    quartic_coeff = st.number_input("quartic coeff", value=0.0015, step=0.0005, format="%.5f")

    st.header("Pricing tiers")
    num_tiers = st.slider("Number of tiers", min_value=1, max_value=4, value=2)

errors: List[str] = []
tier_specs: List[TierSpec] = []
for i in range(num_tiers):
    try:
        tier_specs.append(build_tier_spec_from_ui(i))
    except ValueError as exc:
        errors.append(str(exc))

if q_step <= 0:
    errors.append("q_step must be positive.")
if golden_tol <= 0.0:
    errors.append("golden_tol must be positive.")
if golden_max_iter < 1:
    errors.append("golden_max_iter must be at least 1.")

y_min, y_max = float(y_range[0]), float(y_range[1])
sigma_state_min, sigma_state_max = float(sigma_state_range[0]), float(sigma_state_range[1])

if sigma_state_max <= sigma_state_min:
    errors.append("Volatility state range must have upper bound greater than lower bound.")
if sigma_state_min <= 0.0:
    errors.append("Volatility state must be strictly positive.")
if sigma_bar_state <= 0.0:
    errors.append("Long-run volatility state must be strictly positive.")
if kappa_y < 0.0:
    errors.append("kappa_y must be nonnegative.")
if kappa_nu < 0.0:
    errors.append("kappa_nu must be nonnegative.")
if eta_nu < 0.0:
    errors.append("eta_nu must be nonnegative.")

try:
    q_grid = build_centered_q_grid(float(q_min), float(q_max), float(q_step))
except ValueError as exc:
    errors.append(str(exc))
    q_grid = np.array([], dtype=float)

try:
    y_grid = build_centered_linear_grid(
        float(y_min),
        float(y_max),
        int(n_y_points),
        center=0.0,
        field_name="y_grid",
    )
except ValueError as exc:
    errors.append(str(exc))
    y_grid = np.array([], dtype=float)

try:
    nu_grid, sigma_state_grid, nu_bar = build_centered_nu_grid(
        float(sigma_state_min),
        float(sigma_state_max),
        int(n_sigma_points),
        float(sigma_bar_state),
    )
except ValueError as exc:
    errors.append(str(exc))
    nu_grid = np.array([], dtype=float)
    sigma_state_grid = np.array([], dtype=float)
    nu_bar = float(np.log(max(float(sigma_bar_state), 1e-12)))

if len(q_grid) < 3:
    errors.append("q_grid must contain at least 3 points.")
if len(y_grid) < 3:
    errors.append("y_grid must contain at least 3 points.")
if len(nu_grid) < 3:
    errors.append("nu_grid must contain at least 3 points.")

if len(q_grid) > 0 and not np.isclose(q_grid[len(q_grid) // 2], 0.0):
    errors.append("Internal error: q_grid is not centered at 0.")
if len(y_grid) > 0 and not np.isclose(y_grid[len(y_grid) // 2], 0.0):
    errors.append("Internal error: y_grid is not centered at 0.")
if len(nu_grid) > 0 and not np.isclose(nu_grid[len(nu_grid) // 2], nu_bar):
    errors.append("Internal error: nu_grid is not centered at nu_bar.")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

signature = build_signature(
    q_grid=q_grid,
    y_grid=y_grid,
    nu_grid=nu_grid,
    dt=float(dt),
    n_iter=int(n_iter),
    kappa_y=float(kappa_y),
    kappa_nu=float(kappa_nu),
    nu_bar=float(nu_bar),
    eta_nu=float(eta_nu),
    golden_tol=float(golden_tol),
    golden_max_iter=int(golden_max_iter),
    early_stop=bool(early_stop),
    tol_h=float(tol_h),
    tol_rhs=float(tol_rhs),
    min_iter=int(min_iter),
    consecutive_passes_required=int(consecutive_passes_required),
    risk_aversion=float(risk_aversion),
    sigma_ref=float(sigma_ref),
    tau0=float(tau0),
    cubic_coeff=float(cubic_coeff),
    quartic_coeff=float(quartic_coeff),
    tier_specs=tier_specs,
)

with st.sidebar:
    st.button("Compute / refresh", type="primary", use_container_width=True, key="compute_button")

compute_clicked = st.session_state.get("compute_button", False)

st.caption(
    "Actual solver grids are centered by construction: "
    f"q ∈ [{q_grid[0]:g}, {q_grid[-1]:g}] with {len(q_grid)} points; "
    f"y ∈ [{y_grid[0]:g}, {y_grid[-1]:g}] with {len(y_grid)} points; "
    f"σ ∈ [{sigma_state_grid[0]:.4g}, {sigma_state_grid[-1]:.4g}] with {len(sigma_state_grid)} points, "
    f"centered at σ̄ = {float(np.exp(nu_bar)):.4g} in log-vol space."
)

state_count = len(q_grid) * len(y_grid) * len(nu_grid)
st.caption(
    f"State grid size: {len(q_grid)} q-points × {len(y_grid)} y-points × {len(nu_grid)} ν-points = {state_count} states"
)

need_compute = compute_clicked or "solution" not in st.session_state

if need_compute:
    cpp_tiers = [spec.build_cpp_tier() for spec in tier_specs]

    penalty = lp.PolynomialInventoryPenalty(
        risk_aversion=float(risk_aversion),
        sigma_ref=float(sigma_ref),
        tau0=float(tau0),
        cubic_coeff=float(cubic_coeff),
        quartic_coeff=float(quartic_coeff),
    )

    config = lp.SolverConfig()
    config.q_grid = [float(q) for q in q_grid]
    config.y_grid = [float(y) for y in y_grid]
    config.nu_grid = [float(nu) for nu in nu_grid]
    config.dt = float(dt)
    config.n_iter = int(n_iter)
    config.kappa_y = float(kappa_y)
    config.kappa_nu = float(kappa_nu)
    config.nu_bar = float(nu_bar)
    config.eta_nu = float(eta_nu)
    config.golden_tol = float(golden_tol)
    config.golden_max_iter = int(golden_max_iter)
    config.early_stop = bool(early_stop)
    config.tol_h = float(tol_h)
    config.tol_rhs = float(tol_rhs)
    config.min_iter = int(min_iter)
    config.consecutive_passes_required = int(consecutive_passes_required)

    with st.spinner("Solving HJB in C++ and building drift/vol-aware policies..."):
        solver = lp.HJBLadderSolver(config=config, penalty=penalty, tiers=cpp_tiers)
        solution: lp.HJBSolution = solver.solve()

    st.session_state["solution"] = solution
    st.session_state["tier_specs"] = tier_specs
    st.session_state["signature"] = signature

solution: lp.HJBSolution = st.session_state["solution"]
solved_tier_specs: List[TierSpec] = st.session_state["tier_specs"]
solved_signature = st.session_state["signature"]

if signature != solved_signature:
    st.info("Parameters have changed. Click **Compute / refresh** to update the displayed solution.")

st.success("Solver run complete.")

if solution.converged:
    st.info(
        f"Early stopping triggered after {solution.iterations_used} iterations. "
        f"Final max |Δh| = {solution.final_max_h_change:.2e}, "
        f"final max |rhs| = {solution.final_max_rhs:.2e}."
    )
else:
    st.warning(
        f"Solver reached the iteration cap ({solution.iterations_used}). "
        f"Final max |Δh| = {solution.final_max_h_change:.2e}, "
        f"final max |rhs| = {solution.final_max_rhs:.2e}."
    )

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("q points", len(solution.q_grid))
c2.metric("y points", len(solution.y_grid))
c3.metric("ν points", len(solution.nu_grid))
c4.metric("tiers", len(solution.tiers))
c5.metric("iterations used", solution.iterations_used)

c6, c7, c8 = st.columns(3)
c6.metric("converged", "yes" if solution.converged else "no")
c7.metric("final max |Δh|", f"{solution.final_max_h_change:.2e}")
c8.metric("final max |rhs|", f"{solution.final_max_rhs:.2e}")

available_y = [float(y) for y in solution.y_grid]
available_q = [float(q) for q in solution.q_grid]
available_sigma_state = [float(s) for s in sigma_state_grid_from_solution(solution)]

center_y_value = available_y[len(available_y) // 2]
center_q_value = available_q[len(available_q) // 2]
center_sigma_value = available_sigma_state[len(available_sigma_state) // 2]

c_top1, c_top2 = st.columns(2)
with c_top1:
    global_y_for_h = st.select_slider(
        "Drift level for value-function slice",
        options=available_y,
        value=center_y_value,
        key="global_y_slice",
    )
with c_top2:
    global_sigma_for_h = st.select_slider(
        "Volatility level for value-function slice",
        options=available_sigma_state,
        value=center_sigma_value,
        key="global_sigma_slice",
    )

st.plotly_chart(
    make_h_slice_figure(solution, float(global_y_for_h), float(global_sigma_for_h)),
    use_container_width=True,
)
st.plotly_chart(make_convergence_figure(solution), use_container_width=True)

tab_names = [spec.name for spec in solved_tier_specs]
tabs = st.tabs(tab_names)

for tab, spec, cpp_tier in zip(tabs, solved_tier_specs, solution.tiers):
    with tab:
        st.subheader(f"Tier: {spec.name}")

        st.markdown("**Implied parameters at ladder sizes**")
        st.dataframe(make_parameter_table(spec, cpp_tier), use_container_width=True)

        st.plotly_chart(
            make_flow_curve_figure(cpp_tier, spec),
            use_container_width=True,
        )

        c_q1, c_q2 = st.columns(2)
        with c_q1:
            y_for_inventory = st.select_slider(
                f"Drift level for quotes vs inventory — {spec.name}",
                options=available_y,
                value=center_y_value,
                key=f"y_for_inventory_{spec.name}",
            )
        with c_q2:
            sigma_for_inventory = st.select_slider(
                f"Volatility level for quotes vs inventory — {spec.name}",
                options=available_sigma_state,
                value=center_sigma_value,
                key=f"sigma_for_inventory_{spec.name}",
            )

        st.plotly_chart(
            make_quote_inventory_figure(
                cpp_tier,
                spec,
                float(y_for_inventory),
                float(sigma_for_inventory),
            ),
            use_container_width=True,
        )

        c_v1, c_v2 = st.columns(2)
        with c_v1:
            q_for_vol = st.selectbox(
                f"Inventory level for quotes vs volatility — {spec.name}",
                options=available_q,
                index=len(available_q) // 2,
                key=f"q_for_vol_{spec.name}",
            )
        with c_v2:
            y_for_vol = st.select_slider(
                f"Drift level for quotes vs volatility — {spec.name}",
                options=available_y,
                value=center_y_value,
                key=f"y_for_vol_{spec.name}",
            )

        st.plotly_chart(
            make_quote_volatility_figure(
                cpp_tier,
                spec,
                float(q_for_vol),
                float(y_for_vol),
            ),
            use_container_width=True,
        )

        c_h1, c_h2 = st.columns(2)
        with c_h1:
            z_for_heatmap = st.selectbox(
                f"Size for ask policy heatmap — {spec.name}",
                options=[float(z) for z in spec.sizes],
                index=0,
                key=f"z_heat_{spec.name}",
            )
        with c_h2:
            sigma_for_heatmap = st.select_slider(
                f"Volatility level for ask policy heatmap — {spec.name}",
                options=available_sigma_state,
                value=center_sigma_value,
                key=f"sigma_heat_{spec.name}",
            )

        st.plotly_chart(
            make_ask_policy_heatmap(
                cpp_tier,
                spec,
                float(z_for_heatmap),
                float(sigma_for_heatmap),
            ),
            use_container_width=True,
        )

        c_l1, c_l2, c_l3 = st.columns(3)
        with c_l1:
            q_for_ladder = st.selectbox(
                f"Inventory level for ladder — {spec.name}",
                options=available_q,
                index=len(available_q) // 2,
                key=f"q_ladder_{spec.name}",
            )
        with c_l2:
            y_for_ladder = st.select_slider(
                f"Drift level for ladder — {spec.name}",
                options=available_y,
                value=center_y_value,
                key=f"y_ladder_{spec.name}",
            )
        with c_l3:
            sigma_for_ladder = st.select_slider(
                f"Volatility level for ladder — {spec.name}",
                options=available_sigma_state,
                value=center_sigma_value,
                key=f"sigma_ladder_{spec.name}",
            )

        st.plotly_chart(
            make_ladder_figure(
                cpp_tier,
                spec,
                float(q_for_ladder),
                float(y_for_ladder),
                float(sigma_for_ladder),
            ),
            use_container_width=True,
        )

        st.dataframe(
            make_state_ladder_table(
                cpp_tier,
                spec,
                float(q_for_ladder),
                float(y_for_ladder),
                float(sigma_for_ladder),
            ),
            use_container_width=True,
        )

with st.expander("What this app is solving"):
    st.markdown(
        r"""
The state is now **three-dimensional**: inventory $q$, drift pressure $y$, and log-volatility $\nu$.

Spot evolves as

$$
dS_t = y_t\,dt + e^{\nu_t} dW_t,
$$

with drift-state decay

$$
dy_t = -\kappa_y y_t\,dt,
$$

and OU log-volatility

$$
d\nu_t = \kappa_\nu(\bar{\nu}-\nu_t)\,dt + \eta_\nu\,dB_t.
$$

A fill of size $z$ shifts the drift state by $\alpha(z)$:
- ask fill: $y \to y + \alpha(z)$
- bid fill: $y \to y - \alpha(z)$

Using the ansatz

$$
V(x,s,q,y,\nu)=x+qs+h(q,y,\nu),
$$

the stationary HJB is

$$
0
=
q\,y
-\exp(2\nu)\phi(q)
-\kappa_y y\,\partial_y h
+\kappa_\nu(\bar{\nu}-\nu)\partial_\nu h
+\frac12\eta_\nu^2 \partial_{\nu\nu} h
$$

$$
\qquad
+
\sum_z \sup_{\delta^b}
\lambda(\delta^b,z)
\Big[
z\delta^b + h(q+z,y-\alpha(z),\nu)-h(q,y,\nu)
\Big]
$$

$$
\qquad
+
\sum_z \sup_{\delta^a}
\lambda(\delta^a,z)
\Big[
z\delta^a + h(q-z,y+\alpha(z),\nu)-h(q,y,\nu)
\Big].
$$

The UI shows volatility as

$$
\sigma_{\text{state}} = e^\nu,
$$

so the volatility selectors are easier to interpret than raw log-vol values.
        """
    )

st.caption(
    "Use the Compute / refresh button to run the solver. "
    "The app now constructs centered odd grids automatically so they satisfy the solver invariants."
)