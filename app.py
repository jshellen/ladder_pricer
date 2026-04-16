from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
    markout_base: float
    markout_coeff: float
    delta_min: float
    delta_max: float
    golden_tol: float = 1e-4
    golden_max_iter: int = 32

    def build_cpp_tier(self) -> lp.PriceTier:
        flow = lp.LogisticFlowCurve(
            A0=float(self.flow_A0),
            theta=float(self.flow_theta),
            k=float(self.flow_k),
            m0=float(self.flow_m0),
            m_alpha=float(self.flow_m_alpha),
        )
        markout = lp.SqrtMarkoutModel(
            base=float(self.markout_base),
            coeff=float(self.markout_coeff),
        )
        return lp.PriceTier(
            name=self.name,
            sizes=[float(z) for z in self.sizes],
            flow_curve=flow,
            markout_model=markout,
            delta_min=float(self.delta_min),
            delta_max=float(self.delta_max),
            golden_tol=float(self.golden_tol),
            golden_max_iter=int(self.golden_max_iter),
        )


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
            "markout_base": 0.005,
            "markout_coeff": 0.003,
            "delta_min": -0.5,
            "delta_max": 4.0,
        },
        {
            "name": "aggressive_clients",
            "sizes": "1, 2, 5, 10",
            "flow_A0": 1.30,
            "flow_theta": 0.20,
            "flow_k": 2.40,
            "flow_m0": 0.05,
            "flow_m_alpha": 0.070,
            "markout_base": 0.007,
            "markout_coeff": 0.004,
            "delta_min": -0.5,
            "delta_max": 4.0,
        },
        {
            "name": "sticky_clients",
            "sizes": "1, 2, 5, 10",
            "flow_A0": 0.85,
            "flow_theta": 0.15,
            "flow_k": 1.60,
            "flow_m0": 0.28,
            "flow_m_alpha": 0.090,
            "markout_base": 0.003,
            "markout_coeff": 0.002,
            "delta_min": -0.5,
            "delta_max": 4.0,
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

        st.markdown("**Markout parameters**")
        c5, c6 = st.columns(2)
        markout_base = c5.number_input(
            f"Markout base {i + 1}",
            value=float(defaults["markout_base"]),
            step=0.001,
            format="%.5f",
            key=f"markout_base_{i}",
        )
        markout_coeff = c6.number_input(
            f"Markout coeff {i + 1}",
            value=float(defaults["markout_coeff"]),
            step=0.001,
            format="%.5f",
            key=f"markout_coeff_{i}",
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
        markout_base=float(markout_base),
        markout_coeff=float(markout_coeff),
        delta_min=float(tier_delta_min),
        delta_max=float(tier_delta_max),
    )


def make_flow_parameter_table(spec: TierSpec, cpp_tier: lp.PriceTier) -> pd.DataFrame:
    rows = []
    for z in spec.sizes:
        zf = float(z)
        rows.append(
            {
                "z": zf,
                "A(z)": spec.flow_A0 * zf ** (-spec.flow_theta),
                "m(z)": spec.flow_m0 + spec.flow_m_alpha * zf,
                "k": spec.flow_k,
                "mu(z)": cpp_tier.expected_markout(zf),
            }
        )
    return pd.DataFrame(rows)


def policy_arrays(cpp_tier: lp.PriceTier) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_grid = np.array(cpp_tier.policy.q_grid, dtype=float)
    bid = np.array(cpp_tier.policy.bid, dtype=float)
    ask = np.array(cpp_tier.policy.ask, dtype=float)
    return q_grid, bid, ask


def make_quote_inventory_figure(cpp_tier: lp.PriceTier, spec: TierSpec) -> go.Figure:
    q_grid, bid, ask = policy_arrays(cpp_tier)

    fig = go.Figure()
    for j, z in enumerate(spec.sizes):
        zf = float(z)
        fig.add_trace(
            go.Scatter(
                x=q_grid,
                y=-bid[:, j],
                mode="lines",
                name=f"{zf:g} bid",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=q_grid,
                y=ask[:, j],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{zf:g} ask",
            )
        )

    fig.add_hline(y=0.0)
    fig.update_layout(
        title=f"Quotes vs inventory — {spec.name}",
        xaxis_title="Inventory q",
        yaxis_title="Quote around mid (mid = 0)",
        height=520,
        yaxis=dict(range=[-10, 10])
    )
    return fig


def make_ladder_figure(cpp_tier: lp.PriceTier, spec: TierSpec, q: float) -> go.Figure:
    sizes = [float(z) for z in spec.sizes]
    bid_vals = [-cpp_tier.quote(float(q), z, "bid") for z in sizes]
    ask_vals = [cpp_tier.quote(float(q), z, "ask") for z in sizes]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=bid_vals,
            mode="lines+markers",
            name=f"Bid q={q:g}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=ask_vals,
            mode="lines+markers",
            line=dict(dash="dash"),
            name=f"Ask q={q:g}",
        )
    )

    fig.add_hline(y=0.0)
    fig.update_layout(
        title=f"Ladders — {spec.name} at q = {q:g}",
        xaxis_title="Trade size z",
        yaxis_title="Quote around mid",
        height=520,
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
        height=520,
    )
    return fig


def make_q_ladder_table(cpp_tier: lp.PriceTier, spec: TierSpec, q: float) -> pd.DataFrame:
    rows = []
    for z in spec.sizes:
        zf = float(z)
        bid_delta = cpp_tier.quote(float(q), zf, "bid")
        ask_delta = cpp_tier.quote(float(q), zf, "ask")
        rows.append(
            {
                "q": float(q),
                "z": zf,
                "bid_quote_plot_value": -bid_delta,
                "ask_quote_plot_value": ask_delta,
                "raw_bid_delta": bid_delta,
                "raw_ask_delta": ask_delta,
            }
        )
    return pd.DataFrame(rows)


def make_h_figure(solution: lp.HJBSolution) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(solution.q_grid),
            y=list(solution.h),
            mode="lines+markers",
            name="h(q)",
        )
    )
    fig.update_layout(
        title="Value function h(q)",
        xaxis_title="Inventory q",
        yaxis_title="h(q)",
        height=480,
    )
    return fig


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


# ============================================================
# App
# ============================================================

st.set_page_config(page_title="HJB ladder playground", layout="wide")
st.title("HJB ladder playground")
st.markdown(
    "This version uses the `ladder_pricer` C++ package via pybind11. "
    "The app recomputes automatically whenever parameters change."
)

with st.sidebar:
    st.header("Global parameters")
    q_min = st.number_input("q min", value=-20, step=1)
    q_max = st.number_input("q max", value=20, step=1)
    q_step = st.number_input("q step", value=1, step=1, min_value=1)

    dt = st.number_input("dt", value=0.002, step=0.001, format="%.4f")
    n_iter = st.number_input("n_iter", value=140, step=10, min_value=1)

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
    sigma = st.number_input("sigma", value=0.25, step=0.01, format="%.4f")
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

if q_max <= q_min:
    errors.append("q_max must be greater than q_min.")

q_grid = np.arange(float(q_min), float(q_max) + float(q_step), float(q_step), dtype=float)

if len(q_grid) < 2:
    errors.append("q_grid must contain at least two points.")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

cpp_tiers = [spec.build_cpp_tier() for spec in tier_specs]

penalty = lp.PolynomialInventoryPenalty(
    risk_aversion=float(risk_aversion),
    sigma=float(sigma),
    tau0=float(tau0),
    cubic_coeff=float(cubic_coeff),
    quartic_coeff=float(quartic_coeff),
)

config = lp.SolverConfig()
config.q_grid = [float(q) for q in q_grid]
config.dt = float(dt)
config.n_iter = int(n_iter)
config.early_stop = bool(early_stop)
config.tol_h = float(tol_h)
config.tol_rhs = float(tol_rhs)
config.min_iter = int(min_iter)
config.consecutive_passes_required = int(consecutive_passes_required)

with st.spinner("Solving HJB in C++ and building saved policies..."):
    solver = lp.HJBLadderSolver(config=config, penalty=penalty, tiers=cpp_tiers)
    solution: lp.HJBSolution = solver.solve()

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

c1, c2, c3, c4 = st.columns(4)
c1.metric("q points", len(config.q_grid))
c2.metric("tiers", len(solution.tiers))
c3.metric("iterations used", solution.iterations_used)
c4.metric("converged", "yes" if solution.converged else "no")

c5, c6 = st.columns(2)
c5.metric("final max |Δh|", f"{solution.final_max_h_change:.2e}")
c6.metric("final max |rhs|", f"{solution.final_max_rhs:.2e}")

st.plotly_chart(make_h_figure(solution), use_container_width=True)
st.plotly_chart(make_convergence_figure(solution), use_container_width=True)

tab_names = [spec.name for spec in tier_specs]
tabs = st.tabs(tab_names)

available_q = [float(q) for q in solution.q_grid]

for tab, spec, cpp_tier in zip(tabs, tier_specs, solution.tiers):
    with tab:
        st.subheader(f"Tier: {spec.name}")

        st.markdown("**Implied parameters at ladder sizes**")
        st.dataframe(make_flow_parameter_table(spec, cpp_tier), use_container_width=True)

        st.plotly_chart(
            make_flow_curve_figure(cpp_tier, spec),
            use_container_width=True,
        )

        st.plotly_chart(
            make_quote_inventory_figure(cpp_tier, spec),
            use_container_width=True,
        )

        default_q_single = 0.0 if 0.0 in available_q else available_q[len(available_q) // 2]

        q_for_ladder = st.select_slider(
            f"Inventory level for ladder — {spec.name}",
            options=available_q,
            value=default_q_single,
            key=f"qslider_{spec.name}",
        )

        st.plotly_chart(
            make_ladder_figure(cpp_tier, spec, float(q_for_ladder)),
            use_container_width=True,
        )

        q_for_table = st.selectbox(
            f"q for ladder table — {spec.name}",
            options=available_q,
            index=available_q.index(0.0) if 0.0 in available_q else len(available_q) // 2,
            key=f"qtable_{spec.name}",
        )

        st.dataframe(
            make_q_ladder_table(cpp_tier, spec, float(q_for_table)),
            use_container_width=True,
        )

with st.expander("What this app is solving"):
    st.markdown(
        r"""
For each inventory point $q$, the Bellman right-hand side is

$$
-\phi(q)
+ \sum_{\text{tier}} \sum_z \Big[
\lambda(\delta^b, z) \big(z(\delta^b - \mu(z)) + h(q+z)-h(q)\big)
+
\lambda(\delta^a, z) \big(z(\delta^a - \mu(z)) + h(q-z)-h(q)\big)
\Big].
$$

The C++ backend solves the ladder sequentially:
- first near $q = 0$,
- then for $q > 0$ moving outward,
- then for $q < 0$ moving outward.

For each side and inventory, the ladder is built rung-by-rung in size order using bounded golden-section search.

Hard constraints:
1. Larger sizes cannot be more aggressive than smaller sizes:
   $$
   \delta(q, z_{j+1}) \ge \delta(q, z_j).
   $$
2. Inventory-dependent gap rules:
   - for $q > 0$: ask gaps shrink, bid gaps increase
   - for $q < 0$: ask gaps increase, bid gaps decrease
        """
    )

st.caption(
    "This app recomputes on every widget change. If you want less frequent recomputation, "
    "wrap the sidebar inputs in a form and solve only on submit."
)