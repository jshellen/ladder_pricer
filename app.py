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
    flow_steepness: float
    flow_shift: float
    flow_volume_shift: float
    markout_base: float
    markout_coeff: float
    delta_min: float
    delta_max: float

    def build_cpp_tier(self) -> lp.PriceTier:
        flow = lp.LogisticFlowCurve(
            A0=float(self.flow_A0),
            theta=float(self.flow_theta),
            shift=float(self.flow_shift),
            steepness=float(self.flow_steepness),
            volume_shift=float(self.flow_volume_shift),
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
            "sizes": "1, 2, 3, 5, 10, 20",
            "flow_A0": 1.00,
            "flow_theta": 0.0,
            "flow_steepness": 10.00,
            "flow_shift": 0.50,
            "flow_volume_shift": 0.015,
            "markout_base": 0.000,
            "markout_coeff": 0.000,
            "delta_min": -100.0,
            "delta_max": 100.0,
        },
        {
            "name": "aggressive_clients",
            "sizes": "1, 2, 3, 5, 10, 20",
            "flow_A0": 1.0,
            "flow_theta": 0.0,
            "flow_steepness": 10.0,
            "flow_shift": 0.30,
            "flow_volume_shift": 0.01,
            "markout_base": 0.000,
            "markout_coeff": 0.000,
            "delta_min": -100.0,
            "delta_max": 100.0,
        },
        {
            "name": "sticky_clients",
            "sizes": "1, 2, 3, 5, 10, 15, 20",
            "flow_A0": 0.85,
            "flow_theta": 0.15,
            "flow_steepness": 1.60,
            "flow_shift": 0.28,
            "flow_volume_shift": 0.090,
            "markout_base": 0.000,
            "markout_coeff": 0.000,
            "delta_min": -100.0,
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
        flow_steepness = c3.number_input(
            f"steepness {i + 1}",
            value=float(defaults["flow_steepness"]),
            step=0.05,
            format="%.4f",
            key=f"flow_steepness_{i}",
        )
        flow_shift = c4.number_input(
            f"shift {i + 1}",
            value=float(defaults["flow_shift"]),
            step=0.05,
            format="%.4f",
            key=f"flow_shift_{i}",
        )

        flow_volume_shift = st.number_input(
            f"volume_shift {i + 1}",
            value=float(defaults["flow_volume_shift"]),
            step=0.001,
            format="%.4f",
            key=f"flow_volume_shift_{i}",
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
            step=0.05,
            format="%.4f",
            key=f"tier_delta_min_{i}",
        )
        tier_delta_max = c8.number_input(
            f"Tier delta_max {i + 1}",
            value=float(defaults["delta_max"]),
            step=0.05,
            format="%.4f",
            key=f"tier_delta_max_{i}",
        )

    sizes = parse_float_list(sizes_raw, f"sizes for tier {i + 1}")

    if flow_A0 < 0.0:
        raise ValueError(f"A0 for tier {i + 1} must be nonnegative.")
    if flow_steepness <= 0.0:
        raise ValueError(f"steepness for tier {i + 1} must be positive.")
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
        flow_steepness=float(flow_steepness),
        flow_shift=float(flow_shift),
        flow_volume_shift=float(flow_volume_shift),
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
                "delta_50(z)": spec.flow_shift - spec.flow_volume_shift * (zf - 1.0),
                "steepness": spec.flow_steepness,
                "mu(z)": cpp_tier.expected_markout(zf),
            }
        )
    return pd.DataFrame(rows)


def make_quote_inventory_figure(
    cpp_tier: lp.PriceTier,
    spec: TierSpec,
    spread: float,
    mid_price: float,
) -> go.Figure:
    q_grid = [float(q) for q in cpp_tier.policy.q_grid]

    fig = go.Figure()
    for z in spec.sizes:
        zf = float(z)

        bid_vals = [
            float(cpp_tier.quote_summary(float(q), zf, "bid", float(mid_price), float(spread)).quote_relative_to_mid_pips)
            for q in q_grid
        ]
        ask_vals = [
            float(cpp_tier.quote_summary(float(q), zf, "ask", float(mid_price), float(spread)).quote_relative_to_mid_pips)
            for q in q_grid
        ]

        fig.add_trace(
            go.Scatter(
                x=q_grid,
                y=bid_vals,
                mode="lines",
                name=f"{zf:g} bid",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=q_grid,
                y=ask_vals,
                mode="lines",
                line=dict(dash="dash"),
                name=f"{zf:g} ask",
            )
        )

    fig.add_hline(y=0.0)
    fig.update_layout(
        title=f"Quotes vs inventory — {spec.name}",
        xaxis_title="Inventory q",
        yaxis_title="Quote relative to mid (pips)",
        height=520,
    )
    return fig


def make_ladder_figure(
    cpp_tier: lp.PriceTier,
    spec: TierSpec,
    q: float,
    spread: float,
    mid_price: float,
) -> go.Figure:
    sizes = [float(z) for z in spec.sizes]

    bid_vp = [
        float(cpp_tier.quote_summary(float(q), z, "bid", float(mid_price), float(spread)).volume_premium_pips)
        for z in sizes
    ]
    ask_vp = [
        float(cpp_tier.quote_summary(float(q), z, "ask", float(mid_price), float(spread)).volume_premium_pips)
        for z in sizes
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=bid_vp,
            mode="lines+markers",
            name=f"Bid q={q:g}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=ask_vp,
            mode="lines+markers",
            line=dict(dash="dash"),
            name=f"Ask q={q:g}",
        )
    )

    fig.add_hline(y=0.0)
    fig.update_layout(
        title=f"Volume premium — {spec.name} at q = {q:g}",
        xaxis_title="Trade size z",
        yaxis_title="Premium vs 1M quote (pips)",
        height=520,
    )
    return fig


def make_flow_curve_figure(cpp_tier: lp.PriceTier, spec: TierSpec) -> go.Figure:
    grid = np.linspace(-1.0, +1.0, 100)
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
        xaxis_title="delta (fraction of spread improvement)",
        yaxis_title="arrival rate",
        height=520,
    )
    return fig


def make_q_ladder_table(
    cpp_tier: lp.PriceTier,
    spec: TierSpec,
    q: float,
    spread: float,
    mid_price: float,
) -> pd.DataFrame:
    rows = []

    for z in spec.sizes:
        zf = float(z)
        bid = cpp_tier.quote_summary(float(q), zf, "bid", float(mid_price), float(spread))
        ask = cpp_tier.quote_summary(float(q), zf, "ask", float(mid_price), float(spread))

        rows.append(
            {
                "q": float(q),
                "z": zf,
                "Bid delta": round(float(bid.delta), 6),
                "Ask delta": round(float(ask.delta), 6),
                "Bid improvement [% of spread]": round(float(bid.price_improvement_pct_of_spread), 2),
                "Ask improvement [% of spread]": round(float(ask.price_improvement_pct_of_spread), 2),
                "Bid improvement [pips]": round(float(bid.price_improvement_pips), 3),
                "Ask improvement [pips]": round(float(ask.price_improvement_pips), 3),
                "Bid vs mid [pips]": round(float(bid.quote_relative_to_mid_pips), 3),
                "Ask vs mid [pips]": round(float(ask.quote_relative_to_mid_pips), 3),
                "Bid dist to mid [pips]": round(float(bid.distance_to_mid_pips), 3),
                "Ask dist to mid [pips]": round(float(ask.distance_to_mid_pips), 3),
                "Bid vol premium [pips]": round(float(bid.volume_premium_pips), 3),
                "Ask vol premium [pips]": round(float(ask.volume_premium_pips), 3),
                "Bid quote": round(float(bid.quote_price), 6),
                "Ask quote": round(float(ask.quote_price), 6),
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


def make_convergence_h_figure(solution: lp.HJBSolution) -> go.Figure:
    fig = go.Figure()
    hist_h = list(solution.diagnostics.history_max_h_change)

    if hist_h:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(hist_h) + 1)),
                y=hist_h,
                mode="lines+markers",
                name="max |Δh|",
            )
        )

    fig.update_layout(
        title="Convergence: max |Δh|",
        xaxis_title="Iteration",
        yaxis_title="max |Δh|",
        yaxis_type="log",
        height=420,
    )
    return fig


def make_convergence_rhs_figure(solution: lp.HJBSolution) -> go.Figure:
    fig = go.Figure()
    hist_rhs = list(solution.diagnostics.history_max_rhs)

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
        title="Convergence: max |rhs|",
        xaxis_title="Iteration",
        yaxis_title="max |rhs|",
        yaxis_type="log",
        height=420,
    )
    return fig


# ============================================================
# App
# ============================================================

st.set_page_config(page_title="Trinity 2.0 Pricer", layout="wide")
st.title("Trinity 2.0 Pricer")
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

    spread = st.number_input(
        "reference spread",
        value=20.0 / 10000.0,
        step=1.0 / 10000.0,
        format="%.6f",
        help=(
            "Reference spread used to define reference bid/ask and quote-improvement delta. "
            "ref_bid = mid - 0.5 * spread, ref_ask = mid + 0.5 * spread."
        ),
    )

    mid_price = st.number_input(
        "display mid price",
        value=1.000000,
        step=0.000100,
        format="%.6f",
        help="Used only when displaying absolute bid/ask quotes in the ladder table and charts.",
    )

    st.header("Spot process")
    spot_drift = st.number_input(
        "spot_drift",
        value=0.0,
        step=0.001,
        format="%.5f",
        help=(
            "Constant drift in the spot process dS = spot_drift * dt + sigma * dW. "
            "In the reduced HJB this contributes + spot_drift * q."
        ),
    )

    st.header("Ladder optimization")
    golden_tol = st.number_input("golden_tol", value=1e-4, format="%.1e")
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
    sigma = st.number_input("Volatility [pips / 1min]", value=20.0, step=1.0, format="%.2f")
    sigma = sigma / 10_000.0
    risk_aversion = st.number_input("risk_aversion", value=10.0, step=1.0, format="%.2f")
    tau0 = st.number_input("tau0", value=5.0, step=1.0, format="%.4f")
    tau1 = st.number_input("cubic coeff", value=0.1, step=0.01, format="%.4f")
    tau2 = st.number_input("quartic coeff", value=0.0015, step=0.0005, format="%.5f")

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
if spread <= 0.0:
    errors.append("spread must be positive.")
if golden_tol <= 0.0:
    errors.append("golden_tol must be positive.")

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
    cubic_coeff=float(tau1),
    quartic_coeff=float(tau2),
)

config = lp.SolverConfig()
config.q_grid = [float(q) for q in q_grid]
config.dt = float(dt)
config.n_iter = int(n_iter)
config.spread = float(spread)
config.spot_drift = float(spot_drift)
config.golden_tol = float(golden_tol)
config.golden_max_iter = int(golden_max_iter)
config.early_stop = bool(early_stop)
config.tol_h = float(tol_h)
config.tol_rhs = float(tol_rhs)
config.min_iter = int(min_iter)
config.consecutive_passes_required = int(consecutive_passes_required)

with st.spinner("Solving HJB in C++ and building saved policies..."):
    solver = lp.HJBLadderSolver(config=config, penalty=penalty, tiers=cpp_tiers)
    solution: lp.HJBSolution = solver.solve()

diag = solution.diagnostics

st.success("Solver run complete.")

if diag.converged:
    st.info(
        f"Early stopping triggered after {diag.iterations_used} iterations. "
        f"Final max |Δh| = {diag.final_max_h_change:.2e}, "
        f"final max |rhs| = {diag.final_max_rhs:.2e}."
    )
else:
    st.warning(
        f"Solver reached the iteration cap ({diag.iterations_used}). "
        f"Final max |Δh| = {diag.final_max_h_change:.2e}, "
        f"final max |rhs| = {diag.final_max_rhs:.2e}."
    )

with st.expander("Solver diagnostics", expanded=False):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("q points", len(config.q_grid))
    c2.metric("tiers", len(solution.tiers))
    c3.metric("iterations used", diag.iterations_used)
    c4.metric("converged", "yes" if diag.converged else "no")
    c5.metric("spot drift", f"{config.spot_drift:.5f}")
    c6.metric("spread", f"{config.spread:.6f}")

    c7, c8, c9, c10 = st.columns(4)
    c7.metric("golden tol", f"{config.golden_tol:.1e}")
    c8.metric("golden max iter", int(config.golden_max_iter))
    c9.metric("final max |Δh|", f"{diag.final_max_h_change:.2e}")
    c10.metric("final max |rhs|", f"{diag.final_max_rhs:.2e}")

    st.plotly_chart(make_h_figure(solution), use_container_width=True)
    st.plotly_chart(make_convergence_h_figure(solution), use_container_width=True)
    st.plotly_chart(make_convergence_rhs_figure(solution), use_container_width=True)

tab_names = [spec.name for spec in tier_specs]
tabs = st.tabs(tab_names)

available_q = [float(q) for q in solution.q_grid]

for tab, spec, cpp_tier in zip(tabs, tier_specs, solution.tiers):
    with tab:
        st.subheader(f"Tier: {spec.name}")

        with st.expander("Tier parameters", expanded=False):
            st.dataframe(
                make_flow_parameter_table(spec, cpp_tier),
                use_container_width=True,
            )

        st.plotly_chart(
            make_flow_curve_figure(cpp_tier, spec),
            use_container_width=True,
        )

        st.plotly_chart(
            make_quote_inventory_figure(cpp_tier, spec, float(config.spread), float(mid_price)),
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
            make_ladder_figure(cpp_tier, spec, float(q_for_ladder), float(config.spread), float(mid_price)),
            use_container_width=True,
        )

        q_for_table = st.selectbox(
            f"q for ladder table — {spec.name}",
            options=available_q,
            index=available_q.index(0.0) if 0.0 in available_q else len(available_q) // 2,
            key=f"qtable_{spec.name}",
        )

        st.dataframe(
            make_q_ladder_table(
                cpp_tier,
                spec,
                float(q_for_table),
                float(config.spread),
                float(mid_price),
            ),
            use_container_width=True,
        )

with st.expander("What this app is solving"):
    st.markdown(
        r"""
Assume the spot process is

$$
dS_t = \mu_{\text{spot}}\,dt + \sigma\,dW_t.
$$

Let the reference spread be \(s\), and define

$$
\text{ref bid} = \text{mid} - \tfrac12 s,
\qquad
\text{ref ask} = \text{mid} + \tfrac12 s.
$$

The quote improvement \(\delta\) is measured as a fraction of that spread:

$$
\text{bid quote} = \text{ref bid} + s\,\delta,
\qquad
\text{ask quote} = \text{ref ask} - s\,\delta.
$$

So the quote relative to mid is

$$
\text{bid quote} - \text{mid} = s(\delta - 0.5),
\qquad
\text{ask quote} - \text{mid} = s(0.5 - \delta).
$$

The absolute distance to mid is therefore

$$
|\text{quote} - \text{mid}| = s(0.5 - \delta)
$$

under the normal regime \(\delta \le 0.5\). In the app, these quantities are shown via the `QuoteSummary` object returned by the C++ layer.

For each inventory point \(q\), the Bellman right-hand side is

$$
-\phi(q) + \mu_{\text{spot}} q
+ \sum_{\text{tier}} \sum_z \Big[
\lambda(\delta^b, z)\big(z\,s(0.5-\delta^b) - z\mu(z) + h(q+z)-h(q)\big)
+
\lambda(\delta^a, z)\big(z\,s(0.5-\delta^a) - z\mu(z) + h(q-z)-h(q)\big)
\Big].
$$

So:
- positive spot drift favors long inventory
- negative spot drift favors short inventory
- larger \(\delta\) means a more aggressive quote on both sides
- the model chooses \(\delta(q,z)\) separately for each side, size, inventory point, and tier

The C++ backend solves the ladder sequentially:
- first at the inventory point closest to \(q=0\),
- then for \(q>0\) moving outward,
- then for \(q<0\) moving outward.

For each side and inventory, the ladder is built rung-by-rung in size order using bounded golden-section search.

### What the bounds are doing

Let the sizes be

$$
z_1 < z_2 < \dots < z_n
$$

and let \(\delta_j\) denote the quote improvement for rung \(z_j\).

The first structural rule is monotonicity in size:

$$
\delta_{j+1} \le \delta_j.
$$

Because larger \(\delta\) means a more aggressive quote, this ensures that a larger trade size can never be quoted more aggressively than a smaller one. In practice, when the solver is optimizing rung \(j\), this immediately gives an upper bound:

$$
\delta_j \le \delta_{j-1}.
$$

That is the baseline ladder-shape constraint.

### Gap notation

Define the rung gap as

$$
g_j = \delta_{j-1} - \delta_j \ge 0,
\qquad j = 2,\dots,n.
$$

So:
- small gap means neighboring rungs are close together
- large gap means the ladder widens more quickly in size

The model uses the already-solved neighboring inventory state to decide whether these gaps should shrink or expand as inventory moves away from zero.

### Shrink mode

In shrink mode, the gap at the current inventory should not exceed the previously solved gap:

$$
g_j^{\text{current}} \le g_j^{\text{previous}}.
$$

Since

$$
g_j^{\text{current}} = \delta_{j-1}^{\text{current}} - \delta_j^{\text{current}},
$$

this becomes a lower bound on the current rung:

$$
\delta_j^{\text{current}}
\ge
\delta_{j-1}^{\text{current}} - g_j^{\text{previous}}.
$$

Interpretation:
- the current rung is not allowed to fall too far below the previous rung
- the ladder therefore compresses relative to the neighboring inventory state

This is used when the model wants the ladder to become tighter on the risk-reducing side.

### Expand mode

In expand mode, the current gap must be at least as large as before:

$$
g_j^{\text{current}} \ge g_j^{\text{previous}}.
$$

This becomes an upper bound:

$$
\delta_j^{\text{current}}
\le
\delta_{j-1}^{\text{current}} - g_j^{\text{previous}}.
$$

Interpretation:
- the rung must sit sufficiently below the previous rung
- the ladder therefore widens relative to the neighboring inventory state

This is used when the model wants larger trade sizes to become less competitive on the risk-increasing side.

### Why the solver also reserves room for future rungs

If the current rung is pushed too low, later rungs may become impossible to place while still respecting both:
- the global lower bound \(\delta_{\min}\)
- the required future gap expansion rules

So in expand mode the solver also imposes a feasibility bound of the form

$$
\delta_j \ge \delta_{\min} + \text{(required future gap budget)}.
$$

That budget is the total amount of gap that later rungs will need below the current rung if the ladder is to remain feasible all the way to the largest size.

This is important because otherwise the optimizer could choose a locally optimal \(\delta_j\) that makes rung \(j+1\), \(j+2\), or later rungs infeasible.

### Why rung 1 can also have a special lower bound

For the smallest rung, there is no previous rung within the same ladder, so the monotonicity constraint does not yet apply.

However, in expand mode, the solver may still need to leave enough room below rung 1 for all later mandatory gaps. That gives a lower bound like

$$
\delta_1 \ge \delta_{\min} + \text{(total required ladder width)}.
$$

So even the first rung may be prevented from moving too low if doing so would make the rest of the ladder impossible to fit inside \([\delta_{\min}, \delta_{\max}]\).

### Which side shrinks and which side expands?

The inventory-dependent rule is:

- for \(q > 0\):
  - ask ladder gaps shrink
  - bid ladder gaps expand

- for \(q < 0\):
  - ask ladder gaps expand
  - bid ladder gaps shrink

This matches the intuition that the side which helps reduce inventory should become more competitive, while the side that would add more inventory should become less competitive, especially for larger trade sizes.

### In summary

The solver is not just clipping each rung independently into \([\delta_{\min}, \delta_{\max}]\). It is enforcing a nested feasibility structure:

1. global bounds:
   $$
   \delta_{\min} \le \delta_j \le \delta_{\max}
   $$

2. monotonicity in size:
   $$
   \delta_j \le \delta_{j-1}
   $$

3. shrink / expand gap rules relative to the neighboring inventory state

4. future-feasibility bounds so that later rungs still fit inside the allowed region

That is why the ladder tends to look smooth, ordered, and inventory-aware rather than like a collection of unrelated pointwise optima.
        """
    )

st.caption(
    "This app recomputes on every widget change. If you want less frequent recomputation, "
    "wrap the sidebar inputs in a form and solve only on submit."
)