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
# Inventory-grid helpers
# ============================================================

def build_uniform_centered_q_grid(q_abs_max: float, q_step: float) -> np.ndarray:
    if q_abs_max <= 0.0:
        raise ValueError("q_abs_max must be positive.")
    if q_step <= 0.0:
        raise ValueError("q_step must be positive.")

    pos = np.arange(0.0, q_abs_max + 0.5 * q_step, q_step, dtype=float)
    pos = pos[pos <= q_abs_max + 1e-12]

    if len(pos) == 0 or not np.isclose(pos[-1], q_abs_max, atol=1e-10, rtol=0.0):
        pos = np.append(pos, q_abs_max)

    pos = np.unique(np.round(pos, 12))
    return np.concatenate((-pos[:0:-1], pos))


def build_piecewise_centered_q_grid(
    q_abs_max: float,
    fine_half_width: float,
    fine_step: float,
    coarse_step: float,
) -> np.ndarray:
    if q_abs_max <= 0.0:
        raise ValueError("q_abs_max must be positive.")
    if fine_half_width < 0.0:
        raise ValueError("fine_half_width must be nonnegative.")
    if fine_half_width > q_abs_max:
        raise ValueError("fine_half_width cannot exceed q_abs_max.")
    if fine_step <= 0.0 or coarse_step <= 0.0:
        raise ValueError("fine_step and coarse_step must be positive.")

    inner_end = min(fine_half_width, q_abs_max)

    inner = np.arange(0.0, inner_end + 0.5 * fine_step, fine_step, dtype=float)
    inner = inner[inner <= inner_end + 1e-12]

    outer = np.array([], dtype=float)
    if q_abs_max > inner_end + 1e-12:
        start = inner_end + coarse_step
        outer = np.arange(start, q_abs_max + 0.5 * coarse_step, coarse_step, dtype=float)
        outer = outer[outer <= q_abs_max + 1e-12]

    pos = np.concatenate((inner, outer))
    pos = np.unique(np.round(pos, 12))

    if len(pos) == 0 or not np.isclose(pos[-1], q_abs_max, atol=1e-10, rtol=0.0):
        pos = np.append(pos, q_abs_max)

    return np.concatenate((-pos[:0:-1], pos))


# ============================================================
# Tier specification
# ============================================================

DEFAULT_MDP_SIZES = "1, 2, 3, 5, 10, 20"
DEFAULT_ECN_POLICY = {
    "ecn_delta_start": 0.10,
    "ecn_delta_target": 0.50,
    "ecn_decay": 2.0,
    "ecn_decay_min": 0.25,
    "ecn_decay_max": 10.0,
}


@dataclass
class CommonTierSpec:
    name: str
    flow_A0: float
    flow_theta: float
    flow_steepness: float
    flow_shift: float
    flow_volume_shift: float
    markout_base: float
    markout_coeff: float
    delta_min: float
    delta_max: float

    def build_models(self) -> tuple[lp.LogisticFlowCurve, lp.SqrtMarkoutModel]:
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
        return flow, markout


@dataclass
class MDPTierSpec(CommonTierSpec):
    sizes: list[float]

    def build_cpp_tier(self) -> lp.MDPTier:
        flow, markout = self.build_models()
        return lp.MDPTier(
            name=self.name,
            sizes=[float(z) for z in self.sizes],
            flow_curve=flow,
            markout_model=markout,
            delta_min=float(self.delta_min),
            delta_max=float(self.delta_max),
        )


@dataclass
class ECNTierSpec(CommonTierSpec):
    ecn_delta_start: float
    ecn_delta_target: float
    ecn_decay: float
    ecn_decay_min: float
    ecn_decay_max: float

    def build_cpp_tier(self) -> lp.ECNTier:
        flow, markout = self.build_models()
        ecn_policy = lp.ExponentialECNPolicy(
            delta_start=float(self.ecn_delta_start),
            delta_target=float(self.ecn_delta_target),
            decay=float(self.ecn_decay),
            decay_min=float(self.ecn_decay_min),
            decay_max=float(self.ecn_decay_max),
        )
        return lp.ECNTier(
            name=self.name,
            flow_curve=flow,
            markout_model=markout,
            delta_min=float(self.delta_min),
            delta_max=float(self.delta_max),
            ecn_policy=ecn_policy,
        )


TierSpec = MDPTierSpec | ECNTierSpec


def tier_kind_label(spec: TierSpec) -> str:
    return "ECN" if isinstance(spec, ECNTierSpec) else "MDP"


def tier_sizes(spec: TierSpec) -> list[float]:
    return [1.0] if isinstance(spec, ECNTierSpec) else spec.sizes


def parse_float_list(raw: str, field_name: str) -> List[float]:
    try:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Could not parse {field_name}. Use comma-separated numbers.") from exc

    if not vals:
        raise ValueError(f"{field_name} cannot be empty.")

    return vals


def default_tier_values(i: int) -> dict:
    presets = [
        {
            "kind": "mdp",
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
            "kind": "mdp",
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
            "kind": "ecn",
            "name": "ecn",
            "flow_A0": 0.70,
            "flow_theta": 0.10,
            "flow_steepness": 8.0,
            "flow_shift": 0.18,
            "flow_volume_shift": 0.020,
            "markout_base": 0.000,
            "markout_coeff": 0.000,
            "delta_min": -2.0,
            "delta_max": 2.0,
            **DEFAULT_ECN_POLICY,
        },
        {
            "kind": "mdp",
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
        kind = st.selectbox(
            f"Tier type {i + 1}",
            options=["mdp", "ecn"],
            index=0 if defaults["kind"] == "mdp" else 1,
            key=f"kind_{i}",
        )

        if kind == "mdp":
            sizes_default = defaults.get("sizes", DEFAULT_MDP_SIZES)
            sizes_raw = st.text_input(f"Sizes {i + 1}", value=sizes_default, key=f"sizes_{i}")
            sizes = parse_float_list(sizes_raw, f"sizes for tier {i + 1}")
        else:
            st.text_input(f"Sizes {i + 1}", value="1", key=f"sizes_{i}", disabled=True)
            st.caption("ECN quoted size is fixed to z = 1.")

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

        if kind == "ecn":
            ecn_defaults = {
                key: float(defaults.get(key, DEFAULT_ECN_POLICY[key]))
                for key in DEFAULT_ECN_POLICY
            }

            st.markdown("**ECN exponential policy**")
            c9, c10 = st.columns(2)
            ecn_delta_start = c9.number_input(
                f"ECN delta_start {i + 1}",
                value=ecn_defaults["ecn_delta_start"],
                step=0.01,
                format="%.4f",
                key=f"ecn_delta_start_{i}",
            )
            ecn_delta_target = c10.number_input(
                f"ECN delta_target {i + 1}",
                value=ecn_defaults["ecn_delta_target"],
                step=0.01,
                format="%.4f",
                key=f"ecn_delta_target_{i}",
            )

            c11, c12, c13 = st.columns(3)
            ecn_decay = c11.number_input(
                f"ECN decay init {i + 1}",
                value=ecn_defaults["ecn_decay"],
                step=0.05,
                format="%.4f",
                key=f"ecn_decay_{i}",
            )
            ecn_decay_min = c12.number_input(
                f"ECN decay min {i + 1}",
                value=ecn_defaults["ecn_decay_min"],
                step=0.05,
                format="%.4f",
                key=f"ecn_decay_min_{i}",
            )
            ecn_decay_max = c13.number_input(
                f"ECN decay max {i + 1}",
                value=ecn_defaults["ecn_decay_max"],
                step=0.05,
                format="%.4f",
                key=f"ecn_decay_max_{i}",
            )

            st.caption(
                "The ECN tier quotes only the inventory-reducing side. "
                "Its active-side quote follows a smooth exponential approach toward delta_target, "
                "and the decay parameter is optimized in C++."
            )

    if flow_A0 < 0.0:
        raise ValueError(f"A0 for tier {i + 1} must be nonnegative.")
    if flow_steepness <= 0.0:
        raise ValueError(f"steepness for tier {i + 1} must be positive.")
    if tier_delta_max <= tier_delta_min:
        raise ValueError(f"Tier {i + 1}: delta_max must be greater than delta_min.")

    common_kwargs = dict(
        name=name,
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

    if kind == "mdp":
        if any(z <= 0.0 for z in sizes):
            raise ValueError(f"Tier {i + 1}: all sizes must be positive.")
        if any(sizes[k] >= sizes[k + 1] for k in range(len(sizes) - 1)):
            raise ValueError(f"Tier {i + 1}: sizes must be strictly increasing.")
        return MDPTierSpec(
            sizes=sizes,
            **common_kwargs,
        )

    if ecn_decay_min <= 0.0:
        raise ValueError(f"Tier {i + 1}: ECN decay_min must be positive.")
    if ecn_decay_max <= ecn_decay_min:
        raise ValueError(f"Tier {i + 1}: ECN decay_max must be greater than decay_min.")
    if ecn_decay <= 0.0:
        raise ValueError(f"Tier {i + 1}: ECN decay init must be positive.")
    if ecn_delta_target < ecn_delta_start:
        raise ValueError(f"Tier {i + 1}: ECN delta_target must be >= delta_start.")
    if not (tier_delta_min <= ecn_delta_start <= tier_delta_max):
        raise ValueError(f"Tier {i + 1}: ECN delta_start must lie inside [delta_min, delta_max].")
    if not (tier_delta_min <= ecn_delta_target <= tier_delta_max):
        raise ValueError(f"Tier {i + 1}: ECN delta_target must lie inside [delta_min, delta_max].")

    return ECNTierSpec(
        ecn_delta_start=float(ecn_delta_start),
        ecn_delta_target=float(ecn_delta_target),
        ecn_decay=float(ecn_decay),
        ecn_decay_min=float(ecn_decay_min),
        ecn_decay_max=float(ecn_decay_max),
        **common_kwargs,
    )


# ============================================================
# Helpers
# ============================================================

def is_admissible(cpp_tier: lp.Tier, q: float, z: float, side: str) -> bool:
    return bool(cpp_tier.is_admissible(float(q), float(z), side))


def masked_quote_summary(
    cpp_tier: lp.Tier,
    q: float,
    z: float,
    side: str,
    mid_price: float,
    spread: float,
):
    if not is_admissible(cpp_tier, q, z, side):
        return None
    return cpp_tier.quote_summary(float(q), float(z), side, float(mid_price), float(spread))


def has_ecn_features(cpp_tier: lp.Tier) -> bool:
    return hasattr(cpp_tier, "ecn_policy") and hasattr(cpp_tier, "active_delta")


# ============================================================
# Plots and tables
# ============================================================

def make_flow_parameter_table(spec: TierSpec, cpp_tier: lp.Tier) -> pd.DataFrame:
    rows = []
    for z in tier_sizes(spec):
        zf = float(z)
        row = {
            "type": tier_kind_label(spec),
            "z": zf,
            "A(z)": spec.flow_A0 * zf ** (-spec.flow_theta),
            "delta_50(z)": spec.flow_shift - spec.flow_volume_shift * (zf - 1.0),
            "steepness": spec.flow_steepness,
            "mu(z)": cpp_tier.expected_markout(zf),
        }
        if isinstance(spec, ECNTierSpec) and has_ecn_features(cpp_tier):
            row["ecn_delta_start"] = float(cpp_tier.ecn_policy.delta_start)
            row["ecn_delta_target"] = float(cpp_tier.ecn_policy.delta_target)
            row["ecn_decay_opt"] = float(cpp_tier.ecn_policy.decay)
            row["ecn_decay_min"] = float(cpp_tier.ecn_policy.decay_min)
            row["ecn_decay_max"] = float(cpp_tier.ecn_policy.decay_max)
        rows.append(row)
    return pd.DataFrame(rows)


def make_quote_inventory_figure(
    cpp_tier: lp.Tier,
    spec: TierSpec,
    spread: float,
    mid_price: float,
) -> go.Figure:
    q_grid = [float(q) for q in cpp_tier.policy.q_grid]

    fig = go.Figure()
    for z in tier_sizes(spec):
        zf = float(z)

        bid_vals = []
        ask_vals = []

        for q in q_grid:
            bid = masked_quote_summary(cpp_tier, q, zf, "bid", mid_price, spread)
            ask = masked_quote_summary(cpp_tier, q, zf, "ask", mid_price, spread)

            bid_vals.append(np.nan if bid is None else float(bid.quote_relative_to_mid_pips))
            ask_vals.append(np.nan if ask is None else float(ask.quote_relative_to_mid_pips))

        fig.add_trace(go.Scatter(x=q_grid, y=bid_vals, mode="lines", name=f"{zf:g} bid"))
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
        title=f"Quotes vs inventory — {spec.name} ({tier_kind_label(spec)})",
        xaxis_title="Inventory q",
        yaxis_title="Quote relative to mid (pips)",
        height=520,
    )
    return fig


def make_ladder_figure(
    cpp_tier: lp.Tier,
    spec: TierSpec,
    q: float,
    spread: float,
    mid_price: float,
) -> go.Figure:
    sizes = [float(z) for z in tier_sizes(spec)]

    bid_vp = []
    ask_vp = []

    for z in sizes:
        bid = masked_quote_summary(cpp_tier, q, z, "bid", mid_price, spread)
        ask = masked_quote_summary(cpp_tier, q, z, "ask", mid_price, spread)

        bid_vp.append(np.nan if bid is None else float(bid.volume_premium_pips))
        ask_vp.append(np.nan if ask is None else float(ask.volume_premium_pips))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=bid_vp, mode="lines+markers", name=f"Bid q={q:g}"))
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
        title=f"Volume premium — {spec.name} ({tier_kind_label(spec)}) at q = {q:g}",
        xaxis_title="Trade size z",
        yaxis_title="Premium vs 1M quote (pips)",
        height=520,
    )
    return fig


def make_flow_curve_figure(cpp_tier: lp.Tier, spec: TierSpec) -> go.Figure:
    grid = np.linspace(-1.0, +1.0, 100)
    fig = go.Figure()

    for z in tier_sizes(spec):
        zf = float(z)
        vals = [cpp_tier.arrival_rate(float(d), zf) for d in grid]
        fig.add_trace(go.Scatter(x=grid, y=vals, mode="lines", name=f"{zf:g}"))

    fig.update_layout(
        title=f"Flow curves λ(δ, z) — {spec.name} ({tier_kind_label(spec)})",
        xaxis_title="delta",
        yaxis_title="arrival rate",
        height=520,
    )
    return fig


def make_ecn_decay_figure(cpp_tier: lp.Tier, spread: float) -> go.Figure:
    if not has_ecn_features(cpp_tier):
        raise ValueError("make_ecn_decay_figure requires an ECN tier.")

    q_grid = [float(q) for q in cpp_tier.policy.q_grid]
    q_pos = [q for q in q_grid if q > 0.0]
    q_neg = [q for q in q_grid if q < 0.0]

    ask_vals = [
        10000.0 * spread * (0.5 - float(cpp_tier.active_delta(q)))
        for q in q_pos
    ]
    bid_vals = [
        10000.0 * spread * (float(cpp_tier.active_delta(q)) - 0.5)
        for q in q_neg
    ]

    fig = go.Figure()
    if q_neg:
        fig.add_trace(go.Scatter(x=q_neg, y=bid_vals, mode="lines", name="bid active side"))
    if q_pos:
        fig.add_trace(
            go.Scatter(
                x=q_pos,
                y=ask_vals,
                mode="lines",
                line=dict(dash="dash"),
                name="ask active side",
            )
        )

    fig.add_hline(y=0.0)
    fig.update_layout(
        title="ECN active-side exponential quote vs inventory",
        xaxis_title="Inventory q",
        yaxis_title="Quote relative to mid (pips)",
        height=420,
    )
    return fig


def make_q_ladder_table(
    cpp_tier: lp.Tier,
    spec: TierSpec,
    q: float,
    spread: float,
    mid_price: float,
) -> pd.DataFrame:
    rows = []

    for z in tier_sizes(spec):
        zf = float(z)
        bid_adm = is_admissible(cpp_tier, q, zf, "bid")
        ask_adm = is_admissible(cpp_tier, q, zf, "ask")

        bid = masked_quote_summary(cpp_tier, q, zf, "bid", mid_price, spread)
        ask = masked_quote_summary(cpp_tier, q, zf, "ask", mid_price, spread)

        rows.append(
            {
                "type": tier_kind_label(spec),
                "q": float(q),
                "z": zf,
                "Bid admissible": bid_adm,
                "Ask admissible": ask_adm,
                "Bid delta": np.nan if bid is None else round(float(bid.delta), 6),
                "Ask delta": np.nan if ask is None else round(float(ask.delta), 6),
                "Bid improvement [% of spread]": np.nan if bid is None else round(float(bid.price_improvement_pct_of_spread), 2),
                "Ask improvement [% of spread]": np.nan if ask is None else round(float(ask.price_improvement_pct_of_spread), 2),
                "Bid improvement [pips]": np.nan if bid is None else round(float(bid.price_improvement_pips), 3),
                "Ask improvement [pips]": np.nan if ask is None else round(float(ask.price_improvement_pips), 3),
                "Bid vs mid [pips]": np.nan if bid is None else round(float(bid.quote_relative_to_mid_pips), 3),
                "Ask vs mid [pips]": np.nan if ask is None else round(float(ask.quote_relative_to_mid_pips), 3),
                "Bid dist to mid [pips]": np.nan if bid is None else round(float(bid.distance_to_mid_pips), 3),
                "Ask dist to mid [pips]": np.nan if ask is None else round(float(ask.distance_to_mid_pips), 3),
                "Bid vol premium [pips]": np.nan if bid is None else round(float(bid.volume_premium_pips), 3),
                "Ask vol premium [pips]": np.nan if ask is None else round(float(ask.volume_premium_pips), 3),
                "Bid quote": np.nan if bid is None else round(float(bid.quote_price), 6),
                "Ask quote": np.nan if ask is None else round(float(ask.quote_price), 6),
            }
        )

    return pd.DataFrame(rows)


def make_h_figure(solution: lp.HJBSolution) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(solution.q_grid), y=list(solution.h), mode="lines+markers", name="h(q)"))
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
        fig.add_trace(go.Scatter(x=list(range(1, len(hist_h) + 1)), y=hist_h, mode="lines+markers", name="max |Δh|"))

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
        fig.add_trace(go.Scatter(x=list(range(1, len(hist_rhs) + 1)), y=hist_rhs, mode="lines+markers", name="max |rhs|"))

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
    "MDP tiers use rung-by-rung two-sided ladder controls. "
    "ECN tiers are low-dimensional hedge channels: they always quote size z = 1, "
    "only the inventory-reducing side is active, and the active quote follows an exponential policy whose decay is optimized in C++."
)

with st.sidebar:
    st.header("Global parameters")
    q_grid_mode = st.selectbox(
        "Inventory grid mode",
        options=["uniform", "piecewise"],
        index=1,
    )

    q_abs_max = st.number_input("max |q|", value=20.0, min_value=0.5, step=0.5, format="%.4f")

    if q_grid_mode == "uniform":
        q_step = st.number_input("q step", value=1.0, min_value=0.01, step=0.1, format="%.4f")
        fine_half_width = None
        fine_step = None
        coarse_step = None
    else:
        fine_half_width = st.number_input("fine half-width", value=3.0, min_value=0.0, step=0.5, format="%.4f")
        c_grid1, c_grid2 = st.columns(2)
        fine_step = c_grid1.number_input("fine step", value=0.25, min_value=0.01, step=0.05, format="%.4f")
        coarse_step = c_grid2.number_input("coarse step", value=1.0, min_value=0.01, step=0.1, format="%.4f")
        q_step = None

    dt = st.number_input("dt", value=0.002, step=0.001, format="%.4f")
    n_iter = st.number_input("n_iter", value=140, step=10, min_value=1)

    spread = st.number_input(
        "reference spread",
        value=20.0 / 10000.0,
        step=1.0 / 10000.0,
        format="%.6f",
    )

    mid_price = st.number_input("display mid price", value=1.000000, step=0.000100, format="%.6f")

    st.header("Spot process")
    spot_drift = st.number_input("spot_drift", value=0.0, step=0.001, format="%.5f")

    st.header("Optimization / stopping")
    golden_tol = st.number_input("golden_tol", value=1e-4, format="%.1e")
    golden_max_iter = st.number_input("golden_max_iter", value=32, step=1, min_value=1)
    early_stop = st.checkbox("Enable early stopping", value=True)
    tol_h = st.number_input("tol_h", value=1e-5, format="%.1e")
    tol_rhs = st.number_input("tol_rhs", value=1e-4, format="%.1e")
    min_iter = st.number_input("min_iter", value=5, step=1, min_value=0)
    consecutive_passes_required = st.number_input("consecutive passes required", value=3, step=1, min_value=1)

    st.header("Inventory penalty")
    sigma = st.number_input("Volatility [pips / 1min]", value=20.0, step=1.0, format="%.2f") / 10_000.0
    risk_aversion = st.number_input("risk_aversion", value=10.0, step=1.0, format="%.2f")
    tau0 = st.number_input("tau0", value=5.0, step=1.0, format="%.4f")
    cubic_coeff = st.number_input("cubic coeff", value=0.1, step=0.01, format="%.4f")
    quartic_coeff = st.number_input("quartic coeff", value=0.0015, step=0.0005, format="%.5f")

    st.header("Pricing tiers")
    num_tiers = st.slider("Number of tiers", min_value=1, max_value=4, value=3)

errors: List[str] = []
tier_specs: List[TierSpec] = []
for i in range(num_tiers):
    try:
        tier_specs.append(build_tier_spec_from_ui(i))
    except ValueError as exc:
        errors.append(str(exc))

if spread <= 0.0:
    errors.append("spread must be positive.")
if golden_tol <= 0.0:
    errors.append("golden_tol must be positive.")

try:
    if q_grid_mode == "uniform":
        q_grid = build_uniform_centered_q_grid(float(q_abs_max), float(q_step))
    else:
        q_grid = build_piecewise_centered_q_grid(
            q_abs_max=float(q_abs_max),
            fine_half_width=float(fine_half_width),
            fine_step=float(fine_step),
            coarse_step=float(coarse_step),
        )
except ValueError as exc:
    errors.append(str(exc))
    q_grid = np.array([], dtype=float)

if len(q_grid) < 3:
    errors.append("q_grid must contain at least 3 points.")
if len(q_grid) > 0 and len(q_grid) % 2 == 0:
    errors.append("q_grid must have odd length.")
if len(q_grid) > 0 and not np.isclose(q_grid[len(q_grid) // 2], 0.0, atol=1e-10):
    errors.append("q_grid must contain 0 exactly at the middle index.")
if len(q_grid) > 0:
    mid_idx = len(q_grid) // 2
    for i in range(mid_idx):
        if not np.isclose(q_grid[i] + q_grid[-1 - i], 0.0, atol=1e-10):
            errors.append("q_grid must be symmetric around 0.")
            break

if errors:
    for e in errors:
        st.error(e)
    st.stop()

cpp_tiers: list[lp.Tier] = [spec.build_cpp_tier() for spec in tier_specs]

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
config.spread = float(spread)
config.spot_drift = float(spot_drift)
config.golden_tol = float(golden_tol)
config.golden_max_iter = int(golden_max_iter)
config.early_stop = bool(early_stop)
config.tol_h = float(tol_h)
config.tol_rhs = float(tol_rhs)
config.min_iter = int(min_iter)
config.consecutive_passes_required = int(consecutive_passes_required)

with st.spinner("Solving HJB in C++ and building policies..."):
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

    st.caption(
        f"Inventory grid: {len(config.q_grid)} points, "
        f"center index = {len(config.q_grid)//2}, "
        f"center q = {config.q_grid[len(config.q_grid)//2]:.6f}"
    )

    st.plotly_chart(make_h_figure(solution), use_container_width=True)
    st.plotly_chart(make_convergence_h_figure(solution), use_container_width=True)
    st.plotly_chart(make_convergence_rhs_figure(solution), use_container_width=True)

tab_names = [f"{spec.name} [{tier_kind_label(spec)}]" for spec in tier_specs]
tabs = st.tabs(tab_names)

available_q = [float(q) for q in solution.q_grid]

for tab, spec, cpp_tier in zip(tabs, tier_specs, solution.tiers):
    with tab:
        st.subheader(f"Tier: {spec.name}")
        st.caption(f"Type: {tier_kind_label(spec)}")

        if isinstance(spec, ECNTierSpec) and has_ecn_features(cpp_tier):
            c1, c2, c3 = st.columns(3)
            c1.metric("quoted size", "1.0")
            c2.metric("delta_start", f"{float(cpp_tier.ecn_policy.delta_start):.4f}")
            c3.metric("optimized decay", f"{float(cpp_tier.ecn_policy.decay):.4f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("delta_target", f"{float(cpp_tier.ecn_policy.delta_target):.4f}")
            c5.metric("decay min", f"{float(cpp_tier.ecn_policy.decay_min):.4f}")
            c6.metric("decay max", f"{float(cpp_tier.ecn_policy.decay_max):.4f}")

            st.info(
                "This ECN tier is a one-size hedge channel. "
                "Only the inventory-reducing side is active, and its quote follows a smooth exponential function of |q|."
            )

        with st.expander("Tier parameters", expanded=False):
            st.dataframe(make_flow_parameter_table(spec, cpp_tier), use_container_width=True)

        st.plotly_chart(make_flow_curve_figure(cpp_tier, spec), use_container_width=True)

        if isinstance(spec, ECNTierSpec) and has_ecn_features(cpp_tier):
            st.plotly_chart(
                make_ecn_decay_figure(cpp_tier, float(config.spread)),
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
            make_q_ladder_table(cpp_tier, spec, float(q_for_table), float(config.spread), float(mid_price)),
            use_container_width=True,
        )

with st.expander("What this app is solving"):
    st.markdown(
        r"""
MDP tiers use rung-by-rung two-sided ladder optimization.

ECN tiers are different:

- the quoted size is fixed to \(z = 1\)
- only the inventory-reducing side is admissible
- the active ECN quote is parameterized as an exponential function of \(|q|\)
- the decay parameter is optimized in C++ against the current value function \(h(q)\)

So ECN is treated as a smooth hedge channel rather than a full ladder optimization problem.

The inventory grid must be:

- strictly increasing
- symmetric around 0
- odd-length
- with \(0\) exactly at the middle index
        """
    )

st.caption("After replacing the C++ files, rebuild the extension and restart Streamlit.")