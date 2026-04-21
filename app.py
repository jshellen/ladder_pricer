from __future__ import annotations

from dataclasses import dataclass

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


def validate_centered_q_grid(q_grid: np.ndarray) -> list[str]:
    errors: list[str] = []
    if len(q_grid) < 3:
        errors.append("q_grid must contain at least 3 points.")
        return errors
    if len(q_grid) % 2 == 0:
        errors.append("q_grid must have odd length.")
    mid_idx = len(q_grid) // 2
    if not np.isclose(q_grid[mid_idx], 0.0, atol=1e-10):
        errors.append("q_grid must contain 0 exactly at the middle index.")
    for i in range(mid_idx):
        if not np.isclose(q_grid[i] + q_grid[-1 - i], 0.0, atol=1e-10):
            errors.append("q_grid must be symmetric around 0.")
            break
    return errors


# ============================================================
# Tier specs
# ============================================================

DEFAULT_MDP_SIZES = "1, 2, 3, 5, 10, 20"
DEFAULT_ECN_SETTINGS = {
    "ecn_toxicity": 0.0040,
    "ecn_fee": 0.0000,
    "ecn_convergence_inventory": 4.0,
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
    ecn_toxicity: float
    ecn_fee: float
    ecn_convergence_inventory: float

    def build_cpp_tier(self) -> lp.ECNTier:
        flow, markout = self.build_models()
        adverse_selection = lp.ECNAdverseSelectionModel(
            ecn_toxicity=float(self.ecn_toxicity),
            ecn_fee=float(self.ecn_fee),
            ecn_convergence_inventory=float(self.ecn_convergence_inventory),
        )
        return lp.ECNTier(
            name=self.name,
            flow_curve=flow,
            markout_model=markout,
            delta_min=float(self.delta_min),
            delta_max=float(self.delta_max),
            adverse_selection_model=adverse_selection,
        )


TierSpec = MDPTierSpec | ECNTierSpec
CppTier = lp.MDPTier | lp.ECNTier


def tier_kind_label(spec: TierSpec) -> str:
    return "ECN" if isinstance(spec, ECNTierSpec) else "MDP"


def tier_sizes(spec: TierSpec) -> list[float]:
    return [1.0] if isinstance(spec, ECNTierSpec) else spec.sizes


def parse_float_list(raw: str, field_name: str) -> list[float]:
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
            **DEFAULT_ECN_SETTINGS,
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

        sizes: list[float] | None = None
        if kind == "mdp":
            sizes_raw = st.text_input(
                f"Sizes {i + 1}",
                value=defaults.get("sizes", DEFAULT_MDP_SIZES),
                key=f"sizes_{i}",
            )
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

        ecn_toxicity = ecn_fee = ecn_convergence_inventory = None
        if kind == "ecn":
            ecn_defaults = {
                key: float(defaults.get(key, DEFAULT_ECN_SETTINGS[key]))
                for key in DEFAULT_ECN_SETTINGS
            }
            st.markdown("**ECN settings**")
            c9, c10 = st.columns(2)
            ecn_toxicity = c9.number_input(
                f"ECN toxicity {i + 1}",
                value=ecn_defaults["ecn_toxicity"],
                step=0.0005,
                format="%.5f",
                key=f"ecn_toxicity_{i}",
            )
            ecn_fee = c10.number_input(
                f"ECN fee {i + 1}",
                value=ecn_defaults["ecn_fee"],
                step=0.0005,
                format="%.5f",
                key=f"ecn_fee_{i}",
            )
            ecn_convergence_inventory = st.number_input(
                f"ECN convergence inventory {i + 1}",
                value=ecn_defaults["ecn_convergence_inventory"],
                step=0.25,
                format="%.4f",
                key=f"ecn_convergence_inventory_{i}",
            )
            st.caption(
                "ECN stays one-sided and passive. The quote follows a smooth built-in inventory profile: it starts at delta_min and reaches the mid cap exactly when |q| reaches the convergence inventory."
            )

    if flow_A0 < 0.0:
        raise ValueError(f"A0 for tier {i + 1} must be nonnegative.")
    if flow_steepness <= 0.0:
        raise ValueError(f"steepness for tier {i + 1} must be positive.")
    if tier_delta_max <= tier_delta_min:
        raise ValueError(f"Tier {i + 1}: delta_max must be greater than delta_min.")

    common = dict(
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
        assert sizes is not None
        if any(z <= 0.0 for z in sizes):
            raise ValueError(f"Tier {i + 1}: all sizes must be positive.")
        if any(sizes[k] >= sizes[k + 1] for k in range(len(sizes) - 1)):
            raise ValueError(f"Tier {i + 1}: sizes must be strictly increasing.")
        return MDPTierSpec(sizes=sizes, **common)

    assert ecn_toxicity is not None and ecn_fee is not None and ecn_convergence_inventory is not None
    if ecn_toxicity < 0.0:
        raise ValueError(f"Tier {i + 1}: ECN toxicity must be nonnegative.")
    if ecn_fee < 0.0:
        raise ValueError(f"Tier {i + 1}: ECN fee must be nonnegative.")
    if ecn_convergence_inventory <= 0.0:
        raise ValueError(f"Tier {i + 1}: ECN convergence inventory must be positive.")
    if tier_delta_min >= min(tier_delta_max, 0.5):
        raise ValueError(
            f"Tier {i + 1}: ECN delta_min must be below min(delta_max, 0.5)."
        )

    return ECNTierSpec(
        ecn_toxicity=float(ecn_toxicity),
        ecn_fee=float(ecn_fee),
        ecn_convergence_inventory=float(ecn_convergence_inventory),
        **common,
    )


# ============================================================
# Solver helpers
# ============================================================

def build_solver_config(
    q_grid: np.ndarray,
    dt: float,
    n_iter: int,
    spread: float,
    spot_drift: float,
    golden_tol: float,
    golden_max_iter: int,
    early_stop: bool,
    tol_h: float,
    tol_rhs: float,
    min_iter: int,
    consecutive_passes_required: int,
) -> lp.SolverConfig:
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
    return config


def build_cpp_tiers(specs: list[TierSpec]) -> tuple[list[lp.MDPTier], list[lp.ECNTier]]:
    mdp_tiers: list[lp.MDPTier] = []
    ecn_tiers: list[lp.ECNTier] = []
    for spec in specs:
        cpp_tier = spec.build_cpp_tier()
        if isinstance(spec, MDPTierSpec):
            mdp_tiers.append(cpp_tier)
        else:
            ecn_tiers.append(cpp_tier)
    return mdp_tiers, ecn_tiers


def ordered_solution_tiers(specs: list[TierSpec], solution: lp.HJBSolution) -> list[CppTier]:
    mdp_iter = iter(solution.mdp_tiers)
    ecn_iter = iter(solution.ecn_tiers)
    out: list[CppTier] = []
    for spec in specs:
        out.append(next(ecn_iter) if isinstance(spec, ECNTierSpec) else next(mdp_iter))
    return out


def total_tier_count(solution: lp.HJBSolution) -> int:
    return len(solution.mdp_tiers) + len(solution.ecn_tiers)


def is_ecn_tier(cpp_tier: CppTier) -> bool:
    return isinstance(cpp_tier, lp.ECNTier)


def q_index_lookup(cpp_tier: CppTier) -> dict[float, int]:
    return {float(q): i for i, q in enumerate(cpp_tier.policy.q_grid)}


def is_structurally_admissible(cpp_tier: CppTier, q: float, z: float, side: str) -> bool:
    return bool(cpp_tier.is_admissible(float(q), float(z), side))


def is_policy_active(cpp_tier: CppTier, q_index: int, side: str) -> bool:
    if is_ecn_tier(cpp_tier):
        return bool(cpp_tier.is_policy_active(int(q_index), side))
    return True


def masked_quote_summary(
    cpp_tier: CppTier,
    q_index: int,
    q: float,
    z: float,
    side: str,
    mid_price: float,
    spread: float,
):
    if not is_structurally_admissible(cpp_tier, q, z, side):
        return None
    if not is_policy_active(cpp_tier, q_index, side):
        return None
    return cpp_tier.quote_summary(float(q), float(z), side, float(mid_price), float(spread))


# ============================================================
# Plots and tables
# ============================================================

def make_flow_parameter_table(spec: TierSpec, cpp_tier: CppTier) -> pd.DataFrame:
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
        if isinstance(spec, ECNTierSpec) and is_ecn_tier(cpp_tier):
            row |= {
                "ecn_toxicity": float(cpp_tier.adverse_selection_model.ecn_toxicity),
                "ecn_fee": float(cpp_tier.adverse_selection_model.ecn_fee),
                "ecn_convergence_inventory": float(cpp_tier.adverse_selection_model.ecn_convergence_inventory),
                "ecn_mid_cap": float(cpp_tier.delta_mid_cap()),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def make_flow_curve_figure(cpp_tier: CppTier, spec: TierSpec) -> go.Figure:
    grid = np.linspace(-1.0, 1.0, 150)
    fig = go.Figure()
    for z in tier_sizes(spec):
        zf = float(z)
        vals = [cpp_tier.arrival_rate(float(d), zf) for d in grid]
        fig.add_trace(go.Scatter(x=grid, y=vals, mode="lines", name=f"{zf:g}"))
    fig.update_layout(
        title=f"Flow curves λ(δ, z) — {spec.name} ({tier_kind_label(spec)})",
        xaxis_title="delta",
        yaxis_title="arrival rate",
        height=480,
    )
    return fig


def make_quote_inventory_figure(cpp_tier: CppTier, spec: TierSpec, spread: float, mid_price: float) -> go.Figure:
    q_grid = [float(q) for q in cpp_tier.policy.q_grid]
    fig = go.Figure()
    for z in tier_sizes(spec):
        zf = float(z)
        bid_vals: list[float] = []
        ask_vals: list[float] = []
        for q_index, q in enumerate(q_grid):
            bid = masked_quote_summary(cpp_tier, q_index, q, zf, "bid", mid_price, spread)
            ask = masked_quote_summary(cpp_tier, q_index, q, zf, "ask", mid_price, spread)
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
        height=500,
    )
    return fig


def make_ladder_figure(cpp_tier: CppTier, spec: TierSpec, q: float, spread: float, mid_price: float) -> go.Figure:
    q_idx = q_index_lookup(cpp_tier)[float(q)]
    sizes = [float(z) for z in tier_sizes(spec)]
    bid_vp, ask_vp = [], []
    for z in sizes:
        bid = masked_quote_summary(cpp_tier, q_idx, q, z, "bid", mid_price, spread)
        ask = masked_quote_summary(cpp_tier, q_idx, q, z, "ask", mid_price, spread)
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
        height=480,
    )
    return fig


def make_q_ladder_table(cpp_tier: CppTier, spec: TierSpec, q: float, spread: float, mid_price: float) -> pd.DataFrame:
    q_idx = q_index_lookup(cpp_tier)[float(q)]
    rows = []
    for z in tier_sizes(spec):
        zf = float(z)
        bid = masked_quote_summary(cpp_tier, q_idx, q, zf, "bid", mid_price, spread)
        ask = masked_quote_summary(cpp_tier, q_idx, q, zf, "ask", mid_price, spread)
        rows.append(
            {
                "type": tier_kind_label(spec),
                "q": float(q),
                "z": zf,
                "Bid active": bid is not None,
                "Ask active": ask is not None,
                "Bid delta": np.nan if bid is None else round(float(bid.delta), 6),
                "Ask delta": np.nan if ask is None else round(float(ask.delta), 6),
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
    fig.add_trace(
        go.Scatter(x=list(solution.q_grid), y=list(solution.h), mode="lines+markers", name="h(q)")
    )
    fig.update_layout(title="Value function h(q)", xaxis_title="Inventory q", yaxis_title="h(q)", height=440)
    return fig


def make_convergence_figure(values: list[float], title: str, yaxis_title: str) -> go.Figure:
    fig = go.Figure()
    if values:
        fig.add_trace(go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode="lines+markers"))
    fig.update_layout(title=title, xaxis_title="Iteration", yaxis_title=yaxis_title, yaxis_type="log", height=380)
    return fig


def make_ecn_cap_figure(cpp_tier: lp.ECNTier, spread: float) -> go.Figure:
    q_grid = [float(q) for q in cpp_tier.policy.q_grid]
    fig = go.Figure()

    bid_actual: list[float] = []
    ask_actual: list[float] = []
    bid_cap: list[float] = []
    ask_cap: list[float] = []

    for i, q in enumerate(q_grid):
        cap = float(cpp_tier.delta_cap(abs(q)))
        if q < 0.0:
            bid_cap.append(10000.0 * spread * (cap - 0.5))
            ask_cap.append(np.nan)
            bid_actual.append(np.nan if not cpp_tier.is_policy_active(i, "bid") else 10000.0 * spread * (float(cpp_tier.policy.bid[i][0]) - 0.5))
            ask_actual.append(np.nan)
        elif q > 0.0:
            bid_cap.append(np.nan)
            ask_cap.append(10000.0 * spread * (0.5 - cap))
            bid_actual.append(np.nan)
            ask_actual.append(np.nan if not cpp_tier.is_policy_active(i, "ask") else 10000.0 * spread * (0.5 - float(cpp_tier.policy.ask[i][0])))
        else:
            bid_cap.append(np.nan)
            ask_cap.append(np.nan)
            bid_actual.append(np.nan)
            ask_actual.append(np.nan)

    fig.add_trace(go.Scatter(x=q_grid, y=bid_cap, mode="lines", line=dict(dash="dot"), name="Bid cap"))
    fig.add_trace(go.Scatter(x=q_grid, y=ask_cap, mode="lines", line=dict(dash="dot"), name="Ask cap"))
    fig.add_trace(go.Scatter(x=q_grid, y=bid_actual, mode="lines+markers", name="Bid quote"))
    fig.add_trace(go.Scatter(x=q_grid, y=ask_actual, mode="lines+markers", name="Ask quote"))
    fig.add_hline(y=0.0)
    fig.update_layout(
        title="ECN profile and quote vs inventory",
        xaxis_title="Inventory q",
        yaxis_title="Quote relative to mid (pips)",
        height=480,
    )
    return fig


def make_ecn_activity_table(cpp_tier: lp.ECNTier) -> pd.DataFrame:
    rows = []
    for i, q in enumerate(cpp_tier.policy.q_grid):
        qf = float(q)
        rows.append(
            {
                "q": qf,
                "delta_profile": float(cpp_tier.delta_cap(abs(qf))),
                "bid_active": bool(cpp_tier.is_policy_active(i, "bid")),
                "ask_active": bool(cpp_tier.is_policy_active(i, "ask")),
                "bid_delta": np.nan if qf >= 0.0 else float(cpp_tier.policy.bid[i][0]),
                "ask_delta": np.nan if qf <= 0.0 else float(cpp_tier.policy.ask[i][0]),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# App
# ============================================================

st.set_page_config(page_title="Trinity 2.0 Pricer", layout="wide")
st.title("Trinity 2.0 Pricer")
st.markdown(
    "MDP tiers use rung-by-rung two-sided ladder controls. "
    "ECN tiers quote only size z = 1 on the inventory-reducing side. "
    "Their quote is a direct smooth cap-profile in inventory: it starts at delta_min, reaches the mid cap exactly at the convergence inventory, and never crosses mid."
)

with st.sidebar:
    st.header("Global parameters")

    q_grid_mode = st.selectbox("Inventory grid mode", options=["uniform", "piecewise"], index=1)
    q_abs_max = st.number_input("max |q|", value=20.0, min_value=0.5, step=0.5, format="%.4f")

    q_step = fine_half_width = fine_step = coarse_step = None
    if q_grid_mode == "uniform":
        q_step = st.number_input("q step", value=1.0, min_value=0.01, step=0.1, format="%.4f")
    else:
        fine_half_width = st.number_input("fine half-width", value=3.0, min_value=0.0, step=0.5, format="%.4f")
        c_grid1, c_grid2 = st.columns(2)
        fine_step = c_grid1.number_input("fine step", value=0.25, min_value=0.01, step=0.05, format="%.4f")
        coarse_step = c_grid2.number_input("coarse step", value=1.0, min_value=0.01, step=0.1, format="%.4f")

    dt = st.number_input("dt", value=0.002, step=0.001, format="%.4f")
    n_iter = st.number_input("n_iter", value=140, step=10, min_value=1)
    spread = st.number_input("reference spread", value=20.0 / 10000.0, step=1.0 / 10000.0, format="%.6f")
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
    tau1 = st.number_input("cubic coeff", value=0.1, step=0.01, format="%.4f")
    tau2 = st.number_input("quartic coeff", value=0.0015, step=0.0005, format="%.5f")

    st.header("Pricing tiers")
    num_tiers = st.slider("Number of tiers", min_value=1, max_value=4, value=3)

errors: list[str] = []
tier_specs: list[TierSpec] = []
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
    q_grid = (
        build_uniform_centered_q_grid(float(q_abs_max), float(q_step))
        if q_grid_mode == "uniform"
        else build_piecewise_centered_q_grid(
            q_abs_max=float(q_abs_max),
            fine_half_width=float(fine_half_width),
            fine_step=float(fine_step),
            coarse_step=float(coarse_step),
        )
    )
except ValueError as exc:
    errors.append(str(exc))
    q_grid = np.array([], dtype=float)

errors.extend(validate_centered_q_grid(q_grid))

if errors:
    for error in errors:
        st.error(error)
    st.stop()

mdp_cpp_tiers, ecn_cpp_tiers = build_cpp_tiers(tier_specs)

penalty = lp.PolynomialInventoryPenalty(
    risk_aversion=float(risk_aversion),
    sigma=float(sigma),
    tau0=float(tau0),
    tau1=float(tau1),
    tau2=float(tau2),
)

config = build_solver_config(
    q_grid=q_grid,
    dt=float(dt),
    n_iter=int(n_iter),
    spread=float(spread),
    spot_drift=float(spot_drift),
    golden_tol=float(golden_tol),
    golden_max_iter=int(golden_max_iter),
    early_stop=bool(early_stop),
    tol_h=float(tol_h),
    tol_rhs=float(tol_rhs),
    min_iter=int(min_iter),
    consecutive_passes_required=int(consecutive_passes_required),
)

with st.spinner("Solving HJB in C++ and building policies..."):
    solver = lp.HJBLadderSolver(
        config=config,
        penalty=penalty,
        mdp_tiers=mdp_cpp_tiers,
        ecn_tiers=ecn_cpp_tiers,
    )
    solution: lp.HJBSolution = solver.solve()

solution_tiers = ordered_solution_tiers(tier_specs, solution)
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
    c2.metric("tiers", total_tier_count(solution))
    c3.metric("iterations used", diag.iterations_used)
    c4.metric("converged", "yes" if diag.converged else "no")
    c5.metric("spot drift", f"{config.spot_drift:.5f}")
    c6.metric("spread", f"{config.spread:.6f}")

    st.plotly_chart(make_h_figure(solution), use_container_width=True)
    st.plotly_chart(
        make_convergence_figure(list(diag.history_max_h_change), "Convergence: max |Δh|", "max |Δh|"),
        use_container_width=True,
    )
    st.plotly_chart(
        make_convergence_figure(list(diag.history_max_rhs), "Convergence: max |rhs|", "max |rhs|"),
        use_container_width=True,
    )

tabs = st.tabs([f"{spec.name} [{tier_kind_label(spec)}]" for spec in tier_specs])
available_q = [float(q) for q in solution.q_grid]
default_q = 0.0 if 0.0 in available_q else available_q[len(available_q) // 2]

for idx, (tab, spec, cpp_tier) in enumerate(zip(tabs, tier_specs, solution_tiers)):
    with tab:
        st.subheader(f"Tier: {spec.name}")
        st.caption(f"Type: {tier_kind_label(spec)}")

        if isinstance(spec, ECNTierSpec) and is_ecn_tier(cpp_tier):
            c1, c2, c3 = st.columns(3)
            c1.metric("quoted size", "1.0")
            c2.metric("ECN toxicity", f"{float(cpp_tier.adverse_selection_model.ecn_toxicity):.5f}")
            c3.metric("ECN fee", f"{float(cpp_tier.adverse_selection_model.ecn_fee):.5f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("convergence inventory", f"{float(cpp_tier.adverse_selection_model.ecn_convergence_inventory):.3f}")
            c5.metric("delta_min", f"{float(cpp_tier.delta_min):.4f}")
            c6.metric("mid cap", f"{float(cpp_tier.delta_mid_cap()):.4f}")

            st.info(
                "ECN quotes only the inventory-reducing side. The quote itself follows a smooth built-in inventory profile, reaching the passive mid cap exactly when |q| reaches the convergence inventory."
            )

        with st.expander("Tier parameters", expanded=False):
            st.dataframe(make_flow_parameter_table(spec, cpp_tier), use_container_width=True)

        st.plotly_chart(make_flow_curve_figure(cpp_tier, spec), use_container_width=True)

        if isinstance(spec, ECNTierSpec) and is_ecn_tier(cpp_tier):
            st.plotly_chart(make_ecn_cap_figure(cpp_tier, float(config.spread)), use_container_width=True)

        if isinstance(spec, ECNTierSpec) and is_ecn_tier(cpp_tier):
            pass
        else:
            st.plotly_chart(
                make_quote_inventory_figure(cpp_tier, spec, float(config.spread), float(mid_price)),
                use_container_width=True,
            )

            q_for_ladder = st.select_slider(
                f"Inventory level for ladder — {spec.name}",
                options=available_q,
                value=default_q,
                key=f"qslider_{idx}",
            )

            st.plotly_chart(
                make_ladder_figure(cpp_tier, spec, float(q_for_ladder), float(config.spread), float(mid_price)),
                use_container_width=True,
            )

            q_for_table = st.selectbox(
                f"q for ladder table — {spec.name}",
                options=available_q,
                index=available_q.index(default_q),
                key=f"qtable_{idx}",
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
- the quote itself follows a smooth built-in profile in \(|q|\)
- the profile starts at `delta_min` and reaches the passive mid cap exactly at `ecn_convergence_inventory`
- toxicity and fee still affect the ECN Bellman contribution once the quote is used

So ECN is treated as a simple parametric hedge curve rather than a fully optimized ladder.
        """
    )

st.caption("After replacing the C++ files, rebuild the extension and restart Streamlit.")
