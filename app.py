from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

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
# Defaults
# ============================================================

DEFAULT_MDP_SIZES = "1, 2, 3, 5, 10, 20"
DEFAULT_ECN_SIZES = "1"

DEFAULT_ECN_SETTINGS = {
    "ecn_toxicity": 0.0040,
    "ecn_fee": 0.0,
    "ecn_convergence_inventory": 10.0,
}

DEFAULT_ECN_PASSIVE_IMPACT_SETTINGS = {
    "enabled": False,
    "eta": 0.0,
    "pressure_scale": 1.0,
}

DEFAULT_DARK_POOL_SETTINGS = {
    "enabled": False,
    "arrival_lambda_bid": 2.00,
    "arrival_lambda_ask": 2.00,
    "size_lambda_bid": 2.00,
    "size_lambda_ask": 2.00,
    "fee_per_unit_bid": 0.0,
    "fee_per_unit_ask": 0.0,
    "posted_sizes": "1, 2, 3, 5",
    "allow_both_sides": False,
}


# ============================================================
# Specifications
# ============================================================

@dataclass
class CommonTierSpec:
    enabled: bool
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

    def build_models(self) -> tuple[Any, Any]:
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
    def build_cpp_tier(self) -> Any:
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

    def build_cpp_tier(self) -> Any:
        flow, markout = self.build_models()
        adverse_selection = lp.ECNAdverseSelectionModel(
            ecn_toxicity=float(self.ecn_toxicity),
            ecn_fee=float(self.ecn_fee),
            ecn_convergence_inventory=float(self.ecn_convergence_inventory),
        )
        return lp.ECNTier(
            name=self.name,
            sizes=[float(z) for z in self.sizes],
            flow_curve=flow,
            markout_model=markout,
            delta_min=float(self.delta_min),
            delta_max=float(self.delta_max),
            adverse_selection_model=adverse_selection,
        )


@dataclass
class DarkPoolSpec:
    arrival_lambda_bid: float
    arrival_lambda_ask: float
    size_lambda_bid: float
    size_lambda_ask: float
    fee_per_unit_bid: float
    fee_per_unit_ask: float
    posted_sizes: list[float]
    allow_both_sides: bool

    def build_cpp_venue(self) -> Any:
        return lp.DarkPoolVenue(
            arrival_lambda_bid=float(self.arrival_lambda_bid),
            arrival_lambda_ask=float(self.arrival_lambda_ask),
            size_lambda_bid=float(self.size_lambda_bid),
            size_lambda_ask=float(self.size_lambda_ask),
            fee_per_unit_bid=float(self.fee_per_unit_bid),
            fee_per_unit_ask=float(self.fee_per_unit_ask),
            posted_sizes=[float(u) for u in self.posted_sizes],
            allow_both_sides=bool(self.allow_both_sides),
        )


@dataclass
class ECNPassiveImpactSpec:
    enabled: bool
    eta: float
    pressure_scale: float

    def build_cpp_model(self) -> Any:
        return lp.ECNPassiveImpactModel(
            enabled=bool(self.enabled),
            eta=float(self.eta),
            pressure_scale=float(self.pressure_scale),
        )


TierSpec = MDPTierSpec | ECNTierSpec


# ============================================================
# Parsing / validation helpers
# ============================================================

def tier_kind_label(spec: TierSpec) -> str:
    return "ECN" if isinstance(spec, ECNTierSpec) else "MDP"


def tier_sizes(spec: TierSpec) -> list[float]:
    return [float(z) for z in spec.sizes]


def parse_float_list(raw: str, field_name: str) -> list[float]:
    try:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Could not parse {field_name}. Use comma-separated numbers.") from exc

    if not vals:
        raise ValueError(f"{field_name} cannot be empty.")
    return vals


def try_parse_float_list(raw: str) -> list[float] | None:
    try:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        return None
    return vals or None


def parse_integer_like_list(raw: str, field_name: str) -> list[float]:
    vals = parse_float_list(raw, field_name)
    for v in vals:
        if abs(v - round(v)) > 1e-10:
            raise ValueError(f"{field_name} must contain integer-valued sizes only.")
    return vals


def validate_positive_strictly_increasing(vals: list[float], field_name: str) -> None:
    if any(v <= 0.0 for v in vals):
        raise ValueError(f"{field_name}: all values must be positive.")
    if any(vals[k] >= vals[k + 1] for k in range(len(vals) - 1)):
        raise ValueError(f"{field_name}: values must be strictly increasing.")


def zero_truncated_poisson_mean(size_lambda: float) -> float:
    if size_lambda <= 0.0:
        return np.nan

    positive_mass = 1.0 - math.exp(-size_lambda)
    if positive_mass <= 0.0:
        return np.nan

    return size_lambda / positive_mass


def zero_truncated_poisson_pmf(size_lambda: float, k_max: int) -> pd.DataFrame:
    if size_lambda <= 0.0 or k_max < 1:
        return pd.DataFrame(columns=["k", "probability"])

    positive_mass = 1.0 - math.exp(-size_lambda)
    if positive_mass <= 0.0:
        return pd.DataFrame(columns=["k", "probability"])

    rows = []
    log_lambda = math.log(size_lambda)

    for k in range(1, k_max + 1):
        log_pk = -size_lambda + k * log_lambda - math.lgamma(k + 1)
        rows.append(
            {
                "k": k,
                "probability": math.exp(log_pk) / positive_mass,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# UI specs
# ============================================================

def default_tier_values(i: int) -> dict:
    presets = [
        {
            "enabled": True,
            "kind": "mdp",
            "name": "Tier 1",
            "sizes": DEFAULT_MDP_SIZES,
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
            "enabled": True,
            "kind": "mdp",
            "name": "Tier 2",
            "sizes": DEFAULT_MDP_SIZES,
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
            "enabled": False,
            "kind": "ecn",
            "name": "ecn",
            "sizes": DEFAULT_ECN_SIZES,
            "flow_A0": 0.70,
            "flow_theta": 0.10,
            "flow_steepness": 12.0,
            "flow_shift": 0.4,
            "flow_volume_shift": 0.0,
            "markout_base": 0.000,
            "markout_coeff": 0.000,
            "delta_min": -2.0,
            "delta_max": 2.0,
            **DEFAULT_ECN_SETTINGS,
        },
        {
            "enabled": False,
            "kind": "mdp",
            "name": "Tier 3",
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


def enabled_tier_specs(specs: list[TierSpec]) -> list[TierSpec]:
    return [spec for spec in specs if spec.enabled]


def build_tier_spec_from_ui(i: int) -> TierSpec:
    defaults = default_tier_values(i)

    with st.sidebar.expander(f"Tier {i + 1}", expanded=(i == 0)):
        enabled = st.checkbox(
            f"Enable tier {i + 1}",
            value=bool(defaults.get("enabled", True)),
            key=f"enabled_{i}",
        )

        name = st.text_input(f"Tier name {i + 1}", value=defaults["name"], key=f"name_{i}")
        kind = st.selectbox(
            f"Tier type {i + 1}",
            options=["mdp", "ecn"],
            index=0 if defaults["kind"] == "mdp" else 1,
            key=f"kind_{i}",
        )

        sizes_raw = st.text_input(
            f"Sizes {i + 1}",
            value=defaults.get("sizes", DEFAULT_MDP_SIZES if kind == "mdp" else DEFAULT_ECN_SIZES),
            key=f"sizes_{i}",
            help="ECN now uses the same ladder-style size vector as MDP, according to the current bindings.",
        )

        if enabled:
            sizes = parse_float_list(sizes_raw, f"sizes for tier {i + 1}")
        else:
            sizes = try_parse_float_list(sizes_raw) or [1.0]

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

        st.markdown("**Base markout parameters**")
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

            st.markdown("**ECN controls**")
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
                f"ECN inventory scale {i + 1}",
                value=ecn_defaults["ecn_convergence_inventory"],
                step=0.25,
                format="%.4f",
                key=f"ecn_convergence_inventory_{i}",
            )

            st.caption(
                "ECN now uses the same ladder-shape policy interface as MDP, with ECN-specific "
                "adverse-selection costs and optional passive market impact."
            )

        if not enabled:
            st.caption("Disabled tiers stay editable in the sidebar but are excluded from the solver and output tabs.")

    common = dict(
        enabled=bool(enabled),
        name=name,
        sizes=[float(z) for z in sizes],
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

    if not enabled:
        if kind == "mdp":
            return MDPTierSpec(**common)

        return ECNTierSpec(
            ecn_toxicity=float(ecn_toxicity or 0.0),
            ecn_fee=float(ecn_fee or 0.0),
            ecn_convergence_inventory=float(ecn_convergence_inventory or 1.0),
            **common,
        )

    if flow_A0 < 0.0:
        raise ValueError(f"A0 for tier {i + 1} must be nonnegative.")
    if flow_steepness <= 0.0:
        raise ValueError(f"steepness for tier {i + 1} must be positive.")
    if tier_delta_max <= tier_delta_min:
        raise ValueError(f"Tier {i + 1}: delta_max must be greater than delta_min.")

    validate_positive_strictly_increasing(sizes, f"Tier {i + 1} sizes")

    if kind == "mdp":
        return MDPTierSpec(**common)

    assert ecn_toxicity is not None
    assert ecn_fee is not None
    assert ecn_convergence_inventory is not None

    if ecn_toxicity < 0.0:
        raise ValueError(f"Tier {i + 1}: ECN toxicity must be nonnegative.")
    if ecn_fee < 0.0:
        raise ValueError(f"Tier {i + 1}: ECN fee must be nonnegative.")
    if ecn_convergence_inventory <= 0.0:
        raise ValueError(f"Tier {i + 1}: ECN inventory scale must be positive.")

    return ECNTierSpec(
        ecn_toxicity=float(ecn_toxicity),
        ecn_fee=float(ecn_fee),
        ecn_convergence_inventory=float(ecn_convergence_inventory),
        **common,
    )


def build_ecn_passive_impact_spec_from_ui() -> ECNPassiveImpactSpec:
    defaults = DEFAULT_ECN_PASSIVE_IMPACT_SETTINGS

    with st.sidebar.expander("ECN passive market impact", expanded=False):
        enabled = st.checkbox(
            "Enable ECN passive impact",
            value=bool(defaults["enabled"]),
            help="Only active ECN quotes create passive market pressure. MDP tiers and dark pool are excluded.",
        )

        eta = st.number_input(
            "Impact drift cap η",
            value=float(defaults["eta"]),
            min_value=0.0,
            step=0.0001,
            format="%.6f",
            help="Maximum absolute passive drift contribution from ECN pressure.",
        )

        pressure_scale = st.number_input(
            "Pressure scale",
            value=float(defaults["pressure_scale"]),
            min_value=1e-8,
            step=0.10,
            format="%.6f",
            help="Net ECN pressure scale used in η * tanh(net_pressure / pressure_scale).",
        )

        st.caption(
            "Passive impact drift is η · tanh((ECN bid pressure − ECN ask pressure) / pressure_scale). "
            "This drift is caused only by ECN quotes. MDP customer tiers and dark pool are excluded."
        )

    return ECNPassiveImpactSpec(
        enabled=bool(enabled),
        eta=float(eta),
        pressure_scale=float(pressure_scale),
    )


def build_dark_pool_spec_from_ui() -> DarkPoolSpec | None:
    defaults = DEFAULT_DARK_POOL_SETTINGS

    with st.sidebar.expander("Dark pool venue", expanded=True):
        enabled = st.checkbox("Enable dark pool", value=bool(defaults["enabled"]))
        if not enabled:
            st.caption("Disabled venues are excluded from the solver and output tabs.")
            return None

        allow_both_sides = st.checkbox(
            "Allow both sides simultaneously",
            value=bool(defaults["allow_both_sides"]),
            help="If disabled, the dark pool only posts the inventory-reducing side.",
        )

        st.markdown("**Opportunity arrival intensity**")
        c1, c2 = st.columns(2)
        arrival_lambda_bid = c1.number_input(
            "Arrival λ bid",
            value=float(defaults["arrival_lambda_bid"]),
            min_value=0.0,
            step=0.01,
            format="%.4f",
            help="Poisson intensity of bid-side dark-pool opportunities.",
        )
        arrival_lambda_ask = c2.number_input(
            "Arrival λ ask",
            value=float(defaults["arrival_lambda_ask"]),
            min_value=0.0,
            step=0.01,
            format="%.4f",
            help="Poisson intensity of ask-side dark-pool opportunities.",
        )

        st.markdown("**Incoming size distribution**")
        c3, c4 = st.columns(2)
        size_lambda_bid = c3.number_input(
            "Size λ bid",
            value=float(defaults["size_lambda_bid"]),
            min_value=0.001,
            step=0.05,
            format="%.4f",
            help="Poisson lambda for incoming bid-side contra size, conditional on a dark-pool opportunity.",
        )
        size_lambda_ask = c4.number_input(
            "Size λ ask",
            value=float(defaults["size_lambda_ask"]),
            min_value=0.001,
            step=0.05,
            format="%.4f",
            help="Poisson lambda for incoming ask-side contra size, conditional on a dark-pool opportunity.",
        )

        c5, c6 = st.columns(2)
        fee_per_unit_bid = c5.number_input(
            "Fee per unit bid",
            value=float(defaults["fee_per_unit_bid"]),
            min_value=0.0,
            step=0.001,
            format="%.5f",
        )
        fee_per_unit_ask = c6.number_input(
            "Fee per unit ask",
            value=float(defaults["fee_per_unit_ask"]),
            min_value=0.0,
            step=0.001,
            format="%.5f",
        )

        posted_sizes_raw = st.text_input(
            "Allowed posted sizes",
            value=str(defaults["posted_sizes"]),
            help="Integer-valued posted sizes in the same inventory units as q.",
        )
        posted_sizes = parse_integer_like_list(posted_sizes_raw, "dark-pool posted sizes")

        validate_positive_strictly_increasing(posted_sizes, "Dark-pool posted sizes")

        st.caption(
            "Dark-pool fills occur at mid. Opportunity arrivals are Poisson. "
            "Conditional incoming size is zero-truncated Poisson, and executed size is "
            "min(posted size, incoming size)."
        )

    return DarkPoolSpec(
        arrival_lambda_bid=float(arrival_lambda_bid),
        arrival_lambda_ask=float(arrival_lambda_ask),
        size_lambda_bid=float(size_lambda_bid),
        size_lambda_ask=float(size_lambda_ask),
        fee_per_unit_bid=float(fee_per_unit_bid),
        fee_per_unit_ask=float(fee_per_unit_ask),
        posted_sizes=posted_sizes,
        allow_both_sides=bool(allow_both_sides),
    )


# ============================================================
# Solver / conversion helpers
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
) -> Any:
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


def build_cpp_tiers(specs: list[TierSpec]) -> tuple[list[Any], list[Any]]:
    mdp_tiers: list[Any] = []
    ecn_tiers: list[Any] = []

    for spec in specs:
        if not spec.enabled:
            continue

        cpp_tier = spec.build_cpp_tier()
        if isinstance(spec, MDPTierSpec):
            mdp_tiers.append(cpp_tier)
        else:
            ecn_tiers.append(cpp_tier)

    return mdp_tiers, ecn_tiers


def ordered_solution_tiers(specs: list[TierSpec], solution: Any) -> list[Any]:
    mdp_iter = iter(solution.mdp_tiers)
    ecn_iter = iter(solution.ecn_tiers)

    out: list[Any] = []
    for spec in specs:
        out.append(next(ecn_iter) if isinstance(spec, ECNTierSpec) else next(mdp_iter))
    return out


def total_venue_count(solution: Any) -> int:
    return len(solution.mdp_tiers) + len(solution.ecn_tiers) + (0 if solution.dark_pool is None else 1)


def q_index_for_policy_qgrid(q_grid: list[float] | np.ndarray, q: float) -> int:
    q_arr = np.asarray(list(q_grid), dtype=float)
    if q_arr.size == 0:
        raise ValueError("Policy q_grid is empty.")
    return int(np.argmin(np.abs(q_arr - float(q))))


def q_index_for_tier_policy(cpp_tier: Any, q: float) -> int:
    return q_index_for_policy_qgrid(cpp_tier.policy.q_grid, q)


def is_admissible(cpp_tier: Any, q: float, z: float, side: str) -> bool:
    return bool(cpp_tier.is_admissible(float(q), float(z), side))


def is_active(cpp_tier: Any, q: float, z: float, side: str) -> bool:
    # Current bindings no longer expose ECNTier::is_policy_active.
    # Activity is therefore represented by admissibility.
    return is_admissible(cpp_tier, q, z, side)


def masked_quote_summary(
    cpp_tier: Any,
    q: float,
    z: float,
    side: str,
    mid_price: float,
    spread: float,
):
    if not is_active(cpp_tier, q, z, side):
        return None
    return cpp_tier.quote_summary(float(q), float(z), side, float(mid_price), float(spread))


def is_ecn_tier(cpp_tier: Any) -> bool:
    return isinstance(cpp_tier, lp.ECNTier)


# ============================================================
# Tables and figures
# ============================================================

def make_flow_parameter_table(spec: TierSpec, cpp_tier: Any) -> pd.DataFrame:
    rows = []
    for z in tier_sizes(spec):
        zf = float(z)
        row = {
            "enabled": bool(spec.enabled),
            "type": tier_kind_label(spec),
            "z": zf,
            "A(z)": spec.flow_A0 * zf ** (-spec.flow_theta),
            "delta_50(z)": spec.flow_shift - spec.flow_volume_shift * (zf - 1.0),
            "steepness": spec.flow_steepness,
            "base_mu(z)": cpp_tier.expected_markout(zf),
        }
        if isinstance(spec, ECNTierSpec):
            row |= {
                "ecn_toxicity": float(cpp_tier.adverse_selection_model.ecn_toxicity),
                "ecn_fee": float(cpp_tier.adverse_selection_model.ecn_fee),
                "ecn_inventory_scale": float(
                    cpp_tier.adverse_selection_model.ecn_convergence_inventory
                ),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def make_flow_curve_figure(cpp_tier: Any, spec: TierSpec) -> go.Figure:
    grid = np.linspace(-1.0, 1.0, 160)
    fig = go.Figure()

    for z in tier_sizes(spec):
        zf = float(z)
        vals = [cpp_tier.arrival_rate(float(d), zf) for d in grid]
        fig.add_trace(go.Scatter(x=grid, y=vals, mode="lines", name=f"{zf:g}"))

    fig.update_layout(
        title=f"Flow curves λ(δ, z) — {spec.name} ({tier_kind_label(spec)})",
        xaxis_title="delta",
        yaxis_title="arrival rate",
        height=500,
    )
    return fig


def make_delta_implied_flow_table(cpp_tier: Any, spec: TierSpec) -> pd.DataFrame:
    rows = []
    q_grid = [float(q) for q in cpp_tier.policy.q_grid]
    sizes = [float(z) for z in tier_sizes(spec)]

    for i, q in enumerate(q_grid):
        for j, z in enumerate(sizes):
            for side in ["bid", "ask"]:
                admissible = is_admissible(cpp_tier, q, z, side)
                active = is_active(cpp_tier, q, z, side)

                delta = float(cpp_tier.policy.bid[i][j]) if side == "bid" else float(cpp_tier.policy.ask[i][j])

                if active:
                    a_z = float(cpp_tier.A(z))
                    hit_ratio = float(cpp_tier.hit_ratio(delta, z))
                    arrival_rate = float(cpp_tier.arrival_rate(delta, z))
                    pressure = z * arrival_rate
                else:
                    delta = np.nan
                    a_z = np.nan
                    hit_ratio = np.nan
                    arrival_rate = np.nan
                    pressure = np.nan

                rows.append(
                    {
                        "type": tier_kind_label(spec),
                        "q": q,
                        "z": z,
                        "side": side,
                        "admissible": admissible,
                        "active": active,
                        "delta": delta,
                        "A(z)": a_z,
                        "hit_ratio": hit_ratio,
                        "arrival_rate": arrival_rate,
                        "size_weighted_pressure": pressure,
                    }
                )

    return pd.DataFrame(rows)


def make_delta_implied_flow_figure(cpp_tier: Any, spec: TierSpec) -> go.Figure:
    df = make_delta_implied_flow_table(cpp_tier, spec)
    fig = go.Figure()

    for z in tier_sizes(spec):
        zf = float(z)
        bid_df = df[(df["z"] == zf) & (df["side"] == "bid")]
        ask_df = df[(df["z"] == zf) & (df["side"] == "ask")]

        fig.add_trace(
            go.Scatter(
                x=bid_df["q"],
                y=bid_df["arrival_rate"],
                mode="lines",
                name=f"{zf:g} bid λ(δ*)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ask_df["q"],
                y=ask_df["arrival_rate"],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{zf:g} ask λ(δ*)",
            )
        )

    fig.update_layout(
        title=f"Delta-implied flow rates — {spec.name} ({tier_kind_label(spec)})",
        xaxis_title="Inventory q",
        yaxis_title="Arrival rate λ(δ*(q, z), z)",
        height=500,
    )
    return fig


def make_quote_inventory_figure(
    cpp_tier: Any,
    spec: TierSpec,
    spread: float,
    mid_price: float,
) -> go.Figure:
    q_grid = [float(q) for q in cpp_tier.policy.q_grid]
    fig = go.Figure()

    for z in tier_sizes(spec):
        zf = float(z)
        bid_vals: list[float] = []
        ask_vals: list[float] = []

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
        height=500,
        yaxis=dict(range=[-20, 20]),
    )
    return fig


def make_ladder_figure(
    cpp_tier: Any,
    spec: TierSpec,
    q: float,
    spread: float,
    mid_price: float,
) -> go.Figure:
    sizes = [float(z) for z in tier_sizes(spec)]
    bid_vp, ask_vp = [], []

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
        yaxis_title="Premium vs first-rung quote (pips)",
        height=500,
    )
    return fig


def make_q_ladder_table(
    cpp_tier: Any,
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
        bid_active = is_active(cpp_tier, q, zf, "bid")
        ask_active = is_active(cpp_tier, q, zf, "ask")

        bid = masked_quote_summary(cpp_tier, q, zf, "bid", mid_price, spread)
        ask = masked_quote_summary(cpp_tier, q, zf, "ask", mid_price, spread)

        rows.append(
            {
                "enabled": bool(spec.enabled),
                "type": tier_kind_label(spec),
                "q": float(q),
                "z": zf,
                "Bid admissible": bid_adm,
                "Ask admissible": ask_adm,
                "Bid active": bid_active,
                "Ask active": ask_active,
                "Bid delta": np.nan if bid is None else round(float(bid.delta), 6),
                "Ask delta": np.nan if ask is None else round(float(ask.delta), 6),
                "Bid vs mid [pips]": np.nan if bid is None else round(float(bid.quote_relative_to_mid_pips), 3),
                "Ask vs mid [pips]": np.nan if ask is None else round(float(ask.quote_relative_to_mid_pips), 3),
                "Bid vol premium [pips]": np.nan if bid is None else round(float(bid.volume_premium_pips), 3),
                "Ask vol premium [pips]": np.nan if ask is None else round(float(ask.volume_premium_pips), 3),
            }
        )

    return pd.DataFrame(rows)


def make_ecn_cost_curve_figure(cpp_tier: Any, spec: ECNTierSpec) -> go.Figure:
    delta_grid = np.linspace(float(cpp_tier.delta_min), float(cpp_tier.delta_max), 220)
    fig = go.Figure()

    for z in tier_sizes(spec):
        zf = float(z)

        total_cost = [
            float(cpp_tier.adverse_selection_model.expected_cost(cpp_tier.markout_model, float(d), zf))
            for d in delta_grid
        ]

        fig.add_trace(
            go.Scatter(
                x=delta_grid,
                y=total_cost,
                mode="lines",
                name=f"total cost z={zf:g}",
            )
        )

    fig.update_layout(
        title="ECN total cost vs delta",
        xaxis_title="delta",
        yaxis_title="Cost per fill",
        height=360,
    )
    return fig


def make_ecn_optimization_table(
    solver: Any,
    cpp_tier: Any,
    spec: ECNTierSpec,
    solution: Any,
    spread: float,
    mid_price: float,
) -> pd.DataFrame:
    rows = []
    h_vec = [float(x) for x in solution.h]

    for i, q_raw in enumerate(solution.q_grid):
        q = float(q_raw)

        for j, z in enumerate(tier_sizes(spec)):
            zf = float(z)

            for side in ["bid", "ask"]:
                active = is_active(cpp_tier, q, zf, side)

                delta = np.nan
                arrival_rate = np.nan
                quote_pips = np.nan
                fill_value = np.nan
                impact_drift = np.nan
                impact_value = np.nan
                total_value = np.nan
                candidate_pressure = np.nan

                if active:
                    delta = float(cpp_tier.policy.bid[i][j]) if side == "bid" else float(cpp_tier.policy.ask[i][j])
                    arrival_rate = float(cpp_tier.arrival_rate(delta, zf))

                    quote = masked_quote_summary(cpp_tier, q, zf, side, mid_price, spread)
                    if quote is not None:
                        quote_pips = float(quote.quote_relative_to_mid_pips)

                    candidate_pressure = float(
                        solver.ecn_candidate_net_pressure(cpp_tier, delta, zf, side)
                    )
                    impact_drift = float(
                        solver.ecn_candidate_passive_impact_drift(cpp_tier, delta, zf, side)
                    )
                    impact_value = float(
                        solver.ecn_candidate_passive_impact_value(cpp_tier, q, delta, zf, side)
                    )
                    fill_value = float(
                        solver.ecn_fill_value(cpp_tier, h_vec, q, side, delta, zf)
                    )
                    total_value = float(
                        solver.ecn_total_candidate_value(cpp_tier, h_vec, q, side, delta, zf)
                    )

                rows.append(
                    {
                        "q": q,
                        "z": zf,
                        "side": side,
                        "active": active,
                        "delta": delta,
                        "arrival_rate": arrival_rate,
                        "quote_vs_mid_pips": quote_pips,
                        "candidate_pressure": candidate_pressure,
                        "candidate_impact_drift": impact_drift,
                        "passive_impact_value": impact_value,
                        "fill_value": fill_value,
                        "total_candidate_value": total_value,
                    }
                )

    return pd.DataFrame(rows)


def ecn_pressure_rows(solution: Any) -> pd.DataFrame:
    rows = []

    for i, q_raw in enumerate(solution.q_grid):
        q = float(q_raw)
        bid_pressure = 0.0
        ask_pressure = 0.0

        for tier in solution.ecn_tiers:
            sizes = [float(z) for z in tier.sizes_]

            for j, z in enumerate(sizes):
                if is_admissible(tier, q, z, "bid"):
                    d = float(tier.policy.bid[i][j])
                    bid_pressure += z * float(tier.arrival_rate(d, z))

                if is_admissible(tier, q, z, "ask"):
                    d = float(tier.policy.ask[i][j])
                    ask_pressure += z * float(tier.arrival_rate(d, z))

        net_pressure = bid_pressure - ask_pressure
        passive_drift = float(solution.ecn_passive_impact.drift(net_pressure))

        rows.append(
            {
                "q": q,
                "bid_pressure": bid_pressure,
                "ask_pressure": ask_pressure,
                "net_pressure": net_pressure,
                "passive_drift": passive_drift,
            }
        )

    return pd.DataFrame(rows)


def make_ecn_passive_impact_parameter_table(model: Any) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "enabled": bool(model.enabled),
                "eta": float(model.eta),
                "pressure_scale": float(model.pressure_scale),
                "formula": "eta * tanh((bid_pressure - ask_pressure) / pressure_scale)",
                "included venues": "ECN only",
                "excluded venues": "MDP tiers, dark pool",
            }
        ]
    )


def make_ecn_passive_impact_figure(solution: Any) -> go.Figure:
    df = ecn_pressure_rows(solution)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["q"], y=df["bid_pressure"], mode="lines", name="ECN bid pressure"))
    fig.add_trace(
        go.Scatter(
            x=df["q"],
            y=df["ask_pressure"],
            mode="lines",
            line=dict(dash="dash"),
            name="ECN ask pressure",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["q"],
            y=df["net_pressure"],
            mode="lines",
            line=dict(dash="dot"),
            name="net pressure",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["q"],
            y=df["passive_drift"],
            mode="lines+markers",
            name="passive drift",
            yaxis="y2",
        )
    )

    fig.add_hline(y=0.0)
    fig.update_layout(
        title="ECN passive impact: pressure and induced drift",
        xaxis_title="Inventory q",
        yaxis_title="ECN pressure",
        yaxis2=dict(
            title="passive drift",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        height=460,
    )
    return fig


def make_dark_pool_parameter_table(venue: Any) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "arrival_lambda_bid": float(venue.arrival_lambda_bid),
                "arrival_lambda_ask": float(venue.arrival_lambda_ask),
                "size_lambda_bid": float(venue.size_lambda_bid),
                "size_lambda_ask": float(venue.size_lambda_ask),
                "mean positive size bid": round(zero_truncated_poisson_mean(float(venue.size_lambda_bid)), 4),
                "mean positive size ask": round(zero_truncated_poisson_mean(float(venue.size_lambda_ask)), 4),
                "fee_per_unit_bid": float(venue.fee_per_unit_bid),
                "fee_per_unit_ask": float(venue.fee_per_unit_ask),
                "allow_both_sides": bool(venue.allow_both_sides),
                "posted_sizes": ", ".join(f"{float(u):g}" for u in venue.posted_sizes),
            }
        ]
    )


def make_dark_pool_size_figure(venue: Any) -> go.Figure:
    q_grid = [float(q) for q in venue.policy.q_grid]
    bid_vals = [
        np.nan if not venue.policy.is_active(i, "bid") else float(venue.policy.bid_size[i])
        for i in range(len(q_grid))
    ]
    ask_vals = [
        np.nan if not venue.policy.is_active(i, "ask") else float(venue.policy.ask_size[i])
        for i in range(len(q_grid))
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_grid, y=bid_vals, mode="lines+markers", name="posted bid size"))
    fig.add_trace(
        go.Scatter(
            x=q_grid,
            y=ask_vals,
            mode="lines+markers",
            line=dict(dash="dash"),
            name="posted ask size",
        )
    )
    fig.update_layout(
        title="Dark-pool optimal posted size vs inventory",
        xaxis_title="Inventory q",
        yaxis_title="Posted size",
        height=420,
    )
    return fig


def make_dark_pool_size_distribution_figure(venue: Any) -> go.Figure:
    max_posted_size = max(float(u) for u in venue.posted_sizes)
    max_size_lambda = max(float(venue.size_lambda_bid), float(venue.size_lambda_ask))

    k_max = int(
        max(
            max_posted_size,
            math.ceil(max_size_lambda + 6.0 * math.sqrt(max_size_lambda)),
            10,
        )
    )

    bid_df = zero_truncated_poisson_pmf(float(venue.size_lambda_bid), k_max)
    ask_df = zero_truncated_poisson_pmf(float(venue.size_lambda_ask), k_max)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bid_df["k"],
            y=bid_df["probability"],
            name="bid incoming size",
            opacity=0.75,
        )
    )
    fig.add_trace(
        go.Bar(
            x=ask_df["k"],
            y=ask_df["probability"],
            name="ask incoming size",
            opacity=0.75,
        )
    )

    for u in venue.posted_sizes:
        fig.add_vline(
            x=float(u),
            line_dash="dot",
            annotation_text=f"posted {float(u):g}",
            annotation_position="top",
        )

    fig.update_layout(
        title="Dark-pool incoming size distribution",
        xaxis_title="Incoming size k",
        yaxis_title="P(K = k | K ≥ 1)",
        barmode="overlay",
        height=420,
    )
    return fig


def make_h_figure(solution: Any) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(solution.q_grid), y=list(solution.h), mode="lines+markers", name="h(q)"))
    fig.update_layout(title="Value function h(q)", xaxis_title="Inventory q", yaxis_title="h(q)", height=420)
    return fig


def make_convergence_figure(values: list[float], title: str, yaxis_title: str) -> go.Figure:
    fig = go.Figure()
    if values:
        fig.add_trace(go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode="lines+markers"))
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title=yaxis_title,
        yaxis_type="log",
        height=360,
    )
    return fig


# ============================================================
# App
# ============================================================

st.set_page_config(page_title="Trinity 2.0 Pricer", layout="wide")
st.title("Trinity 2.0 Pricer")
st.markdown(
    "MDP tiers use rung-by-rung two-sided ladder controls. "
    "ECN tiers now use the same ladder-style size interface as MDP, with ECN-specific costs. "
    "The dark pool trades at mid with a posted-size control. "
    "Only ECN quotes can create passive market-impact drift."
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

try:
    ecn_passive_impact_spec = build_ecn_passive_impact_spec_from_ui()
except ValueError as exc:
    errors.append(str(exc))
    ecn_passive_impact_spec = ECNPassiveImpactSpec(enabled=False, eta=0.0, pressure_scale=1.0)

try:
    dark_pool_spec = build_dark_pool_spec_from_ui()
except ValueError as exc:
    errors.append(str(exc))
    dark_pool_spec = None

active_tier_specs = enabled_tier_specs(tier_specs)
active_mdp_count = sum(isinstance(spec, MDPTierSpec) for spec in active_tier_specs)
active_ecn_count = sum(isinstance(spec, ECNTierSpec) for spec in active_tier_specs)
disabled_tier_names = [spec.name for spec in tier_specs if not spec.enabled]

if spread <= 0.0:
    errors.append("spread must be positive.")
if golden_tol <= 0.0:
    errors.append("golden_tol must be positive.")
if ecn_passive_impact_spec.eta < 0.0:
    errors.append("ECN passive impact eta must be nonnegative.")
if ecn_passive_impact_spec.pressure_scale <= 0.0:
    errors.append("ECN passive impact pressure_scale must be positive.")

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

if not active_tier_specs and dark_pool_spec is None:
    errors.append("Enable at least one tier or the dark-pool venue.")

if errors:
    for error in errors:
        st.error(error)
    st.stop()

if ecn_passive_impact_spec.enabled and active_ecn_count == 0:
    st.warning("ECN passive impact is enabled, but there are no active ECN tiers.")

mdp_cpp_tiers, ecn_cpp_tiers = build_cpp_tiers(active_tier_specs)
dark_pool_cpp = None if dark_pool_spec is None else dark_pool_spec.build_cpp_venue()
ecn_passive_impact_cpp = ecn_passive_impact_spec.build_cpp_model()

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
        dark_pool=dark_pool_cpp,
        ecn_passive_impact=ecn_passive_impact_cpp,
    )
    solution = solver.solve()

diag = solution.diagnostics
solution_tiers = ordered_solution_tiers(active_tier_specs, solution)

st.success("Solver run complete.")

summary_cols = st.columns(6)
summary_cols[0].metric("Configured tiers", len(tier_specs))
summary_cols[1].metric("Active MDP tiers", active_mdp_count)
summary_cols[2].metric("Active ECN tiers", active_ecn_count)
summary_cols[3].metric("Dark pool", "on" if dark_pool_spec is not None else "off")
summary_cols[4].metric("ECN impact", "on" if bool(solution.ecn_passive_impact.enabled) else "off")
summary_cols[5].metric("Disabled tiers", len(disabled_tier_names))

if disabled_tier_names:
    st.caption("Excluded from solve: " + ", ".join(disabled_tier_names))

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
    c2.metric("venues", total_venue_count(solution))
    c3.metric("iterations used", diag.iterations_used)
    c4.metric("converged", "yes" if diag.converged else "no")
    c5.metric("spot drift", f"{config.spot_drift:.5f}")
    c6.metric("spread", f"{config.spread:.6f}")

    st.plotly_chart(make_h_figure(solution), use_container_width=True)
    st.plotly_chart(
        make_convergence_figure(
            list(solution.diagnostics.history_max_h_change),
            "Convergence: max |Δh|",
            "max |Δh|",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        make_convergence_figure(
            list(solution.diagnostics.history_max_rhs),
            "Convergence: max |rhs|",
            "max |rhs|",
        ),
        use_container_width=True,
    )

with st.expander("ECN passive impact diagnostics", expanded=False):
    st.dataframe(make_ecn_passive_impact_parameter_table(solution.ecn_passive_impact), use_container_width=True)

    if active_ecn_count == 0:
        st.info("No active ECN tiers. ECN passive pressure is zero.")
    else:
        st.plotly_chart(make_ecn_passive_impact_figure(solution), use_container_width=True)
        st.dataframe(ecn_pressure_rows(solution).round(8), use_container_width=True)

tab_names = [f"{spec.name} [{tier_kind_label(spec)}]" for spec in active_tier_specs]
if solution.dark_pool is not None:
    tab_names.append("Dark Pool")

if tab_names:
    tabs = st.tabs(tab_names)
    available_q = [float(q) for q in solution.q_grid]
    default_q = 0.0 if 0.0 in available_q else available_q[len(available_q) // 2]

    for idx, (spec, cpp_tier) in enumerate(zip(active_tier_specs, solution_tiers)):
        with tabs[idx]:
            st.subheader(f"Tier: {spec.name}")
            st.caption(f"Type: {tier_kind_label(spec)}")

            is_ecn = isinstance(spec, ECNTierSpec)

            if is_ecn:
                c1, c2, c3 = st.columns(3)
                c1.metric("ECN toxicity", f"{float(cpp_tier.adverse_selection_model.ecn_toxicity):.5f}")
                c2.metric("ECN fee", f"{float(cpp_tier.adverse_selection_model.ecn_fee):.5f}")
                c3.metric("inventory scale", f"{float(cpp_tier.adverse_selection_model.ecn_convergence_inventory):.4f}")

                c4, c5, c6 = st.columns(3)
                c4.metric("delta_min", f"{float(cpp_tier.delta_min):.4f}")
                c5.metric("delta_max", f"{float(cpp_tier.delta_max):.4f}")
                c6.metric("sizes", ", ".join(f"{float(z):g}" for z in cpp_tier.sizes_))

                st.caption(
                    "This ECN tier uses the current ladder-style ECN binding: sizes_, bid/ask policy matrices, "
                    "ECN adverse-selection costs, and optional passive impact."
                )

            with st.expander("Tier parameters", expanded=False):
                st.dataframe(make_flow_parameter_table(spec, cpp_tier), use_container_width=True)

            st.plotly_chart(make_flow_curve_figure(cpp_tier, spec), use_container_width=True)

            with st.expander("Delta-implied flow rates", expanded=False):
                st.caption(
                    "These are the realized flow rates implied by the optimized policy deltas: "
                    "λ(δ*(q, z), z). The table also shows A(z), hit ratio, admissibility, and size-weighted pressure."
                )
                st.plotly_chart(make_delta_implied_flow_figure(cpp_tier, spec), use_container_width=True)
                st.dataframe(make_delta_implied_flow_table(cpp_tier, spec).round(8), use_container_width=True)

            st.plotly_chart(
                make_quote_inventory_figure(cpp_tier, spec, float(config.spread), float(mid_price)),
                use_container_width=True,
            )

            if is_ecn:
                st.plotly_chart(make_ecn_cost_curve_figure(cpp_tier, spec), use_container_width=True)

                with st.expander("ECN optimization diagnostics", expanded=False):
                    st.dataframe(
                        make_ecn_optimization_table(
                            solver,
                            cpp_tier,
                            spec,
                            solution,
                            float(config.spread),
                            float(mid_price),
                        ).round(8),
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

    if solution.dark_pool is not None:
        venue = solution.dark_pool
        with tabs[-1]:
            st.subheader("Dark pool venue")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("arrival λ bid", f"{float(venue.arrival_lambda_bid):.4f}")
            c2.metric("arrival λ ask", f"{float(venue.arrival_lambda_ask):.4f}")
            c3.metric("size λ bid", f"{float(venue.size_lambda_bid):.4f}")
            c4.metric("size λ ask", f"{float(venue.size_lambda_ask):.4f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("mean positive size bid", f"{zero_truncated_poisson_mean(float(venue.size_lambda_bid)):.3f}")
            c6.metric("mean positive size ask", f"{zero_truncated_poisson_mean(float(venue.size_lambda_ask)):.3f}")
            c7.metric("fee bid", f"{float(venue.fee_per_unit_bid):.5f}")
            c8.metric("both sides allowed", "yes" if bool(venue.allow_both_sides) else "no")

            c9, c10 = st.columns(2)
            c9.metric("fee ask", f"{float(venue.fee_per_unit_ask):.5f}")
            c10.metric("posted size candidates", ", ".join(f"{float(u):g}" for u in venue.posted_sizes))

            with st.expander("Dark-pool parameters", expanded=False):
                st.dataframe(make_dark_pool_parameter_table(venue), use_container_width=True)

            st.plotly_chart(make_dark_pool_size_figure(venue), use_container_width=True)

            with st.expander("Incoming size distribution", expanded=False):
                st.caption(
                    "The incoming dark-pool size is modeled as zero-truncated Poisson: "
                    "K ~ Poisson(size_lambda) conditional on K ≥ 1. "
                    "Executed size is min(posted size, K)."
                )
                st.plotly_chart(make_dark_pool_size_distribution_figure(venue), use_container_width=True)

with st.expander("What this app is solving"):
    st.markdown(
        r"""
MDP tiers:
- use rung-by-rung two-sided ladder optimization
- use the tier's `sizes_` / `sizes` vector
- optimize deltas over `[delta_min, delta_max]`

ECN tiers:
- now use the same ladder-style policy shape as MDP tiers
- have their own `sizes_` vector
- use ECN fill economics: spread capture, adverse selection, fees, continuation value
- may contribute passive market-impact drift
- no longer use `is_policy_active`, `bid_active`, `ask_active`, or `delta_mid_cap`

Delta-implied flow diagnostics:
- for each MDP and ECN tier, the app reports the realized arrival rate implied by the optimized delta
- the displayed value is $\lambda(\delta^*(q,z), z)$
- the table also shows $A(z)$, the logistic hit ratio, and size-weighted pressure

ECN passive market impact:
- only ECN quotes create passive price pressure
- MDP customer tiers do not create passive price pressure
- dark pool does not create passive price pressure
- ECN bid pressure and ask pressure are computed from ECN policy-implied flow rates
- passive drift is `eta * tanh((bid_pressure - ask_pressure) / pressure_scale)`
- this drift enters the HJB through the reduced-form term `q * drift`

Dark-pool venue:
- execution price is the mid
- there is no quote-price optimization
- opportunity arrivals are controlled by `arrival_lambda_bid` and `arrival_lambda_ask`
- conditional incoming order size is zero-truncated Poisson
- if posted size is $u$ and incoming size is $Z$, then executed size is $\min(u, Z)$

The inventory grid must be:
- strictly increasing
- symmetric around 0
- odd-length
- with $0$ exactly at the middle index
        """
    )

st.caption("After replacing the C++ files, rebuild the extension and restart Streamlit.")