from __future__ import annotations

import math
from dataclasses import dataclass, field

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
# Specifications
# ============================================================

DEFAULT_MDP_SIZES = "1, 2, 3, 5, 10, 20"
DEFAULT_DARK_POOL_SETTINGS = {
    "enabled": False,
    "dist_type": "Geometric",
    "lambda_bid": 2.00,
    "lambda_ask": 2.00,
    # Geometric
    "p_bid": 0.50,
    "p_ask": 0.50,
    # Zero-inflated Poisson
    "mu_bid": 2.0,
    "mu_ask": 2.0,
    "p0_bid": 0.1,
    "p0_ask": 0.1,
    "fee_per_unit_bid": 0.0,
    "fee_per_unit_ask": 0.0,
    "posted_sizes": "1, 2, 3, 5",
    "allow_both_sides": False,
}


@dataclass
class SqrtMarkoutSpec:
    base: float = 0.0        # pips
    coeff: float = 0.0       # pips
    trading_cost: float = 0.0  # EUR / million

    def build(self, spot: float) -> lp.SqrtMarkoutModel:
        return lp.SqrtMarkoutModel(
            base=self.base / 10_000 + self.trading_cost * spot / 1_000_000,
            coeff=self.coeff / 10_000,
        )


@dataclass
class SqrtTimeMarkoutSpec:
    a0: float = -0.1785    # pips
    a1: float = -0.1819    # pips
    a2: float = 0.0052    # pips
    b0: float = 0.0089    # pips
    b1: float = 0.0068    # pips
    b2: float = -0.0030    # pips

    def build(self, spot: float) -> lp.SqrtTimeMarkoutModel:
        return lp.SqrtTimeMarkoutModel(
            a0=self.a0 / 10_000,
            a1=self.a1 / 10_000,
            a2=self.a2 / 10_000,
            b0=self.b0 / 10_000,
            b1=self.b1 / 10_000,
            b2=self.b2 / 10_000,
        )


MarkoutSpec = SqrtMarkoutSpec | SqrtTimeMarkoutSpec


@dataclass
class CommonTierSpec:
    enabled: bool
    name: str
    flow_A0: float
    flow_theta: float
    flow_beta: float
    flow_steepness: float
    flow_shift: float
    flow_volume_shift: float
    markout_spec: MarkoutSpec
    use_markout: bool
    delta_min: float
    delta_max: float

    def build_models(self, spot: float):
        flow = lp.LogisticFlowCurve(
            A0=float(self.flow_A0),
            theta=float(self.flow_theta),
            beta=float(self.flow_beta),
            shift=float(self.flow_shift),
            steepness=float(self.flow_steepness),
            volume_shift=float(self.flow_volume_shift),
        )
        return flow, self.markout_spec.build(spot)


@dataclass
class MDPTierSpec(CommonTierSpec):
    sizes: list[float]

    def build_cpp_tier(self, spot: float) -> lp.MDPTier:
        flow, markout = self.build_models(spot)
        return lp.MDPTier(
            name=self.name,
            sizes=[float(z) for z in self.sizes],
            flow_curve=flow,
            markout_model=markout,
            delta_min=float(self.delta_min),
            delta_max=float(self.delta_max),
            use_markout=bool(self.use_markout),
        )


@dataclass
class DarkPoolSpec:
    dist_type: str  # "Geometric" or "Zero-inflated Poisson"
    lambda_bid: float
    lambda_ask: float
    # Geometric params
    p_bid: float = 0.5
    p_ask: float = 0.5
    # Zero-inflated Poisson params
    mu_bid: float = 2.0
    mu_ask: float = 2.0
    p0_bid: float = 0.1
    p0_ask: float = 0.1
    fee_per_unit_bid: float = 0.0
    fee_per_unit_ask: float = 0.0
    posted_sizes: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 5.0])
    allow_both_sides: bool = False

    def build_cpp_venue(self, spot: float) -> lp.DarkPoolVenue:
        fee_bid = float(self.fee_per_unit_bid) * spot / 1_000_000
        fee_ask = float(self.fee_per_unit_ask) * spot / 1_000_000
        sizes = [float(u) for u in self.posted_sizes]
        if self.dist_type == "Geometric":
            return lp.DarkPoolVenue(
                lambda_bid=float(self.lambda_bid),
                lambda_ask=float(self.lambda_ask),
                p_bid=float(self.p_bid),
                p_ask=float(self.p_ask),
                fee_per_unit_bid=fee_bid,
                fee_per_unit_ask=fee_ask,
                posted_sizes=sizes,
                allow_both_sides=bool(self.allow_both_sides),
            )
        else:
            return lp.DarkPoolVenue(
                dist_bid=lp.ZeroInflatedPoissonArrivalDist(float(self.lambda_bid), float(self.mu_bid), float(self.p0_bid)),
                dist_ask=lp.ZeroInflatedPoissonArrivalDist(float(self.lambda_ask), float(self.mu_ask), float(self.p0_ask)),
                fee_per_unit_bid=fee_bid,
                fee_per_unit_ask=fee_ask,
                posted_sizes=sizes,
                allow_both_sides=bool(self.allow_both_sides),
            )


TierSpec = MDPTierSpec
CppTier = lp.MDPTier


def tier_kind_label() -> str:
    return "MDP"


def tier_sizes(spec: TierSpec) -> list[float]:
    return spec.sizes


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


def default_tier_values(i: int) -> dict:
    _markout_defaults = {
        "markout_model_type": "sqrt_time",
        "markout_base": 0.0,
        "markout_coeff": 0.0,
        "trading_cost": 0.0,
        "sqrt_t_a0": 0.0,
        "sqrt_t_a1": 0.0,
        "sqrt_t_a2": 0.0,
        "sqrt_t_b0": 0.0,
        "sqrt_t_b1": 0.0,
        "sqrt_t_b2": 0.0,
    }
    presets = [
        {
            "enabled": True,
            "kind": "mdp",
            "name": "Tier 1",
            "sizes": "1, 2, 3, 5, 10, 20",
            "flow_A0": 0.0155,
            "flow_theta": 0.144,
            "flow_beta": 0.0857,
            "flow_steepness": 8.42,
            "flow_shift": 0.52,
            "flow_volume_shift": 0.026,
            **_markout_defaults,
            "sqrt_t_a0": -0.1785,
            "sqrt_t_a1": -0.1819,
            "sqrt_t_a2":  0.0052,
            "sqrt_t_b0":  0.0089,
            "sqrt_t_b1":  0.0068,
            "sqrt_t_b2": -0.0030,
            "delta_min": -100.0,
            "delta_max": 100.0,
        },
        {
            "enabled": True,
            "kind": "mdp",
            "name": "Tier 2",
            "sizes": "1, 2, 3, 5, 10, 20",
            "flow_A0": 0.0232,
            "flow_theta": 0.303,
            "flow_beta": 0.122,
            "flow_steepness": 2.86,
            "flow_shift": 0.48,
            "flow_volume_shift": 0.02,
            **_markout_defaults,
            "sqrt_t_a0": -0.16,
            "sqrt_t_a1": -0.156,
            "sqrt_t_a2":  0.0044,
            "sqrt_t_b0":  0.011,
            "sqrt_t_b1": -0.00046,
            "sqrt_t_b2":  0.0,
            "delta_min": -100.0,
            "delta_max": 100.0,
        },
        {
            "enabled": False,
            "kind": "mdp",
            "name": "Tier 3",
            "sizes": "1, 2, 3, 5, 10, 15, 20",
            "flow_A0": 0.85,
            "flow_theta": 0.15,
            "flow_beta": 0.0,
            "flow_steepness": 1.60,
            "flow_shift": 0.28,
            "flow_volume_shift": 0.090,
            **_markout_defaults,
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

        sizes: list[float] | None = None
        sizes_raw = st.text_input(
            f"Sizes {i + 1}",
            value=defaults.get("sizes", DEFAULT_MDP_SIZES),
            key=f"sizes_{i}",
        )
        if enabled:
            sizes = parse_float_list(sizes_raw, f"sizes for tier {i + 1}")
        else:
            sizes = try_parse_float_list(sizes_raw) or [1.0]

        st.markdown("**Flow curve parameters**")
        c1, c2, c3 = st.columns(3)
        flow_A0 = c1.number_input(
            f"A0 {i + 1}",
            value=float(defaults["flow_A0"]),
            step=0.05,
            format="%.4f",
            key=f"flow_A0_{i}",
            help="Overall flow intensity: expected number of client RFQs per minute at δ = 50 % and z = 1.",
        )
        flow_theta = c2.number_input(
            f"theta {i + 1}",
            value=float(defaults["flow_theta"]),
            step=0.05,
            format="%.4f",
            key=f"flow_theta_{i}",
        )
        flow_beta = c3.number_input(
            f"beta {i + 1}",
            value=float(defaults.get("flow_beta", 0.0)),
            step=0.005,
            format="%.4f",
            key=f"flow_beta_{i}",
        )

        c4, c5 = st.columns(2)
        flow_steepness = c4.number_input(
            f"steepness {i + 1}",
            value=float(defaults["flow_steepness"]),
            step=0.05,
            format="%.4f",
            key=f"flow_steepness_{i}",
        )
        flow_shift = c5.number_input(
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

        st.markdown("**Markout model**")
        use_markout = st.checkbox(
            f"Price in markout {i + 1}",
            value=bool(defaults.get("use_markout", True)),
            key=f"use_markout_{i}",
        )
        markout_model_type = st.selectbox(
            f"Markout model type {i + 1}",
            options=["sqrt", "sqrt_time"],
            index=0 if defaults.get("markout_model_type", "sqrt") == "sqrt" else 1,
            key=f"markout_model_type_{i}",
            format_func=lambda x: "base + coeff·√z" if x == "sqrt" else "α(z)·√t + β(z)·t",
        )

        if markout_model_type == "sqrt":
            c5, c6 = st.columns(2)
            markout_base = c5.number_input(
                f"Markout base {i + 1}",
                value=float(defaults["markout_base"]),
                step=0.1,
                format="%.2f",
                key=f"markout_base_{i}",
                help="Constant adverse-selection cost in pips (1 pip = 1/10 000).",
            )
            markout_coeff = c6.number_input(
                f"Markout coeff {i + 1}",
                value=float(defaults["markout_coeff"]),
                step=0.1,
                format="%.2f",
                key=f"markout_coeff_{i}",
                help="Size-dependent coefficient in pips. Total markout = base + coeff × √z pips.",
            )
            st.markdown("**Trading cost (EUR / million)**")
            trading_cost = st.number_input(
                f"Trading cost {i + 1}",
                value=float(defaults.get("trading_cost", 0.0)),
                step=0.5,
                format="%.2f",
                key=f"trading_cost_{i}",
                help="Flat execution cost in EUR per million traded. Added to markout base after converting via spot.",
            )
            markout_spec: MarkoutSpec = SqrtMarkoutSpec(
                base=float(markout_base),
                coeff=float(markout_coeff),
                trading_cost=float(trading_cost),
            )
        else:
            st.caption("α(z) = a₀ + a₁z + a₂z²,   β(z) = b₀ + b₁z + b₂z²   (coefficients in pips)")
            st.markdown("α(z) coefficients")
            ca0, ca1, ca2 = st.columns(3)
            sqrt_t_a0 = ca0.number_input(f"a0 {i+1}", value=float(defaults.get("sqrt_t_a0", 0.0)), step=0.1, format="%.4f", key=f"sqrt_t_a0_{i}")
            sqrt_t_a1 = ca1.number_input(f"a1 {i+1}", value=float(defaults.get("sqrt_t_a1", 0.0)), step=0.01, format="%.4f", key=f"sqrt_t_a1_{i}")
            sqrt_t_a2 = ca2.number_input(f"a2 {i+1}", value=float(defaults.get("sqrt_t_a2", 0.0)), step=0.001, format="%.5f", key=f"sqrt_t_a2_{i}")
            st.markdown("β(z) coefficients")
            cb0, cb1, cb2 = st.columns(3)
            sqrt_t_b0 = cb0.number_input(f"b0 {i+1}", value=float(defaults.get("sqrt_t_b0", 0.0)), step=0.1, format="%.4f", key=f"sqrt_t_b0_{i}")
            sqrt_t_b1 = cb1.number_input(f"b1 {i+1}", value=float(defaults.get("sqrt_t_b1", 0.0)), step=0.01, format="%.4f", key=f"sqrt_t_b1_{i}")
            sqrt_t_b2 = cb2.number_input(f"b2 {i+1}", value=float(defaults.get("sqrt_t_b2", 0.0)), step=0.001, format="%.5f", key=f"sqrt_t_b2_{i}")
            markout_spec = SqrtTimeMarkoutSpec(
                a0=float(sqrt_t_a0), a1=float(sqrt_t_a1), a2=float(sqrt_t_a2),
                b0=float(sqrt_t_b0), b1=float(sqrt_t_b1), b2=float(sqrt_t_b2),
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

        if not enabled:
            st.caption("Disabled tiers stay editable in the sidebar but are excluded from the solver and output tabs.")

    common = dict(
        enabled=bool(enabled),
        name=name,
        flow_A0=float(flow_A0),
        flow_theta=float(flow_theta),
        flow_beta=float(flow_beta),
        flow_steepness=float(flow_steepness),
        flow_shift=float(flow_shift),
        flow_volume_shift=float(flow_volume_shift),
        markout_spec=markout_spec,
        use_markout=bool(use_markout),
        delta_min=float(tier_delta_min),
        delta_max=float(tier_delta_max),
    )

    assert sizes is not None

    if not enabled:
        return MDPTierSpec(sizes=sizes, **common)

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

    return MDPTierSpec(sizes=sizes, **common)


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

        dist_type = st.selectbox(
            "Arrival distribution",
            options=["Geometric", "Zero-inflated Poisson"],
            index=["Geometric", "Zero-inflated Poisson"].index(defaults["dist_type"]),
            help="Distribution of incoming dark-pool order sizes.",
        )

        c1, c2 = st.columns(2)
        lambda_bid = c1.number_input(
            "Dark pool λ bid",
            value=float(defaults["lambda_bid"]),
            min_value=0.0,
            step=0.01,
            format="%.4f",
        )
        lambda_ask = c2.number_input(
            "Dark pool λ ask",
            value=float(defaults["lambda_ask"]),
            min_value=0.0,
            step=0.01,
            format="%.4f",
        )

        c3, c4 = st.columns(2)
        if dist_type == "Geometric":
            p_bid = c3.number_input(
                "p bid",
                value=float(defaults["p_bid"]),
                min_value=0.001,
                max_value=1.0,
                step=0.01,
                format="%.4f",
                help="Success probability per unit. Mean fill size = 1/p.",
            )
            p_ask = c4.number_input(
                "p ask",
                value=float(defaults["p_ask"]),
                min_value=0.001,
                max_value=1.0,
                step=0.01,
                format="%.4f",
                help="Success probability per unit. Mean fill size = 1/p.",
            )
            mu_bid = mu_ask = p0_bid = p0_ask = 0.0
        else:
            mu_bid = c3.number_input(
                "μ bid",
                value=float(defaults["mu_bid"]),
                min_value=0.01,
                step=0.1,
                format="%.3f",
                help="Poisson rate — mean fill size conditional on a non-zero fill.",
            )
            mu_ask = c4.number_input(
                "μ ask",
                value=float(defaults["mu_ask"]),
                min_value=0.01,
                step=0.1,
                format="%.3f",
                help="Poisson rate — mean fill size conditional on a non-zero fill.",
            )
            c5, c6 = st.columns(2)
            p0_bid = c5.number_input(
                "p₀ bid",
                value=float(defaults["p0_bid"]),
                min_value=0.0,
                max_value=0.999,
                step=0.01,
                format="%.3f",
                help="Zero-inflation probability — chance of no fill regardless of arrival.",
            )
            p0_ask = c6.number_input(
                "p₀ ask",
                value=float(defaults["p0_ask"]),
                min_value=0.0,
                max_value=0.999,
                step=0.01,
                format="%.3f",
                help="Zero-inflation probability — chance of no fill regardless of arrival.",
            )
            p_bid = p_ask = 0.5

        c_fee1, c_fee2 = st.columns(2)
        fee_per_unit_bid = c_fee1.number_input(
            "Fee / rebate bid [EUR/M]",
            value=float(defaults["fee_per_unit_bid"]),
            step=0.5,
            format="%.2f",
            help="Positive = fee, negative = rebate. EUR per million of dark-pool fill.",
        )
        fee_per_unit_ask = c_fee2.number_input(
            "Fee / rebate ask [EUR/M]",
            value=float(defaults["fee_per_unit_ask"]),
            step=0.5,
            format="%.2f",
            help="Positive = fee, negative = rebate. EUR per million of dark-pool fill.",
        )

        posted_sizes_raw = st.text_input(
            "Allowed posted sizes",
            value=str(defaults["posted_sizes"]),
            help="Integer-valued posted sizes in the same inventory units as q.",
        )
        posted_sizes = parse_integer_like_list(posted_sizes_raw, "dark-pool posted sizes")

        if any(u <= 0.0 for u in posted_sizes):
            raise ValueError("Dark pool posted sizes must be positive.")
        if any(posted_sizes[k] >= posted_sizes[k + 1] for k in range(len(posted_sizes) - 1)):
            raise ValueError("Dark pool posted sizes must be strictly increasing.")

        st.caption(
            "Dark-pool fills occur at mid. Arrival intensity is constant. "
            "The executed size is min(posted size, incoming size)."
        )

    return DarkPoolSpec(
        dist_type=str(dist_type),
        lambda_bid=float(lambda_bid),
        lambda_ask=float(lambda_ask),
        p_bid=float(p_bid),
        p_ask=float(p_ask),
        mu_bid=float(mu_bid),
        mu_ask=float(mu_ask),
        p0_bid=float(p0_bid),
        p0_ask=float(p0_ask),
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
    spot: float,
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
    config.spot = float(spot)
    config.spot_drift = float(spot_drift)
    config.golden_tol = float(golden_tol)
    config.golden_max_iter = int(golden_max_iter)
    config.early_stop = bool(early_stop)
    config.tol_h = float(tol_h)
    config.tol_rhs = float(tol_rhs)
    config.min_iter = int(min_iter)
    config.consecutive_passes_required = int(consecutive_passes_required)
    return config


def build_cpp_tiers(specs: list[TierSpec], spot: float) -> list[lp.MDPTier]:
    return [spec.build_cpp_tier(spot) for spec in specs if spec.enabled]


def ordered_solution_tiers(specs: list[TierSpec], solution: lp.HJBSolution) -> list[CppTier]:
    mdp_iter = iter(solution.mdp_tiers)
    return [next(mdp_iter) for _ in specs]


def total_venue_count(solution: lp.HJBSolution) -> int:
    return len(solution.mdp_tiers) + (0 if solution.dark_pool is None else 1)


def q_index_for_policy_qgrid(q_grid: list[float] | np.ndarray, q: float) -> int:
    q_arr = np.asarray(list(q_grid), dtype=float)
    if q_arr.size == 0:
        raise ValueError("Policy q_grid is empty.")
    return int(np.argmin(np.abs(q_arr - float(q))))


def q_index_for_tier_policy(cpp_tier: CppTier, q: float) -> int:
    return q_index_for_policy_qgrid(cpp_tier.policy.q_grid, q)


def is_admissible(cpp_tier: CppTier, q: float, z: float, side: str) -> bool:
    return bool(cpp_tier.is_admissible(float(q), float(z), side))


def masked_quote_summary(
    cpp_tier: CppTier,
    q: float,
    z: float,
    side: str,
    mid_price: float,
    spread: float,
):
    if not is_admissible(cpp_tier, q, z, side):
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
            "enabled": bool(spec.enabled),
            "type": tier_kind_label(),
            "z": zf,
            "A(z)": spec.flow_A0 * zf ** (-spec.flow_theta - spec.flow_beta * zf),
            "delta_50(z)": spec.flow_shift - spec.flow_volume_shift * (zf - 1.0),
            "steepness": spec.flow_steepness,
            "mu(z, t=1min)": cpp_tier.expected_markout(zf, 1.0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def make_flow_curve_figure(cpp_tier: CppTier, spec: TierSpec) -> go.Figure:
    grid = np.linspace(-1.0, 1.0, 160)
    fig = go.Figure()

    for z in tier_sizes(spec):
        zf = float(z)
        vals = [cpp_tier.arrival_rate(float(d), zf) for d in grid]
        fig.add_trace(go.Scatter(x=grid, y=vals, mode="lines", name=f"{zf:g}"))

    fig.update_layout(
        title=f"Flow curves λ(δ, z) — {spec.name} ({tier_kind_label()})",
        xaxis_title="delta",
        yaxis_title="arrival rate",
        height=500,
    )
    return fig


def make_hit_ratio_figure(cpp_tier: CppTier, spec: TierSpec) -> go.Figure:
    grid = np.linspace(-1.0, 1.0, 160)
    fig = go.Figure()
    for z in tier_sizes(spec):
        zf = float(z)
        vals = [cpp_tier.hit_ratio(float(d), zf) for d in grid]
        fig.add_trace(go.Scatter(x=grid, y=vals, mode="lines", name=f"{zf:g}"))
    fig.update_layout(
        title=f"Hit ratios HR(δ, z) — {spec.name}",
        xaxis_title="delta",
        yaxis_title="hit ratio",
        yaxis={"range": [0.0, 1.0]},
        height=500,
    )
    return fig


def make_implied_hit_ratio_figure(cpp_tier: CppTier, spec: TierSpec) -> go.Figure:
    q_grid = [float(q) for q in cpp_tier.policy.q_grid]
    sizes = [float(z) for z in tier_sizes(spec)]
    n = len(sizes)
    fig = go.Figure()

    for i, zf in enumerate(sizes):
        t = 1.0 - i / max(n - 1, 1)
        shade = int((0.3 + 0.6 * t) * 255)
        bid_color = f"rgb(0, {shade // 2}, {shade})"
        ask_color = f"rgb({shade}, {shade // 4}, 0)"

        bid_hr, ask_hr = [], []
        for q in q_grid:
            if is_admissible(cpp_tier, q, zf, "bid"):
                d = cpp_tier.quote(q, zf, "bid")
                bid_hr.append(cpp_tier.hit_ratio(float(d), zf))
            else:
                bid_hr.append(np.nan)

            if is_admissible(cpp_tier, q, zf, "ask"):
                d = cpp_tier.quote(q, zf, "ask")
                ask_hr.append(cpp_tier.hit_ratio(float(d), zf))
            else:
                ask_hr.append(np.nan)

        fig.add_trace(go.Scatter(
            x=q_grid, y=bid_hr, mode="lines",
            name=f"{zf:g} bid", line=dict(color=bid_color),
        ))
        fig.add_trace(go.Scatter(
            x=q_grid, y=ask_hr, mode="lines",
            name=f"{zf:g} ask", line=dict(color=ask_color, dash="dash"),
        ))

    fig.update_layout(
        title=f"Implied hit ratios vs inventory — {spec.name}",
        xaxis_title="Inventory q",
        yaxis_title="Hit ratio",
        yaxis={"range": [0.0, 1.0]},
        height=500,
    )
    return fig


def make_quote_inventory_figure(
    cpp_tier: CppTier,
    spec: TierSpec,
    spread: float,
    mid_price: float,
) -> go.Figure:
    q_grid = [float(q) for q in cpp_tier.policy.q_grid]
    sizes = [float(z) for z in tier_sizes(spec)]
    n = len(sizes)
    fig = go.Figure()

    for i, zf in enumerate(sizes):
        # shade from light (small rung) to dark (large rung), range [0.3, 0.9]
        t = 1.0 - i / max(n - 1, 1)
        shade = int((0.3 + 0.6 * t) * 255)
        bid_color = f"rgb(0, {shade // 2}, {shade})"
        ask_color = f"rgb({shade}, {shade // 4}, 0)"

        bid_vals: list[float] = []
        ask_vals: list[float] = []
        for q in q_grid:
            bid = masked_quote_summary(cpp_tier, q, zf, "bid", mid_price, spread)
            ask = masked_quote_summary(cpp_tier, q, zf, "ask", mid_price, spread)
            bid_vals.append(np.nan if bid is None else float(bid.quote_relative_to_mid_pips))
            ask_vals.append(np.nan if ask is None else float(ask.quote_relative_to_mid_pips))

        fig.add_trace(go.Scatter(
            x=q_grid, y=bid_vals, mode="lines",
            name=f"{zf:g} bid", line=dict(color=bid_color),
        ))
        fig.add_trace(go.Scatter(
            x=q_grid, y=ask_vals, mode="lines",
            name=f"{zf:g} ask", line=dict(color=ask_color, dash="dash"),
        ))

    fig.add_hline(y=0.0)
    fig.update_layout(
        title=f"Quotes vs inventory — {spec.name} ({tier_kind_label()})",
        xaxis_title="Inventory q",
        yaxis_title="Quote relative to mid (pips)",
        height=500,
        yaxis=dict(range=[-20, 20]),
    )
    return fig


def make_quote_surface_figure(
    cpp_tier: CppTier,
    spec: TierSpec,
    spread: float,
    mid_price: float,
    side: str,
) -> go.Figure:
    q_grid = [float(q) for q in cpp_tier.policy.q_grid]
    sizes = [float(z) for z in tier_sizes(spec)]

    surface_z = []
    for zf in sizes:
        row = []
        for q in q_grid:
            qs = masked_quote_summary(cpp_tier, q, zf, side, mid_price, spread)
            row.append(np.nan if qs is None else float(qs.quote_relative_to_mid_pips))
        surface_z.append(row)

    colorscale = "Blues" if side == "bid" else "Reds"
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=q_grid, y=sizes, z=surface_z,
        colorscale=colorscale, opacity=0.85,
        showscale=False,
        contours=dict(
            x=dict(show=True, color="white", width=1),
            y=dict(show=True, color="white", width=1),
        ),
    ))
    fig.update_layout(
        title=f"{side.capitalize()} quote surface — {spec.name}",
        scene=dict(
            xaxis_title="Inventory q",
            yaxis_title="Rung size z",
            zaxis_title="Quote vs mid (pips)",
        ),
        height=500,
    )
    return fig


def make_ladder_figure(
    cpp_tier: CppTier,
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
        title=f"Volume premium — {spec.name} ({tier_kind_label()}) at q = {q:g}",
        xaxis_title="Trade size z",
        yaxis_title="Premium vs 1M quote (pips)",
        height=500,
    )
    return fig


def make_q_ladder_table(
    cpp_tier: CppTier,
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
        bid_active = bid_adm
        ask_active = ask_adm

        bid = masked_quote_summary(cpp_tier, q, zf, "bid", mid_price, spread)
        ask = masked_quote_summary(cpp_tier, q, zf, "ask", mid_price, spread)

        rows.append(
            {
                "enabled": bool(spec.enabled),
                "type": tier_kind_label(),
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


def make_dark_pool_parameter_table(venue: lp.DarkPoolVenue) -> pd.DataFrame:
    row: dict = {
        "dist_bid": venue.dist_bid.name(),
        "dist_ask": venue.dist_ask.name(),
        "lambda_bid": float(venue.lambda_bid),
        "lambda_ask": float(venue.lambda_ask),
        "fee_per_unit_bid": float(venue.fee_per_unit_bid),
        "fee_per_unit_ask": float(venue.fee_per_unit_ask),
        "allow_both_sides": bool(venue.allow_both_sides),
        "posted_sizes": ", ".join(f"{float(u):g}" for u in venue.posted_sizes),
    }
    if isinstance(venue.dist_bid, lp.GeometricArrivalDist):
        row["p_bid"] = float(venue.p_bid)
        row["mean fill size bid"] = round(1.0 / float(venue.p_bid), 4)
    if isinstance(venue.dist_ask, lp.GeometricArrivalDist):
        row["p_ask"] = float(venue.p_ask)
        row["mean fill size ask"] = round(1.0 / float(venue.p_ask), 4)
    return pd.DataFrame([row])


def make_dark_pool_size_figure(venue: lp.DarkPoolVenue) -> go.Figure:
    q_grid = [float(q) for q in venue.policy.q_grid]
    bid_vals = [
        float(venue.policy.bid_size[i]) if venue.policy.is_active(i, "bid") else 0.0
        for i in range(len(q_grid))
    ]
    ask_vals = [
        float(venue.policy.ask_size[i]) if venue.policy.is_active(i, "ask") else 0.0
        for i in range(len(q_grid))
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=q_grid, y=bid_vals, name="posted bid size", opacity=0.7))
    fig.add_trace(go.Bar(x=q_grid, y=ask_vals, name="posted ask size", opacity=0.7))
    fig.update_layout(
        title="Dark-pool optimal posted size vs inventory",
        xaxis_title="Inventory q",
        yaxis_title="Posted size",
        barmode="overlay",
        height=420,
    )
    return fig


def make_dark_pool_posted_size_table(venue: lp.DarkPoolVenue) -> pd.DataFrame:
    q_grid = [float(q) for q in venue.policy.q_grid]
    rows = []
    for i, q in enumerate(q_grid):
        bid_active = venue.policy.is_active(i, "bid")
        ask_active = venue.policy.is_active(i, "ask")
        rows.append({
            "Inventory": q,
            "Bid posted size": float(venue.policy.bid_size[i]) if bid_active else None,
            "Ask posted size": float(venue.policy.ask_size[i]) if ask_active else None,
        })
    return pd.DataFrame(rows).set_index("Inventory")


def _geom_pmf(p: float, k_values: list[int]) -> list[float]:
    r = 1.0 - p
    return [p * r ** (k - 1) for k in k_values]


def _zip_pmf(dist: lp.ZeroInflatedPoissonArrivalDist, k_values: list[int]) -> list[float]:
    """P(fill=k) for k in k_values (all >= 1) under the truncated ZIP used by the solver.

    The cap u = max(k_values). Normalization is over {1,...,u} as in the C++ implementation.
    """
    mu = float(dist.mu)
    p0 = float(dist.p0)
    u = max(k_values)
    log_mu = math.log(mu) if mu > 0.0 else float("-inf")
    raw = [math.exp(-mu + k * log_mu - math.lgamma(k + 1)) for k in range(1, u + 1)]
    Z = sum(raw)
    if Z < 1e-15:
        return [0.0] * len(k_values)
    return [(1.0 - p0) * raw[k - 1] / Z for k in k_values]


def _dist_sides(
    dist_bid: lp.ArrivalDistribution, dist_ask: lp.ArrivalDistribution
) -> list[tuple[str, lp.ArrivalDistribution]]:
    """Return [(label, dist), ...] deduplicating bid/ask if they have identical params."""
    if dist_bid.name() == dist_ask.name():
        if isinstance(dist_bid, lp.GeometricArrivalDist) and isinstance(dist_ask, lp.GeometricArrivalDist):
            if dist_bid.p == dist_ask.p:
                return [("bid/ask", dist_bid)]
        elif isinstance(dist_bid, lp.ZeroInflatedPoissonArrivalDist) and isinstance(dist_ask, lp.ZeroInflatedPoissonArrivalDist):
            if dist_bid.mu == dist_ask.mu and dist_bid.p0 == dist_ask.p0:
                return [("bid/ask", dist_bid)]
    return [("bid", dist_bid), ("ask", dist_ask)]


def make_dark_pool_arrival_figure(venue: lp.DarkPoolVenue) -> go.Figure:
    posted_sizes = sorted(int(round(float(u))) for u in venue.posted_sizes)
    max_size = max(posted_sizes)
    fill_sizes = list(range(1, max_size + 1))
    x_labels = [str(k) for k in fill_sizes]

    fig = go.Figure()
    for side, dist in _dist_sides(venue.dist_bid, venue.dist_ask):
        if isinstance(dist, lp.GeometricArrivalDist):
            y = _geom_pmf(float(dist.p), fill_sizes)
        elif isinstance(dist, lp.ZeroInflatedPoissonArrivalDist):
            y = _zip_pmf(dist, fill_sizes)
        else:
            fig.add_annotation(text=f"PMF not implemented for {dist.name()}", showarrow=False)
            break
        fig.add_trace(go.Bar(x=x_labels, y=y, name=side, opacity=0.7))

    fig.update_layout(
        title=f"Fill size density ({venue.dist_bid.name()})",
        xaxis_title="Fill size k",
        yaxis_title="P(fill = k)",
        height=400,
    )
    return fig


def make_dark_pool_full_fill_figure(venue: lp.DarkPoolVenue) -> go.Figure:
    posted_sizes = sorted(int(round(float(u))) for u in venue.posted_sizes)
    x_labels = [str(u) for u in posted_sizes]

    fig = go.Figure()
    for side, dist in _dist_sides(venue.dist_bid, venue.dist_ask):
        if isinstance(dist, lp.GeometricArrivalDist):
            r = 1.0 - float(dist.p)
            y = [r ** (u - 1) for u in posted_sizes]
        elif isinstance(dist, lp.ZeroInflatedPoissonArrivalDist):
            # P(fill = u | cap u) — last term of the truncated ZIP
            y = [_zip_pmf(dist, list(range(1, u + 1)))[-1] for u in posted_sizes]
        else:
            fig.add_annotation(text=f"P(full fill) not implemented for {dist.name()}", showarrow=False)
            break
        fig.add_trace(go.Bar(x=x_labels, y=y, name=side, opacity=0.7))

    fig.update_layout(
        title="P(full fill) by posted size",
        xaxis_title="Posted size u",
        yaxis_title="P(fill = u)",
        height=400,
    )
    return fig


def make_internalization_time_figure(
    internalization_time: lp.PolynomialInternalizationTime,
    q_abs_max: float,
) -> go.Figure:
    q_abs = np.linspace(0.0, q_abs_max, 300)
    t_vals = [internalization_time.value(float(q)) for q in q_abs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_abs, y=t_vals, mode="lines", name="t(|q|)"))
    fig.update_layout(
        title="Internalization time t(|q|) = τ₀ + τ₁|q| + τ₂|q|²",
        xaxis_title="|q| (absolute inventory)",
        yaxis_title="t(|q|)",
        height=420,
    )
    return fig


def make_markout_figure(cpp_tier: CppTier, spec: TierSpec) -> go.Figure:
    t_grid = np.linspace(0.0, 10.0, 300)
    fig = go.Figure()
    for z in tier_sizes(spec):
        zf = float(z)
        mu_vals = [cpp_tier.expected_markout(zf, float(t)) * 10_000 for t in t_grid]
        fig.add_trace(go.Scatter(x=list(t_grid), y=mu_vals, mode="lines", name=f"z={zf:g}"))
    fig.update_layout(
        title=f"Markout — {spec.name}",
        xaxis_title="Time since trade [minutes]",
        yaxis_title="Market impact [pips]",
        height=500,
    )
    return fig


def make_h_figure(solution: lp.HJBSolution) -> go.Figure:
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
    "The dark pool trades at mid with fee or rebate, and the only control is posted size."
)

with st.sidebar:
    with st.sidebar.expander("Global parameters", expanded=False):
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

        mid_price = st.number_input("display mid price", value=1.000000, step=0.000100, format="%.6f")

    with st.sidebar.expander("Spot process", expanded=False):
        spot = st.number_input("spot (quote CCY per base CCY)", value=11.5, min_value=0.001, step=0.1, format="%.4f")
        spot_drift = st.number_input("spot_drift", value=0.0, step=0.001, format="%.5f")
        spread = st.number_input("reference spread", value=20.0 / 10000.0, step=1.0 / 10000.0, format="%.6f")

    with st.sidebar.expander("Optimization / stopping", expanded=False):
        golden_tol = st.number_input("golden_tol", value=1e-4, format="%.1e")
        golden_max_iter = st.number_input("golden_max_iter", value=32, step=1, min_value=1)
        early_stop = st.checkbox("Enable early stopping", value=True)
        tol_h = st.number_input("tol_h", value=1e-5, format="%.1e")
        tol_rhs = st.number_input("tol_rhs", value=1e-4, format="%.1e")
        min_iter = st.number_input("min_iter", value=5, step=1, min_value=0)
        consecutive_passes_required = st.number_input("consecutive passes required", value=3, step=1, min_value=1)

    with st.sidebar.expander("Carry cost", expanded=False):
        sigma = st.number_input(
            "Volatility σ [pips / √min]",
            value=20.0,
            step=1.0,
            format="%.2f",
            help="One-minute volatility in pips. Used in both carry terms.",
        ) / 10_000.0
        risk_aversion = st.number_input(
            "γ (risk aversion)",
            value=10.0, step=1.0, format="%.2f",
            help="Quadratic penalty: γσ²·q²·t(q).",
        )
    with st.sidebar.expander("Internalization time", expanded=False):
        st.caption("t(q) = τ₀ + τ₁|q| + τ₂|q|²  —  time in minutes")
        tau0 = st.number_input("tau0 (constant) [min]", value=4.0, step=1.0, format="%.4f",
                               help="Minimum internalization time at zero inventory, in minutes.")
        tau1 = st.number_input("tau1 (linear) [min / lot]", value=0.070, step=0.01, format="%.4f",
                               help="Additional minutes per unit of absolute inventory.")
        tau2 = st.number_input("tau2 (quadratic) [min / lot²]", value=0.0084, step=0.0005, format="%.5f",
                               help="Quadratic growth in minutes per lot².")

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
    dark_pool_spec = build_dark_pool_spec_from_ui()
except ValueError as exc:
    errors.append(str(exc))
    dark_pool_spec = None

active_tier_specs = enabled_tier_specs(tier_specs)
active_mdp_count = sum(isinstance(spec, MDPTierSpec) for spec in active_tier_specs)
disabled_tier_names = [spec.name for spec in tier_specs if not spec.enabled]

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

if not active_tier_specs and dark_pool_spec is None:
    errors.append("Enable at least one tier or the dark-pool venue.")

if errors:
    for error in errors:
        st.error(error)
    st.stop()

global_internalization_time = lp.PolynomialInternalizationTime(
    tau0=float(tau0), tau1=float(tau1), tau2=float(tau2)
)

mdp_cpp_tiers = build_cpp_tiers(active_tier_specs, float(spot))
dark_pool_cpp = None if dark_pool_spec is None else dark_pool_spec.build_cpp_venue(float(spot))

penalty = lp.PolynomialInventoryPenalty(
    carry_cost=lp.CarryCost(risk_aversion=float(risk_aversion), sigma=float(sigma)),
    internalization_time=global_internalization_time,
)

config = build_solver_config(
    q_grid=q_grid,
    dt=float(dt),
    n_iter=int(n_iter),
    spread=float(spread),
    spot=float(spot),
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
        dark_pool=dark_pool_cpp,
    )
    solution: lp.HJBSolution = solver.solve()

diag = solution.diagnostics
solution_tiers = ordered_solution_tiers(active_tier_specs, solution)

st.success("Solver run complete.")

summary_cols = st.columns(4)
summary_cols[0].metric("Configured tiers", len(tier_specs))
summary_cols[1].metric("Active MDP tiers", active_mdp_count)
summary_cols[2].metric("Dark pool", "on" if dark_pool_spec is not None else "off")
summary_cols[3].metric("Disabled tiers", len(disabled_tier_names))

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

tab_names = ["Internalization time"] + [f"{spec.name} [{tier_kind_label()}]" for spec in active_tier_specs]
if solution.dark_pool is not None:
    tab_names.append("Dark Pool")

if tab_names:
    tabs = st.tabs(tab_names)
    available_q = [float(q) for q in solution.q_grid]
    default_q = 0.0 if 0.0 in available_q else available_q[len(available_q) // 2]

    with tabs[0]:
        internalization_time_model = lp.PolynomialInternalizationTime(
            tau0=float(tau0), tau1=float(tau1), tau2=float(tau2)
        )
        st.plotly_chart(
            make_internalization_time_figure(internalization_time_model, float(q_abs_max)),
            use_container_width=True,
        )

    for idx, (spec, cpp_tier) in enumerate(zip(active_tier_specs, solution_tiers)):
        with tabs[idx + 1]:
            st.subheader(f"Tier: {spec.name}")
            st.caption(f"Type: {tier_kind_label()}")

            with st.expander("Tier parameters", expanded=False):
                st.dataframe(make_flow_parameter_table(spec, cpp_tier), use_container_width=True)

            with st.expander("Flow curves", expanded=False):
                st.plotly_chart(make_flow_curve_figure(cpp_tier, spec), use_container_width=True)

            with st.expander("Hit ratios", expanded=False):
                st.plotly_chart(make_hit_ratio_figure(cpp_tier, spec), use_container_width=True)

            with st.expander("Implied hit ratios vs inventory", expanded=False):
                st.plotly_chart(make_implied_hit_ratio_figure(cpp_tier, spec), use_container_width=True)

            with st.expander("Markouts", expanded=False):
                st.plotly_chart(make_markout_figure(cpp_tier, spec), use_container_width=True)

            with st.expander("Quotes vs inventory", expanded=False):
                st.plotly_chart(
                    make_quote_inventory_figure(cpp_tier, spec, float(config.spread), float(mid_price)),
                    use_container_width=True,
                )

            with st.expander("Quote surface (3D)", expanded=False):
                col_bid, col_ask = st.columns(2)
                with col_bid:
                    st.plotly_chart(
                        make_quote_surface_figure(cpp_tier, spec, float(config.spread), float(mid_price), "bid"),
                        use_container_width=True,
                    )
                with col_ask:
                    st.plotly_chart(
                        make_quote_surface_figure(cpp_tier, spec, float(config.spread), float(mid_price), "ask"),
                        use_container_width=True,
                    )

            with st.expander("Volume premium", expanded=False):
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

            with st.expander("What this tab is solving"):
                st.markdown(
                    r"""
**MDP tier — rung-by-rung two-sided ladder**

The dealer solves a stationary HJB equation. With value decomposition $V(x, q, m) = x + qm + h(q)$, the HJB reduces to a fixed-point problem in the inventory value-adjustment $h(q)$.

**Fill payoff for a bid quote at rung $z$, inventory $q$:**

$$
\Pi^{\text{bid}}(z, q, \delta) = \lambda(\delta, z)\Bigl[z s(0.5 - \delta) + z\,\mu(z,\,t(q+z)) + h(q+z) - h(q)\Bigr]
$$

where $s$ is the spread, $\delta$ is the quoted delta, $\mu(z, t)$ is the expected markout, and $t(q)$ is the internalization time.

**Ask is symmetric** (inventory decreases by $z$, markout evaluated at $t(q-z)$).

**Carry cost penalty:**

$$
\Pi(q) = -\gamma\sigma^2 q^2\, t(q)
$$

**Flow curve:** $\lambda(\delta, z) = A(z)\cdot\sigma\!\left(\kappa\bigl(\delta - \delta_{50}(z)\bigr)\right)$, with $A(z) = A_0\,z^{-\theta - \beta z}$.

**Markout model:** $\mu(z, t) = \alpha(z)\sqrt{t} + \beta(z)\,t$, with $\alpha(z) = a_0 + a_1 z + a_2 z^2$ and $\beta(z) = b_0 + b_1 z + b_2 z^2$.

**Internalization time:** $t(q) = \tau_0 + \tau_1|q| + \tau_2 q^2$.
                    """
                )

    if solution.dark_pool is not None:
        dark_pool_tab = tabs[-1]
        venue = solution.dark_pool
        with dark_pool_tab:
            st.subheader("Dark pool venue")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("λ bid", f"{float(venue.lambda_bid):.4f}")
            c2.metric("λ ask", f"{float(venue.lambda_ask):.4f}")
            if isinstance(venue.dist_bid, lp.GeometricArrivalDist):
                c3.metric("mean fill size bid", f"{1.0 / float(venue.p_bid):.3f}")
            else:
                c3.metric("dist bid", venue.dist_bid.name())
            if isinstance(venue.dist_ask, lp.GeometricArrivalDist):
                c4.metric("mean fill size ask", f"{1.0 / float(venue.p_ask):.3f}")
            else:
                c4.metric("dist ask", venue.dist_ask.name())

            c5, c6, c7 = st.columns(3)
            c5.metric("fee / rebate bid", f"{float(venue.fee_per_unit_bid):.5f}")
            c6.metric("fee / rebate ask", f"{float(venue.fee_per_unit_ask):.5f}")
            c7.metric("both sides allowed", "yes" if bool(venue.allow_both_sides) else "no")

            with st.expander("Dark-pool parameters", expanded=False):
                st.dataframe(make_dark_pool_parameter_table(venue), use_container_width=True)

            with st.expander("Arrival size density", expanded=False):
                st.plotly_chart(make_dark_pool_arrival_figure(venue), use_container_width=True)
            with st.expander("P(full fill) by posted size", expanded=False):
                st.plotly_chart(make_dark_pool_full_fill_figure(venue), use_container_width=True)
            with st.expander("Dark pool policy", expanded=False):
                st.plotly_chart(make_dark_pool_size_figure(venue), use_container_width=True)
                st.dataframe(make_dark_pool_posted_size_table(venue), use_container_width=True)

            with st.expander("What this tab is solving"):
                st.markdown(
                    r"""
**Dark-pool venue**

The dealer posts a fixed bid size $u^{\text{bid}}$ and/or ask size $u^{\text{ask}}$. Execution is at the current mid price — no quote-price optimisation.

**Arrival process:** Poisson with constant intensity $\lambda^{\text{bid}}$ / $\lambda^{\text{ask}}$.

**Order-size distribution:** Pluggable via `ArrivalDistribution`. The default is geometric with success probability $p$ (mean fill size $= 1/p$); zero-inflated Poisson is also supported.

**Executed size:** $\min(u, Z)$ where $Z$ is the incoming order size drawn from the distribution and $u$ is the posted size.

**Fill payoff for bid, posted size $u$:**

$$
\Pi^{\text{bid}}(u) = \sum_{k=1}^{u} P(\text{fill}=k)\,\bigl[h(q+k) - h(q) - \text{fee}\cdot k\bigr]
$$

**Optimizer chooses** $u^{\text{bid}}$, $u^{\text{ask}}$, or both, subject to the venue's allow-both-sides flag.

**Inventory grid constraints:**
- strictly increasing, symmetric around 0, odd-length, with 0 at the centre index
                    """
                )

st.caption("After replacing the C++ files, rebuild the extension and restart Streamlit.")