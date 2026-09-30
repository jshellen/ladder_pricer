from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Always prefer the package shipped next to this app.  Streamlit keeps the Python
# process alive across reruns, so an older editable/site-packages copy can otherwise
# remain in sys.modules after the project directory is replaced.
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_PACKAGE_DIR = (_PROJECT_ROOT / "ladder_pricer").resolve()
_project_root_str = str(_PROJECT_ROOT)
if not sys.path or sys.path[0] != _project_root_str:
    sys.path = [_project_root_str] + [p for p in sys.path if p != _project_root_str]
importlib.invalidate_caches()

_existing_lp = sys.modules.get("ladder_pricer")
if _existing_lp is not None:
    _existing_file = getattr(_existing_lp, "__file__", None)
    _existing_is_local = False
    if _existing_file:
        try:
            _existing_is_local = Path(_existing_file).resolve().parent == _LOCAL_PACKAGE_DIR
        except OSError:
            _existing_is_local = False
    if not _existing_is_local:
        del sys.modules["ladder_pricer"]

try:
    import ladder_pricer as lp
except ImportError as exc:
    raise ImportError(
        "Could not import the local 'ladder_pricer' package shipped with app.py."
    ) from exc

# A same-path module can also be stale after hot-replacing project files.  Reload
# once if the benchmark-version sentinel is missing.
if not hasattr(lp, "PNL_BENCHMARK_VERSION"):
    lp = importlib.reload(lp)

_loaded_lp_file = Path(getattr(lp, "__file__", "")).resolve()
if _loaded_lp_file.parent != _LOCAL_PACKAGE_DIR:
    raise RuntimeError(
        "Streamlit imported ladder_pricer from the wrong location: "
        f"{_loaded_lp_file}. Expected {_LOCAL_PACKAGE_DIR}. Restart Streamlit "
        "from this project directory."
    )
if not hasattr(lp, "PNL_BENCHMARK_VERSION"):
    raise RuntimeError(
        "The local ladder_pricer package is stale: PNL_BENCHMARK_VERSION is missing. "
        "Restart Streamlit from this project directory."
    )

_PNL_BENCHMARK_VERSION = lp.PNL_BENCHMARK_VERSION


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

SESSION_START_LABEL = "07:45"
SESSION_END_LABEL = "17:15"
SESSION_HORIZON_MINUTES = 9.5 * 60.0  # 570 minutes
BASE_CCY_LABEL = "EUR"
QUOTE_CCY_LABEL = "SEK"

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
class SaturatingMarkoutSpec:
    impact_scale_pips: float = 1.0
    size_exponent: float = 0.5
    tau_minutes: float = 0.5

    def build(self, spot: float) -> lp.SaturatingMarkoutModel:
        del spot  # kept in the spec API for consistency with other tier builders
        return lp.SaturatingMarkoutModel(
            impact_scale=self.impact_scale_pips / 10_000.0,
            size_exponent=self.size_exponent,
            tau=self.tau_minutes,
        )


MarkoutSpec = SaturatingMarkoutSpec


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

    def build_tier(self, spot: float) -> lp.MDPTier:
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

    def build_venue(self, spot: float) -> lp.DarkPoolVenue:
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
PricerTier = lp.MDPTier


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
        "impact_scale_pips": 1.0,
        "impact_size_exponent": 0.5,
        "impact_tau_minutes": 0.5,
        "use_markout": True,
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
            "delta_min": -10.0,
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
            "delta_min": -10.0,
            "delta_max": 100.0,
        },
        {
            "enabled": False,
            "kind": "mdp",
            "name": "Tier 3",
            "sizes": "1",
            "flow_A0": 1.0,
            "flow_theta": 0.144,
            "flow_beta": 0.0857,
            "flow_steepness": 20.0,
            "flow_shift": 0.52,
            "flow_volume_shift": 0.026,
            "impact_scale_pips": 4.0,
            "impact_size_exponent": 0.5,
            "impact_tau_minutes": 0.10,
            "use_markout": True,
            "delta_min": -10.0,
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
            help="RFQ intensity scale [1/min]. The resulting fill intensity is A(z) × hit_ratio(δ,z), also in fills/minute.",
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
            f"Include adverse markout {i + 1}",
            value=bool(defaults.get("use_markout", True)),
            key=f"use_markout_{i}",
        )
        st.caption(r"$m(z,t)=a z^{\beta}(1-e^{-t/\tau})$ — positive adverse-selection cost")
        c5, c6, c7 = st.columns(3)
        impact_scale_pips = c5.number_input(
            f"Impact scale a {i + 1} [pips]",
            value=float(defaults.get("impact_scale_pips", 1.0)),
            min_value=0.0,
            step=0.1,
            format="%.3f",
            key=f"impact_scale_pips_{i}",
            help="Eventual markout for a 1M trade, in pips.",
        )
        impact_size_exponent = c6.number_input(
            f"Size exponent β {i + 1}",
            value=float(defaults.get("impact_size_exponent", 0.5)),
            min_value=0.0,
            step=0.05,
            format="%.3f",
            key=f"impact_size_exponent_{i}",
            help="Controls how eventual impact grows with trade size: M(z)=a·z^β.",
        )
        impact_tau_minutes = c7.number_input(
            f"Impact τ {i + 1} [min]",
            value=float(defaults.get("impact_tau_minutes", 0.5)),
            min_value=0.001,
            step=0.1,
            format="%.2f",
            key=f"impact_tau_minutes_{i}",
            help="Time scale of impact. About 63% of eventual impact is reached after one τ.",
        )
        markout_spec: MarkoutSpec = SaturatingMarkoutSpec(
            impact_scale_pips=float(impact_scale_pips),
            size_exponent=float(impact_size_exponent),
            tau_minutes=float(impact_tau_minutes),
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
            "Dark pool λ bid [1/min]",
            value=float(defaults["lambda_bid"]),
            min_value=0.0,
            step=0.01,
            format="%.4f",
        )
        lambda_ask = c2.number_input(
            "Dark pool λ ask [1/min]",
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
    spread: float,
    spot: float,
    spot_drift: float,
) -> lp.SolverConfig:
    config = lp.SolverConfig()
    config.q_grid = [float(q) for q in q_grid]
    config.spread = float(spread)
    config.spot = float(spot)
    config.spot_drift = float(spot_drift)
    return config


def build_tiers(specs: list[TierSpec], spot: float) -> list[lp.MDPTier]:
    return [spec.build_tier(spot) for spec in specs if spec.enabled]


def ordered_solution_tiers(specs: list[TierSpec], solution: lp.HJBSolution) -> list[PricerTier]:
    mdp_iter = iter(solution.mdp_tiers)
    return [next(mdp_iter) for _ in specs]


def total_venue_count(solution: lp.HJBSolution) -> int:
    return len(solution.mdp_tiers) + (0 if solution.dark_pool is None else 1)


def _rounded_flat(values, digits: int = 14) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=float).ravel()
    return tuple(float(x) for x in np.round(arr, digits))


def monte_carlo_model_signature(
    solver: lp.HJBLadderSolver,
    *,
    horizon_minutes: float,
    n_paths: int,
    initial_inventory: float,
    seed: int,
    sigma: float,
) -> tuple:
    """Signature for cached Monte Carlo + closed-form results.

    Include the benchmark implementation version, solved policies and every
    model input used directly by the simulation/benchmark. This intentionally
    invalidates cached results after PnL-accounting changes even when the HJB
    policy itself is unchanged.
    """
    tier_sig = []
    for tier in solver.mdp_tiers:
        tier_sig.append((
            tier.name,
            tuple(float(z) for z in tier.sizes_),
            float(tier.flow_curve.A0), float(tier.flow_curve.theta),
            float(tier.flow_curve.beta), float(tier.flow_curve.shift),
            float(tier.flow_curve.steepness), float(tier.flow_curve.volume_shift),
            bool(tier.use_markout),
            float(tier.markout_model.impact_scale),
            float(tier.markout_model.size_exponent),
            float(tier.markout_model.tau),
            float(tier.delta_min), float(tier.delta_max),
            _rounded_flat(tier.policy.bid),
            _rounded_flat(tier.policy.ask),
        ))

    dark_sig = None
    if solver.dark_pool is not None:
        venue = solver.dark_pool
        dark_sig = (
            repr(venue.dist_bid), repr(venue.dist_ask),
            float(venue.fee_per_unit_bid), float(venue.fee_per_unit_ask),
            tuple(float(x) for x in venue.posted_sizes),
            bool(venue.allow_both_sides),
            _rounded_flat(venue.policy.bid_size),
            _rounded_flat(venue.policy.ask_size),
            tuple(bool(x) for x in venue.policy.bid_active),
            tuple(bool(x) for x in venue.policy.ask_active),
        )

    return (
        _PNL_BENCHMARK_VERSION,
        tuple(float(q) for q in solver.solve_q_grid_),
        float(solver.config.spot), float(solver.config.spot_drift),
        float(solver.config.spread),
        tuple(tier_sig), dark_sig,
        float(horizon_minutes), int(n_paths), float(initial_inventory),
        int(seed), float(sigma),
    )




def efficient_frontier_signature(
    *,
    q_grid: np.ndarray,
    spot: float,
    spot_drift: float,
    spread: float,
    sigma: float,
    horizon_minutes: float,
    initial_inventory: float,
    tier_specs: list[TierSpec],
    dark_pool_spec: DarkPoolSpec | None,
    tau0: float,
    tau1: float,
    tau2: float,
    gammas: np.ndarray,
) -> tuple:
    tier_sig = []
    for spec in tier_specs:
        tier_sig.append((
            type(spec).__name__,
            bool(spec.enabled),
            str(spec.name),
            tuple(float(z) for z in tier_sizes(spec)),
            float(spec.flow_A0), float(spec.flow_theta), float(spec.flow_beta),
            float(spec.flow_steepness), float(spec.flow_shift), float(spec.flow_volume_shift),
            bool(spec.use_markout),
            float(spec.markout_spec.impact_scale_pips),
            float(spec.markout_spec.size_exponent),
            float(spec.markout_spec.tau_minutes),
            float(spec.delta_min), float(spec.delta_max),
        ))

    dark_sig = None
    if dark_pool_spec is not None:
        dark_sig = (
            str(dark_pool_spec.dist_type),
            float(dark_pool_spec.lambda_bid), float(dark_pool_spec.lambda_ask),
            float(dark_pool_spec.p_bid), float(dark_pool_spec.p_ask),
            float(dark_pool_spec.mu_bid), float(dark_pool_spec.mu_ask),
            float(dark_pool_spec.p0_bid), float(dark_pool_spec.p0_ask),
            float(dark_pool_spec.fee_per_unit_bid), float(dark_pool_spec.fee_per_unit_ask),
            tuple(float(x) for x in dark_pool_spec.posted_sizes),
            bool(dark_pool_spec.allow_both_sides),
        )

    return (
        'efficient_frontier_v1',
        _PNL_BENCHMARK_VERSION,
        tuple(float(q) for q in q_grid),
        float(spot), float(spot_drift), float(spread), float(sigma),
        float(horizon_minutes), float(initial_inventory),
        float(tau0), float(tau1), float(tau2),
        tuple(float(x) for x in np.asarray(gammas, dtype=float)),
        tuple(tier_sig),
        dark_sig,
    )


def compute_efficient_frontier(
    *,
    q_grid: np.ndarray,
    spot: float,
    spot_drift: float,
    spread: float,
    sigma: float,
    horizon_minutes: float,
    initial_inventory: float,
    tier_specs: list[TierSpec],
    dark_pool_spec: DarkPoolSpec | None,
    tau0: float,
    tau1: float,
    tau2: float,
    gammas: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for gamma in np.asarray(gammas, dtype=float):
        local_tiers = build_tiers(tier_specs, float(spot))
        local_dark_pool = None if dark_pool_spec is None else dark_pool_spec.build_venue(float(spot))
        local_penalty = lp.QuadraticInventoryPenalty(
            carry_cost=lp.CarryCost(risk_aversion=float(gamma), sigma=float(sigma)),
        )
        local_config = build_solver_config(
            q_grid=q_grid,
            spread=float(spread),
            spot=float(spot),
            spot_drift=float(spot_drift),
        )
        local_internalization_time = lp.PolynomialInternalizationTime(
            tau0=float(tau0), tau1=float(tau1), tau2=float(tau2)
        )
        local_solver = lp.HJBLadderSolver(
            config=local_config,
            penalty=local_penalty,
            internalization_time=local_internalization_time,
            mdp_tiers=local_tiers,
            dark_pool=local_dark_pool,
        )
        local_solution = local_solver.solve()
        local_stats = local_solver.closed_form_pnl_statistics(
            horizon_minutes=float(horizon_minutes),
            sigma=float(sigma),
            initial_inventory=float(initial_inventory),
        )
        rows.append({
            'gamma': float(gamma),
            'expected_pnl_base_ccy': float(local_stats.expected_pnl_base_ccy_at_reference_spot),
            'std_pnl_base_ccy': float(local_stats.std_pnl_base_ccy_at_reference_spot),
            'expected_pnl_quote_ccy': float(local_stats.expected_pnl_quote_ccy),
            'std_pnl_quote_ccy': float(local_stats.std_pnl_quote_ccy),
            'average_reward': float(local_solution.average_reward),
            'n_states': float(len(local_solution.q_grid)),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['std_pnl_base_ccy', 'gamma']).reset_index(drop=True)
    return df

def q_index_for_policy_qgrid(q_grid: list[float] | np.ndarray, q: float) -> int:
    q_arr = np.asarray(list(q_grid), dtype=float)
    if q_arr.size == 0:
        raise ValueError("Policy q_grid is empty.")
    return int(np.argmin(np.abs(q_arr - float(q))))


def q_index_for_tier_policy(cpp_tier: PricerTier, q: float) -> int:
    return q_index_for_policy_qgrid(cpp_tier.policy.q_grid, q)


def is_admissible(cpp_tier: PricerTier, q: float, z: float, side: str) -> bool:
    return bool(cpp_tier.is_admissible(float(q), float(z), side))


def masked_quote_summary(
    cpp_tier: PricerTier,
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

def make_flow_parameter_table(spec: TierSpec, cpp_tier: PricerTier) -> pd.DataFrame:
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
            "markout(z, 1 min) [pips]": cpp_tier.expected_markout(zf, 1.0) * 10_000.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def make_flow_curve_figure(cpp_tier: PricerTier, spec: TierSpec) -> go.Figure:
    grid = np.linspace(-1.0, 1.0, 160)
    fig = go.Figure()

    for z in tier_sizes(spec):
        zf = float(z)
        vals = [cpp_tier.arrival_rate(float(d), zf) for d in grid]
        fig.add_trace(go.Scatter(x=grid, y=vals, mode="lines", name=f"{zf:g}"))

    fig.update_layout(
        title=f"Flow curves λ(δ, z) — {spec.name} ({tier_kind_label()})",
        xaxis_title="delta",
        yaxis_title="Fill intensity [1/min]",
        height=500,
    )
    return fig


def make_hit_ratio_figure(cpp_tier: PricerTier, spec: TierSpec) -> go.Figure:
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


def make_implied_hit_ratio_figure(cpp_tier: PricerTier, spec: TierSpec) -> go.Figure:
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
    cpp_tier: PricerTier,
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
    cpp_tier: PricerTier,
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
    cpp_tier: PricerTier,
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
    cpp_tier: PricerTier,
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

    The cap u = max(k_values). Normalization is over {1,...,u}.
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


def make_markout_figure(cpp_tier: PricerTier, spec: TierSpec) -> go.Figure:
    horizon_minutes = 15.0
    t_minutes = np.linspace(0.0, horizon_minutes, 300)
    fig = go.Figure()
    for z in tier_sizes(spec):
        zf = float(z)
        mu_vals = [
            cpp_tier.expected_markout(zf, float(t)) * 10_000.0
            for t in t_minutes
        ]
        fig.add_trace(go.Scatter(x=list(t_minutes), y=mu_vals, mode="lines", name=f"{zf:g}M"))
    fig.update_layout(
        title=f"Saturating markout — {spec.name}",
        xaxis_title="Time since trade [minutes]",
        yaxis_title="Adverse markout [pips]",
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


def make_pnl_distribution_figure(
    result: lp.MonteCarloResult,
    closed_form_value: float,
) -> go.Figure:
    pnl = np.asarray(result.pnl_base_ccy, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl,
        nbinsx=max(20, min(80, int(math.sqrt(max(1, len(pnl)))) * 2)),
        name="Monte Carlo PnL",
        opacity=0.78,
    ))
    if len(pnl):
        fig.add_vline(
            x=float(np.mean(pnl)), line_dash="dash",
            annotation_text="MC mean", annotation_position="top left",
        )
    fig.add_vline(
        x=float(closed_form_value), line_dash="dot",
        annotation_text="Closed-form mean", annotation_position="top right",
    )
    fig.update_layout(
        title=f"Trading-session PnL distribution — {SESSION_START_LABEL}–{SESSION_END_LABEL}",
        xaxis_title=f"PnL [{BASE_CCY_LABEL}]",
        yaxis_title="Path count",
        bargap=0.03,
        height=500,
    )
    return fig


def make_inventory_paths_distribution_figure(result: lp.MonteCarloResult) -> go.Figure:
    fig = go.Figure()

    # A handful of actual paths give intuition for the jump dynamics. The
    # percentile envelope itself is computed from every simulated path.
    for i, path in enumerate(result.sample_paths):
        fig.add_trace(go.Scatter(
            x=path.inventory_event_times,
            y=path.inventory_event_values,
            mode="lines",
            line_shape="hv",
            name=f"sample {i + 1}",
            opacity=0.22,
            showlegend=False,
            hovertemplate="%{x:.1f} min<br>q=%{y:.2f}M<extra></extra>",
        ))

    times = np.asarray(result.inventory_sample_times, dtype=float)
    lo = np.asarray(result.inventory_ci_lower, dtype=float)
    med = np.asarray(result.inventory_median, dtype=float)
    hi = np.asarray(result.inventory_ci_upper, dtype=float)
    if len(times):
        fig.add_trace(go.Scatter(
            x=times, y=lo, mode="lines",
            line=dict(width=0),
            name="2.5th percentile",
            showlegend=False,
            hovertemplate="%{x:.1f} min<br>2.5%=%{y:.2f}M<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=times, y=hi, mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="95% interval",
            hovertemplate="%{x:.1f} min<br>97.5%=%{y:.2f}M<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=times, y=med, mode="lines",
            name="Median inventory",
            line=dict(width=2.5),
            hovertemplate="%{x:.1f} min<br>median=%{y:.2f}M<extra></extra>",
        ))

    tickvals = [0.0, 120.0, 240.0, 360.0, 480.0, SESSION_HORIZON_MINUTES]
    ticktext = ["07:45", "09:45", "11:45", "13:45", "15:45", "17:15"]
    fig.add_hline(y=0.0, line_dash="dot")
    fig.update_layout(
        title=f"Inventory paths — median and 95% simulation interval — {SESSION_START_LABEL}–{SESSION_END_LABEL}",
        xaxis_title="Session time",
        yaxis_title="Inventory [M]",
        xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext),
        height=430,
        hovermode="x unified",
    )
    return fig




def make_efficient_frontier_figure(frontier: pd.DataFrame, current_gamma: float) -> go.Figure:
    fig = go.Figure()
    if frontier.empty:
        fig.update_layout(
            title='Efficient frontier',
            xaxis_title=f'Closed-form stdev [{BASE_CCY_LABEL}]',
            yaxis_title=f'Closed-form expected PnL [{BASE_CCY_LABEL}]',
            height=520,
        )
        return fig

    df = frontier.sort_values(['std_pnl_base_ccy', 'gamma']).reset_index(drop=True)
    hover = [
        f"γ={row.gamma:.4g}<br>mean={_format_ccy(row.expected_pnl_base_ccy)} {BASE_CCY_LABEL}<br>std={_format_ccy(row.std_pnl_base_ccy)} {BASE_CCY_LABEL}"
        for row in df.itertuples()
    ]
    fig.add_trace(go.Scatter(
        x=df['std_pnl_base_ccy'],
        y=df['expected_pnl_base_ccy'],
        mode='lines+markers',
        name='Closed-form frontier',
        customdata=np.array(hover, dtype=object),
        hovertemplate='%{customdata}<extra></extra>',
    ))

    idx_current = int(np.argmin(np.abs(df['gamma'].to_numpy(dtype=float) - float(current_gamma))))
    current = df.iloc[idx_current]
    fig.add_trace(go.Scatter(
        x=[float(current['std_pnl_base_ccy'])],
        y=[float(current['expected_pnl_base_ccy'])],
        mode='markers',
        name=f'Current γ={float(current["gamma"]):.4g}',
        marker=dict(symbol='star', size=15, line=dict(width=1.5)),
        hovertemplate=(
            f"current γ={float(current['gamma']):.4g}<br>mean={_format_ccy(float(current['expected_pnl_base_ccy']))} {BASE_CCY_LABEL}"
            f"<br>std={_format_ccy(float(current['std_pnl_base_ccy']))} {BASE_CCY_LABEL}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title='Efficient frontier — closed-form mean vs. standard deviation',
        xaxis_title=f'Closed-form PnL stdev [{BASE_CCY_LABEL}]',
        yaxis_title=f'Closed-form expected PnL [{BASE_CCY_LABEL}]',
        height=520,
        hovermode='closest',
    )
    return fig



def make_phi_risk_adjusted_figure(frontier: pd.DataFrame, current_gamma: float) -> go.Figure:
    """Plot the frontier parameter phi against expected PnL / PnL stdev.

    The current implementation uses the same risk-aversion parameter that is
    called gamma in the HJB solver.  The plot labels it phi because phi is the
    economic frontier parameter shown to the user.
    """
    fig = go.Figure()
    if frontier.empty:
        fig.update_layout(
            title='Risk-adjusted return vs. φ',
            xaxis_title='φ (risk aversion)',
            yaxis_title='Expected PnL / stdev',
            height=460,
        )
        return fig

    df = frontier.sort_values('gamma').reset_index(drop=True).copy()
    std = df['std_pnl_base_ccy'].to_numpy(dtype=float)
    mean = df['expected_pnl_base_ccy'].to_numpy(dtype=float)
    ratio = np.divide(mean, std, out=np.full_like(mean, np.nan), where=np.abs(std) > 1e-15)
    df['mean_over_std'] = ratio

    fig.add_trace(go.Scatter(
        x=df['gamma'],
        y=df['mean_over_std'],
        mode='lines+markers',
        name='Expected PnL / stdev',
        customdata=np.column_stack((df['expected_pnl_base_ccy'], df['std_pnl_base_ccy'])),
        hovertemplate=(
            'φ=%{x:.4g}<br>'
            'E[PnL]/Std=%{y:.4f}<br>'
            f'E[PnL]=%{{customdata[0]:,.0f}} {BASE_CCY_LABEL}<br>'
            f'Std=%{{customdata[1]:,.0f}} {BASE_CCY_LABEL}<extra></extra>'
        ),
    ))

    idx_current = int(np.argmin(np.abs(df['gamma'].to_numpy(dtype=float) - float(current_gamma))))
    current = df.iloc[idx_current]
    fig.add_trace(go.Scatter(
        x=[float(current['gamma'])],
        y=[float(current['mean_over_std'])],
        mode='markers',
        name=f'Current φ={float(current["gamma"]):.4g}',
        marker=dict(symbol='star', size=15, line=dict(width=1.5)),
        hovertemplate=(
            f'current φ={float(current["gamma"]):.4g}<br>'
            f'E[PnL]/Std={float(current["mean_over_std"]):.4f}<extra></extra>'
        ),
    ))
    fig.add_hline(y=0.0, line_dash='dot')
    fig.update_layout(
        title='Risk-adjusted return across φ',
        xaxis_title='φ (risk aversion)',
        yaxis_title='Expected PnL / stdev',
        height=460,
        hovermode='closest',
    )
    return fig

SESSION_TICKVALS = [0.0, 120.0, 240.0, 360.0, 480.0, SESSION_HORIZON_MINUTES]
SESSION_TICKTEXT = ["07:45", "09:45", "11:45", "13:45", "15:45", "17:15"]


def _session_clock_label(elapsed_minutes: float, with_seconds: bool = True) -> str:
    total_seconds = int(round((7 * 60 + 45 + float(elapsed_minutes)) * 60))
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if with_seconds else f"{hours:02d}:{minutes:02d}"


def make_inventory_path_figure(path: lp.MonteCarloSamplePath) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=path.inventory_event_times,
        y=path.inventory_event_values,
        mode="lines",
        line_shape="hv",
        name="Inventory",
    ))
    fig.add_hline(y=0.0, line_dash="dot")
    fig.update_layout(
        title="Inventory path",
        xaxis_title="Session time",
        yaxis_title="Inventory [M]",
        xaxis=dict(tickmode="array", tickvals=SESSION_TICKVALS, ticktext=SESSION_TICKTEXT),
        height=360,
        showlegend=False,
    )
    return fig


def make_spot_quote_path_figure(path: lp.MonteCarloSamplePath, quote_size: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=path.times, y=path.spots, mode="lines", name="Spot", line=dict(width=2.5),
    ))

    matched_tiers: set[int] = set()
    for series in path.quote_series:
        if not math.isclose(series.size, float(quote_size), rel_tol=0.0, abs_tol=1e-12):
            continue
        matched_tiers.add(series.tier_idx)
        side_label = "Bid" if series.side == "bid" else "Ask"
        fig.add_trace(go.Scatter(
            x=path.times,
            y=series.prices,
            mode="lines",
            name=f"{series.tier_name} {side_label} {series.size:g}M",
            line=dict(dash="dash" if series.side == "bid" else "dot"),
        ))

    # Actual execution markers, split by tier and side. Plot all fill sizes: the
    # hover text makes it clear which rung executed.
    groups: dict[tuple[str, str], list[lp.MonteCarloFillEvent]] = {}
    for fill in path.fills:
        groups.setdefault((fill.tier_name, fill.side), []).append(fill)
    for (tier_name, side), fills in groups.items():
        symbol = "triangle-up" if side == "bid" else "triangle-down"
        side_label = "Bid fills" if side == "bid" else "Ask fills"
        hover = [
            f"{_session_clock_label(f.time_minutes)}<br>{f.tier_name} {f.side} {f.size:g}M"
            f"<br>exec={f.execution_price:.6f}<br>q: {f.inventory_before:g} → {f.inventory_after:g}"
            for f in fills
        ]
        fig.add_trace(go.Scatter(
            x=[f.time_minutes for f in fills],
            y=[f.execution_price for f in fills],
            mode="markers",
            name=f"{tier_name} {side_label}",
            marker=dict(symbol=symbol, size=8),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Spot, {quote_size:g}M quotes and fills",
        xaxis_title="Session time",
        yaxis_title="Price",
        xaxis=dict(tickmode="array", tickvals=SESSION_TICKVALS, ticktext=SESSION_TICKTEXT),
        height=520,
        legend=dict(orientation="h"),
    )
    return fig


def make_fill_counts_figure(path: lp.MonteCarloSamplePath) -> go.Figure:
    counts: dict[tuple[str, str], int] = {}
    for fill in path.fills:
        counts[(fill.tier_name, fill.side)] = counts.get((fill.tier_name, fill.side), 0) + 1
    tiers = list(dict.fromkeys(fill.tier_name for fill in path.fills))
    fig = go.Figure()
    for side in ("bid", "ask"):
        fig.add_trace(go.Bar(
            x=tiers,
            y=[counts.get((tier, side), 0) for tier in tiers],
            name=side.capitalize(),
        ))
    fig.update_layout(
        title="Fills by tier and side",
        xaxis_title="Tier",
        yaxis_title="Fill count",
        barmode="group",
        height=350,
    )
    return fig


def fill_tape_frame(path: lp.MonteCarloSamplePath) -> pd.DataFrame:
    rows = []
    for fill in path.fills:
        rows.append({
            "Time": _session_clock_label(fill.time_minutes),
            "Tier": fill.tier_name,
            "Side": fill.side.capitalize(),
            "Size [M]": fill.size,
            "Execution price": fill.execution_price,
            "Spot": fill.spot_before_fill,
            "Inventory before": fill.inventory_before,
            "Inventory after": fill.inventory_after,
            "Delta": fill.delta,
        })
    return pd.DataFrame(rows)


def _format_ccy(x: float) -> str:
    ax = abs(float(x))
    if ax >= 1_000_000:
        return f"{x / 1_000_000:,.2f}m"
    if ax >= 1_000:
        return f"{x / 1_000:,.1f}k"
    return f"{x:,.0f}"


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
        q_abs_max = st.number_input(
            "operational max |q|", value=20.0, min_value=0.5, step=0.5, format="%.4f",
            help=(
                "Inventory range shown in the pricing GUI. The Howard solve automatically "
                "adds a hidden buffer equal to the largest allowed trade size, so displayed "
                "quotes never rely on extrapolated continuation values."
            ),
        )

        q_step = fine_half_width = fine_step = coarse_step = None
        if q_grid_mode == "uniform":
            q_step = st.number_input("q step", value=1.0, min_value=0.01, step=0.1, format="%.4f")
        else:
            fine_half_width = st.number_input("fine half-width", value=3.0, min_value=0.0, step=0.5, format="%.4f")
            c_grid1, c_grid2 = st.columns(2)
            fine_step = c_grid1.number_input("fine step", value=0.25, min_value=0.01, step=0.05, format="%.4f")
            coarse_step = c_grid2.number_input("coarse step", value=1.0, min_value=0.01, step=0.1, format="%.4f")

        mid_price = st.number_input("display mid price", value=1.000000, step=0.000100, format="%.6f")

    with st.sidebar.expander("Spot process", expanded=False):
        spot = st.number_input("spot (quote CCY per base CCY)", value=11.5, min_value=0.001, step=0.1, format="%.4f")
        spot_drift = st.number_input("spot_drift", value=0.0, step=0.001, format="%.5f")
        spread = st.number_input("reference spread", value=20.0 / 10000.0, step=1.0 / 10000.0, format="%.6f")

    with st.sidebar.expander("Carry cost", expanded=False):
        sigma = st.number_input(
            "Volatility σ [pips / √min]",
            value=20.0,
            step=1.0,
            format="%.2f",
            help="One-minute volatility in pips. The running inventory penalty is γσ²q².",
        ) / 10_000.0
        risk_aversion = st.number_input(
            "γ (risk aversion)",
            value=0.1, step=0.01, format="%.2f",
            help="Standard quadratic running penalty: γσ²·q².",
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

mdp_tiers = build_tiers(active_tier_specs, float(spot))
dark_pool_venue = None if dark_pool_spec is None else dark_pool_spec.build_venue(float(spot))

penalty = lp.QuadraticInventoryPenalty(
    carry_cost=lp.CarryCost(risk_aversion=float(risk_aversion), sigma=float(sigma)),
)

config = build_solver_config(
    q_grid=q_grid,
    spread=float(spread),
    spot=float(spot),
    spot_drift=float(spot_drift),
)

with st.spinner("Solving HJB with Howard policy iteration..."):
    solver = lp.HJBLadderSolver(
        config=config,
        penalty=penalty,
        internalization_time=global_internalization_time,
        mdp_tiers=mdp_tiers,
        dark_pool=dark_pool_venue,
    )
    try:
        solution: lp.HJBSolution = solver.solve()
    except RuntimeError as exc:
        st.error(
            "Howard fixed-policy evaluation became singular/non-ergodic. "
            "This can happen when extremely defensive quote bounds make transition rates "
            "numerically zero, so a fixed policy no longer has a single recurrent class. "
            "Adjust the economic quote bounds or model specification. "
            f"Details: {exc}"
        )
        st.stop()

diag = solution.diagnostics
solution_tiers = ordered_solution_tiers(active_tier_specs, solution)

st.success("Solver run complete.")

main_policy_tab, monte_carlo_tab = st.tabs(["Policy", "Monte Carlo"])

with main_policy_tab:
    summary_cols = st.columns(4)
    summary_cols[0].metric("Configured tiers", len(tier_specs))
    summary_cols[1].metric("Active MDP tiers", active_mdp_count)
    summary_cols[2].metric("Dark pool", "on" if dark_pool_spec is not None else "off")
    summary_cols[3].metric("Disabled tiers", len(disabled_tier_names))

    if disabled_tier_names:
        st.caption("Excluded from solve: " + ", ".join(disabled_tier_names))

    if diag.converged:
        st.info(
            f"Howard converged after {diag.iterations_used} policy iterations. "
            f"Bellman residual = {diag.final_max_rhs:.2e}; "
            f"average reward ρ = {solution.average_reward:.6g}."
        )
    else:
        st.warning(
            f"Howard did not reach machine-precision convergence. "
            f"Iterations = {diag.iterations_used}; Bellman residual = {diag.final_max_rhs:.2e}."
        )

    with st.expander("Solver diagnostics", expanded=False):
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("pricing q points", len(config.q_grid))
        c2.metric("solve q points", len(solution.solve_q_grid))
        c3.metric("hard |q|", f"{solution.hard_inventory_limit:.2f}")
        c4.metric("venues", total_venue_count(solution))
        c5.metric("iterations used", diag.iterations_used)
        c6.metric("converged", "yes" if diag.converged else "no")
        c7.metric("spread", f"{config.spread:.6f}")
        st.caption(
            "The displayed pricing grid is padded internally by the largest allowed trade size. "
            "Howard solves h(q) on that hidden domain; transitions beyond its hard edge are inadmissible, "
            "so no continuation value is obtained by extrapolation."
        )

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
                "Convergence: Bellman residual",
                "Bellman residual",
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
    \Pi^{\text{bid}}(z, q, \delta) = \lambda(\delta, z)\Bigl[z s(0.5 - \delta) - z\,m(z,\,t(q+z)) + h(q+z) - h(q)\Bigr]
    $$

    where $s$ is the spread, $\delta$ is the quoted delta, $m(z,t)\ge 0$ is the adverse markout cost, and $t(q)$ is the internalization horizon.

    **Ask is symmetric** (inventory decreases by $z$, markout evaluated at $t(q-z)$).

    **Carry cost penalty:**

    $$
    \Pi(q) = -\gamma\sigma^2 q^2
    $$

    **Flow curve:** $\lambda(\delta, z) = A(z)\cdot\sigma\!\left(\kappa\bigl(\delta - \delta_{50}(z)\bigr)\right)$, with $A(z) = A_0\,z^{-\theta - \beta z}$.

    **Markout model:** $m(z,t)=a z^{\beta}\left(1-e^{-t/\tau}\right)$. Larger trades have larger eventual impact $a z^{\beta}$; impact arrives quickly and then saturates with time scale $\tau$.

    **Internalization time:** $t(q) = \tau_0 + \tau_1|q| + \tau_2 q^2$.

    **Inventory ladder shape:** moving farther from zero inventory changes both ladder level and slope. On the inventory-increasing side every rung becomes less aggressive and volume-premium gaps may only widen; on the inventory-reducing side every rung becomes more aggressive and gaps may only flatten. Thus a long book shifts bids lower and steepens them, while shifting asks lower and flattening them; a short book is the mirror image.

    **Inventory boundary:** the GUI range is the operational pricing range. The solver automatically extends the hidden Howard grid by the largest allowed trade size, so every fill from a displayed state lands on a solved continuation state. At the hidden hard edge, further inventory-increasing fills are inadmissible rather than valued by extrapolating $h(q)$.
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
    - the operational grid is strictly increasing, symmetric around 0, odd-length, with 0 at the centre index
    - the Howard solve adds a hidden buffer equal to the largest allowed fill size
    - transitions beyond the hidden hard inventory bound are inadmissible; $h(q)$ is never extrapolated outside the solved domain
                        """
                    )


with monte_carlo_tab:
    st.subheader("Monte Carlo PnL")
    st.caption(
        "Realized PnL is execution cashflow plus mark-to-market inventory PnL. "
        "The quadratic inventory penalty is not subtracted from realized PnL; gamma affects PnL only through the solved policy."
    )

    available_mc_q = [float(q) for q in solution.q_grid]
    st.info(
        f"Core trading session: **{SESSION_START_LABEL}–{SESSION_END_LABEL}** "
        f"(**{SESSION_HORIZON_MINUTES:.0f} minutes**). All flow intensities are interpreted per minute."
    )
    mc_horizon = SESSION_HORIZON_MINUTES
    mc_c1, mc_c2, mc_c3 = st.columns(3)
    mc_paths = mc_c1.number_input(
        "Paths", min_value=100, max_value=20_000, value=1_000, step=500, key="mc_paths"
    )
    mc_q0 = mc_c2.selectbox(
        "Initial inventory", options=available_mc_q,
        index=available_mc_q.index(0.0) if 0.0 in available_mc_q else len(available_mc_q) // 2,
        key="mc_q0",
    )
    mc_seed = mc_c3.number_input(
        "Random seed", min_value=0, value=12345, step=1, key="mc_seed"
    )

    st.markdown(r"""
The simulated spot process is

$$
dS_t = \mu\,dt + \sigma\,dW_t + \sum_k b_k(t-T_k)\,dt,
$$

where an RFQ fill at time $T_k$ adds an adverse drift kernel

$$
b_k(u)= -\operatorname{sign}(\Delta q_k)\,\frac{a z_k^\beta}{\tau}e^{-u/\tau},\qquad u\ge0.
$$

Integrating this drift gives exactly the markout curve
$m(z,u)=a z^\beta(1-e^{-u/\tau})$. Brownian spot noise remains on top of that expected drift.
    """)

    model_signature = monte_carlo_model_signature(
        solver,
        horizon_minutes=float(mc_horizon),
        n_paths=int(mc_paths),
        initial_inventory=float(mc_q0),
        seed=int(mc_seed),
        sigma=float(sigma),
    )

    # Streamlit keeps session_state across code hot-reloads. Explicitly discard
    # results produced by any older benchmark implementation or model state.
    if st.session_state.get("mc_signature") != model_signature:
        st.session_state.pop("mc_result", None)
        st.session_state.pop("mc_closed_form", None)
        st.session_state.pop("mc_signature", None)

    if st.button("Run Monte Carlo", type="primary", key="run_mc"):
        with st.spinner("Simulating spot, fills and mark-to-market PnL..."):
            try:
                mc_result = solver.simulate_pnl(
                    horizon_minutes=float(mc_horizon),
                    n_paths=int(mc_paths),
                    sigma=float(sigma),
                    initial_inventory=float(mc_q0),
                    initial_spot=float(config.spot),
                    seed=int(mc_seed),
                    sample_paths=8,
                    sample_points=381,
                )
                closed_form = solver.closed_form_pnl_statistics(
                    horizon_minutes=float(mc_horizon),
                    sigma=float(sigma),
                    initial_inventory=float(mc_q0),
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                st.error(f"Monte Carlo / closed-form benchmark failed: {exc}")
            else:
                st.session_state["mc_result"] = mc_result
                st.session_state["mc_closed_form"] = closed_form
                st.session_state["mc_signature"] = model_signature

    if st.session_state.get("mc_signature") == model_signature:
        mc_result = st.session_state.get("mc_result")
        closed_form = st.session_state.get("mc_closed_form")
        if mc_result is not None and closed_form is not None:
            pnl = np.asarray(mc_result.pnl_base_ccy, dtype=float)
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric(f"MC mean PnL [{BASE_CCY_LABEL}]", _format_ccy(float(np.mean(pnl))))
            m2.metric(
                f"Closed-form mean [{BASE_CCY_LABEL}]",
                _format_ccy(closed_form.expected_pnl_base_ccy_at_reference_spot),
                help=(
                    f"Impact-aware finite-horizon SEK expectation using benchmark {_PNL_BENCHMARK_VERSION}, "
                    f"converted at reference EURSEK spot {closed_form.reference_spot:.6f}."
                ),
            )
            m3.metric(f"MC stdev [{BASE_CCY_LABEL}]", _format_ccy(float(np.std(pnl, ddof=1))))
            m4.metric(
                f"Closed-form stdev [{BASE_CCY_LABEL}]",
                _format_ccy(closed_form.std_pnl_base_ccy_at_reference_spot),
                help=(
                    "Exact terminal SEK-PnL standard deviation under the fixed-policy model, "
                    "including fill randomness, overlapping impact and Brownian inventory risk; "
                    f"converted to EUR at reference EURSEK spot {closed_form.reference_spot:.6f}. "
                    "It is not the exact standard deviation of the nonlinear pathwise conversion PnL_SEK / S_T."
                ),
            )
            m5.metric(f"5% PnL [{BASE_CCY_LABEL}]", _format_ccy(float(np.percentile(pnl, 5))))
            m6.metric("P(PnL < 0)", f"{100.0 * float(np.mean(pnl < 0.0)):.1f}%")

            distribution_tab, frontier_tab, path_explorer_tab = st.tabs(["Distribution", "Efficient frontier", "Path explorer"])

            with distribution_tab:
                st.plotly_chart(
                    make_pnl_distribution_figure(mc_result, closed_form.expected_pnl_base_ccy_at_reference_spot),
                    use_container_width=True,
                )
                st.plotly_chart(make_inventory_paths_distribution_figure(mc_result), use_container_width=True)

                d1, d2, d3 = st.columns(3)
                d1.metric("Mean final inventory", f"{np.mean(mc_result.final_inventory):.2f}")
                d2.metric("RMS final inventory", f"{math.sqrt(np.mean(np.square(mc_result.final_inventory))):.2f}")
                d3.metric("Mean fills / path", f"{np.mean(mc_result.trade_count):.1f}")

                st.caption(
                    f"Monte Carlo PnL is reported in {BASE_CCY_LABEL} path by path as quote-currency PnL divided by the final simulated spot. "
                    f"The closed-form benchmark uses the same exponential post-fill spot-impact dynamics as the Monte Carlo and is exact for expected {QUOTE_CCY_LABEL} PnL. "
                    f"It is then converted to {BASE_CCY_LABEL} at the reference spot; because Monte Carlo converts path by path using final spot, the displayed {BASE_CCY_LABEL} means can still differ slightly. "
                    "The artificial inventory penalty is excluded from both."
                )


            with frontier_tab:
                st.markdown(
                    "The efficient frontier below is computed from the **closed-form benchmark only**. "
                    "Each point resolves the full HJB for a different risk-aversion parameter γ, then plots "
                    "closed-form terminal PnL standard deviation on the horizontal axis and expected terminal PnL on the vertical axis."
                )
                fc1, fc2, fc3, fc4 = st.columns(4)
                gamma_min = fc1.number_input(
                    "γ min",
                    min_value=1e-6,
                    value=max(1e-4, float(risk_aversion) / 10.0),
                    step=0.01,
                    format="%.4f",
                    key="frontier_gamma_min",
                )
                gamma_max = fc2.number_input(
                    "γ max",
                    min_value=1e-6,
                    value=max(float(risk_aversion) * 5.0, float(risk_aversion) + 0.01),
                    step=0.05,
                    format="%.4f",
                    key="frontier_gamma_max",
                )
                frontier_points = int(fc3.number_input(
                    "Frontier points",
                    min_value=3,
                    max_value=25,
                    value=9,
                    step=1,
                    key="frontier_points",
                ))
                log_gamma_grid = fc4.checkbox("Log-spaced γ grid", value=True, key="frontier_log_grid")

                if gamma_max <= gamma_min:
                    st.error("γ max must be larger than γ min.")
                else:
                    if log_gamma_grid:
                        gamma_grid = np.geomspace(float(gamma_min), float(gamma_max), num=int(frontier_points))
                    else:
                        gamma_grid = np.linspace(float(gamma_min), float(gamma_max), num=int(frontier_points))
                    gamma_grid = np.unique(np.round(np.append(gamma_grid, float(risk_aversion)), 12))

                    frontier_signature = efficient_frontier_signature(
                        q_grid=q_grid,
                        spot=float(spot),
                        spot_drift=float(spot_drift),
                        spread=float(spread),
                        sigma=float(sigma),
                        horizon_minutes=float(mc_horizon),
                        initial_inventory=float(mc_q0),
                        tier_specs=active_tier_specs,
                        dark_pool_spec=dark_pool_spec,
                        tau0=float(tau0),
                        tau1=float(tau1),
                        tau2=float(tau2),
                        gammas=gamma_grid,
                    )

                    if st.session_state.get("frontier_signature") != frontier_signature:
                        st.session_state.pop("frontier_result", None)
                        st.session_state.pop("frontier_signature", None)

                    if st.button("Compute frontier", key="run_frontier"):
                        with st.spinner("Solving closed-form frontier across γ..."):
                            try:
                                frontier_df = compute_efficient_frontier(
                                    q_grid=q_grid,
                                    spot=float(spot),
                                    spot_drift=float(spot_drift),
                                    spread=float(spread),
                                    sigma=float(sigma),
                                    horizon_minutes=float(mc_horizon),
                                    initial_inventory=float(mc_q0),
                                    tier_specs=active_tier_specs,
                                    dark_pool_spec=dark_pool_spec,
                                    tau0=float(tau0),
                                    tau1=float(tau1),
                                    tau2=float(tau2),
                                    gammas=gamma_grid,
                                )
                            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                                st.error(f"Efficient frontier failed: {exc}")
                            else:
                                st.session_state["frontier_result"] = frontier_df
                                st.session_state["frontier_signature"] = frontier_signature

                    if st.session_state.get("frontier_signature") == frontier_signature and st.session_state.get("frontier_result") is not None:
                        frontier_df = st.session_state["frontier_result"].copy()
                        st.plotly_chart(make_efficient_frontier_figure(frontier_df, float(risk_aversion)), use_container_width=True)
                        st.plotly_chart(make_phi_risk_adjusted_figure(frontier_df, float(risk_aversion)), use_container_width=True)
                        display_df = frontier_df.rename(columns={
                            'gamma': 'γ',
                            'expected_pnl_base_ccy': f'Expected PnL [{BASE_CCY_LABEL}]',
                            'std_pnl_base_ccy': f'Stdev [{BASE_CCY_LABEL}]',
                            'average_reward': 'Average reward',
                        })[[
                            'γ',
                            f'Expected PnL [{BASE_CCY_LABEL}]',
                            f'Stdev [{BASE_CCY_LABEL}]',
                            'Average reward',
                        ]]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        st.caption(
                            f"All frontier points use the same session horizon ({SESSION_HORIZON_MINUTES:.0f} min), "
                            f"initial inventory {float(mc_q0):.2f}M, and the current tier / dark-pool configuration. "
                            f"Expected PnL and stdev are closed-form {BASE_CCY_LABEL} figures converted at the reference spot. "
                            "In the second chart, φ is the same risk-aversion parameter currently named γ in the solver."
                        )
                    else:
                        st.info("Choose a γ range and click **Compute frontier**.")

            with path_explorer_tab:
                if not mc_result.sample_paths:
                    st.info("No detailed sample paths were retained for this run.")
                else:
                    pc1, pc2 = st.columns(2)
                    selected_path_idx = pc1.selectbox(
                        "Path",
                        options=list(range(len(mc_result.sample_paths))),
                        format_func=lambda i: f"Path {i + 1}",
                        key="mc_selected_path",
                    )
                    selected_path = mc_result.sample_paths[int(selected_path_idx)]
                    quote_sizes = sorted({series.size for series in selected_path.quote_series})
                    quote_size = pc2.selectbox(
                        "Quote size to display",
                        options=quote_sizes,
                        index=0,
                        format_func=lambda z: f"{z:g}M",
                        key="mc_quote_size",
                    )

                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric(
                        f"Path PnL [{BASE_CCY_LABEL}]",
                        _format_ccy(float(mc_result.pnl_base_ccy[int(selected_path_idx)])),
                    )
                    pm2.metric("Final inventory", f"{mc_result.final_inventory[int(selected_path_idx)]:.2f}M")
                    pm3.metric("Fills", f"{mc_result.trade_count[int(selected_path_idx)]}")
                    pm4.metric("Final spot", f"{mc_result.final_spot[int(selected_path_idx)]:.6f}")

                    st.plotly_chart(make_inventory_path_figure(selected_path), use_container_width=True)
                    st.plotly_chart(
                        make_spot_quote_path_figure(selected_path, float(quote_size)),
                        use_container_width=True,
                    )

                    fc1, fc2 = st.columns([1, 2])
                    with fc1:
                        st.plotly_chart(make_fill_counts_figure(selected_path), use_container_width=True)
                    with fc2:
                        st.markdown("**Fill tape**")
                        tape = fill_tape_frame(selected_path)
                        if tape.empty:
                            st.info("No fills on this path.")
                        else:
                            st.dataframe(tape, use_container_width=True, hide_index=True, height=350)

                    st.caption(
                        "Inventory is event-level and steps exactly at fills. Spot and quote lines are regular display snapshots; "
                        "the underlying fill simulation remains continuous-time/event-driven. Fill markers show the actual execution price and size."
                    )
    else:
        st.info("Choose the Monte Carlo settings and click **Run Monte Carlo**.")

st.caption("Pure Python + NumPy implementation. Restart Streamlit after editing the model code.")