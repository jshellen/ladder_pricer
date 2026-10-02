"""Pure-Python HJB ladder pricer.

Howard policy iteration for a continuous-time, average-reward inventory-control
problem. RFQ quote improvement uses the analytical logistic/Lambert-W solution.
The operational pricing grid is padded internally so continuation values are never
obtained by inventory extrapolation. Within each inventory row, larger sizes cannot be quoted tighter. As absolute
inventory grows, every rung moves monotonically with inventory: quotes on the
inventory-increasing side become less aggressive while quotes on the
inventory-reducing side become more aggressive. At the same time adjacent rung
gaps (volume premia) widen or stay equal on the inventory-increasing side and
flatten or stay equal on the inventory-reducing side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import copy
import math
from typing import Iterable, Optional, Sequence

import numpy as np


# Increment whenever the Monte Carlo PnL accounting or closed-form benchmark
# semantics change. The Streamlit app includes this in its cache signature so
# results produced by an older implementation can never be displayed as current.
PNL_BENCHMARK_VERSION = "impact-aware-v3-second-moment"


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

class Side(Enum):
    Bid = "bid"
    Ask = "ask"


def _side(side: Side | str) -> Side:
    if isinstance(side, Side):
        return side
    s = str(side).lower()
    if s == "bid":
        return Side.Bid
    if s == "ask":
        return Side.Ask
    raise ValueError("side must be 'bid' or 'ask'.")


def _strictly_increasing(x: Sequence[float], name: str) -> None:
    if not x:
        raise ValueError(f"{name} cannot be empty.")
    if any(float(x[i]) <= float(x[i - 1]) for i in range(1, len(x))):
        raise ValueError(f"{name} must be strictly increasing.")


def _positive_strictly_increasing(x: Sequence[float], name: str) -> None:
    _strictly_increasing(x, name)
    if any(float(v) <= 0.0 for v in x):
        raise ValueError(f"{name} must contain positive values only.")


def _validate_centered_symmetric_grid(x: Sequence[float], name: str, tol: float = 1e-10) -> None:
    _strictly_increasing(x, name)
    if len(x) < 3:
        raise ValueError(f"{name} must contain at least 3 points.")
    if len(x) % 2 == 0:
        raise ValueError(f"{name} must have odd length so that 0 is exactly in the middle.")
    mid = len(x) // 2
    if abs(float(x[mid])) > tol:
        raise ValueError(f"{name} must contain 0 exactly at the middle index.")
    for i in range(mid):
        if abs(float(x[i]) + float(x[-1 - i])) > tol:
            raise ValueError(f"{name} must be symmetric around 0.")


def _locate_segment_with_weight(grid: Sequence[float], x: float) -> tuple[int, float]:
    n = len(grid)
    if n == 0:
        raise ValueError("grid cannot be empty")
    if n == 1:
        return 0, 0.0
    g = np.asarray(grid, dtype=float)
    if x <= g[0]:
        i = 0
    elif x >= g[-1]:
        i = n - 2
    else:
        i = int(np.searchsorted(g, x, side="right") - 1)
    t = (x - g[i]) / (g[i + 1] - g[i])
    return i, float(t)


def _interp_linear(grid: Sequence[float], vals: Sequence[float], x: float) -> float:
    if len(grid) != len(vals):
        raise ValueError("grid and vals size mismatch")
    if len(grid) == 1:
        return float(vals[0])
    i, t = _locate_segment_with_weight(grid, x)
    return float(vals[i] + t * (vals[i + 1] - vals[i]))


def _interp_weights(grid: Sequence[float], x: float) -> tuple[int, float, int, float]:
    """Linear interpolation/extrapolation weights matching _interp_linear."""
    if len(grid) == 1:
        return 0, 1.0, 0, 0.0
    i, t = _locate_segment_with_weight(grid, x)
    return i, 1.0 - t, i + 1, t


def _locate_segment_with_weight_bounded(
    grid: Sequence[float], x: float, tol: float = 1e-10
) -> tuple[int, float]:
    """Locate ``x`` for interpolation, rejecting extrapolation outside ``grid``."""
    if not grid:
        raise ValueError("grid cannot be empty")
    g = np.asarray(grid, dtype=float)
    x = float(x)
    if x < g[0] - tol or x > g[-1] + tol:
        raise ValueError(
            f"inventory state {x:g} lies outside solved domain [{g[0]:g}, {g[-1]:g}]"
        )
    # Clamp tiny floating-point overshoots back onto the solved domain.
    x = min(max(x, float(g[0])), float(g[-1]))
    if len(g) == 1:
        return 0, 0.0
    if x >= g[-1]:
        return len(g) - 2, 1.0
    i = int(np.searchsorted(g, x, side="right") - 1)
    i = max(0, min(i, len(g) - 2))
    t = (x - g[i]) / (g[i + 1] - g[i])
    return i, float(t)


def _interp_linear_bounded(grid: Sequence[float], vals: Sequence[float], x: float) -> float:
    if len(grid) != len(vals):
        raise ValueError("grid and vals size mismatch")
    if len(grid) == 1:
        return float(vals[0])
    i, t = _locate_segment_with_weight_bounded(grid, x)
    return float(vals[i] + t * (vals[i + 1] - vals[i]))


def _interp_weights_bounded(grid: Sequence[float], x: float) -> tuple[int, float, int, float]:
    if len(grid) == 1:
        return 0, 1.0, 0, 0.0
    i, t = _locate_segment_with_weight_bounded(grid, x)
    return i, 1.0 - t, i + 1, t


# ---------------------------------------------------------------------------
# Economic model components
# ---------------------------------------------------------------------------

@dataclass
class LogisticFlowCurve:
    """RFQ flow curve with all intensities expressed per minute.

    ``A0`` is the exogenous RFQ-arrival intensity scale. ``A(z)`` is the
    customer RFQ arrival rate for size ``z``; the logistic curve is the
    conditional probability that our quote wins that RFQ.  Their product,
    ``arrival_rate(delta, z)``, is the derived won-trade intensity.
    """
    A0: float = 1.0
    theta: float = 0.0
    beta: float = 0.0
    shift: float = 0.20
    steepness: float = 10.0
    volume_shift: float = 0.05
    z_floor: float = 1e-8

    def A(self, z: float) -> float:
        z = max(float(z), self.z_floor)
        return self.A0 * z ** (-self.theta - self.beta * z)

    def center(self, z: float) -> float:
        return self.shift - self.volume_shift * (float(z) - 1.0)

    def hit_ratio(self, delta: float, z: float) -> float:
        y = (float(delta) - self.shift + self.volume_shift * (float(z) - 1.0)) * self.steepness
        if y >= 0.0:
            e = math.exp(-y) if y < 745.0 else 0.0
            return 1.0 / (1.0 + e)
        e = math.exp(y) if y > -745.0 else 0.0
        return e / (1.0 + e)

    def rfq_arrival_rate(self, z: float) -> float:
        return self.A(z)

    def win_probability(self, delta: float, z: float) -> float:
        return self.hit_ratio(delta, z)

    def arrival_rate(self, delta: float, z: float) -> float:
        return self.rfq_arrival_rate(z) * self.win_probability(delta, z)

    @staticmethod
    def lambert_w_exp(x: float) -> float:
        """Return W_0(exp(x)) without SciPy and without forming exp(x)."""
        if math.isnan(x):
            return math.nan
        if x == -math.inf:
            return 0.0
        if x == math.inf:
            return math.inf
        y = x if x <= 0.0 else math.log1p(x)
        eps = np.finfo(float).eps
        for _ in range(20):
            ey = math.exp(y) if y > -745.0 else 0.0
            step = (y + ey - x) / (1.0 + ey)
            y -= step
            if abs(step) <= 8.0 * eps * max(1.0, abs(y)):
                break
        return math.exp(y) if y < 709.0 else math.inf

    def optimal_delta(self, z: float, spread: float, additive_value: float) -> float:
        z = float(z)
        spread = float(spread)
        if z <= 0.0:
            raise ValueError("z must be positive")
        if spread <= 0.0:
            raise ValueError("spread must be positive")
        if self.steepness <= 0.0:
            raise ValueError("optimal_delta requires positive steepness")
        b = 0.5 + float(additive_value) / (z * spread)
        x = self.steepness * (b - self.center(z)) - 1.0
        w = self.lambert_w_exp(x)
        return b - (1.0 + w) / self.steepness


@dataclass
class SaturatingMarkoutModel:
    """Positive adverse-selection cost with saturating time impact.

    m(z, t) = impact_scale * z**size_exponent * (1 - exp(-t / tau))

    ``impact_scale`` is the eventual markout of a unit-size trade in price
    units, ``size_exponent`` controls how total price impact grows with trade
    size. Both ``t`` and ``tau`` are expressed in minutes, matching the internalization-time model.
    """

    impact_scale: float = 1.0 / 10_000.0
    size_exponent: float = 0.5
    tau: float = 0.5  # minutes

    def __post_init__(self) -> None:
        if self.impact_scale < 0.0:
            raise ValueError("impact_scale must be nonnegative")
        if self.size_exponent < 0.0:
            raise ValueError("size_exponent must be nonnegative")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")

    def asymptotic_markout(self, z: float) -> float:
        z = float(z)
        if z <= 0.0:
            raise ValueError("z must be positive")
        return self.impact_scale * z ** self.size_exponent

    def expected_markout(self, z: float, t: float) -> float:
        t = float(t)
        if t <= 0.0:
            return 0.0
        return self.asymptotic_markout(z) * (-math.expm1(-t / self.tau))


@dataclass
class CarryCost:
    risk_aversion: float = 2.0
    sigma: float = 0.25

    def value(self) -> float:
        return self.risk_aversion * self.sigma * self.sigma


@dataclass
class PolynomialInternalizationTime:
    tau0: float = 2.0
    tau1: float = 0.1
    tau2: float = 0.0015

    def value(self, q: float) -> float:
        x = abs(float(q))
        return self.tau0 + self.tau1 * x + self.tau2 * x * x


@dataclass
class QuadraticInventoryPenalty:
    carry_cost: CarryCost = field(default_factory=CarryCost)

    def value(self, q: float) -> float:
        q = float(q)
        return self.carry_cost.value() * q * q


# ---------------------------------------------------------------------------
# Quote analytics / policy containers
# ---------------------------------------------------------------------------

@dataclass
class QuoteSummary:
    delta: float = 0.0
    price_improvement_frac: float = 0.0
    price_improvement_pct_of_spread: float = 0.0
    price_improvement_pips: float = 0.0
    quote_relative_to_mid: float = 0.0
    quote_relative_to_mid_pips: float = 0.0
    distance_to_mid: float = 0.0
    distance_to_mid_pips: float = 0.0
    reference_delta: float = 0.0
    volume_premium_pips: float = 0.0
    quote_price: float = 0.0


class QuoteMetrics:
    pips_per_unit = 10000.0

    @staticmethod
    def price_improvement(delta: float, spread: float) -> float:
        return spread * delta

    @staticmethod
    def price_improvement_pct_of_spread(delta: float) -> float:
        return 100.0 * delta

    @staticmethod
    def price_improvement_pips(delta: float, spread: float) -> float:
        return QuoteMetrics.pips_per_unit * spread * delta

    @staticmethod
    def quote_relative_to_mid(delta: float, side: Side | str, spread: float) -> float:
        return spread * (delta - 0.5) if _side(side) is Side.Bid else spread * (0.5 - delta)

    @staticmethod
    def quote_relative_to_mid_pips(delta: float, side: Side | str, spread: float) -> float:
        return QuoteMetrics.pips_per_unit * QuoteMetrics.quote_relative_to_mid(delta, side, spread)

    @staticmethod
    def distance_to_mid(delta: float, spread: float) -> float:
        return abs(spread * (0.5 - delta))

    @staticmethod
    def distance_to_mid_pips(delta: float, spread: float) -> float:
        return QuoteMetrics.pips_per_unit * QuoteMetrics.distance_to_mid(delta, spread)

    @staticmethod
    def volume_premium_pips(delta_ref: float, delta_cur: float, spread: float) -> float:
        return QuoteMetrics.pips_per_unit * spread * (delta_ref - delta_cur)

    @staticmethod
    def quote_price(mid: float, delta: float, side: Side | str, spread: float) -> float:
        return mid + QuoteMetrics.quote_relative_to_mid(delta, side, spread)

    @staticmethod
    def make_summary(delta: float, delta_ref: float, side: Side | str, mid: float, spread: float) -> QuoteSummary:
        return QuoteSummary(
            delta=delta,
            price_improvement_frac=QuoteMetrics.price_improvement(delta, spread),
            price_improvement_pct_of_spread=QuoteMetrics.price_improvement_pct_of_spread(delta),
            price_improvement_pips=QuoteMetrics.price_improvement_pips(delta, spread),
            quote_relative_to_mid=QuoteMetrics.quote_relative_to_mid(delta, side, spread),
            quote_relative_to_mid_pips=QuoteMetrics.quote_relative_to_mid_pips(delta, side, spread),
            distance_to_mid=QuoteMetrics.distance_to_mid(delta, spread),
            distance_to_mid_pips=QuoteMetrics.distance_to_mid_pips(delta, spread),
            reference_delta=delta_ref,
            volume_premium_pips=QuoteMetrics.volume_premium_pips(delta_ref, delta, spread),
            quote_price=QuoteMetrics.quote_price(mid, delta, side, spread),
        )


@dataclass
class QuotePolicy:
    q_grid: list[float] = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)
    bid: list[list[float]] = field(default_factory=list)
    ask: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.q_grid or self.sizes or self.bid or self.ask:
            self.validate()

    @staticmethod
    def resize_matrix(mat: list[list[float]], rows: int, cols: int) -> list[list[float]]:
        return [[0.0] * cols for _ in range(rows)]

    def reset_shape(self, q_grid: Sequence[float], sizes: Sequence[float]) -> None:
        self.q_grid = list(map(float, q_grid))
        self.sizes = list(map(float, sizes))
        self.bid = [[0.0] * len(self.sizes) for _ in self.q_grid]
        self.ask = [[0.0] * len(self.sizes) for _ in self.q_grid]

    def validate(self) -> None:
        _strictly_increasing(self.q_grid, "QuotePolicy.q_grid")
        _positive_strictly_increasing(self.sizes, "QuotePolicy.sizes")
        if len(self.bid) != len(self.q_grid) or len(self.ask) != len(self.q_grid):
            raise ValueError("QuotePolicy: row count mismatch")
        if any(len(r) != len(self.sizes) for r in self.bid + self.ask):
            raise ValueError("QuotePolicy: column count mismatch")

    def _matrix(self, side: Side | str) -> list[list[float]]:
        return self.bid if _side(side) is Side.Bid else self.ask

    def _interp_q_column(self, mat: list[list[float]], j: int, q: float) -> float:
        if len(self.q_grid) == 1:
            return mat[0][j]
        i, t = _locate_segment_with_weight_bounded(self.q_grid, q)
        return mat[i][j] + t * (mat[i + 1][j] - mat[i][j])

    def delta(self, q: float, z: float, side: Side | str) -> float:
        mat = self._matrix(side)
        if len(self.sizes) == 1:
            return self._interp_q_column(mat, 0, q)
        j, tz = _locate_segment_with_weight(self.sizes, z)
        v0 = self._interp_q_column(mat, j, q)
        v1 = self._interp_q_column(mat, j + 1, q)
        return v0 + tz * (v1 - v0)

    def reference_delta(self, q: float, side: Side | str) -> float:
        return self.delta(q, self.sizes[0], side)

    def quote_summary(self, q: float, z: float, side: Side | str, mid: float, spread: float) -> QuoteSummary:
        d = self.delta(q, z, side)
        return QuoteMetrics.make_summary(d, self.reference_delta(q, side), side, mid, spread)


@dataclass
class DarkPoolPolicy:
    q_grid: list[float] = field(default_factory=list)
    bid_size: list[float] = field(default_factory=list)
    ask_size: list[float] = field(default_factory=list)
    bid_active: list[int] = field(default_factory=list)
    ask_active: list[int] = field(default_factory=list)

    def __init__(self, q_grid: Optional[Sequence[float]] = None):
        self.q_grid = []
        self.bid_size = []
        self.ask_size = []
        self.bid_active = []
        self.ask_active = []
        if q_grid is not None:
            self.reset_shape(q_grid)

    def reset_shape(self, q_grid: Sequence[float]) -> None:
        self.q_grid = list(map(float, q_grid))
        n = len(self.q_grid)
        self.bid_size = [0.0] * n
        self.ask_size = [0.0] * n
        self.bid_active = [0] * n
        self.ask_active = [0] * n

    def validate(self) -> None:
        _strictly_increasing(self.q_grid, "DarkPoolPolicy.q_grid")
        n = len(self.q_grid)
        if any(len(v) != n for v in [self.bid_size, self.ask_size, self.bid_active, self.ask_active]):
            raise ValueError("DarkPoolPolicy: vector length mismatch")

    def is_active(self, q_index: int, side: Side | str) -> bool:
        flags = self.bid_active if _side(side) is Side.Bid else self.ask_active
        return 0 <= q_index < len(flags) and bool(flags[q_index])

    def set_posted_size(self, q_index: int, side: Side | str, size: float, active: bool) -> None:
        if _side(side) is Side.Bid:
            self.bid_size[q_index] = float(size); self.bid_active[q_index] = int(bool(active))
        else:
            self.ask_size[q_index] = float(size); self.ask_active[q_index] = int(bool(active))

    def posted_size(self, q: float, side: Side | str) -> float:
        vals = self.bid_size if _side(side) is Side.Bid else self.ask_size
        if len(self.q_grid) == 1:
            return vals[0]
        i, t = _locate_segment_with_weight_bounded(self.q_grid, q)
        return vals[i] if t <= 0.5 else vals[i + 1]


# ---------------------------------------------------------------------------
# Tiers and dark pool
# ---------------------------------------------------------------------------

@dataclass
class MDPTier:
    name: str = ""
    sizes_: list[float] = field(default_factory=list)
    flow_curve: LogisticFlowCurve = field(default_factory=LogisticFlowCurve)
    markout_model: SaturatingMarkoutModel = field(default_factory=SaturatingMarkoutModel)
    delta_min: float = -5.0
    delta_max: float = 5.0
    use_markout: bool = True
    policy: QuotePolicy = field(default_factory=QuotePolicy)

    def __init__(self, name: str = "", sizes: Optional[Sequence[float]] = None,
                 flow_curve: Optional[LogisticFlowCurve] = None,
                 markout_model: Optional[SaturatingMarkoutModel] = None,
                 delta_min: float = -5.0, delta_max: float = 5.0, use_markout: bool = True):
        self.name = name
        self.sizes_ = list(map(float, sizes or []))
        self.flow_curve = flow_curve if flow_curve is not None else LogisticFlowCurve()
        self.markout_model = markout_model if markout_model is not None else SaturatingMarkoutModel()
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.use_markout = bool(use_markout)
        self.policy = QuotePolicy()
        if self.sizes_:
            self.validate()

    def tier_type_name(self) -> str:
        return "mdp"

    def sizes(self) -> list[float]:
        return self.sizes_

    def A(self, z: float) -> float:
        return self.flow_curve.A(z)

    def hit_ratio(self, delta: float, z: float) -> float:
        return self.flow_curve.hit_ratio(delta, z)

    def arrival_rate(self, delta: float, z: float) -> float:
        return self.flow_curve.arrival_rate(delta, z)

    def expected_markout(self, z: float, t: float) -> float:
        return self.markout_model.expected_markout(z, t)

    def is_admissible(self, q: float, z: float, side: Side | str) -> bool:
        return True

    def validate(self) -> None:
        _positive_strictly_increasing(self.sizes_, "MDPTier.sizes")
        if self.delta_max <= self.delta_min:
            raise ValueError("MDPTier: delta_max must be > delta_min")

    def reset_policy_shape(self, q_grid: Sequence[float]) -> None:
        self.policy.reset_shape(q_grid, self.sizes_)

    def quote(self, q: float, z: float, side: Side | str) -> float:
        return self.policy.delta(q, z, side)

    def quote_summary(self, q: float, z: float, side: Side | str, mid: float, spread: float) -> QuoteSummary:
        return self.policy.quote_summary(q, z, side, mid, spread)


Tier = MDPTier


class ArrivalDistribution:
    def validate(self) -> None:
        raise NotImplementedError
    def name(self) -> str:
        raise NotImplementedError
    def fill_rates(self, u: int) -> list[tuple[int, float]]:
        """Return (fill_size, event_rate) pairs for positive fills."""
        raise NotImplementedError
    def expected_fill_value(self, h_interp, q: float, hq: float, u: int, fee: float, direction: float) -> float:
        return sum(rate * (h_interp(q + direction * k) - hq - fee * k)
                   for k, rate in self.fill_rates(u))


@dataclass
class ZeroInflatedPoissonArrivalDist(ArrivalDistribution):
    lambda_arr: float = 1.0
    mu: float = 2.0
    p0: float = 0.1

    def validate(self) -> None:
        if self.lambda_arr < 0.0: raise ValueError("lambda_arr must be nonnegative")
        if self.mu <= 0.0: raise ValueError("mu must be positive")
        if not (0.0 <= self.p0 < 1.0): raise ValueError("p0 must lie in [0,1)")

    def name(self) -> str:
        return "zero_inflated_poisson"

    def fill_rates(self, u: int) -> list[tuple[int, float]]:
        """Transition rates induced by F=min(X,u).

        Arrivals occur at ``lambda_arr``.  X is zero with probability ``p0``;
        conditional on X>0 it is zero-truncated Poisson(mu).  The full-fill
        bucket u therefore absorbs the entire upper tail X>=u.
        """
        if u <= 0:
            return []
        positive_norm = 1.0 - math.exp(-self.mu)
        if positive_norm <= 1e-15 or self.lambda_arr <= 0.0:
            return []
        positive_mass = 1.0 - self.p0
        out=[]
        cumulative=0.0
        for k in range(1,u):
            pk=math.exp(-self.mu+k*math.log(self.mu)-math.lgamma(k+1))/positive_norm
            cumulative += pk
            out.append((k,self.lambda_arr*positive_mass*pk))
        out.append((u,self.lambda_arr*positive_mass*max(0.0,1.0-cumulative)))
        return out

    def sample_incoming_size(self, rng: np.random.Generator) -> int:
        if float(rng.random()) < self.p0:
            return 0
        x=0
        while x <= 0:
            x=int(rng.poisson(self.mu))
        return x


class DarkPoolVenue:
    def __init__(self, *args, **kwargs):
        self.policy = DarkPoolPolicy()
        self.min_fill_value = float(kwargs.pop("min_fill_value", 0.0))
        self.allow_both_sides = bool(kwargs.pop("allow_both_sides", False))

        if "dist_bid" in kwargs or (args and isinstance(args[0], ArrivalDistribution)):
            if args:
                self.dist_bid, self.dist_ask, fee_b, fee_a, sizes, *rest = args
                self.allow_both_sides = bool(rest[0]) if len(rest) > 0 else self.allow_both_sides
                self.min_fill_value = float(rest[1]) if len(rest) > 1 else self.min_fill_value
            else:
                self.dist_bid = kwargs.pop("dist_bid")
                self.dist_ask = kwargs.pop("dist_ask")
                fee_b = kwargs.pop("fee_per_unit_bid")
                fee_a = kwargs.pop("fee_per_unit_ask")
                sizes = kwargs.pop("posted_sizes")
        else:
            if args:
                lb, la, mub, mua, p0b, p0a, fee_b, fee_a, sizes, *rest = args
                self.allow_both_sides = bool(rest[0]) if len(rest) > 0 else self.allow_both_sides
                self.min_fill_value = float(rest[1]) if len(rest) > 1 else self.min_fill_value
            else:
                lb = kwargs.pop("lambda_bid", 1.0); la = kwargs.pop("lambda_ask", 1.0)
                mub = kwargs.pop("mu_bid", 2.0); mua = kwargs.pop("mu_ask", 2.0)
                p0b = kwargs.pop("p0_bid", 0.1); p0a = kwargs.pop("p0_ask", 0.1)
                fee_b = kwargs.pop("fee_per_unit_bid", 0.0); fee_a = kwargs.pop("fee_per_unit_ask", 0.0)
                sizes = kwargs.pop("posted_sizes", [1.0, 2.0, 3.0])
            self.dist_bid = ZeroInflatedPoissonArrivalDist(lb, mub, p0b)
            self.dist_ask = ZeroInflatedPoissonArrivalDist(la, mua, p0a)

        if kwargs:
            raise TypeError(f"unexpected arguments: {', '.join(kwargs)}")
        if not isinstance(self.dist_bid, ZeroInflatedPoissonArrivalDist) or not isinstance(self.dist_ask, ZeroInflatedPoissonArrivalDist):
            raise TypeError("dark pool supports ZeroInflatedPoissonArrivalDist only")
        self.fee_per_unit_bid = float(fee_b)
        self.fee_per_unit_ask = float(fee_a)
        self.posted_sizes = list(map(float, sizes))
        self.validate()

    @property
    def lambda_bid(self) -> float:
        return self.dist_bid.lambda_arr
    @property
    def lambda_ask(self) -> float:
        return self.dist_ask.lambda_arr

    @staticmethod
    def is_integer_like(x: float, tol: float = 1e-10) -> bool:
        return abs(x - round(x)) <= tol

    def validate(self) -> None:
        self.dist_bid.validate(); self.dist_ask.validate()
        _positive_strictly_increasing(self.posted_sizes, "DarkPoolVenue.posted_sizes")
        if any(not self.is_integer_like(u) for u in self.posted_sizes):
            raise ValueError("DarkPoolVenue.posted_sizes must be integer-valued")

    def reset_policy_shape(self, q_grid: Sequence[float]) -> None:
        self.policy.reset_shape(q_grid)

    def is_admissible(self, q: float, side: Side | str, tol: float = 1e-12) -> bool:
        if self.allow_both_sides:
            return True
        s = _side(side)
        return (q > tol and s is Side.Ask) or (q < -tol and s is Side.Bid)


# ---------------------------------------------------------------------------
# Solver data objects
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    q_grid: list[float] = field(default_factory=list)
    spot: float = 1.0
    spot_drift: float = 0.0
    spread: float = 20.0 / 10000.0

    def validate(self) -> None:
        _validate_centered_symmetric_grid(self.q_grid, "SolverConfig.q_grid")
        if self.spot <= 0.0: raise ValueError("SolverConfig.spot must be positive")
        if self.spread <= 0.0: raise ValueError("SolverConfig.spread must be positive")


@dataclass
class SolverGridMeta:
    nq: int = 0
    q0_idx: int = 0
    q_min: float = 0.0
    q_max: float = 0.0


def build_solver_grid_meta(q_grid: Sequence[float]) -> SolverGridMeta:
    if not q_grid: raise ValueError("q_grid cannot be empty")
    return SolverGridMeta(len(q_grid), len(q_grid)//2, float(q_grid[0]), float(q_grid[-1]))


@dataclass
class SolverDiagnostics:
    converged: bool = False
    iterations_used: int = 0
    final_max_h_change: float = math.inf
    final_max_rhs: float = math.inf
    history_max_h_change: list[float] = field(default_factory=list)
    history_max_rhs: list[float] = field(default_factory=list)

    def record_iteration(self, iteration: int, max_h_change: float, bellman_residual: float) -> None:
        self.iterations_used = iteration
        self.final_max_h_change = float(max_h_change)
        self.final_max_rhs = float(bellman_residual)
        self.history_max_h_change.append(float(max_h_change))
        self.history_max_rhs.append(float(bellman_residual))


@dataclass
class HJBSolution:
    # Public/operational grid shown to users and used for quote inspection.
    h: list[float] = field(default_factory=list)
    q_grid: list[float] = field(default_factory=list)
    mdp_tiers: list[MDPTier] = field(default_factory=list)
    dark_pool: Optional[DarkPoolVenue] = None
    diagnostics: SolverDiagnostics = field(default_factory=SolverDiagnostics)
    average_reward: float = 0.0
    # Hidden grid used by Howard evaluation. This is padded beyond q_grid so
    # every transition from the operational range lands on a solved state.
    solve_q_grid: list[float] = field(default_factory=list)
    h_solve: list[float] = field(default_factory=list)
    hard_inventory_limit: float = 0.0



@dataclass
class ClosedFormPnLBenchmark:
    horizon_minutes: float
    initial_inventory: float
    reference_spot: float
    expected_pnl_quote_ccy: float
    expected_pnl_rate_quote_ccy_per_min: float
    # Base-currency benchmark obtained by converting the quote-currency
    # expectation at the reference spot. Unlike Monte Carlo pathwise base PnL,
    # this is not E[PnL_quote / S_T].
    expected_pnl_base_ccy_at_reference_spot: float
    expected_pnl_rate_base_ccy_per_min_at_reference_spot: float


@dataclass
class ClosedFormPnLStatistics:
    """Exact quote-currency first two moments of terminal PnL.

    ``std_pnl_quote_ccy`` is the exact standard deviation under the fixed-policy
    Monte Carlo dynamics, including RFQ/dark-pool jump randomness, overlapping
    exponential impact kernels, and Brownian inventory exposure.  The base-
    currency fields are reference-spot conversions of the quote-currency
    moments, matching the convention used by :class:`ClosedFormPnLBenchmark`;
    they are not moments of the nonlinear random quantity ``PnL_quote / S_T``.
    """

    horizon_minutes: float
    initial_inventory: float
    reference_spot: float
    expected_pnl_quote_ccy: float
    variance_pnl_quote_ccy: float
    std_pnl_quote_ccy: float
    expected_pnl_base_ccy_at_reference_spot: float
    variance_pnl_base_ccy_at_reference_spot: float
    std_pnl_base_ccy_at_reference_spot: float


@dataclass
class MonteCarloFillEvent:
    time_minutes: float
    kind: str
    tier_idx: int | None
    tier_name: str
    side: str
    size: float
    execution_price: float
    spot_before_fill: float
    inventory_before: float
    inventory_after: float
    delta: float | None = None


@dataclass
class MonteCarloQuoteSeries:
    tier_idx: int
    tier_name: str
    side: str
    size: float
    prices: list[float] = field(default_factory=list)


@dataclass
class MonteCarloSamplePath:
    times: list[float] = field(default_factory=list)
    spots: list[float] = field(default_factory=list)
    inventories: list[float] = field(default_factory=list)
    inventory_event_times: list[float] = field(default_factory=list)
    inventory_event_values: list[float] = field(default_factory=list)
    quote_series: list[MonteCarloQuoteSeries] = field(default_factory=list)
    fills: list[MonteCarloFillEvent] = field(default_factory=list)

    def quote_prices(self, tier_idx: int, side: Side | str, size: float) -> list[float]:
        side_value = _side(side).value
        for series in self.quote_series:
            if (
                series.tier_idx == int(tier_idx)
                and series.side == side_value
                and math.isclose(series.size, float(size), rel_tol=0.0, abs_tol=1e-12)
            ):
                return series.prices
        return []


@dataclass
class MonteCarloResult:
    horizon_minutes: float
    initial_inventory: float
    pnl_quote_ccy: list[float] = field(default_factory=list)
    # Exact pathwise PnL in base currency, obtained by converting final
    # quote-currency wealth at the path's final simulated spot.
    pnl_base_ccy: list[float] = field(default_factory=list)
    final_inventory: list[float] = field(default_factory=list)
    final_spot: list[float] = field(default_factory=list)
    trade_count: list[int] = field(default_factory=list)
    sample_paths: list[MonteCarloSamplePath] = field(default_factory=list)
    inventory_sample_times: list[float] = field(default_factory=list)
    inventory_median: list[float] = field(default_factory=list)
    inventory_ci_lower: list[float] = field(default_factory=list)
    inventory_ci_upper: list[float] = field(default_factory=list)

    # Compatibility accessors for callers that used the original sample arrays.
    @property
    def sample_times(self) -> list[list[float]]:
        return [path.times for path in self.sample_paths]

    @property
    def sample_spots(self) -> list[list[float]]:
        return [path.spots for path in self.sample_paths]

    @property
    def mean_pnl(self) -> float:
        """Backward-compatible quote-currency mean PnL."""
        return float(np.mean(self.pnl_quote_ccy)) if self.pnl_quote_ccy else 0.0

    @property
    def std_pnl(self) -> float:
        """Backward-compatible quote-currency PnL standard deviation."""
        return float(np.std(self.pnl_quote_ccy, ddof=1)) if len(self.pnl_quote_ccy) > 1 else 0.0

    def percentile(self, p: float) -> float:
        """Backward-compatible quote-currency PnL percentile."""
        if not self.pnl_quote_ccy:
            return 0.0
        return float(np.percentile(np.asarray(self.pnl_quote_ccy, dtype=float), float(p)))

    @property
    def mean_pnl_base_ccy(self) -> float:
        return float(np.mean(self.pnl_base_ccy)) if self.pnl_base_ccy else 0.0

    @property
    def std_pnl_base_ccy(self) -> float:
        return float(np.std(self.pnl_base_ccy, ddof=1)) if len(self.pnl_base_ccy) > 1 else 0.0

    def percentile_base_ccy(self, p: float) -> float:
        if not self.pnl_base_ccy:
            return 0.0
        return float(np.percentile(np.asarray(self.pnl_base_ccy, dtype=float), float(p)))

# ---------------------------------------------------------------------------
# Howard solver
# ---------------------------------------------------------------------------

class HJBLadderSolver:
    def __init__(self, config: SolverConfig, penalty: QuadraticInventoryPenalty,
                 internalization_time: PolynomialInternalizationTime,
                 mdp_tiers: Optional[Sequence[MDPTier]] = None,
                 dark_pool: Optional[DarkPoolVenue] = None):
        self.config = config
        self.penalty = penalty
        self.internalization_time = internalization_time
        self.mdp_tiers = list(mdp_tiers or [])
        self.dark_pool = dark_pool
        self.grid_meta_ = SolverGridMeta()
        self.solve_q_grid_: list[float] = []
        self.hard_inventory_limit_: float = 0.0
        self.config.validate()

    def _max_inventory_jump(self) -> float:
        jumps = [float(z) for tier in self.mdp_tiers for z in tier.sizes_]
        if self.dark_pool is not None:
            jumps.extend(float(u) for u in self.dark_pool.posted_sizes)
        return max(jumps, default=0.0)

    def _build_padded_solve_grid(self) -> list[float]:
        """Extend the operational grid by one maximum allowed inventory jump.

        ``config.q_grid`` is the range for which the user wants prices. The
        hidden Howard grid extends to ``Q_operational + z_max``. Thus every
        RFQ/DP fill from an operational state lands inside a solved state. The
        outer edge is a hard risk bound: transitions that would move inventory
        farther out are inadmissible rather than extrapolated.
        """
        base = np.asarray(self.config.q_grid, dtype=float)
        q_op = float(base[-1])
        pad = self._max_inventory_jump()
        q_hard = q_op + pad
        if pad <= 0.0:
            return base.tolist()
        outer_step = float(base[-1] - base[-2])
        if outer_step <= 0.0:
            raise ValueError("q_grid must have positive outer spacing")
        extra = np.arange(q_op + outer_step, q_hard + 0.5 * outer_step, outer_step)
        extra = extra[extra < q_hard - 1e-10]
        pos = np.concatenate([base[len(base)//2:], extra, np.array([q_hard])])
        pos = np.unique(np.round(pos, 12))
        return np.concatenate((-pos[:0:-1], pos)).astype(float).tolist()

    def _inventory_transition_admissible(self, q: float, delta_q: float, tol: float = 1e-10) -> bool:
        target = float(q) + float(delta_q)
        return (
            target >= -self.hard_inventory_limit_ - tol
            and target <= self.hard_inventory_limit_ + tol
        )

    def _mdp_transition_admissible(self, q: float, z: float, side: Side | str) -> bool:
        direction = 1.0 if _side(side) is Side.Bid else -1.0
        return self._inventory_transition_admissible(q, direction * float(z))

    def _dark_pool_post_admissible(self, q: float, u: int, side: Side | str) -> bool:
        direction = 1.0 if _side(side) is Side.Bid else -1.0
        # Fill distributions can realize any k up to u. Requiring the largest
        # possible fill to stay inside the hard domain guarantees all fills do.
        return self._inventory_transition_admissible(q, direction * float(u))

    @staticmethod
    def repair_or_throw_bounds(lower: float, upper: float, tier_name: str, side_name: str) -> tuple[float, float]:
        tol = 1e-12
        if upper < lower:
            if lower - upper > tol:
                raise RuntimeError(f"Infeasible ladder bounds for MDP tier '{tier_name}', side '{side_name}'.")
            upper = lower
        return lower, upper

    def validate_problem_definition(self) -> None:
        self.config.validate()
        for tier in self.mdp_tiers: tier.validate()
        if self.dark_pool is not None: self.dark_pool.validate()

    def initialize_policy_shapes(self) -> None:
        for tier in self.mdp_tiers: tier.reset_policy_shape(self.solve_q_grid_)
        if self.dark_pool is not None: self.dark_pool.reset_policy_shape(self.solve_q_grid_)

    def prepare_solve_context(self) -> None:
        self.validate_problem_definition()
        self.solve_q_grid_ = self._build_padded_solve_grid()
        self.hard_inventory_limit_ = float(self.solve_q_grid_[-1])
        self.grid_meta_ = build_solver_grid_meta(self.solve_q_grid_)
        self.initialize_policy_shapes()

    def mdp_fill_payoff(self, lam: float, z: float, d: float, mu: float, dh: float) -> float:
        return lam * (z * self.config.spread * (0.5 - d) - z * mu + dh)

    @staticmethod
    def _ladder_gaps(row: Sequence[float]) -> list[float]:
        """Adjacent volume premia: delta[j-1] - delta[j]."""
        return [float(row[j - 1]) - float(row[j]) for j in range(1, len(row))]

    def mdp_bounds_for_rung(
        self,
        tier: MDPTier,
        rung_idx: int,
        current_delta: Sequence[float],
        q: float,
        side: Side | str,
        prev_delta: Optional[Sequence[float]] = None,
        min_gaps: Optional[Sequence[float]] = None,
        max_gaps: Optional[Sequence[float]] = None,
        rung_lower_bounds: Optional[Sequence[float]] = None,
        rung_upper_bounds: Optional[Sequence[float]] = None,
    ) -> tuple[float, float]:
        """Bounds for one greedy rung.

        The admissible ladder has two kinds of trader constraints.

        1. Within one inventory row, larger tickets cannot be tighter:

               delta[j] >= delta[j+1].

        2. Moving one inventory state farther away from zero:

           * on the inventory-increasing side every individual rung becomes
             no more aggressive and adjacent gaps may only widen;
           * on the inventory-reducing side every individual rung becomes
             no less aggressive and adjacent gaps may only flatten.

        ``rung_lower_bounds`` / ``rung_upper_bounds`` impose the individual-rung
        monotonicity relative to the adjacent inner inventory row. ``min_gaps``
        / ``max_gaps`` impose the corresponding adjacent-gap monotonicity.
        """
        nz = len(tier.sizes_)
        expected = max(0, nz - 1)

        mins = list(map(float, min_gaps or []))
        maxs = list(map(float, max_gaps or []))
        lowers = list(map(float, rung_lower_bounds or []))
        uppers = list(map(float, rung_upper_bounds or []))

        for name, gaps in (("min_gaps", mins), ("max_gaps", maxs)):
            if gaps and len(gaps) != expected:
                raise ValueError(
                    f"{name} for tier '{tier.name}' must have length {expected}."
                )
            if any(g < -1e-12 for g in gaps):
                raise ValueError(f"{name} must be nonnegative")

        for name, bounds in (("rung_lower_bounds", lowers), ("rung_upper_bounds", uppers)):
            if bounds and len(bounds) != nz:
                raise ValueError(
                    f"{name} for tier '{tier.name}' must have length {nz}."
                )

        mins = [max(0.0, g) for g in mins]
        maxs = [max(0.0, g) for g in maxs]
        if mins and maxs:
            for j, (gmin, gmax) in enumerate(zip(mins, maxs)):
                if gmin > gmax + 1e-12:
                    raise ValueError(
                        f"inconsistent gap bounds at rung {j}: min {gmin} > max {gmax}"
                    )

        # Reserve enough room above delta_min for all remaining mandatory gaps.
        lower = tier.delta_min + (sum(mins[rung_idx:]) if mins else 0.0)
        upper = tier.delta_max

        if lowers:
            lower = max(lower, lowers[rung_idx])
        if uppers:
            upper = min(upper, uppers[rung_idx])

        if rung_idx > 0:
            previous_rung = float(current_delta[rung_idx - 1])

            required_min_gap = mins[rung_idx - 1] if mins else 0.0
            upper = min(upper, previous_rung - required_min_gap)

            if maxs:
                allowed_max_gap = maxs[rung_idx - 1]
                lower = max(lower, previous_rung - allowed_max_gap)

        return self.repair_or_throw_bounds(lower, upper, tier.name, _side(side).value)

    def _build_mdp_ladder(
        self,
        tier: MDPTier,
        h: np.ndarray,
        q: float,
        side: Side,
        min_gaps: Optional[Sequence[float]] = None,
        max_gaps: Optional[Sequence[float]] = None,
        rung_lower_bounds: Optional[Sequence[float]] = None,
        rung_upper_bounds: Optional[Sequence[float]] = None,
    ) -> list[float]:
        row: list[float] = []
        direction = 1.0 if side is Side.Bid else -1.0
        hq = _interp_linear_bounded(self.solve_q_grid_, h, q)
        for j, z in enumerate(tier.sizes_):
            lo, hi = self.mdp_bounds_for_rung(
                tier, j, row, q, side,
                min_gaps=min_gaps, max_gaps=max_gaps,
                rung_lower_bounds=rung_lower_bounds,
                rung_upper_bounds=rung_upper_bounds,
            )
            q_next = q + direction * z
            if not self._mdp_transition_admissible(q, z, side):
                # Hard outer inventory bound: this RFQ size is not quoted. The
                # stored delta is only a display placeholder; Bellman/reward
                # calculations skip the transition entirely.
                row.append(tier.delta_min)
                continue
            tau = self.internalization_time.value(q_next)
            mu = tier.markout_model.expected_markout(z, tau) if tier.use_markout else 0.0
            additive = -z * mu + _interp_linear_bounded(self.solve_q_grid_, h, q_next) - hq
            d = tier.flow_curve.optimal_delta(z, self.config.spread, additive)
            row.append(min(max(d, lo), hi))
        return row

    def _project_ladder_to_bounds(
        self,
        tier: MDPTier,
        row: Sequence[float],
        min_gaps: Optional[Sequence[float]] = None,
        max_gaps: Optional[Sequence[float]] = None,
        rung_lower_bounds: Optional[Sequence[float]] = None,
        rung_upper_bounds: Optional[Sequence[float]] = None,
    ) -> list[float]:
        """Project a row into the sequential trader-admissible set."""
        out: list[float] = []
        for j, raw in enumerate(row):
            lo, hi = self.mdp_bounds_for_rung(
                tier, j, out, 0.0, Side.Bid,
                min_gaps=min_gaps, max_gaps=max_gaps,
                rung_lower_bounds=rung_lower_bounds,
                rung_upper_bounds=rung_upper_bounds,
            )
            out.append(min(max(float(raw), lo), hi))
        return out

    def _ladder_hamiltonian(self, tier: MDPTier, h: np.ndarray, q: float, side: Side, row: Sequence[float]) -> float:
        direction = 1.0 if side is Side.Bid else -1.0
        hq = _interp_linear_bounded(self.solve_q_grid_, h, q)
        total = 0.0
        for d, z in zip(row, tier.sizes_):
            if not self._mdp_transition_admissible(q, z, side):
                continue
            q_next = q + direction * z
            tau = self.internalization_time.value(q_next)
            mu = tier.markout_model.expected_markout(z, tau) if tier.use_markout else 0.0
            dh = _interp_linear_bounded(self.solve_q_grid_, h, q_next) - hq
            total += self.mdp_fill_payoff(tier.flow_curve.arrival_rate(d, z), z, d, mu, dh)
        return total

    def _choose_howard_safe_row(
        self,
        tier: MDPTier,
        h: np.ndarray,
        q: float,
        side: Side,
        current: Sequence[float],
        candidate: Sequence[float],
        min_gaps: Optional[Sequence[float]] = None,
        max_gaps: Optional[Sequence[float]] = None,
        rung_lower_bounds: Optional[Sequence[float]] = None,
        rung_upper_bounds: Optional[Sequence[float]] = None,
    ) -> list[float]:
        baseline = list(map(float, current))
        if any(x is not None for x in (min_gaps, max_gaps, rung_lower_bounds, rung_upper_bounds)):
            baseline = self._project_ladder_to_bounds(
                tier, baseline,
                min_gaps=min_gaps, max_gaps=max_gaps,
                rung_lower_bounds=rung_lower_bounds,
                rung_upper_bounds=rung_upper_bounds,
            )
        if self._ladder_hamiltonian(tier, h, q, side, candidate) >= self._ladder_hamiltonian(
            tier, h, q, side, baseline
        ):
            return list(map(float, candidate))
        return baseline

    def _build_mdp_policy(self, tier: MDPTier, h: np.ndarray, side: Side) -> None:
        """Improve one side with monotone level and volume-premium constraints.

        Relative to the adjacent inventory state closer to zero:

        * inventory-increasing side: every rung becomes no more aggressive and
          adjacent gaps may only widen;
        * inventory-reducing side: every rung becomes no less aggressive and
          adjacent gaps may only flatten.

        Cross-inventory bounds use the *previous complete policy* rather than a
        row updated earlier in the same Howard pass. This Jacobi-style update
        avoids cascading order dependence. At convergence the fixed policy
        satisfies the desired neighboring-state inequalities exactly.
        """
        mat = tier.policy.bid if side is Side.Bid else tier.policy.ask
        q_grid = self.solve_q_grid_
        q0 = self.grid_meta_.q0_idx
        old = [list(map(float, row)) for row in mat]
        improved = [list(row) for row in old]

        # q=0 has no inventory-direction constraint; only within-row size order.
        candidate0 = self._build_mdp_ladder(tier, h, q_grid[q0], side)
        improved[q0] = self._choose_howard_safe_row(
            tier, h, q_grid[q0], side, old[q0], candidate0
        )

        def improve_state(i: int, inner_i: int, inventory_increasing: bool, delta_direction: str) -> None:
            inner = old[inner_i]
            inner_gaps = self._ladder_gaps(inner)
            kwargs: dict[str, Sequence[float]] = {}
            if inventory_increasing:
                kwargs["min_gaps"] = inner_gaps
            else:
                kwargs["max_gaps"] = inner_gaps

            if delta_direction == "down":
                # Bid price lower or ask price higher, depending on side: in
                # delta-space the current rung may not exceed the inner row.
                kwargs["rung_upper_bounds"] = inner
            elif delta_direction == "up":
                kwargs["rung_lower_bounds"] = inner
            else:
                raise ValueError("delta_direction must be 'down' or 'up'")

            candidate = self._build_mdp_ladder(tier, h, q_grid[i], side, **kwargs)
            improved[i] = self._choose_howard_safe_row(
                tier, h, q_grid[i], side, old[i], candidate, **kwargs
            )

        q_operational_max = float(self.config.q_grid[-1])

        # Positive inventory, moving q upward away from zero. Trader shape
        # constraints are enforced only on the operational pricing range. The
        # hidden buffer exists solely to value continuation states and is free
        # to choose its own mean-reverting policy subject to the hard bound.
        for i in range(q0 + 1, len(q_grid)):
            if q_grid[i] > q_operational_max + 1e-10:
                candidate = self._build_mdp_ladder(tier, h, q_grid[i], side)
                improved[i] = self._choose_howard_safe_row(
                    tier, h, q_grid[i], side, old[i], candidate
                )
                continue
            inner_i = i - 1
            if side is Side.Bid:
                # Long book: bids add inventory -> lower bid deltas + wider gaps.
                improve_state(i, inner_i, inventory_increasing=True, delta_direction="down")
            else:
                # Long book: asks reduce inventory -> higher ask deltas + flatter gaps.
                improve_state(i, inner_i, inventory_increasing=False, delta_direction="up")

        # Negative inventory, moving q downward away from zero.
        for i in range(q0 - 1, -1, -1):
            if q_grid[i] < -q_operational_max - 1e-10:
                candidate = self._build_mdp_ladder(tier, h, q_grid[i], side)
                improved[i] = self._choose_howard_safe_row(
                    tier, h, q_grid[i], side, old[i], candidate
                )
                continue
            inner_i = i + 1
            if side is Side.Bid:
                # Short book: bids reduce inventory -> higher bid deltas + flatter gaps.
                improve_state(i, inner_i, inventory_increasing=False, delta_direction="up")
            else:
                # Short book: asks add shorts -> lower ask deltas + wider gaps.
                improve_state(i, inner_i, inventory_increasing=True, delta_direction="down")

        for i, row in enumerate(improved):
            mat[i] = row

    def _dark_pool_fill_value(self, h: np.ndarray, q: float, hq: float, u: int,
                              dist: ArrivalDistribution, fee: float, direction: float) -> float:
        return dist.expected_fill_value(
            lambda x: _interp_linear_bounded(self.solve_q_grid_, h, x), q, hq, u, fee, direction
        )

    def build_dark_pool_policy(self, venue: DarkPoolVenue, h_vec: Sequence[float]) -> None:
        h = np.asarray(h_vec, dtype=float)
        for side, direction, dist, fee in [
            (Side.Bid, +1.0, venue.dist_bid, venue.fee_per_unit_bid),
            (Side.Ask, -1.0, venue.dist_ask, venue.fee_per_unit_ask),
        ]:
            for i, q in enumerate(self.solve_q_grid_):
                if not venue.is_admissible(q, side):
                    venue.policy.set_posted_size(i, side, 0.0, False)
                    continue
                hq = _interp_linear_bounded(self.solve_q_grid_, h, q)
                best_size, best_value = 0.0, -math.inf
                for u_raw in venue.posted_sizes:
                    u = int(round(u_raw))
                    if not self._dark_pool_post_admissible(q, u, side):
                        continue
                    value = self._dark_pool_fill_value(h, q, hq, u, dist, fee, direction)
                    if value > best_value:
                        best_size, best_value = u_raw, value
                active = best_value > venue.min_fill_value
                venue.policy.set_posted_size(i, side, best_size if active else 0.0, active)

    def update_policies(self, h_vec: Sequence[float]) -> None:
        h = np.asarray(h_vec, dtype=float)
        for tier in self.mdp_tiers:
            self._build_mdp_policy(tier, h, Side.Bid)
            self._build_mdp_policy(tier, h, Side.Ask)
        if self.dark_pool is not None:
            self.build_dark_pool_policy(self.dark_pool, h)

    def initialize_stabilizing_policy(self) -> None:
        zero = np.zeros(self.grid_meta_.nq, dtype=float)
        for tier in self.mdp_tiers:
            zero_bid = self._build_mdp_ladder(tier, zero, 0.0, Side.Bid)
            zero_ask = self._build_mdp_ladder(tier, zero, 0.0, Side.Ask)

            # Start from a mean-reverting policy that already satisfies the
            # structural inventory rules. Wrong-way ladders are shifted
            # defensively without changing their q=0 gaps; right-way ladders
            # start at the q=0 ladder.
            def defensive_shift(row: Sequence[float]) -> list[float]:
                shift = max(0.0, float(row[-1]) - tier.delta_min)
                return [max(tier.delta_min, float(d) - shift) for d in row]

            defensive_bid = defensive_shift(zero_bid)
            defensive_ask = defensive_shift(zero_ask)
            for i, q in enumerate(self.solve_q_grid_):
                if q > 0.0:
                    tier.policy.bid[i] = list(defensive_bid)
                    tier.policy.ask[i] = list(zero_ask)
                elif q < 0.0:
                    tier.policy.bid[i] = list(zero_bid)
                    tier.policy.ask[i] = list(defensive_ask)
                else:
                    tier.policy.bid[i] = list(zero_bid)
                    tier.policy.ask[i] = list(zero_ask)
        if self.dark_pool is not None:
            venue = self.dark_pool
            for i, q in enumerate(self.solve_q_grid_):
                venue.policy.set_posted_size(i, Side.Bid, 0.0, False)
                venue.policy.set_posted_size(i, Side.Ask, 0.0, False)
                if not venue.posted_sizes or q == 0.0: continue
                u = venue.posted_sizes[-1]
                if q < 0 and venue.is_admissible(q, Side.Bid): venue.policy.set_posted_size(i, Side.Bid, u, True)
                if q > 0 and venue.is_admissible(q, Side.Ask): venue.policy.set_posted_size(i, Side.Ask, u, True)

    def _add_transition(self, L: np.ndarray, row: int, target_q: float, rate: float) -> None:
        if rate == 0.0:
            return
        if target_q < -self.hard_inventory_limit_ - 1e-10 or target_q > self.hard_inventory_limit_ + 1e-10:
            raise RuntimeError("attempted transition outside the solved inventory domain")
        i0, w0, i1, w1 = _interp_weights_bounded(self.solve_q_grid_, target_q)
        L[row, i0] += rate * w0
        L[row, i1] += rate * w1
        L[row, row] -= rate

    def fixed_policy_affine_operator(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.grid_meta_.nq
        c = np.zeros(n, dtype=float)
        L = np.zeros((n, n), dtype=float)

        for i, q in enumerate(self.solve_q_grid_):
            c[i] = -self.penalty.value(q) + self.config.spot_drift * q

            for tier in self.mdp_tiers:
                for side, direction, row in [
                    (Side.Bid, +1.0, tier.policy.bid[i]),
                    (Side.Ask, -1.0, tier.policy.ask[i]),
                ]:
                    for j, z in enumerate(tier.sizes_):
                        if not self._mdp_transition_admissible(q, z, side):
                            continue
                        d = row[j]
                        lam = tier.flow_curve.arrival_rate(d, z)
                        q_next = q + direction * z
                        tau = self.internalization_time.value(q_next)
                        mu = tier.markout_model.expected_markout(z, tau) if tier.use_markout else 0.0
                        c[i] += lam * (z * self.config.spread * (0.5 - d) - z * mu)
                        self._add_transition(L, i, q_next, lam)

            if self.dark_pool is not None:
                venue = self.dark_pool
                for side, direction, dist, fee, size_vec in [
                    (Side.Bid, +1.0, venue.dist_bid, venue.fee_per_unit_bid, venue.policy.bid_size),
                    (Side.Ask, -1.0, venue.dist_ask, venue.fee_per_unit_ask, venue.policy.ask_size),
                ]:
                    if not venue.policy.is_active(i, side):
                        continue
                    u = int(round(size_vec[i]))
                    if u <= 0 or not self._dark_pool_post_admissible(q, u, side):
                        continue
                    for k, rate in dist.fill_rates(u):
                        c[i] -= rate * fee * k
                        self._add_transition(L, i, q + direction * k, rate)
        return c, L

    def evaluate_current_policy(self) -> tuple[np.ndarray, float]:
        c, L = self.fixed_policy_affine_operator()
        n = len(c)
        A = np.zeros((n + 1, n + 1), dtype=float)
        b = np.zeros(n + 1, dtype=float)
        A[:n, :n] = L
        A[:n, n] = -1.0
        b[:n] = -c
        A[n, self.grid_meta_.q0_idx] = 1.0
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Howard policy evaluation produced a singular/non-ergodic linear system."
            ) from exc
        if not np.all(np.isfinite(x)):
            raise RuntimeError("Howard policy evaluation produced non-finite values.")
        return x[:n], float(x[n])

    def mdp_bellman_contribution_side(self, tier: MDPTier, q: float, q_index: int,
                                      h_vec: Sequence[float], side: Side | str) -> float:
        h = np.asarray(h_vec, dtype=float)
        s = _side(side)
        direction = 1.0 if s is Side.Bid else -1.0
        row = tier.policy.bid[q_index] if s is Side.Bid else tier.policy.ask[q_index]
        hq = _interp_linear_bounded(self.solve_q_grid_, h, q)
        total = 0.0
        for j, z in enumerate(tier.sizes_):
            if not self._mdp_transition_admissible(q, z, s):
                continue
            d = row[j]
            q_next = q + direction * z
            mu = tier.markout_model.expected_markout(z, self.internalization_time.value(q_next)) if tier.use_markout else 0.0
            total += self.mdp_fill_payoff(tier.flow_curve.arrival_rate(d, z), z, d, mu,
                                          _interp_linear_bounded(self.solve_q_grid_, h, q_next) - hq)
        return total

    def dark_pool_bellman_contribution(self, venue: DarkPoolVenue, q: float, q_index: int,
                                       h_vec: Sequence[float]) -> float:
        h = np.asarray(h_vec, dtype=float)
        hq = _interp_linear_bounded(self.solve_q_grid_, h, q)
        total = 0.0
        if venue.policy.is_active(q_index, Side.Bid) and venue.policy.bid_size[q_index] > 0:
            total += self._dark_pool_fill_value(h, q, hq, int(round(venue.policy.bid_size[q_index])),
                                                venue.dist_bid, venue.fee_per_unit_bid, +1.0)
        if venue.policy.is_active(q_index, Side.Ask) and venue.policy.ask_size[q_index] > 0:
            total += self._dark_pool_fill_value(h, q, hq, int(round(venue.policy.ask_size[q_index])),
                                                venue.dist_ask, venue.fee_per_unit_ask, -1.0)
        return total

    def bellman_rhs(self, h_vec: Sequence[float]) -> np.ndarray:
        h = np.asarray(h_vec, dtype=float)
        rhs = np.zeros(self.grid_meta_.nq, dtype=float)
        for i, q in enumerate(self.solve_q_grid_):
            val = -self.penalty.value(q) + self.config.spot_drift * q
            for tier in self.mdp_tiers:
                val += self.mdp_bellman_contribution_side(tier, q, i, h, Side.Bid)
                val += self.mdp_bellman_contribution_side(tier, q, i, h, Side.Ask)
            if self.dark_pool is not None:
                val += self.dark_pool_bellman_contribution(self.dark_pool, q, i, h)
            rhs[i] = val
        return rhs

    def bellman_rhs_from_policies(self, h_vec: Sequence[float]) -> list[float]:
        if len(h_vec) != len(self.solve_q_grid_):
            raise ValueError("h_vec size must match q_grid")
        self.prepare_solve_context()
        self.update_policies(h_vec)
        return self.bellman_rhs(h_vec).tolist()

    @staticmethod
    def _machine_convergence_scale(rho: float, rhs: np.ndarray) -> float:
        scale = max(1.0, abs(rho), float(np.max(np.abs(rhs))) if rhs.size else 0.0)
        return math.sqrt(np.finfo(float).eps) * scale

    def _policy_vector(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        for tier in self.mdp_tiers:
            parts.append(np.asarray(tier.policy.bid, dtype=float).ravel())
            parts.append(np.asarray(tier.policy.ask, dtype=float).ravel())
        if self.dark_pool is not None:
            parts.append(np.asarray(self.dark_pool.policy.bid_size, dtype=float))
            parts.append(np.asarray(self.dark_pool.policy.ask_size, dtype=float))
            parts.append(np.asarray(self.dark_pool.policy.bid_active, dtype=float))
            parts.append(np.asarray(self.dark_pool.policy.ask_active, dtype=float))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

    @staticmethod
    def _machine_policy_scale(policy: np.ndarray) -> float:
        scale = max(1.0, float(np.max(np.abs(policy))) if policy.size else 0.0)
        return math.sqrt(np.finfo(float).eps) * scale


    @staticmethod
    def _expm_pade13(A: np.ndarray) -> np.ndarray:
        """Real matrix exponential using scaling-and-squaring Padé(13).

        This is the standard Higham-style algorithm specialized to dense
        NumPy arrays.  It avoids eigendecomposition, which can be numerically
        fragile for non-normal Markov generators even when exp(A) is perfectly
        well conditioned and real.
        """
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix exponential requires a square matrix")
        n = A.shape[0]
        if n == 0:
            return A.copy()

        # Higham 2005 threshold for Padé order 13 in IEEE double precision.
        theta13 = 5.371920351148152
        norm1 = float(np.linalg.norm(A, 1))
        if norm1 == 0.0:
            return np.eye(n, dtype=float)
        s = max(0, int(math.ceil(math.log2(norm1 / theta13)))) if norm1 > theta13 else 0
        As = A / (2.0 ** s)

        b = np.array([
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ], dtype=float)

        I = np.eye(n, dtype=float)
        A2 = As @ As
        A4 = A2 @ A2
        A6 = A4 @ A2

        U = As @ (
            A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
            + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I
        )
        V = (
            A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2)
            + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I
        )

        # (V-U) R = V+U is numerically preferable to explicit inversion.
        R = np.linalg.solve(V - U, V + U)
        for _ in range(s):
            R = R @ R
        return np.asarray(R, dtype=float)

    def _mc_mean_augmented_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the exact first-moment system for the Monte Carlo PnL model.

        The Monte Carlo does *not* charge markout as a deterministic cost on the
        filled quantity.  Instead, each RFQ fill creates an exponentially
        decaying adverse drift in spot.  That drift marks the desk's *entire
        current inventory* while it is active.  Consequently a reward term
        ``-z * expected_markout`` is not the correct finite-horizon mean once
        inventory before/after the fill is nonzero.

        Because the impact kernel is exponential, its remaining future move is
        a linear state variable.  For every distinct impact timescale ``tau`` we
        track the first moment of that impact conditional on the inventory
        state.  Together with the inventory CTMC this gives a finite-dimensional
        linear ODE whose matrix exponential yields the exact expected
        quote-currency PnL used by ``simulate_pnl`` (Brownian noise has zero
        mean and therefore does not enter this system).

        Returns
        -------
        A : ndarray
            Row-vector evolution matrix for
            ``[inventory probabilities, impact moments..., cumulative PnL]``.
        y0_template : ndarray
            Zero initial state template. The caller inserts the initial
            inventory distribution into the first block.
        """
        if not self.solve_q_grid_:
            raise RuntimeError("solve context is not initialized")

        n = self.grid_meta_.nq
        L = np.zeros((n, n), dtype=float)
        # Actual instantaneous wealth reward excluding post-trade impact.
        # RFQ impact is represented dynamically below, exactly as in Monte Carlo.
        r0 = np.zeros(n, dtype=float)
        impact_jump_mats: dict[float, np.ndarray] = {}

        def add_impact_jump(row: int, target_q: float, rate: float, tau: float, jump: float) -> None:
            mat = impact_jump_mats.setdefault(float(tau), np.zeros((n, n), dtype=float))
            i0, w0, i1, w1 = _interp_weights_bounded(self.solve_q_grid_, target_q)
            mat[row, i0] += rate * w0 * jump
            mat[row, i1] += rate * w1 * jump

        for i, q in enumerate(self.solve_q_grid_):
            # Holding inventory earns/loses the exogenous spot drift.
            r0[i] = self.config.spot_drift * q

            for tier in self.mdp_tiers:
                for side, direction, row in [
                    (Side.Bid, +1.0, tier.policy.bid[i]),
                    (Side.Ask, -1.0, tier.policy.ask[i]),
                ]:
                    for j, z in enumerate(tier.sizes_):
                        if not self._mdp_transition_admissible(q, z, side):
                            continue
                        d = float(row[j])
                        lam = tier.flow_curve.arrival_rate(d, z)
                        if lam <= 0.0:
                            continue
                        q_next = q + direction * z

                        # At the fill instant, marking the new inventory at the
                        # contemporaneous spot leaves only spread capture as a
                        # wealth jump.
                        r0[i] += lam * z * self.config.spread * (0.5 - d)
                        self._add_transition(L, i, q_next, lam)

                        if tier.use_markout:
                            tau = float(tier.markout_model.tau)
                            # Same signed remaining future price move inserted
                            # by simulate_pnl after a fill.
                            impact_jump = -direction * tier.markout_model.asymptotic_markout(z)
                            add_impact_jump(i, q_next, lam, tau, impact_jump)

            if self.dark_pool is not None:
                venue = self.dark_pool
                for side, direction, dist, fee, size_vec in [
                    (Side.Bid, +1.0, venue.dist_bid, venue.fee_per_unit_bid, venue.policy.bid_size),
                    (Side.Ask, -1.0, venue.dist_ask, venue.fee_per_unit_ask, venue.policy.ask_size),
                ]:
                    if not venue.policy.is_active(i, side):
                        continue
                    u = int(round(size_vec[i]))
                    if u <= 0 or not self._dark_pool_post_admissible(q, u, side):
                        continue
                    for k, rate in dist.fill_rates(u):
                        if rate <= 0.0:
                            continue
                        r0[i] -= rate * fee * k
                        self._add_transition(L, i, q + direction * k, rate)

        taus = sorted(impact_jump_mats)
        n_blocks = 1 + len(taus)
        dim = n_blocks * n + 1
        pnl_idx = dim - 1
        A = np.zeros((dim, dim), dtype=float)

        # Row-vector convention: p'(t) = p(t) L.
        A[:n, :n] = L
        A[:n, pnl_idx] = r0

        q_vec = np.asarray(self.solve_q_grid_, dtype=float)
        for k, tau in enumerate(taus):
            sl = slice((k + 1) * n, (k + 2) * n)
            # A fill transports probability into the new inventory state and
            # simultaneously adds the signed remaining impact jump.
            A[:n, sl] = impact_jump_mats[tau]
            # Existing impact moments move with inventory transitions and decay
            # exponentially between fills.
            A[sl, sl] = L - np.eye(n, dtype=float) / tau
            # Spot drift generated by remaining impact is X/tau.  It marks the
            # full current inventory q, hence the q * X / tau PnL rate.
            A[sl, pnl_idx] = q_vec / tau

        return A, np.zeros(dim, dtype=float)

    @staticmethod
    def _moment_monomials(n_variables: int) -> list[tuple[int, ...]]:
        """All monomials of total degree <= 2 in ``n_variables`` variables."""
        d = int(n_variables)
        if d <= 0:
            raise ValueError("n_variables must be positive")
        zero = (0,) * d
        out: list[tuple[int, ...]] = [zero]
        for r in range(d):
            e = [0] * d
            e[r] = 1
            out.append(tuple(e))
        for r in range(d):
            for u in range(r, d):
                e = [0] * d
                e[r] += 1
                e[u] += 1
                out.append(tuple(e))
        return out

    @staticmethod
    def _shifted_monomial_coefficients(
        exponent: tuple[int, ...], shift: np.ndarray
    ) -> dict[tuple[int, ...], float]:
        """Expand ``prod_r (x_r + shift_r)**exponent_r`` in monomials."""
        d = len(exponent)
        terms: dict[tuple[int, ...], float] = {(0,) * d: 1.0}
        for r, power in enumerate(exponent):
            if power == 0:
                continue
            new_terms: dict[tuple[int, ...], float] = {}
            for base_exp, base_coeff in terms.items():
                for k in range(power + 1):
                    e = list(base_exp)
                    e[r] += k
                    e_tuple = tuple(e)
                    coeff = (
                        base_coeff
                        * math.comb(power, k)
                        * float(shift[r]) ** (power - k)
                    )
                    new_terms[e_tuple] = new_terms.get(e_tuple, 0.0) + coeff
            terms = new_terms
        return terms

    def _mc_second_moment_system(
        self, sigma: float
    ) -> tuple[np.ndarray, list[tuple[int, ...]], int]:
        """Build an exact degree-two moment system for terminal quote PnL.

        Continuous variables are the remaining signed impact moves for each
        distinct exponential decay time and terminal marked-to-market PnL ``P``
        in *million quote currency*.  Conditional on inventory, their dynamics
        are affine and fills add deterministic jumps. Therefore all moments of
        total degree <= 2 form a closed finite-dimensional linear system.

        Brownian spot noise contributes the Ito term ``(sigma*q)**2`` to the
        evolution of ``E[P**2]``. No discretisation in time is required.
        """
        if not self.solve_q_grid_:
            raise RuntimeError("solve context is not initialized")
        sigma = float(sigma)
        if sigma < 0.0:
            raise ValueError("sigma must be nonnegative")

        n = self.grid_meta_.nq
        taus = sorted({
            float(tier.markout_model.tau)
            for tier in self.mdp_tiers
            if tier.use_markout
        })
        tau_to_var = {tau: k for k, tau in enumerate(taus)}
        n_impact = len(taus)
        pnl_var = n_impact
        n_variables = n_impact + 1
        monomials = self._moment_monomials(n_variables)
        monomial_index = {e: m for m, e in enumerate(monomials)}
        n_monomials = len(monomials)
        dim = n * n_monomials
        A = np.zeros((dim, dim), dtype=float)

        def state_moment_index(q_idx: int, m_idx: int) -> int:
            return q_idx * n_monomials + m_idx

        # Between fills: impact components decay, while PnL accumulates base
        # drift + impact drift and Brownian inventory PnL.
        for i, q in enumerate(self.solve_q_grid_):
            drift_const = np.zeros(n_variables, dtype=float)
            drift_linear = np.zeros((n_variables, n_variables), dtype=float)
            for k, tau in enumerate(taus):
                drift_linear[k, k] = -1.0 / tau
            drift_const[pnl_var] = self.config.spot_drift * q
            for k, tau in enumerate(taus):
                drift_linear[pnl_var, k] = q / tau
            pnl_diffusion_variance = (sigma * q) ** 2

            # For each target monomial phi, expand its infinitesimal generator
            # G phi into the same degree-two basis. Row-vector convention:
            # moment'(target) = sum_source moment(source) * A[source,target].
            for target_m, exponent in enumerate(monomials):
                for r, power in enumerate(exponent):
                    if power == 0:
                        continue
                    reduced = list(exponent)
                    reduced[r] -= 1
                    if drift_const[r] != 0.0:
                        source_exp = tuple(reduced)
                        A[
                            state_moment_index(i, monomial_index[source_exp]),
                            state_moment_index(i, target_m),
                        ] += power * drift_const[r]
                    for u in range(n_variables):
                        coeff = drift_linear[r, u]
                        if coeff == 0.0:
                            continue
                        source_exp = list(reduced)
                        source_exp[u] += 1
                        source_tuple = tuple(source_exp)
                        A[
                            state_moment_index(i, monomial_index[source_tuple]),
                            state_moment_index(i, target_m),
                        ] += power * coeff

                pnl_power = exponent[pnl_var]
                if pnl_power >= 2 and pnl_diffusion_variance > 0.0:
                    source_exp = list(exponent)
                    source_exp[pnl_var] -= 2
                    source_tuple = tuple(source_exp)
                    A[
                        state_moment_index(i, monomial_index[source_tuple]),
                        state_moment_index(i, target_m),
                    ] += (
                        0.5
                        * pnl_power
                        * (pnl_power - 1)
                        * pnl_diffusion_variance
                    )

            events: list[tuple[float, float, np.ndarray]] = []
            for tier in self.mdp_tiers:
                for side, direction, row in [
                    (Side.Bid, +1.0, tier.policy.bid[i]),
                    (Side.Ask, -1.0, tier.policy.ask[i]),
                ]:
                    for j, z in enumerate(tier.sizes_):
                        if not self._mdp_transition_admissible(q, z, side):
                            continue
                        delta = float(row[j])
                        rate = tier.flow_curve.arrival_rate(delta, z)
                        if rate <= 0.0:
                            continue
                        shift = np.zeros(n_variables, dtype=float)
                        # Wealth jump at the contemporaneous spot: spread capture.
                        shift[pnl_var] = z * self.config.spread * (0.5 - delta)
                        if tier.use_markout:
                            tau = float(tier.markout_model.tau)
                            impact_var = tau_to_var[tau]
                            shift[impact_var] = (
                                -direction * tier.markout_model.asymptotic_markout(z)
                            )
                        events.append((q + direction * z, float(rate), shift))

            if self.dark_pool is not None:
                venue = self.dark_pool
                for side, direction, dist, fee, size_vec in [
                    (Side.Bid, +1.0, venue.dist_bid, venue.fee_per_unit_bid, venue.policy.bid_size),
                    (Side.Ask, -1.0, venue.dist_ask, venue.fee_per_unit_ask, venue.policy.ask_size),
                ]:
                    if not venue.policy.is_active(i, side):
                        continue
                    posted = int(round(size_vec[i]))
                    if posted <= 0 or not self._dark_pool_post_admissible(q, posted, side):
                        continue
                    for fill_size, rate in dist.fill_rates(posted):
                        if rate <= 0.0:
                            continue
                        shift = np.zeros(n_variables, dtype=float)
                        shift[pnl_var] = -float(fee) * float(fill_size)
                        events.append((q + direction * fill_size, float(rate), shift))

            # Fill jumps transport the inventory state and shift impact/PnL.
            for target_q, rate, shift in events:
                i0, w0, i1, w1 = _interp_weights_bounded(self.solve_q_grid_, target_q)
                target_states = ((i0, w0), (i1, w1))
                for target_m, exponent in enumerate(monomials):
                    source_same = state_moment_index(i, target_m)
                    A[source_same, source_same] -= rate
                    expansion = self._shifted_monomial_coefficients(exponent, shift)
                    for target_i, weight in target_states:
                        if weight == 0.0:
                            continue
                        target_col = state_moment_index(target_i, target_m)
                        for source_exp, coeff in expansion.items():
                            source_m = monomial_index[source_exp]
                            A[
                                state_moment_index(i, source_m), target_col
                            ] += rate * weight * coeff

        return A, monomials, pnl_var

    def closed_form_pnl_statistics(
        self,
        horizon_minutes: float,
        sigma: float,
        initial_inventory: float = 0.0,
    ) -> ClosedFormPnLStatistics:
        """Exact finite-horizon quote-currency mean and standard deviation.

        The first two quote-currency moments match the fixed-policy dynamics of
        :meth:`simulate_pnl`: fill/event randomness, spread capture, dark-pool
        fees, overlapping exponential markout drift, exogenous spot drift, and
        Brownian inventory exposure are all included.

        For display in base currency, both mean and standard deviation are
        divided by the reference spot ``S0``. This is a reference-spot
        conversion, not the exact moments of the pathwise Monte Carlo quantity
        ``PnL_quote / S_T``.
        """
        T = float(horizon_minutes)
        if T < 0.0:
            raise ValueError("horizon_minutes must be nonnegative")

        A, monomials, pnl_var = self._mc_second_moment_system(float(sigma))
        n = self.grid_meta_.nq
        n_monomials = len(monomials)
        zero_exp = (0,) * (pnl_var + 1)
        pnl_exp = [0] * (pnl_var + 1)
        pnl_exp[pnl_var] = 1
        pnl2_exp = [0] * (pnl_var + 1)
        pnl2_exp[pnl_var] = 2
        zero_m = monomials.index(zero_exp)
        pnl_m = monomials.index(tuple(pnl_exp))
        pnl2_m = monomials.index(tuple(pnl2_exp))

        y0 = np.zeros(A.shape[0], dtype=float)
        i0, w0, i1, w1 = _interp_weights_bounded(
            self.solve_q_grid_, float(initial_inventory)
        )
        y0[i0 * n_monomials + zero_m] += w0
        y0[i1 * n_monomials + zero_m] += w1

        if T == 0.0:
            mean_million_quote = 0.0
            second_moment_million_quote2 = 0.0
        else:
            yT = y0 @ self._expm_pade13(A * T)
            mean_million_quote = float(sum(
                yT[i * n_monomials + pnl_m] for i in range(n)
            ))
            second_moment_million_quote2 = float(sum(
                yT[i * n_monomials + pnl2_m] for i in range(n)
            ))

        if not math.isfinite(mean_million_quote) or not math.isfinite(second_moment_million_quote2):
            raise RuntimeError("closed-form Monte Carlo moments produced a non-finite value")
        variance_million_quote2 = second_moment_million_quote2 - mean_million_quote ** 2
        roundoff_scale = max(1.0, abs(second_moment_million_quote2), mean_million_quote ** 2)
        if variance_million_quote2 < -1e-10 * roundoff_scale:
            raise RuntimeError(
                "closed-form PnL variance became materially negative; moment system is numerically unstable"
            )
        variance_million_quote2 = max(0.0, variance_million_quote2)
        std_million_quote = math.sqrt(variance_million_quote2)

        mean_quote = 1_000_000.0 * mean_million_quote
        variance_quote = (1_000_000.0 ** 2) * variance_million_quote2
        std_quote = 1_000_000.0 * std_million_quote
        reference_spot = float(self.config.spot)
        if reference_spot <= 0.0:
            raise ValueError("reference spot must be positive for base-currency PnL conversion")
        mean_base = mean_quote / reference_spot
        std_base = std_quote / reference_spot
        variance_base = variance_quote / (reference_spot ** 2)

        return ClosedFormPnLStatistics(
            horizon_minutes=T,
            initial_inventory=float(initial_inventory),
            reference_spot=reference_spot,
            expected_pnl_quote_ccy=mean_quote,
            variance_pnl_quote_ccy=variance_quote,
            std_pnl_quote_ccy=std_quote,
            expected_pnl_base_ccy_at_reference_spot=mean_base,
            variance_pnl_base_ccy_at_reference_spot=variance_base,
            std_pnl_base_ccy_at_reference_spot=std_base,
        )

    def closed_form_expected_pnl(
        self, horizon_minutes: float, initial_inventory: float = 0.0
    ) -> ClosedFormPnLBenchmark:
        """Finite-horizon expected PnL consistent with ``simulate_pnl``.

        RFQ markout is treated exactly as in Monte Carlo: each fill creates an
        exponentially decaying expected drift in spot and that drift marks the
        desk's current inventory.  The augmented first-moment system therefore
        includes cross-inventory effects from overlapping impact kernels.

        Brownian spot noise has zero quote-currency mean, so it does not alter
        this expectation.  The inventory penalty is excluded from actual PnL.
        """
        T = float(horizon_minutes)
        if T < 0.0:
            raise ValueError("horizon_minutes must be nonnegative")

        A, y0 = self._mc_mean_augmented_system()
        n = self.grid_meta_.nq
        i0, w0, i1, w1 = _interp_weights_bounded(self.solve_q_grid_, float(initial_inventory))
        y0[i0] += w0
        y0[i1] += w1

        if T == 0.0:
            expected_million_quote = 0.0
        else:
            yT = y0 @ self._expm_pade13(A * T)
            expected_million_quote = float(yT[-1])
            if not math.isfinite(expected_million_quote):
                raise RuntimeError("closed-form Monte Carlo mean produced a non-finite value")

        expected_quote = 1_000_000.0 * expected_million_quote
        rate = expected_quote / T if T > 0.0 else 0.0
        reference_spot = float(self.config.spot)
        if reference_spot <= 0.0:
            raise ValueError("reference spot must be positive for base-currency PnL conversion")
        expected_base = expected_quote / reference_spot
        base_rate = expected_base / T if T > 0.0 else 0.0
        return ClosedFormPnLBenchmark(
            horizon_minutes=T,
            initial_inventory=float(initial_inventory),
            reference_spot=reference_spot,
            expected_pnl_quote_ccy=expected_quote,
            expected_pnl_rate_quote_ccy_per_min=rate,
            expected_pnl_base_ccy_at_reference_spot=expected_base,
            expected_pnl_rate_base_ccy_per_min_at_reference_spot=base_rate,
        )

    def _mc_events(self, q: float) -> list[dict]:
        """All fill events and rates available at the current inventory."""
        events: list[dict] = []
        q = float(q)
        for tier_idx, tier in enumerate(self.mdp_tiers):
            for side, direction in ((Side.Bid, +1.0), (Side.Ask, -1.0)):
                for z in tier.sizes_:
                    if not self._mdp_transition_admissible(q, z, side):
                        continue
                    d = tier.policy.delta(q, z, side)
                    rate = tier.flow_curve.arrival_rate(d, z)
                    if rate <= 0.0:
                        continue
                    events.append({
                        "kind": "mdp", "tier_idx": tier_idx, "side": side,
                        "direction": direction, "size": float(z), "delta": float(d),
                        "rate": float(rate),
                    })

        if self.dark_pool is not None:
            venue = self.dark_pool
            for side, direction, dist, fee in (
                (Side.Bid, +1.0, venue.dist_bid, venue.fee_per_unit_bid),
                (Side.Ask, -1.0, venue.dist_ask, venue.fee_per_unit_ask),
            ):
                u = int(round(venue.policy.posted_size(q, side)))
                if u <= 0 or not self._dark_pool_post_admissible(q, u, side):
                    continue
                rate=float(dist.lambda_arr)
                if rate > 0.0:
                    events.append({
                        "kind": "dark", "side": side, "direction": direction,
                        "size": float(u), "fee": float(fee), "rate": rate,
                        "dist": dist,
                    })
        return events

    def _mc_quote_snapshot(self, q: float, spot: float) -> list[tuple[int, str, str, float, float]]:
        """Return all displayed MDP quote prices at one simulated state.

        Each tuple is ``(tier_idx, tier_name, side, size, price)``. Quotes are
        evaluated from the solved policy at the current simulated inventory and
        shifted by the current simulated spot.
        """
        snapshot: list[tuple[int, str, str, float, float]] = []
        for tier_idx, tier in enumerate(self.mdp_tiers):
            for side in (Side.Bid, Side.Ask):
                for z in tier.sizes_:
                    d = tier.policy.delta(float(q), float(z), side)
                    price = QuoteMetrics.quote_price(float(spot), d, side, self.config.spread)
                    snapshot.append((tier_idx, tier.name, side.value, float(z), float(price)))
        return snapshot

    @staticmethod
    def _append_mc_snapshot(
        path: MonteCarloSamplePath,
        time_minutes: float,
        spot: float,
        inventory: float,
        quotes: list[tuple[int, str, str, float, float]],
    ) -> None:
        path.times.append(float(time_minutes))
        path.spots.append(float(spot))
        path.inventories.append(float(inventory))
        if not path.quote_series:
            path.quote_series = [
                MonteCarloQuoteSeries(
                    tier_idx=tier_idx, tier_name=tier_name, side=side, size=size, prices=[price]
                )
                for tier_idx, tier_name, side, size, price in quotes
            ]
        else:
            if len(path.quote_series) != len(quotes):
                raise RuntimeError("Monte Carlo quote snapshot shape changed within a path")
            for series, quote in zip(path.quote_series, quotes):
                tier_idx, tier_name, side, size, price = quote
                if (
                    series.tier_idx != tier_idx
                    or series.tier_name != tier_name
                    or series.side != side
                    or not math.isclose(series.size, size, rel_tol=0.0, abs_tol=1e-12)
                ):
                    raise RuntimeError("Monte Carlo quote snapshot ordering changed within a path")
                series.prices.append(float(price))

    @staticmethod
    def _advance_mc_spot(
        spot: float,
        dt: float,
        sigma: float,
        base_drift: float,
        impact_components: dict[float, float],
        rng: np.random.Generator,
    ) -> float:
        if dt <= 0.0:
            return float(spot)
        deterministic = float(base_drift) * dt
        for tau, remaining in list(impact_components.items()):
            decay = math.exp(-dt / tau)
            deterministic += remaining * (1.0 - decay)
            new_remaining = remaining * decay
            if abs(new_remaining) < 1e-15:
                impact_components.pop(tau, None)
            else:
                impact_components[tau] = new_remaining
        brownian = float(sigma) * math.sqrt(dt) * float(rng.normal()) if sigma > 0.0 else 0.0
        return float(spot) + deterministic + brownian

    def simulate_pnl(
        self,
        horizon_minutes: float,
        n_paths: int,
        sigma: float,
        initial_inventory: float = 0.0,
        initial_spot: Optional[float] = None,
        seed: int = 12345,
        sample_paths: int = 8,
        sample_points: int = 121,
    ) -> MonteCarloResult:
        """Simulate execution + mark-to-market PnL under the solved policy.

        RFQ and dark-pool fills are simulated in continuous event time. Spot is
        advanced exactly between event/snapshot times for Brownian increments
        plus the integrated exponential post-trade impact drift. For the first
        ``sample_paths`` paths, the result also records inventory, spot, all MDP
        quote prices and the complete fill tape for GUI path inspection.

        A fill of size z creates an adverse expected spot move

            +/- a*z**beta * (1-exp(-u/tau)),

        matching the saturating markout curve. The quadratic inventory penalty
        is never subtracted from realized PnL.
        """
        T = float(horizon_minutes)
        n_paths = int(n_paths)
        if T <= 0.0:
            raise ValueError("horizon_minutes must be positive")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if sigma < 0.0:
            raise ValueError("sigma must be nonnegative")
        q0 = float(initial_inventory)
        _locate_segment_with_weight_bounded(self.solve_q_grid_, q0)
        s0 = float(self.config.spot if initial_spot is None else initial_spot)
        if s0 <= 0.0:
            raise ValueError("initial_spot must be positive")
        rng = np.random.default_rng(int(seed))
        result = MonteCarloResult(horizon_minutes=T, initial_inventory=q0)
        n_sample = max(0, min(int(sample_paths), n_paths))
        sample_grid = np.linspace(0.0, T, max(2, int(sample_points)))
        # Inventory is cheap to retain on the regular display grid.  Keeping it
        # for every Monte Carlo path lets the GUI show an empirical pathwise
        # confidence band rather than a band based only on the few detailed
        # paths retained for inspection. float32 keeps even the 20k-path GUI
        # maximum comfortably small in memory.
        inventory_samples = np.empty((n_paths, len(sample_grid)), dtype=np.float32)

        for path_idx in range(n_paths):
            q = q0
            spot = s0
            cash = 0.0  # million quote currency
            t = 0.0
            impacts: dict[float, float] = {}
            trades = 0

            record = path_idx < n_sample
            path = MonteCarloSamplePath() if record else None
            sample_idx = 1
            inventory_sample_idx = 1
            inventory_samples[path_idx, 0] = float(q)
            if path is not None:
                path.inventory_event_times = [0.0]
                path.inventory_event_values = [float(q)]
                self._append_mc_snapshot(
                    path, 0.0, spot, q, self._mc_quote_snapshot(q, spot)
                )

            while t < T - 1e-14:
                events = self._mc_events(q)
                total_rate = sum(e["rate"] for e in events)
                wait = float(rng.exponential(1.0 / total_rate)) if total_rate > 0.0 else math.inf
                event_time = t + wait
                stop_time = min(event_time, T)

                # Inventory is piecewise constant between fills. Record its
                # pre-fill value at every regular session sample time for every
                # Monte Carlo path. If a sample time coincides with a fill, it
                # intentionally records the pre-fill state, matching the
                # detailed path snapshots below.
                while (
                    inventory_sample_idx < len(sample_grid)
                    and sample_grid[inventory_sample_idx] <= stop_time + 1e-14
                ):
                    inventory_samples[path_idx, inventory_sample_idx] = float(q)
                    inventory_sample_idx += 1

                if path is not None:
                    while sample_idx < len(sample_grid) and sample_grid[sample_idx] <= stop_time + 1e-14:
                        target = float(sample_grid[sample_idx])
                        spot = self._advance_mc_spot(
                            spot, target - t, sigma, self.config.spot_drift, impacts, rng
                        )
                        t = target
                        self._append_mc_snapshot(
                            path, t, spot, q, self._mc_quote_snapshot(q, spot)
                        )
                        sample_idx += 1
                if t < stop_time - 1e-14:
                    spot = self._advance_mc_spot(
                        spot, stop_time - t, sigma, self.config.spot_drift, impacts, rng
                    )
                    t = stop_time

                if event_time > T or not events:
                    break

                # Select the continuous-time fill event. If a regular snapshot
                # coincides with the event time, it intentionally records the
                # pre-fill state and quote before the execution.
                weights = np.asarray([e["rate"] for e in events], dtype=float)
                weights /= float(np.sum(weights))
                event = events[int(rng.choice(len(events), p=weights))]
                z = float(event["size"])
                if event["kind"] == "dark":
                    incoming=int(event["dist"].sample_incoming_size(rng))
                    z=float(min(incoming,int(round(z))))
                    if z <= 0.0:
                        continue
                direction = float(event["direction"])
                q_before = float(q)
                spot_before = float(spot)

                if event["kind"] == "mdp":
                    delta = float(event["delta"])
                    side = event["side"]
                    tier_idx = int(event["tier_idx"])
                    tier = self.mdp_tiers[tier_idx]
                    execution_price = QuoteMetrics.quote_price(
                        spot, delta, side, self.config.spread
                    )
                    if side is Side.Bid:
                        cash -= z * execution_price
                    else:
                        cash += z * execution_price
                    q += direction * z
                    if tier.use_markout:
                        tau = float(tier.markout_model.tau)
                        adverse_total_move = -direction * tier.markout_model.asymptotic_markout(z)
                        impacts[tau] = impacts.get(tau, 0.0) + adverse_total_move
                    if path is not None:
                        path.fills.append(MonteCarloFillEvent(
                            time_minutes=float(t),
                            kind="MDP",
                            tier_idx=tier_idx,
                            tier_name=tier.name,
                            side=side.value,
                            size=z,
                            execution_price=float(execution_price),
                            spot_before_fill=spot_before,
                            inventory_before=q_before,
                            inventory_after=float(q),
                            delta=delta,
                        ))
                else:
                    fee = float(event["fee"])
                    side = event["side"]
                    execution_price = float(spot)
                    if side is Side.Bid:
                        cash -= z * spot + fee * z
                    else:
                        cash += z * spot - fee * z
                    q += direction * z
                    if path is not None:
                        path.fills.append(MonteCarloFillEvent(
                            time_minutes=float(t),
                            kind="Dark pool",
                            tier_idx=None,
                            tier_name="Dark pool",
                            side=side.value,
                            size=z,
                            execution_price=execution_price,
                            spot_before_fill=spot_before,
                            inventory_before=q_before,
                            inventory_after=float(q),
                            delta=None,
                        ))
                trades += 1
                if path is not None:
                    path.inventory_event_times.append(float(t))
                    path.inventory_event_values.append(float(q))

            # Ensure a recorded path reaches T if floating-point comparisons
            # skipped the final regular snapshot.
            if path is not None and (not path.times or path.times[-1] < T - 1e-12):
                spot = self._advance_mc_spot(spot, T - t, sigma, self.config.spot_drift, impacts, rng)
                t = T
                self._append_mc_snapshot(
                    path, T, spot, q, self._mc_quote_snapshot(q, spot)
                )

            # Defensive fill in case numerical endpoint comparisons left a
            # final regular inventory sample unset.
            while inventory_sample_idx < len(sample_grid):
                inventory_samples[path_idx, inventory_sample_idx] = float(q)
                inventory_sample_idx += 1

            pnl_million_quote = cash + q * spot - q0 * s0
            pnl_quote = 1_000_000.0 * pnl_million_quote
            if spot <= 0.0:
                raise RuntimeError("Monte Carlo final spot must be positive for base-currency PnL conversion")
            pnl_base = pnl_quote / spot
            result.pnl_quote_ccy.append(float(pnl_quote))
            result.pnl_base_ccy.append(float(pnl_base))
            result.final_inventory.append(float(q))
            result.final_spot.append(float(spot))
            result.trade_count.append(int(trades))
            if path is not None:
                result.sample_paths.append(path)

        lower, median, upper = np.percentile(
            inventory_samples.astype(float), [2.5, 50.0, 97.5], axis=0
        )
        result.inventory_sample_times = sample_grid.astype(float).tolist()
        result.inventory_ci_lower = lower.astype(float).tolist()
        result.inventory_median = median.astype(float).tolist()
        result.inventory_ci_upper = upper.astype(float).tolist()
        return result

    def solve(self) -> HJBSolution:
        self.prepare_solve_context()
        h = np.zeros(self.grid_meta_.nq, dtype=float)
        self.initialize_stabilizing_policy()
        diag = SolverDiagnostics()
        rho = 0.0

        # Internal safety guard only, not a solver/meta parameter exposed to users.
        for it in range(1, 257):
            h_prev = h.copy()
            h, rho = self.evaluate_current_policy()
            policy_before = self._policy_vector()
            self.update_policies(h)
            policy_after = self._policy_vector()
            policy_change = (
                float(np.max(np.abs(policy_after - policy_before)))
                if policy_after.size else 0.0
            )
            rhs = self.bellman_rhs(h)
            residual = float(np.max(np.abs(rhs - rho)))
            max_h_change = float(np.max(np.abs(h - h_prev)))
            diag.record_iteration(it, max_h_change, residual)
            if (
                residual <= self._machine_convergence_scale(rho, rhs)
                and policy_change <= self._machine_policy_scale(policy_after)
            ):
                diag.converged = True
                break

        # Return the user-requested operational range, while retaining the
        # hidden solved grid for diagnostics. The operational points are a
        # literal subset of the padded grid by construction.
        solve_arr = np.asarray(self.solve_q_grid_, dtype=float)
        op_indices: list[int] = []
        for q in self.config.q_grid:
            hits = np.where(np.isclose(solve_arr, float(q), atol=1e-10, rtol=0.0))[0]
            if len(hits) != 1:
                raise RuntimeError(f"operational inventory state {q} missing from solve grid")
            op_indices.append(int(hits[0]))

        out_tiers = copy.deepcopy(self.mdp_tiers)
        for tier in out_tiers:
            tier.policy.q_grid = list(map(float, self.config.q_grid))
            tier.policy.bid = [tier.policy.bid[i] for i in op_indices]
            tier.policy.ask = [tier.policy.ask[i] for i in op_indices]

            # Snap the returned operational surface exactly onto the structural
            # trader constraints. Howard has already converged; this only
            # removes O(sqrt(machine-eps)) lag from the Jacobi propagation.
            q0_out = len(tier.policy.q_grid) // 2
            for i in range(q0_out + 1, len(tier.policy.q_grid)):
                inner_bid = tier.policy.bid[i - 1]
                inner_ask = tier.policy.ask[i - 1]
                tier.policy.bid[i] = self._project_ladder_to_bounds(
                    tier, tier.policy.bid[i],
                    min_gaps=self._ladder_gaps(inner_bid),
                    rung_upper_bounds=inner_bid,
                )
                tier.policy.ask[i] = self._project_ladder_to_bounds(
                    tier, tier.policy.ask[i],
                    max_gaps=self._ladder_gaps(inner_ask),
                    rung_lower_bounds=inner_ask,
                )
            for i in range(q0_out - 1, -1, -1):
                inner_bid = tier.policy.bid[i + 1]
                inner_ask = tier.policy.ask[i + 1]
                tier.policy.bid[i] = self._project_ladder_to_bounds(
                    tier, tier.policy.bid[i],
                    max_gaps=self._ladder_gaps(inner_bid),
                    rung_lower_bounds=inner_bid,
                )
                tier.policy.ask[i] = self._project_ladder_to_bounds(
                    tier, tier.policy.ask[i],
                    min_gaps=self._ladder_gaps(inner_ask),
                    rung_upper_bounds=inner_ask,
                )

        out_dark = copy.deepcopy(self.dark_pool)
        if out_dark is not None:
            p = out_dark.policy
            p.q_grid = list(map(float, self.config.q_grid))
            p.bid_size = [p.bid_size[i] for i in op_indices]
            p.ask_size = [p.ask_size[i] for i in op_indices]
            p.bid_active = [p.bid_active[i] for i in op_indices]
            p.ask_active = [p.ask_active[i] for i in op_indices]

        return HJBSolution(
            h=[float(h[i]) for i in op_indices],
            q_grid=list(map(float, self.config.q_grid)),
            mdp_tiers=out_tiers, dark_pool=out_dark,
            diagnostics=diag, average_reward=float(rho),
            solve_q_grid=list(map(float, self.solve_q_grid_)),
            h_solve=h.tolist(),
            hard_inventory_limit=float(self.hard_inventory_limit_),
        )


# Aliases matching the old extension namespace.
Bid = Side.Bid
Ask = Side.Ask

__all__ = [
    "Side", "Bid", "Ask", "LogisticFlowCurve", "SaturatingMarkoutModel",
    "CarryCost", "PolynomialInternalizationTime", "QuadraticInventoryPenalty", "QuoteSummary",
    "QuoteMetrics", "QuotePolicy", "DarkPoolPolicy", "MDPTier", "Tier", "ArrivalDistribution",
    "ZeroInflatedPoissonArrivalDist", "DarkPoolVenue", "SolverConfig",
    "SolverGridMeta", "build_solver_grid_meta", "SolverDiagnostics", "HJBSolution", "HJBLadderSolver",
    "PNL_BENCHMARK_VERSION",
]
