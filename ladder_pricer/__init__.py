"""Pure-Python HJB ladder pricer.

Howard policy iteration for a continuous-time, average-reward inventory-control
problem. RFQ quote improvement uses the analytical logistic/Lambert-W solution.
Within each inventory row, larger sizes cannot be quoted tighter. As absolute
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


# ---------------------------------------------------------------------------
# Economic model components
# ---------------------------------------------------------------------------

@dataclass
class LogisticFlowCurve:
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

    def arrival_rate(self, delta: float, z: float) -> float:
        return self.A(z) * self.hit_ratio(delta, z)

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
        i, t = _locate_segment_with_weight(self.q_grid, q)
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
        i, t = _locate_segment_with_weight(self.q_grid, q)
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
class GeometricArrivalDist(ArrivalDistribution):
    lambda_: float = 1.0
    p: float = 0.5

    def __init__(self, lambda_: float = 1.0, p: float = 0.5, **kwargs):
        if "lambda" in kwargs:
            lambda_ = kwargs["lambda"]
        self.lambda_ = float(lambda_)
        self.p = float(p)
        self.validate()

    # Compatibility with pybind field named "lambda" via getattr/setattr.
    def __getattr__(self, name: str):
        if name == "lambda":
            return self.lambda_
        raise AttributeError(name)
    def __setattr__(self, name: str, value):
        if name == "lambda":
            name = "lambda_"
        object.__setattr__(self, name, value)

    def validate(self) -> None:
        if self.lambda_ < 0.0:
            raise ValueError("lambda must be nonnegative")
        if not (0.0 < self.p <= 1.0):
            raise ValueError("p must lie in (0,1]")

    def name(self) -> str:
        return "geometric"

    def fill_rates(self, u: int) -> list[tuple[int, float]]:
        r = 1.0 - self.p
        out = [(k, self.lambda_ * self.p * r ** (k - 1)) for k in range(1, u)]
        out.append((u, self.lambda_ * r ** (u - 1)))
        return out


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
        # Preserve the original implementation: truncate to 1..u and renormalize.
        pmf = []
        p = math.exp(-self.mu)
        for k in range(0, u + 1):
            if k == 0:
                pk = p
            else:
                p *= self.mu / k
                pk = p
            pmf.append(pk)
        Z = sum(pmf[1:])
        if Z <= 1e-15:
            return []
        scale = self.lambda_arr * (1.0 - self.p0) / Z
        return [(k, scale * pmf[k]) for k in range(1, u + 1)]


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
                lb, la, pb, pa, fee_b, fee_a, sizes, *rest = args
                self.allow_both_sides = bool(rest[0]) if len(rest) > 0 else self.allow_both_sides
                self.min_fill_value = float(rest[1]) if len(rest) > 1 else self.min_fill_value
            else:
                lb = kwargs.pop("lambda_bid", 1.0); la = kwargs.pop("lambda_ask", 1.0)
                pb = kwargs.pop("p_bid", 0.5); pa = kwargs.pop("p_ask", 0.5)
                fee_b = kwargs.pop("fee_per_unit_bid", 0.0); fee_a = kwargs.pop("fee_per_unit_ask", 0.0)
                sizes = kwargs.pop("posted_sizes", [1.0, 2.0, 3.0])
            self.dist_bid = GeometricArrivalDist(lb, pb)
            self.dist_ask = GeometricArrivalDist(la, pa)

        if kwargs:
            raise TypeError(f"unexpected arguments: {', '.join(kwargs)}")
        self.fee_per_unit_bid = float(fee_b)
        self.fee_per_unit_ask = float(fee_a)
        self.posted_sizes = list(map(float, sizes))
        self.validate()

    @property
    def lambda_bid(self) -> float:
        return self.dist_bid.lambda_ if isinstance(self.dist_bid, GeometricArrivalDist) else self.dist_bid.lambda_arr
    @property
    def lambda_ask(self) -> float:
        return self.dist_ask.lambda_ if isinstance(self.dist_ask, GeometricArrivalDist) else self.dist_ask.lambda_arr
    @property
    def p_bid(self) -> float:
        if not isinstance(self.dist_bid, GeometricArrivalDist): raise RuntimeError("p_bid only defined for geometric")
        return self.dist_bid.p
    @property
    def p_ask(self) -> float:
        if not isinstance(self.dist_ask, GeometricArrivalDist): raise RuntimeError("p_ask only defined for geometric")
        return self.dist_ask.p

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
    h: list[float] = field(default_factory=list)
    q_grid: list[float] = field(default_factory=list)
    mdp_tiers: list[MDPTier] = field(default_factory=list)
    dark_pool: Optional[DarkPoolVenue] = None
    diagnostics: SolverDiagnostics = field(default_factory=SolverDiagnostics)
    average_reward: float = 0.0


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
        self.config.validate()

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
        for tier in self.mdp_tiers: tier.reset_policy_shape(self.config.q_grid)
        if self.dark_pool is not None: self.dark_pool.reset_policy_shape(self.config.q_grid)

    def prepare_solve_context(self) -> None:
        self.validate_problem_definition()
        self.grid_meta_ = build_solver_grid_meta(self.config.q_grid)
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
        hq = _interp_linear(self.config.q_grid, h, q)
        for j, z in enumerate(tier.sizes_):
            lo, hi = self.mdp_bounds_for_rung(
                tier, j, row, q, side,
                min_gaps=min_gaps, max_gaps=max_gaps,
                rung_lower_bounds=rung_lower_bounds,
                rung_upper_bounds=rung_upper_bounds,
            )
            q_next = q + direction * z
            tau = self.internalization_time.value(q_next)
            mu = tier.markout_model.expected_markout(z, tau) if tier.use_markout else 0.0
            additive = -z * mu + _interp_linear(self.config.q_grid, h, q_next) - hq
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
        hq = _interp_linear(self.config.q_grid, h, q)
        total = 0.0
        for d, z in zip(row, tier.sizes_):
            q_next = q + direction * z
            tau = self.internalization_time.value(q_next)
            mu = tier.markout_model.expected_markout(z, tau) if tier.use_markout else 0.0
            dh = _interp_linear(self.config.q_grid, h, q_next) - hq
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
        q_grid = self.config.q_grid
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

        # Positive inventory, moving q upward away from zero.
        for i in range(q0 + 1, len(q_grid)):
            inner_i = i - 1
            if side is Side.Bid:
                # Long book: bids add inventory -> lower bid deltas + wider gaps.
                improve_state(i, inner_i, inventory_increasing=True, delta_direction="down")
            else:
                # Long book: asks reduce inventory -> higher ask deltas + flatter gaps.
                improve_state(i, inner_i, inventory_increasing=False, delta_direction="up")

        # Negative inventory, moving q downward away from zero.
        for i in range(q0 - 1, -1, -1):
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
            lambda x: _interp_linear(self.config.q_grid, h, x), q, hq, u, fee, direction
        )

    def build_dark_pool_policy(self, venue: DarkPoolVenue, h_vec: Sequence[float]) -> None:
        h = np.asarray(h_vec, dtype=float)
        for side, direction, dist, fee in [
            (Side.Bid, +1.0, venue.dist_bid, venue.fee_per_unit_bid),
            (Side.Ask, -1.0, venue.dist_ask, venue.fee_per_unit_ask),
        ]:
            for i, q in enumerate(self.config.q_grid):
                if not venue.is_admissible(q, side):
                    venue.policy.set_posted_size(i, side, 0.0, False)
                    continue
                hq = _interp_linear(self.config.q_grid, h, q)
                best_size, best_value = 0.0, -math.inf
                for u_raw in venue.posted_sizes:
                    u = int(round(u_raw))
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
            for i, q in enumerate(self.config.q_grid):
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
            for i, q in enumerate(self.config.q_grid):
                venue.policy.set_posted_size(i, Side.Bid, 0.0, False)
                venue.policy.set_posted_size(i, Side.Ask, 0.0, False)
                if not venue.posted_sizes or q == 0.0: continue
                u = venue.posted_sizes[-1]
                if q < 0 and venue.is_admissible(q, Side.Bid): venue.policy.set_posted_size(i, Side.Bid, u, True)
                if q > 0 and venue.is_admissible(q, Side.Ask): venue.policy.set_posted_size(i, Side.Ask, u, True)

    def _add_transition(self, L: np.ndarray, row: int, target_q: float, rate: float) -> None:
        if rate == 0.0:
            return
        i0, w0, i1, w1 = _interp_weights(self.config.q_grid, target_q)
        L[row, i0] += rate * w0
        L[row, i1] += rate * w1
        L[row, row] -= rate

    def fixed_policy_affine_operator(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.grid_meta_.nq
        c = np.zeros(n, dtype=float)
        L = np.zeros((n, n), dtype=float)

        for i, q in enumerate(self.config.q_grid):
            c[i] = -self.penalty.value(q) + self.config.spot_drift * q

            for tier in self.mdp_tiers:
                for side, direction, row in [
                    (Side.Bid, +1.0, tier.policy.bid[i]),
                    (Side.Ask, -1.0, tier.policy.ask[i]),
                ]:
                    for j, z in enumerate(tier.sizes_):
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
                    if u <= 0:
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
        hq = _interp_linear(self.config.q_grid, h, q)
        total = 0.0
        for j, z in enumerate(tier.sizes_):
            d = row[j]
            q_next = q + direction * z
            mu = tier.markout_model.expected_markout(z, self.internalization_time.value(q_next)) if tier.use_markout else 0.0
            total += self.mdp_fill_payoff(tier.flow_curve.arrival_rate(d, z), z, d, mu,
                                          _interp_linear(self.config.q_grid, h, q_next) - hq)
        return total

    def dark_pool_bellman_contribution(self, venue: DarkPoolVenue, q: float, q_index: int,
                                       h_vec: Sequence[float]) -> float:
        h = np.asarray(h_vec, dtype=float)
        hq = _interp_linear(self.config.q_grid, h, q)
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
        for i, q in enumerate(self.config.q_grid):
            val = -self.penalty.value(q) + self.config.spot_drift * q
            for tier in self.mdp_tiers:
                val += self.mdp_bellman_contribution_side(tier, q, i, h, Side.Bid)
                val += self.mdp_bellman_contribution_side(tier, q, i, h, Side.Ask)
            if self.dark_pool is not None:
                val += self.dark_pool_bellman_contribution(self.dark_pool, q, i, h)
            rhs[i] = val
        return rhs

    def bellman_rhs_from_policies(self, h_vec: Sequence[float]) -> list[float]:
        if len(h_vec) != len(self.config.q_grid):
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

        return HJBSolution(
            h=h.tolist(), q_grid=list(map(float, self.config.q_grid)),
            mdp_tiers=copy.deepcopy(self.mdp_tiers), dark_pool=copy.deepcopy(self.dark_pool),
            diagnostics=diag, average_reward=float(rho),
        )


# Aliases matching the old extension namespace.
Bid = Side.Bid
Ask = Side.Ask

__all__ = [
    "Side", "Bid", "Ask", "LogisticFlowCurve", "SaturatingMarkoutModel",
    "CarryCost", "PolynomialInternalizationTime", "QuadraticInventoryPenalty", "QuoteSummary",
    "QuoteMetrics", "QuotePolicy", "DarkPoolPolicy", "MDPTier", "Tier", "ArrivalDistribution",
    "GeometricArrivalDist", "ZeroInflatedPoissonArrivalDist", "DarkPoolVenue", "SolverConfig",
    "SolverGridMeta", "build_solver_grid_meta", "SolverDiagnostics", "HJBSolution", "HJBLadderSolver",
]
