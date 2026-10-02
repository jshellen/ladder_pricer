from __future__ import annotations

from copy import deepcopy


ECN_DISTANCE_STEP_PIPS = 0.01

def ecn_distance_grid(min_distance_pips: float, max_distance_pips: float) -> list[float]:
    lo = float(min_distance_pips)
    hi = float(max_distance_pips)
    if lo < 0.0:
        raise ValueError("ECN minimum distance from mid must be nonnegative")
    if hi < lo:
        raise ValueError("ECN maximum distance from mid must be >= minimum distance")
    n = int(round((hi - lo) / ECN_DISTANCE_STEP_PIPS))
    if abs(lo + n * ECN_DISTANCE_STEP_PIPS - hi) > 1e-9:
        raise ValueError("ECN min/max distances must lie on the 0.5-pip grid")
    return [round(lo + ECN_DISTANCE_STEP_PIPS * i, 10) for i in range(n + 1)]

ECN_DISTANCE_GRID = ecn_distance_grid(0.0, 20.0)

_DEFAULT_CONFIG = {
    "spot": 11.5,
    "spreadPips": 20.0,
    "spotDrift": 0.0,
    "sigmaPips": 20.0,
    "gamma": 0.1,
    "grid": {
        "mode": "uniform",
        "maxAbs": 20.0,
        "step": 1.0,
    },
    "internalization": {"tau0": 4.0, "tau1": 0.070, "tau2": 0.0084},
    "tiers": [
        {
            "enabled": True,
            "name": "Tier 1",
            "sizes": [1, 2, 3, 5, 10, 20],
            "flow": {"A0": 0.0155, "theta": 0.144, "beta": 0.0857, "steepness": 8.42, "shift": 0.52, "volumeShift": 0.026},
            "markout": {"impactScalePips": 1.0, "sizeExponent": 0.5, "tauMinutes": 0.5},
            "useMarkout": True,
            "feePips": 0.0,
            "deltaMin": -10.0,
            "deltaMax": 100.0,
        },
        {
            "enabled": True,
            "name": "Tier 2",
            "sizes": [1, 2, 3, 5, 10, 20],
            "flow": {"A0": 0.0232, "theta": 0.303, "beta": 0.122, "steepness": 2.86, "shift": 0.48, "volumeShift": 0.020},
            "markout": {"impactScalePips": 1.0, "sizeExponent": 0.5, "tauMinutes": 0.5},
            "useMarkout": True,
            "feePips": 0.0,
            "deltaMin": -10.0,
            "deltaMax": 100.0,
        },
        {
            "enabled": False,
            "name": "Tier 3",
            "sizes": [1],
            "flow": {"A0": 1.0, "theta": 0.144, "beta": 0.0857, "steepness": 20.0, "shift": 0.52, "volumeShift": 0.026},
            "markout": {"impactScalePips": 4.0, "sizeExponent": 0.5, "tauMinutes": 0.10},
            "useMarkout": True,
            "feePips": 0.0,
            "deltaMin": -10.0,
            "deltaMax": 100.0,
        },
    ],
    "darkPool": {
        "enabled": False,
        # Symmetric zero-inflated Poisson fill-size model. lambda is the raw
        # arrival intensity per side, mu controls positive fill size, and p0 is
        # the zero-fill mass. Dark-pool orders are hedge-only and may not cross
        # through flat. feePips is applied symmetrically to buy and sell fills.
        "lambda": 0.0012,
        "mu": 2.9,
        "p0": 0.7172,
        "feePips": 0.0,
        "postedSizes": [1, 2, 3, 4, 5],
    },
    "passiveEcn": {
        "enabled": True,
        # Hedge-only ECN quoting. Delta is the absolute distance from the moving
        # reference/mid, measured directly in pips. The optimizer chooses NONE or
        # one point on the 0.5-pip grid between minDistancePips and maxDistancePips.
        "minDistancePips": 0.0,
        "maxDistancePips": 20.0,
        "deltas": ECN_DISTANCE_GRID,
        # Empirical-style exponential arrival curve in pip distance from mid:
        # ECN market trades arrive independently on each side at rate A. Their
        # distance D from mid is Exp(k), so an active quote at depth delta has
        # fill intensity A * P(D >= delta) = A * exp(-k * delta).
        "flow": {"A": 4.0, "k": 8.4},
        "quoteSize": 1.0,
        "makerFeePips": 3.0,
    },
}


def default_config() -> dict:
    return deepcopy(_DEFAULT_CONFIG)
