from __future__ import annotations

from copy import deepcopy



# Temporary reference FX mids used by the dashboard when configuring crossed
# customer-flow sources.  In production this should be replaced by a market-
# data/database lookup.  Values are deliberately centralized here so the UI
# and model configuration never hard-code reference rates.
DEFAULT_FX_MIDS: dict[str, float] = {
    "EURSEK": 11.50,
    "USDSEK": 9.7458,
    "GBPSEK": 13.0682,
    "NOKSEK": 0.9914,
    "DKKSEK": 1.5416,
    "EURNOK": 11.60,
    "USDNOK": 9.8305,
    "GBPNOK": 13.1818,
    "SEKNOK": 1.0087,
    "DKKNOK": 1.5549,
    "EURDKK": 7.46,
    "USDDKK": 6.3220,
    "GBPDKK": 8.4773,
    "SEKDKK": 0.6487,
    "NOKDKK": 0.6431,
    "EURUSD": 1.18,
    "GBPUSD": 1.3409,
    "AUDUSD": 0.6600,
    "NZDUSD": 0.5750,
    "USDCAD": 1.3600,
    "USDCHF": 0.8000,
    "USDJPY": 147.00,
    "EURGBP": 0.8800,
    "EURCHF": 0.9440,
    "EURJPY": 173.46,
    "GBPCHF": 1.0727,
    "GBPJPY": 197.11,
    "AUDJPY": 97.02,
    "NZDJPY": 84.53,
}


def default_fx_mid(pair: str | None) -> float | None:
    """Return a temporary reference mid for *pair*, including inverse pairs.

    The dashboard may derive a cross in an orientation that is not normally
    quoted in the market (for example ``USDEUR``).  Keeping reciprocal handling
    here lets callers ask for the economically required orientation directly.
    """
    key = str(pair or "").upper().strip()
    if len(key) != 6 or not key.isalpha():
        return None
    if key in DEFAULT_FX_MIDS:
        return float(DEFAULT_FX_MIDS[key])
    inverse = key[3:] + key[:3]
    value = DEFAULT_FX_MIDS.get(inverse)
    if value is None or value <= 0.0:
        return None
    return 1.0 / float(value)

ECN_DELTA_STEP = 0.01

def ecn_delta_grid(min_delta: float, max_delta: float) -> list[float]:
    """Return the passive-ECN price-improvement grid.

    ECN delta uses the same convention as customer tiers: d=0 is the
    same-side touch and d=0.5 is mid.  One grid step (0.01) is one
    percentage point of the displayed spread.  Negative deltas are allowed
    for quotes outside the touch; passive quotes may not cross through mid.
    """
    lo = float(min_delta)
    hi = float(max_delta)
    if hi < lo:
        raise ValueError("ECN maximum delta must be >= minimum delta")
    if hi > 0.5 + 1e-12:
        raise ValueError("ECN maximum delta must be <= 0.5 (mid)")
    for value in (lo, hi):
        snapped = round(value / ECN_DELTA_STEP) * ECN_DELTA_STEP
        if abs(snapped - value) > 1e-9:
            raise ValueError("ECN min/max deltas must lie on the 0.01 grid")
    n = int(round((hi - lo) / ECN_DELTA_STEP))
    return [round(lo + ECN_DELTA_STEP * i, 10) for i in range(n + 1)]

ECN_DELTA_GRID = ecn_delta_grid(0.0, 0.5)

_DEFAULT_CONFIG = {
    "targetPair": "EURSEK",
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
        # Hedge-only ECN quoting using the same normalized delta convention as
        # customer tiers: d=0 is the same-side touch and d=0.5 is mid.  The
        # optimizer chooses NONE or one point on a 0.01 (one percentage-point)
        # grid.  Negative d is allowed if the user wants to quote outside touch.
        "minDelta": 0.0,
        "maxDelta": 0.5,
        "deltas": ECN_DELTA_GRID,
        # A is the fill/market-trade intensity at mid. Moving away from mid by
        # x=0.5-d spread fractions reduces intensity exponentially:
        # lambda(d) = A * exp(-k * (0.5 - d)).
        "flow": {"A": 4.0, "k": 8.4},
        "quoteSize": 1.0,
        "makerFeePips": 3.0,
    },
}


def default_config() -> dict:
    return deepcopy(_DEFAULT_CONFIG)
