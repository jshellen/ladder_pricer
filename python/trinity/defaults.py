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



# Customer RFQ request-size distributions are calibrated independently of
# pricing knots. totalRfqRate is the pooled bid+ask request rate [RFQ/min];
# rfqSizes are source-base-currency notionals in millions and probabilities
# sum to one. These defaults are the exact discrete 1M-grid equivalents of
# the pre-refactor A0/theta/beta calibrations, so Step 5 changes configuration
# representation rather than baseline economics.
DEFAULT_RFQ_SIZES_1_TO_20 = [float(x) for x in range(1, 21)]
TIER1_RFQ_PROBABILITIES = [
    0.21887176643980527, 0.1758915120548416, 0.14086931659796414, 0.11145689474702523,
    0.08710225837745503, 0.06729930665560926, 0.05146610610223016, 0.0389929676799421,
    0.029293520883484213, 0.021836837534594877, 0.01616249159390996, 0.0118838633720731,
    0.008684355862944412, 0.006309905987122827, 0.004560025677193113, 0.003278741254971594,
    0.002346194782902506, 0.0016712719948773097, 0.0011853721953480028, 0.000837290205705274,
]
TIER2_RFQ_PROBABILITIES = [
    0.3136701863886962, 0.21468902039849305, 0.15041083512168163, 0.104771651299432,
    0.07216284188024215, 0.049100588546606073, 0.03301374631116151, 0.021949323966710118,
    0.01444040874095255, 0.009407478663408483, 0.00607270626461731, 0.003886511706966847,
    0.002467369826774755, 0.0015545681378698554, 0.0009724574021369573, 0.0006042027270333202,
    0.0003729891132545595, 0.00022884806873664417, 0.00013959184708587386, 8.467358814014033e-05,
]

ECN_TRADE_SIZES = [1.0, 0.75, 0.50, 0.25, 0.10]

# Generic fallback retained for ECN source pairs without a pair-specific
# empirical size distribution.  This is the nearest-pillar discretization of
# the former Exp(mean=1M) parent-size model.
ECN_TRADE_SIZE_PROBABILITIES_FALLBACK = [
    0.4168620196785084,   # 1.00M
    0.11839940884048189,  # 0.75M
    0.15202785027198196,  # 0.50M
    0.15216774197823513,  # 0.25M
    0.16054297923079264,  # 0.10M residual
]

# Empirical source-pair defaults.  The 100k probability is the residual
# 1 - [P(1M) + P(750k) + P(500k) + P(250k)].
ECN_TRADE_SIZE_PROBABILITIES_BY_PAIR: dict[str, list[float]] = {
    "EURSEK": [0.35, 0.08, 0.22, 0.16, 0.19],
    "USDSEK": [0.34, 0.135, 0.216, 0.13, 0.179],
}


def default_ecn_trade_size_probabilities(pair: str | None) -> list[float]:
    """Return pair-specific ECN parent-size probabilities when available."""
    key = str(pair or "").upper().strip()
    probs = ECN_TRADE_SIZE_PROBABILITIES_BY_PAIR.get(
        key, ECN_TRADE_SIZE_PROBABILITIES_FALLBACK
    )
    return list(probs)


# Backwards-compatible module-level default.  The dashboard's default target
# pair is EURSEK, so this now intentionally resolves to the EURSEK empirical
# size distribution.
ECN_TRADE_SIZE_PROBABILITIES = default_ecn_trade_size_probabilities("EURSEK")

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
    # Legacy scalar volatility is retained only for old configs.  The active
    # model uses the calibrated intraday CTMC below.
    "sigmaPips": 10.171908058405807,
    "volatilityModel": {
        "enabled": True,
        "initialState": 2,
        "sigmaStatesPips": [
            5,
            10,
            20,
            30,
            35 ,
        ],
        "generatorPerMinute": [
            [-0.04620107444359171, 0.031005372217958557, 0.011511895625479662, 0.0029163468917881813, 0.0007674597083653108],
            [0.031538461538461536, -0.07830769230769231, 0.03292307692307692, 0.010461538461538461, 0.003384615384615385],
            [0.010363495746326373, 0.033255993812838364, -0.0866202629543697, 0.03758700696055684, 0.005413766434648105],
            [0.002007722007722008, 0.009575289575289575, 0.036756756756756756, -0.07521235521235521, 0.02687258687258687],
            [0.0009309542280837859, 0.0021722265321955005, 0.006051202482544608, 0.025446082234290148, -0.03460046547711404],
        ],
    },
    "gamma": 0.75,
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
            "flow": {
                "totalRfqRate": 0.1416354448280368,
                "rfqSizes": list(DEFAULT_RFQ_SIZES_1_TO_20),
                "rfqProbabilities": list(TIER1_RFQ_PROBABILITIES),
                "steepness": 8.42, "shift": 0.52, "volumeShift": 0.026,
            },
            "markout": {"asymptoticEurPerEurM": 8.695652173913043, "tauMinutes": 0.5},
            "useMarkout": True,
            "feeEurPerEurM": 3.0,
            "deltaMin": -10.0,
            "deltaMax": 100.0,
        },
        {
            "enabled": True,
            "name": "Tier 2",
            "sizes": [1, 2, 3, 5, 10, 20],
            "flow": {
                "totalRfqRate": 0.1479260765398395,
                "rfqSizes": list(DEFAULT_RFQ_SIZES_1_TO_20),
                "rfqProbabilities": list(TIER2_RFQ_PROBABILITIES),
                "steepness": 2.86, "shift": 0.48, "volumeShift": 0.020,
            },
            "markout": {"asymptoticEurPerEurM": 8.695652173913043, "tauMinutes": 0.5},
            "useMarkout": True,
            "feeEurPerEurM": 3.0,
            "deltaMin": -10.0,
            "deltaMax": 100.0,
        },
        {
            "enabled": False,
            "name": "Tier 3",
            "sizes": [1],
            "flow": {
                "totalRfqRate": 2.0,
                "rfqSizes": [1.0],
                "rfqProbabilities": [1.0],
                "steepness": 20.0, "shift": 0.52, "volumeShift": 0.026,
            },
            "markout": {"asymptoticEurPerEurM": 34.78260869565217, "tauMinutes": 0.10},
            "useMarkout": True,
            "feeEurPerEurM": 3.0,
            "deltaMin": -10.0,
            "deltaMax": 100.0,
        },
    ],
    "darkPool": {
        "enabled": False,
        # Symmetric zero-inflated Poisson fill-size model. lambda is the raw
        # arrival intensity per side, mu controls positive fill size, and p0 is
        # the zero-fill mass. Dark-pool orders are hedge-only and may not cross
        # through flat. feeEurPerEurM is applied symmetrically to buy and sell fills.
        "lambda": 0.0012,
        "mu": 2.9,
        "p0": 0.7172,
        "feeEurPerEurM": 0.0,
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
        # Discrete parent ECN trade-size probabilities for fixed source-base
        # currency pillars [1M, 750k, 500k, 250k, 100k].  The UI edits the
        # first four probabilities and defines the 100k probability as the
        # residual 1 - sum(other probabilities).
        "tradeSizeProbabilities": list(ECN_TRADE_SIZE_PROBABILITIES),
        "quoteSize": 1.0,
        "makerFeeEurPerEurM": 3.0,  # EUR per EURm traded
    },
}


def default_config() -> dict:
    return deepcopy(_DEFAULT_CONFIG)
