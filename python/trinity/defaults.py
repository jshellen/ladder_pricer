from __future__ import annotations

from copy import deepcopy



# Temporary reference FX mids used by the dashboard when configuring crossed
# customer-flow sources.  In production this should be replaced by a market-
# data/database lookup.  Values are deliberately centralized here so the UI
# and model configuration never hard-code reference rates.
# Reference FX pair market data is maintained in fx_pair_defaults.json.
# These are illustrative offline defaults, not current executable FX quotes.
# Changing the target pair reloads its spot and spread from that file.
from pathlib import Path
import json
import math


def _load_pair_market_defaults() -> dict[str, dict[str, float]]:
    path = Path(__file__).with_name("fx_pair_defaults.json")
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    pairs = raw["pairs"]
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError("FX pair market defaults JSON must contain a nonempty 'pairs' object")
    result: dict[str, dict[str, float]] = {}
    for pair, values in pairs.items():
        if len(pair) != 6 or not pair.isalpha() or pair.upper() != pair:
            raise ValueError(f"Invalid FX pair key in market defaults: {pair!r}")
        spot = float(values["spot"])
        spread = float(values["spreadPips"])
        if not math.isfinite(spot) or spot <= 0 or not math.isfinite(spread) or spread <= 0:
            raise ValueError(f"Invalid spot or spread in FX pair defaults for {pair}")
        result[pair] = {"spot": spot, "spreadPips": spread}
    return result


def _validated_volatility_regimes(model: dict, pair: str) -> dict:
    """Parse 2–12-state Markov volatility parameters from JSON; return a fresh copy."""
    try:
        sigma = [float(v) for v in model["sigmaPips"]]
        rates = [[float(v) for v in row] for row in model["ratesPerMinute"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid volatility regimes for {pair}: expected sigmaPips and ratesPerMinute") from exc
    n = len(sigma)
    if not 2 <= n <= 12 or len(rates) != n or any(len(row) != n for row in rates):
        raise ValueError(f"Volatility matrix for {pair} must be square with 2–12 states")
    if any(not math.isfinite(v) or v <= 0 for v in sigma) or any(a >= b for a,b in zip(sigma,sigma[1:])):
        raise ValueError(f"Volatility sigmas for {pair} must increase strictly and be positive")
    if any(not math.isfinite(v) or v < 0 for row in rates for v in row):
        raise ValueError(f"Volatility matrix for {pair} must contain finite, nonnegative rates")
    if any(abs(rates[i][i]) > 1e-12 for i in range(n)):
        raise ValueError(f"Volatility matrix diagonal for {pair} must be zero (rates, not generator)")
    if any(sum(rates[i][j] for j in range(n) if j != i) <= 0 for i in range(n)):
        raise ValueError(f"Every volatility regime for {pair} needs a positive exit rate")
    return {"sigmaPips": sigma, "ratesPerMinute": rates}


def _load_pair_volatility_defaults(json_path: Path | None = None) -> dict[str, dict]:
    """An explicit per-pair override or the common JSON default; never UI inputs."""
    source = json_path or Path(__file__).with_name("fx_pair_defaults.json")
    with source.open(encoding="utf-8") as file:
        document = json.load(file)
    fallback = document["volatilityRegimes"]
    return {
        pair: _validated_volatility_regimes(values.get("volatilityRegimes", fallback), pair)
        for pair, values in document["pairs"].items()
    }


DEFAULT_FX_PAIR_VOLATILITY = _load_pair_volatility_defaults()

# The calibration notebook writes pair-specific snapshots here.  These are
# intentionally not cached: a new calibration becomes effective on the next
# Solve without a dashboard restart.  The packaged JSON remains the fallback.
CALIBRATION_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "output"


def calibration_artifact_path(pair: str) -> Path:
    key = str(pair or "").upper().strip()
    if key not in DEFAULT_FX_PAIR_VOLATILITY:
        raise ValueError(f"Unknown volatility target pair {key!r}")
    return CALIBRATION_OUTPUT_DIR / f"{key}_volatility_regimes.json"


def _volatility_from_calibration_artifact(document: dict, pair: str) -> dict:
    """Translate an offline calibration's full Q into native zero-diagonal rates.

    No attempt is made to infer sigma levels from Q: they must be calibrated
    alongside Q and supplied in the same artifact.  Reject inconsistent or
    truncated models instead of using plausible-looking fallback parameters.
    """
    if not isinstance(document, dict):
        raise ValueError(f"Calibration for {pair} must be a JSON object")
    if "pair" in document and document["pair"] != pair:
        raise ValueError(f"Calibration for {pair} contains pair {document['pair']!r}")
    if "generatorQPerMinute" not in document:
        # Read old standalone exports as well as the canonical Q format.
        return _validated_volatility_regimes(document, pair)
    try:
        sigma = [float(x) for x in document["sigmaPips"]]
        q = [[float(x) for x in row] for row in document["generatorQPerMinute"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Calibration for {pair} requires sigmaPips and generatorQPerMinute") from exc
    n = len(q)
    if not 2 <= n <= 12 or len(sigma) != n or any(len(row) != n for row in q):
        raise ValueError(f"Calibration Q for {pair} must be NxN (2-12) matching {n} sigma levels")
    rates = [[0.0 if i == j else q[i][j] for j in range(n)] for i in range(n)]
    model = _validated_volatility_regimes({"sigmaPips": sigma, "ratesPerMinute": rates}, pair)
    for i, row in enumerate(q):
        if not math.isfinite(row[i]) or row[i] >= 0:
            raise ValueError(f"Calibration Q for {pair} has an invalid diagonal in row {i}")
        if abs(sum(row)) > max(1e-9, 1e-8 * abs(row[i])):
            raise ValueError(f"Calibration Q for {pair} must have zero row sums (row {i})")
        if not math.isclose(row[i], -sum(rates[i]), rel_tol=1e-8, abs_tol=1e-9):
            raise ValueError(f"Calibration Q for {pair} has inconsistent diagonal in row {i}")
    if "ratesPerMinute" in document:
        canonical = _validated_volatility_regimes(document, pair)
        for left, right in zip(canonical["ratesPerMinute"], rates):
            if any(not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10) for a, b in zip(left, right)):
                raise ValueError(f"Calibration Q and ratesPerMinute disagree for {pair}")
    return model


def volatility_calibration_source(pair: str | None) -> str:
    """Readable source indicator for the dashboard."""
    path = calibration_artifact_path(str(pair or ""))
    return str(path) if path.is_file() else "fx_pair_defaults.json (fallback)"


def default_fx_pair_volatility(pair: str | None) -> dict:
    """Return exactly the configured volatility model for a target FX pair."""
    key = str(pair or "").upper().strip()
    if key not in DEFAULT_FX_PAIR_VOLATILITY:
        raise ValueError(f"Missing JSON volatility model for target FX pair {key!r}")
    path = calibration_artifact_path(key)
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as stream:
                artifact = json.load(stream)
            return _volatility_from_calibration_artifact(artifact, key)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid volatility calibration file {path}: {exc}") from exc
    return deepcopy(DEFAULT_FX_PAIR_VOLATILITY[key])


DEFAULT_FX_PAIR_MARKET_DATA = _load_pair_market_defaults()
DEFAULT_FX_MIDS: dict[str, float] = {
    pair: values["spot"] for pair, values in DEFAULT_FX_PAIR_MARKET_DATA.items()
}


def default_fx_pair_market_data(pair: str | None) -> dict[str, float] | None:
    """Pair-specific starting inputs for the target FX pair (fresh copy)."""
    key = str(pair or "").upper().strip()
    values = DEFAULT_FX_PAIR_MARKET_DATA.get(key)
    return dict(values) if values is not None else None



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
    "spot": default_fx_pair_market_data("EURSEK")["spot"],
    "spreadPips": default_fx_pair_market_data("EURSEK")["spreadPips"],
    "sigmaPips": default_fx_pair_volatility("EURSEK")["sigmaPips"][1],
    # JSON is authoritative for all target-pair volatility regimes.
    "volatilityRegimes": default_fx_pair_volatility("EURSEK"),
    "gamma": 0.1,
    "grid": {
        "mode": "uniform",
        "maxAbs": 20.0,
        "step": 1.0,
    },
    "tiers": [
        {
            "enabled": True,
            "name": "Tier 1",
            "sizes": [1, 2, 3, 5, 10, 20],
            # One-sided exogenous RFQ-arrival intensity curve:
            # lambda_RFQ(z) = A0 * z^(-theta-beta*z).  Monte Carlo evaluates
            # this curve only on the listed pricing knots; they also define
            # the possible RFQ sizes.
            "flow": {"A0": 0.0155, "theta": 0.144, "beta": 0.0857, "steepness": 8.42, "shift": 0.52, "volumeShift": 0.026},
            "useMarkout": True,
            # Signed adverse markout, EUR per traded EUR 1M; independent of RFQ size.
            "markout": {"asymptoticEurPerMillion": -50.8, "tauSeconds": 36.7},
            "feePips": 0.0,
            "deltaMin": -10.0,
            "deltaMax": 100.0,
        },
        {
            "enabled": True,
            "name": "Tier 2",
            "sizes": [1, 2, 3, 5, 10, 20],
            "flow": {"A0": 0.0232, "theta": 0.303, "beta": 0.122, "steepness": 2.86, "shift": 0.48, "volumeShift": 0.020},
            "useMarkout": True,
            # Signed adverse markout, EUR per traded EUR 1M; independent of RFQ size.
            "markout": {"asymptoticEurPerMillion": -50.8, "tauSeconds": 36.7},
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
        # Discrete parent ECN trade-size probabilities for fixed source-base
        # currency pillars [1M, 750k, 500k, 250k, 100k].  The UI edits the
        # first four probabilities and defines the 100k probability as the
        # residual 1 - sum(other probabilities).
        "tradeSizeProbabilities": list(ECN_TRADE_SIZE_PROBABILITIES),
        "quoteSize": 1.0,
        "makerFeePips": 3.0,
    },
}


def default_config() -> dict:
    cfg = deepcopy(_DEFAULT_CONFIG)
    # Do not retain the import-time 3-state default if notebook was rerun.
    model = default_fx_pair_volatility(cfg["targetPair"])
    cfg["volatilityRegimes"] = model
    cfg["sigmaPips"] = model["sigmaPips"][len(model["sigmaPips"]) // 2]
    return cfg
