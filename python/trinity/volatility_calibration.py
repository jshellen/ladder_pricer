"""Estimate the existing HJB volatility-regime inputs from intraday FX closes.

The C++ solver expects sigmaPips in quote-price units of 1e-4 per sqrt(min)
and zero-diagonal ratesPerMinute (not a full generator).  The sampling grid
must be short enough to observe a meaningful number of regime transitions.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

COPENHAGEN = "Europe/Copenhagen"
SOLVER_PIP_SIZE = 0.0001  # The current C++ solver divides sigmaPips by 10000.


@dataclass
class CalibrationResult:
    model: dict
    generator: np.ndarray
    transition_counts: np.ndarray
    observed_transition_probabilities: np.ndarray
    fitted_transition_probabilities: np.ndarray
    vol_frame: pd.DataFrame
    regime_frame: pd.DataFrame
    thresholds: np.ndarray
    sample_minutes: int
    window_minutes: int
    optimization_success: bool
    regime_method: str = "equal_frequency"
    regime_quantiles: tuple[float, ...] = ()
    extreme_quantile: float | None = None


def yahoo_fx_closes(pair: str, *, lookback_days: int = 55, interval_minutes: int = 5) -> pd.Series:
    """Download current intraday data from Yahoo; perform NO synthetic fallback.

    FX ticker: EURSEK -> EURSEK=X.  The intraday retrieval window is limited
    by Yahoo to approximately 60 days; the requested start is kept inside it.
    Yahoo's timestamp must be timezone-aware so DST is handled correctly.
    """
    if not isinstance(pair, str) or len(pair) != 6 or not pair.isalpha():
        raise ValueError("pair must be a six-letter FX pair such as EURSEK")
    if not (3 <= lookback_days <= 58):
        raise ValueError("lookback_days must be in [3,58] for intraday Yahoo data")
    if interval_minutes not in (1, 2, 5, 15, 30, 60):
        raise ValueError("unsupported Yahoo intraday interval")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError('Install calibration dependencies: pip install -e ".[calibration]"') from exc
    ticker = pair.upper() + "=X"
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker, start=start, interval=f"{interval_minutes}m",
        auto_adjust=False, progress=False, threads=False, ignore_tz=False,
    )
    return normalize_yahoo_close(raw, ticker)


def normalize_yahoo_close(data: pd.DataFrame, ticker: str = "EURSEK=X") -> pd.Series:
    """Accept both yfinance flat and MultiIndex single-ticker return layouts."""
    if data is None or data.empty:
        raise ValueError(f"Yahoo returned no intraday Close prices for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        hits = [col for col in data.columns if "Close" in col and ticker in col]
        if len(hits) != 1:
            raise ValueError(f"Cannot identify the Close column for {ticker} in Yahoo response")
        close = data[hits[0]]
    else:
        if "Close" not in data:
            raise ValueError(f"Yahoo response for {ticker} has no Close column")
        close = data["Close"]
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError("Intraday timestamps must be timezone-aware (check yfinance ignore_tz=False)")
    close = pd.to_numeric(close, errors="coerce").astype(float)
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close = close[close > 0].sort_index()
    close = close[~close.index.duplicated(keep="last")]
    if close.empty:
        raise ValueError(f"No valid positive FX closes for {ticker}")
    close.index = close.index.tz_convert(COPENHAGEN)
    close.name = "close"
    return close


def realized_volatility(
    closes: pd.Series,
    *,
    interval_minutes: int = 5,
    window_minutes: int = 60,
    session_start: str = "07:45",
    session_end: str = "17:15",
    pip_size: float = SOLVER_PIP_SIZE,
) -> pd.DataFrame:
    """Rolling sqrt(sum of squared consecutive price changes / elapsed minutes).

    Rolling estimates reset each local Copenhagen session. Missing bars break
    the rolling window. Weekends and overnight returns are never used.
    Yahoo timestamps are treated as candle *starts*.
    """
    if not isinstance(closes.index, pd.DatetimeIndex) or closes.index.tz is None:
        raise ValueError("closes must have a timezone-aware DatetimeIndex")
    if interval_minutes <= 0 or window_minutes <= 0 or window_minutes % interval_minutes:
        raise ValueError("window_minutes must be a positive multiple of interval_minutes")
    if not (math.isfinite(pip_size) and pip_size > 0):
        raise ValueError("pip_size must be positive and finite")
    idx = closes.index.tz_convert(COPENHAGEN)
    series = pd.Series(pd.to_numeric(closes, errors="coerce").to_numpy(dtype=float), index=idx)
    series = series.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series[series > 0]
    # Filtering by local clock preserves session windows across DST switches.
    series = series[(series.index.dayofweek < 5)].between_time(session_start, session_end, inclusive="left")
    if series.empty:
        raise ValueError("No valid close prices within the Copenhagen session")
    rows = []
    bar_count = window_minutes // interval_minutes
    for _, day in series.groupby(series.index.date, sort=True):
        delta_minutes = day.index.to_series().diff().dt.total_seconds().to_numpy() / 60.0
        pip_return = day.diff() / pip_size
        pip_return.iloc[~np.isclose(delta_minutes, interval_minutes)] = np.nan
        sum_sq = pip_return.pow(2).rolling(bar_count, min_periods=bar_count).sum()
        sigma = np.sqrt(sum_sq / window_minutes)
        rows.append(pd.DataFrame({
            "close": day,
            "returnPips": pip_return,
            "sigmaPips": sigma,
        }, index=day.index))
    frame = pd.concat(rows)
    frame.index.name = "timeCopenhagen"
    frame["sessionDate"] = frame.index.date
    return frame


def regime_quantile_boundaries(
    regimes: int, *, method: str = "equal_frequency", extreme_quantile: float = 0.95,
) -> np.ndarray:
    """Return *percentile* boundaries, independent of the volatility sample.

    In tail-weighted mode the final boundary is ALWAYS extreme_quantile,
    independent of the number of states.  The lower N-1 buckets are obtained
    by interpolating a six-state reference allocation of 30/20/20/15/10% of
    the lower 95% distribution (at extreme_quantile=.95).  Thus for N=6 the
    complete occupancies are 30/20/20/15/10/5%, while for N=3 the highest
    state still means the upper 5%, not the upper third.

    We keep a fixed *quantile* threshold, not a fixed absolute sigma: the
    numerical threshold can change when the calibration sample changes.
    """
    if (not isinstance(regimes, (int, np.integer))
            or isinstance(regimes, (bool, np.bool_)) or not 2 <= regimes <= 12):
        raise ValueError("regimes must be an integer between 2 and 12 (C++ solver limit)")
    if method not in ("equal_frequency", "tail_weighted"):
        raise ValueError("method must be 'equal_frequency' or 'tail_weighted'")
    if not (isinstance(extreme_quantile, (int, float, np.integer, np.floating))
            and np.isfinite(extreme_quantile) and 0.80 <= extreme_quantile <= 0.99):
        raise ValueError("extreme_quantile must be between 0.80 and 0.99")
    if method == "equal_frequency":
        return np.arange(1, regimes, dtype=float) / regimes
    # Reference anchors are the cumulative mass allocations for N=6 when
    # extreme_quantile=.95. Rescale the lower population to an adjustable
    # extreme_quantile while keeping the *shape* of the bulk allocation.
    anchors = np.array([0., .30, .50, .70, .85, .95]) / .95
    sample_positions = np.linspace(0, len(anchors) - 1, regimes)
    positions = np.interp(sample_positions, np.arange(len(anchors)), anchors)
    result = positions[1:] * extreme_quantile
    result[-1] = float(extreme_quantile)
    return result


def label_volatility_regimes(
    frame: pd.DataFrame, *, regimes: int = 3,
    method: str = "equal_frequency", extreme_quantile: float = .95,
):
    """Label rolling sigma by requested quantiles; state sigma is conditional RMS.

    'tail_weighted' uses the same top-tail quantile for every state count.
    'equal_frequency' preserves the legacy calibration exactly.
    """
    quantiles = regime_quantile_boundaries(
        regimes, method=method, extreme_quantile=extreme_quantile,
    )
    valid = frame.dropna(subset=["sigmaPips"]).copy()
    if len(valid) < regimes * 30:
        raise ValueError("Too few valid rolling-volatility observations for regime calibration")
    values = valid["sigmaPips"].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Rolling volatility must be strictly positive and finite")
    thresholds = np.quantile(values, quantiles)
    if (not np.all(np.diff(thresholds) > 0) or values.min() >= thresholds[0]
            or values.max() <= thresholds[-1]):
        raise ValueError("Data cannot distinguish requested volatility regimes; use fewer states or better data")
    valid["regime"] = np.searchsorted(thresholds, values, side="right")
    sigma_levels = []
    for i in range(regimes):
        sub = valid.loc[valid["regime"] == i, "sigmaPips"].to_numpy()
        if len(sub) < 20:
            raise ValueError(
                f"Too few observations ({len(sub)}) in volatility regime {i+1}; "
                "use fewer regimes, a longer sample, or a less extreme tail cutoff"
            )
        sigma_levels.append(float(np.sqrt(np.mean(sub ** 2))))
    if any(b <= a for a, b in zip(sigma_levels, sigma_levels[1:])):
        raise ValueError("Volatility levels are not strictly increasing")
    return valid, np.asarray(sigma_levels), thresholds


def count_regime_transitions(regime_frame: pd.DataFrame, *, interval_minutes: int = 5, regimes: int = 3) -> np.ndarray:
    """Only adjacent *intra-session* observations contribute transitions."""
    counts = np.zeros((regimes, regimes), dtype=int)
    times = regime_frame.index
    if not isinstance(times, pd.DatetimeIndex) or times.tz is None:
        raise ValueError("regime_frame requires timezone-aware timestamps")
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("regime_frame timestamps must be unique and sorted")
    states = regime_frame["regime"].to_numpy(dtype=int)
    for i in range(1, len(states)):
        if times[i].date() != times[i - 1].date():
            continue
        if not np.isclose((times[i] - times[i - 1]).total_seconds() / 60, interval_minutes):
            continue
        a, b = int(states[i - 1]), int(states[i])
        if not (0 <= a < regimes and 0 <= b < regimes):
            raise ValueError("invalid regime label")
        counts[a, b] += 1
    if np.any(counts.sum(axis=1) == 0):
        raise ValueError("Every regime must have valid in-session consecutive samples")
    return counts


def _rate_matrix_to_generator(rate_matrix: np.ndarray) -> np.ndarray:
    q = np.array(rate_matrix, dtype=float, copy=True)
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))
    return q


def fit_ctmc_generator(
    counts: np.ndarray, *, interval_minutes: int = 5,
    transition_topology: str = "auto",
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Panel-observed CTMC MLE: P(Δ)=expm(Q Δ), not P/Δ or log(P)/Δ.

    Positivity/irreducibility follow from exp-rate parametrization. For
    >3 regimes, 'auto' selects an adjacent-state (birth-death) generator to
    keep the estimation identifiable with short Yahoo history. Multiple
    state moves during one bar are STILL possible via expm(ΔQ). For 2/3
    regimes the old full generator is preserved. 'full' and 'adjacent'
    can be selected explicitly; 'full' is expensive for many regimes.
    """
    c = np.asarray(counts, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not 2 <= c.shape[0] <= 12:
        raise ValueError("transition counts must be square, between 2x2 and 12x12")
    if not np.isfinite(c).all() or (c < 0).any() or (c.sum(axis=1) == 0).any():
        raise ValueError("transition counts must be nonnegative with populated rows")
    if not (interval_minutes > 0):
        raise ValueError("interval_minutes must be positive")
    if transition_topology not in ("auto", "full", "adjacent"):
        raise ValueError("transition_topology must be 'auto', 'full' or 'adjacent'")
    n = len(c)
    topology = ("adjacent" if n > 3 else "full") if transition_topology == "auto" else transition_topology
    indices = [(i, j) for i in range(n) for j in range(n)
               if i != j and (topology == "full" or abs(i-j) == 1)]
    maximum_rate = 2.0 / interval_minutes
    min_rate = 1e-8
    crude = c / (c.sum(axis=1, keepdims=True) * interval_minutes)
    initial = np.log([np.clip(crude[i, j], 1e-4, maximum_rate * 0.9) for i, j in indices])

    def assemble(log_rates):
        rates = np.zeros((n, n))
        for x, (i, j) in zip(log_rates, indices):
            rates[i, j] = np.exp(x)
        return _rate_matrix_to_generator(rates)

    def loss(x):
        p = expm(assemble(x) * interval_minutes)
        if not np.isfinite(p).all():
            return 1e100
        return float(-(c * np.log(np.clip(p, 1e-300, 1.0))).sum())

    limits = [(np.log(min_rate), np.log(maximum_rate))] * len(indices)
    best = None
    for guess in (initial, np.log(np.full(len(indices), 0.02))):
        result = minimize(loss, np.clip(guess, limits[0][0], limits[0][1]),
                          method="L-BFGS-B", bounds=limits,
                          options={"maxiter": 1000, "ftol": 1e-12})
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("CTMC transition rate optimization failed")
    generator = assemble(best.x)
    # No silent acceptance of a numerically failed optimization.
    if not best.success:
        raise RuntimeError(f"CTMC optimization did not converge: {best.message}")
    return generator, expm(generator * interval_minutes), bool(best.success)


def calibrate_from_closes(
    closes: pd.Series, *, regimes: int = 3, interval_minutes: int = 5,
    window_minutes: int = 60, pip_size: float = SOLVER_PIP_SIZE,
    transition_topology: str = "auto",
    regime_method: str = "equal_frequency", extreme_quantile: float = .95,
) -> CalibrationResult:
    frame = realized_volatility(
        closes, interval_minutes=interval_minutes,
        window_minutes=window_minutes, pip_size=pip_size,
    )
    tagged, sigmas, boundaries = label_volatility_regimes(
        frame, regimes=regimes, method=regime_method, extreme_quantile=extreme_quantile,
    )
    counts = count_regime_transitions(tagged, interval_minutes=interval_minutes, regimes=regimes)
    q, fitted_p, success = fit_ctmc_generator(
        counts, interval_minutes=interval_minutes, transition_topology=transition_topology,
    )
    if not np.all(np.isfinite(q)) or np.any(np.diag(q) >= 0):
        raise ValueError("Calibrated CTMC has an invalid generator")
    rates = np.maximum(q, 0.0)
    np.fill_diagonal(rates, 0.0)
    model = {
        "sigmaPips": [float(s) for s in sigmas],
        "ratesPerMinute": rates.tolist(),
    }
    empirical_p = counts / counts.sum(axis=1, keepdims=True)
    quantiles = regime_quantile_boundaries(
        regimes, method=regime_method, extreme_quantile=extreme_quantile,
    )
    return CalibrationResult(model, q, counts, empirical_p, fitted_p, frame, tagged,
                             boundaries, interval_minutes, window_minutes, success,
                             regime_method, tuple(float(x) for x in quantiles),
                             float(extreme_quantile) if regime_method == "tail_weighted" else None)


def update_pair_default(pair: str, model: dict, *, path: Path | None = None) -> Path:
    """Explicit opt-in: replace only this pair's volatility override, atomically."""
    from .defaults import _validated_volatility_regimes

    pair = pair.strip().upper()
    json_path = Path(path) if path is not None else Path(__file__).with_name("fx_pair_defaults.json")
    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if pair not in payload["pairs"]:
        raise ValueError(f"FX pair {pair} is not defined in {json_path}")
    payload["pairs"][pair]["volatilityRegimes"] = _validated_volatility_regimes(model, pair)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=json_path.parent,
                                         prefix=".vol_calibration_", suffix=".json", delete=False) as f:
            temp_name = f.name
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temp_name, json_path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
    return json_path


def save_calibration_artifact(
    pair: str, result: CalibrationResult, *, output_dir: Path | None = None,
) -> Path:
    """Persist complete calibrated Q + sigma levels for Dash auto-discovery.

    The matrix stored in `generatorQPerMinute` is the genuine generator with
    *negative diagonal*; the Dash defaults loader converts it into the
    solver's zero-diagonal `ratesPerMinute` on every new Solve.  Atomic
    replacement prevents dashboard readers from seeing partially written JSON.
    """
    from .defaults import CALIBRATION_OUTPUT_DIR, _volatility_from_calibration_artifact

    pair = pair.strip().upper()
    if len(pair) != 6 or not pair.isalpha():
        raise ValueError("pair must be a six-letter FX pair")
    q = np.asarray(result.generator, dtype=float)
    document = {
        "schemaVersion": 1,
        "pair": pair,
        "sigmaPips": [float(x) for x in result.model["sigmaPips"]],
        "generatorQPerMinute": q.tolist(),
        "regimeThresholdsPips": [float(x) for x in result.thresholds],
        "calibration": {
            "method": "yfinance-rolling-realized-volatility-ctmc-mle",
            "barMinutes": int(result.sample_minutes),
            "volWindowMinutes": int(result.window_minutes),
            "sampleStart": result.regime_frame.index.min().isoformat(),
            "sampleEnd": result.regime_frame.index.max().isoformat(),
            "observations": int(len(result.regime_frame)),
            "regimeMethod": result.regime_method,
            "regimeQuantiles": list(result.regime_quantiles),
            "extremeQuantile": result.extreme_quantile,
        },
    }
    # Validate BEFORE overwriting an existing good calibration.
    _volatility_from_calibration_artifact(document, pair)
    directory = Path(output_dir) if output_dir is not None else CALIBRATION_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{pair}_volatility_regimes.json"
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=f".{pair}_", suffix=".json", delete=False) as stream:
            temp_name = stream.name
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp_name, destination)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
    return destination
