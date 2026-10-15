"""Offline regression coverage for the optional yfinance volatility notebook."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from trinity.volatility_calibration import (
    SOLVER_PIP_SIZE,
    calibrate_from_closes,
    count_regime_transitions,
    fit_ctmc_generator,
    normalize_yahoo_close,
    realized_volatility,
    update_pair_default,
)


def _prices(days=3, step=5):
    # Local days include session boundaries; no synthetic weekend returns.
    times = []
    for date in pd.bdate_range("2026-10-05", periods=days):
        times.extend(pd.date_range(date.strftime("%Y-%m-%d") + " 07:45",
                                   date.strftime("%Y-%m-%d") + " 17:10",
                                   freq=f"{step}min", tz="Europe/Copenhagen"))
    idx = pd.DatetimeIndex(times)
    return pd.Series(11.5 + np.arange(len(idx)) * SOLVER_PIP_SIZE, index=idx)


def test_normalize_yahoo_multilevel_and_timezone_and_invalid():
    closes = _prices(days=1)
    df = pd.DataFrame({("Close", "EURSEK=X"): closes, ("Open", "EURSEK=X"): closes})
    result = normalize_yahoo_close(df, "EURSEK=X")
    assert len(result) == len(closes)
    assert str(result.index.tz) == "Europe/Copenhagen"
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_yahoo_close(pd.DataFrame({"Close": [11.5]}, index=[pd.Timestamp("2026-10-01")]))


def test_realized_sigma_pips_and_no_overnight_jump():
    closes = _prices(days=3)
    # Every 5-minute change is 1 pip, so sigma = 1/sqrt(5) pips/sqrt(min).
    frame = realized_volatility(closes, window_minutes=60)
    sigma = frame["sigmaPips"].dropna().to_numpy()
    assert len(sigma) == 3 * (len(_prices(days=1)) - 12)
    assert sigma == pytest.approx(np.full(len(sigma), 1.0 / np.sqrt(5)), rel=1e-9)
    assert np.isnan(frame.iloc[0]["returnPips"])


def test_missing_bar_invalidates_only_window_containing_gap():
    closes = _prices(days=1)
    closes = closes.drop(index=closes.index[55])
    frame = realized_volatility(closes, window_minutes=60)
    # Gap: the following 12-window rolling estimates must be null.
    assert frame.loc[closes.index[56], "sigmaPips"] != frame.loc[closes.index[56], "sigmaPips"]
    assert np.isfinite(frame.iloc[-1]["sigmaPips"])


def test_transitions_do_not_count_cross_day_or_gaps():
    idx = pd.to_datetime([
        "2026-10-05 08:00", "2026-10-05 08:05", "2026-10-05 08:10",
        "2026-10-05 08:20", "2026-10-06 08:00", "2026-10-06 08:05",
    ]).tz_localize("Europe/Copenhagen")
    states = pd.DataFrame({"regime": [0, 1, 0, 1, 0, 1]}, index=idx)
    with pytest.raises(ValueError, match="Every regime"):
        count_regime_transitions(states, regimes=3)
    counts = count_regime_transitions(states, regimes=2)
    assert counts[0, 1] == 2
    assert counts[1, 0] == 1
    assert counts.sum() == 3


def test_ctmc_fit_recovers_transition_probability_with_discrete_sampling():
    q = np.array([[-.018, .018, 0.0], [.012, -.020, .008], [0.0, .025, -.025]])
    probability = expm(q * 5)
    row_samples = np.array([20000, 24000, 16000])
    # A deterministic large synthetic panel avoids Monte Carlo randomness.
    counts = np.vstack([np.round(row_samples[i] * probability[i]).astype(int) for i in range(3)])
    fitted, model_p, success = fit_ctmc_generator(counts, interval_minutes=5)
    assert success
    assert np.allclose(fitted.sum(axis=1), 0, atol=1e-12)
    assert np.all(np.diag(fitted) < 0)
    assert np.min(fitted - np.diag(np.diag(fitted))) >= 0
    assert np.allclose(model_p, counts / counts.sum(axis=1, keepdims=True), atol=.003)


def test_full_calibration_synthetic_prices_produces_solver_schema():
    idx = _prices(days=22).index
    rng = np.random.default_rng(12)
    # A slow-changing conditional σ process with within-session price moves.
    sigma = np.array([8.0, 15.0, 30.0])
    ret_pips = np.zeros(len(idx))
    state = 1
    for i in range(len(idx)):
        if i % 95 == 0:
            state = (state + 1) % 3
        ret_pips[i] = rng.normal(0, sigma[state] * np.sqrt(5))
    price = pd.Series(11.5 + np.cumsum(ret_pips) * SOLVER_PIP_SIZE, index=idx)
    result = calibrate_from_closes(price)
    assert len(result.model["sigmaPips"]) == 3
    assert list(result.model["sigmaPips"]) == sorted(result.model["sigmaPips"])
    assert result.model["sigmaPips"][0] > 0
    assert result.generator.shape == (3, 3)
    assert result.transition_counts.sum() > 1200
    assert np.allclose(np.diag(result.model["ratesPerMinute"]), 0)
    assert np.allclose(result.generator.sum(axis=1), 0)


def test_update_pair_default_preserves_other_pairs_and_fallback(tmp_path):
    path = tmp_path / "fx_pair_defaults.json"
    content = {
        "pairs": {"EURSEK": {"spot": 11.5}, "USDSEK": {"spot": 9.7}},
        "volatilityRegimes": {"sigmaPips": [12, 20, 35], "ratesPerMinute": [[0, .1, 0], [.1, 0, .1], [0, .1, 0]]},
    }
    path.write_text(json.dumps(content), encoding="utf-8")
    model = {"sigmaPips": [7., 16., 25.], "ratesPerMinute": [[0, .01, 0], [.03, 0, .02], [0, .04, 0]]}
    update_pair_default("EURSEK", model, path=path)
    result = json.loads(path.read_text())
    assert result["pairs"]["EURSEK"]["volatilityRegimes"] == model
    assert result["volatilityRegimes"] == content["volatilityRegimes"]
    assert result["pairs"]["USDSEK"] == content["pairs"]["USDSEK"]
    with pytest.raises(ValueError, match="not defined"):
        update_pair_default("EURZAR", model, path=path)

@pytest.mark.parametrize("regimes", [2, 3, 4, 5, 7, 12])
def test_choose_arbitrary_regime_count_and_ordered_levels(regimes):
    from trinity.volatility_calibration import label_volatility_regimes

    # Distinct positive observations, sufficient for all 12 possible buckets.
    frame = pd.DataFrame({"sigmaPips": np.linspace(3., 40., 1200)})
    tagged, sigmas, thresholds = label_volatility_regimes(frame, regimes=regimes)
    assert len(sigmas) == regimes
    assert len(thresholds) == regimes - 1
    assert tagged.regime.nunique() == regimes
    assert sigmas.tolist() == sorted(sigmas)
    assert min(tagged.regime) == 0
    assert max(tagged.regime) == regimes - 1


@pytest.mark.parametrize("invalid", [0, 1, 13, 3.5, True])
def test_calibration_rejects_unsupported_regime_count(invalid):
    from trinity.volatility_calibration import label_volatility_regimes

    with pytest.raises(ValueError, match="2 and 12"):
        label_volatility_regimes(pd.DataFrame({"sigmaPips": np.arange(200) + 1.}), regimes=invalid)


@pytest.mark.parametrize("regimes", [4, 5, 7, 12])
def test_ctmc_fit_multiregime_nearest_neighbor_generator(regimes):
    q = np.zeros((regimes, regimes))
    for i in range(regimes):
        if i > 0:
            q[i, i - 1] = 0.025
        if i + 1 < regimes:
            q[i, i + 1] = 0.03
        q[i, i] = -q[i].sum()
    samples = 14000
    observations = expm(q * 5)
    counts = np.round(observations * samples).astype(int)
    fitted, fitted_p, success = fit_ctmc_generator(counts, interval_minutes=5)
    assert success
    assert fitted.shape == (regimes, regimes)
    assert np.allclose(fitted.sum(axis=1), 0, atol=1e-12)
    assert np.all(np.diag(fitted) < 0)
    assert np.all(fitted[np.abs(np.subtract.outer(np.arange(regimes), np.arange(regimes))) > 1] == 0)
    assert np.allclose(fitted_p, counts / counts.sum(axis=1, keepdims=True), atol=0.004)


def test_multiregime_calibration_full_pipeline_and_solver_json(tmp_path):
    from trinity.defaults import _validated_volatility_regimes

    idx = _prices(days=30).index
    rng = np.random.default_rng(478)
    volatility = np.tile(np.array([6., 12., 22., 33., 44.]), len(idx) // 5 + 1)[:len(idx)]
    # Slowly varying intraday volatility with non-trivial dynamic range.
    volatility = np.repeat(np.array([6., 12., 22., 33., 44.]), 91)
    volatility = np.tile(volatility, len(idx) // len(volatility) + 1)[:len(idx)]
    ret = rng.normal(0, volatility * np.sqrt(5))
    closes = pd.Series(11.5 + np.cumsum(ret) * SOLVER_PIP_SIZE, index=idx)
    result = calibrate_from_closes(closes, regimes=5)
    assert result.generator.shape == (5, 5)
    assert len(result.model["sigmaPips"]) == 5
    assert _validated_volatility_regimes(result.model, "EURSEK") == result.model
    assert result.transition_counts.sum() > 2000
    assert np.allclose(np.diag(result.model["ratesPerMinute"]), 0)
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"pairs": {"EURSEK": {"spot": 11.5}}, "volatilityRegimes": result.model}))
    update_pair_default("EURSEK", result.model, path=path)
    assert len(json.loads(path.read_text())["pairs"]["EURSEK"]["volatilityRegimes"]["sigmaPips"]) == 5
