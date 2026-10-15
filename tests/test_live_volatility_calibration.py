"""Q-artifact → dashboard defaults/config; no native module or Yahoo network needed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import trinity.defaults as defaults
from trinity.volatility_calibration import CalibrationResult, save_calibration_artifact
from trinity.ui_config import build_config


def _result(n):
    sigma = list(np.linspace(7, 40, n))
    q = np.zeros((n, n))
    for i in range(n):
        if i:
            q[i, i - 1] = .01 + .001 * i
        if i < n - 1:
            q[i, i + 1] = .02 + .001 * i
        q[i, i] = -q[i].sum()
    rate = np.maximum(q, 0).tolist()
    now = pd.date_range("2026-10-01 10:00", periods=120, freq="5min", tz="Europe/Copenhagen")
    df = pd.DataFrame({"regime": np.arange(120) % n}, index=now)
    return CalibrationResult(
        model={"sigmaPips": sigma, "ratesPerMinute": rate}, generator=q,
        transition_counts=np.ones((n, n)),
        observed_transition_probabilities=np.eye(n),
        fitted_transition_probabilities=np.eye(n),
        vol_frame=df, regime_frame=df,
        thresholds=np.linspace(10, 30, n - 1), sample_minutes=5,
        window_minutes=60, optimization_success=True,
    )


def _values():
    # build a full, realistic dashboard config, not just the loader function.
    from test_ui_config import ui_values_from_default
    out = ui_values_from_default()
    out["target-pair"] = "EURSEK"
    return out


def test_export_autoloads_q_and_dimensions_without_restarting(monkeypatch, tmp_path):
    monkeypatch.setattr(defaults, "CALIBRATION_OUTPUT_DIR", tmp_path)
    default_model = defaults.default_fx_pair_volatility("EURSEK")
    assert len(default_model["sigmaPips"]) == 3

    for n in (2, 5, 12):
        result = _result(n)
        destination = save_calibration_artifact("EURSEK", result, output_dir=tmp_path)
        document = json.loads(destination.read_text())
        assert document["generatorQPerMinute"] == result.generator.tolist()
        assert document["sigmaPips"] == result.model["sigmaPips"]
        assert np.allclose(np.asarray(document["generatorQPerMinute"]).sum(axis=1), 0)
        model = defaults.default_fx_pair_volatility("EURSEK")
        assert len(model["sigmaPips"]) == n
        assert np.allclose(model["ratesPerMinute"], np.maximum(result.generator, 0))
        assert defaults.default_config()["volatilityRegimes"] == model
        # Build the very configuration passed to native solving, not simply defaults.
        cfg = build_config(_values())
        assert len(cfg["volatilityRegimes"]["sigmaPips"]) == n
        assert cfg["volatilityRegimes"] == model
        assert cfg["sigmaPips"] == model["sigmaPips"][n // 2]
        assert defaults.volatility_calibration_source("EURSEK") == str(destination)
        assert len(defaults.default_fx_pair_volatility("USDSEK")["sigmaPips"]) == 3

    destination.unlink()
    assert defaults.default_fx_pair_volatility("EURSEK") == default_model
    assert "fallback" in defaults.volatility_calibration_source("EURSEK")


@pytest.mark.parametrize("fault", ["wrong_size", "nonzero_rows", "wrong_pair", "negative_rate", "rates_disagree", "missing_sigma"])
def test_bad_q_artifact_fails_explicitly(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(defaults, "CALIBRATION_OUTPUT_DIR", tmp_path)
    p = save_calibration_artifact("EURSEK", _result(5), output_dir=tmp_path)
    content = json.loads(p.read_text())
    if fault == "wrong_size":
        content["sigmaPips"].pop()
    elif fault == "nonzero_rows":
        content["generatorQPerMinute"][1][1] = -0.01
    elif fault == "wrong_pair":
        content["pair"] = "USDSEK"
    elif fault == "negative_rate":
        content["generatorQPerMinute"][1][2] = -0.01
    elif fault == "rates_disagree":
        content["ratesPerMinute"] = _result(5).model["ratesPerMinute"]
        content["ratesPerMinute"][1][2] = .08
    elif fault == "missing_sigma":
        content.pop("sigmaPips")
    p.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Invalid volatility calibration file"):
        defaults.default_fx_pair_volatility("EURSEK")


def test_legacy_export_still_loads(monkeypatch, tmp_path):
    monkeypatch.setattr(defaults, "CALIBRATION_OUTPUT_DIR", tmp_path)
    p = tmp_path / "EURSEK_volatility_regimes.json"
    model = _result(4).model
    p.write_text(json.dumps(model))
    assert defaults.default_fx_pair_volatility("EURSEK") == model
