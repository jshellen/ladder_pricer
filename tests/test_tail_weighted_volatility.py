"""Tail-weighted regime constraints and actual Q artifact/dashboard integration."""
import json
import numpy as np
import pandas as pd
import pytest

import trinity.defaults as defaults
from trinity.volatility_calibration import (
    SOLVER_PIP_SIZE,
    calibrate_from_closes,
    label_volatility_regimes,
    regime_quantile_boundaries,
    save_calibration_artifact,
)


def _sample_vol_frame():
    return pd.DataFrame({"sigmaPips": np.linspace(3., 40., 2400)})


@pytest.mark.parametrize('count', [2, 3, 4, 5, 6, 8, 12])
def test_top_5pct_is_same_state_regardless_of_regime_count(count):
    frame = _sample_vol_frame()
    observations, levels, cut = label_volatility_regimes(
        frame, regimes=count, method='tail_weighted', extreme_quantile=.95,
    )
    quantiles = regime_quantile_boundaries(count, method='tail_weighted')
    assert quantiles[-1] == .95
    assert np.allclose(cut[-1], np.quantile(frame['sigmaPips'], .95))
    assert sum(observations['regime'] == count - 1) == 120
    assert levels[-1] == pytest.approx(
        np.sqrt(np.mean(frame['sigmaPips'].to_numpy()[-120:] ** 2))
    )
    assert np.all(np.diff(levels) > 0)


def test_six_state_reference_weights_and_legacy_equal_frequency():
    assert regime_quantile_boundaries(6, method='tail_weighted') == pytest.approx(
        [0.30, 0.50, 0.70, 0.85, 0.95], abs=1e-14)
    assert regime_quantile_boundaries(3, method='tail_weighted') == pytest.approx(
        [.60, .95], abs=1e-14)
    assert regime_quantile_boundaries(6, method='equal_frequency') == pytest.approx(
        np.arange(1, 6) / 6)
    assert regime_quantile_boundaries(6, method='tail_weighted', extreme_quantile=.90)[-1] == pytest.approx(.90)


@pytest.mark.parametrize('args', [
    {'regimes': 1}, {'regimes': 13}, {'regimes': True},
    {'regimes': 4, 'method': 'unknown'},
    {'regimes': 4, 'method': 'tail_weighted', 'extreme_quantile': .9999},
])
def test_tail_configs_rejected(args):
    with pytest.raises(ValueError):
        regime_quantile_boundaries(**args)


def test_tail_calibration_q_roundtrip_to_dashboard(monkeypatch, tmp_path):
    monkeypatch.setattr(defaults, 'CALIBRATION_OUTPUT_DIR', tmp_path)
    rng = np.random.default_rng(3491)
    times = []
    for day in pd.bdate_range('2026-08-20', periods=40):
        times.extend(pd.date_range(day.strftime('%Y-%m-%d') + ' 07:45',
                                   periods=114, freq='5min', tz='Europe/Copenhagen'))
    idx = pd.DatetimeIndex(times)
    level = np.repeat(np.array([5., 10., 17., 30., 45.]), 125)
    level = np.tile(level, len(idx)//len(level)+1)[:len(idx)]
    closes = pd.Series(11.5 + np.cumsum(rng.normal(0, level*np.sqrt(5))) * SOLVER_PIP_SIZE,
                       index=idx)
    result = calibrate_from_closes(closes, regimes=6, regime_method='tail_weighted')
    assert len(result.model['sigmaPips']) == 6
    assert result.regime_quantiles[-1] == .95
    path = save_calibration_artifact('EURSEK', result, output_dir=tmp_path)
    saved = json.loads(path.read_text())
    assert saved['calibration']['regimeMethod'] == 'tail_weighted'
    assert saved['calibration']['extremeQuantile'] == .95
    assert saved['calibration']['regimeQuantiles'][-1] == .95
    assert len(saved['generatorQPerMinute']) == 6
    model = defaults.default_fx_pair_volatility('EURSEK')
    assert model['sigmaPips'] == result.model['sigmaPips']
    assert np.allclose(model['ratesPerMinute'], result.model['ratesPerMinute'])
    assert np.allclose(result.generator.sum(axis=1), 0.)
