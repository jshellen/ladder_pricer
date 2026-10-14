"""Monte Carlo, Markov moment analytics and frontier use the same q x regime policy."""

from copy import deepcopy
import math
from statistics import mean, pstdev

import pytest

from trinity.defaults import default_config
from trinity.engine import Engine
from trinity.ui_config import build_config
from test_ui_config import ui_values_from_default


def small_config(regimes: int, *, dark: bool = True):
    cfg = default_config()
    cfg["grid"]["maxAbs"] = 6.0
    for tier in cfg["tiers"]:
        tier["sizes"] = [1, 2]
    cfg["darkPool"]["enabled"] = dark
    if regimes == 2:
        cfg["volatilityRegimes"] = {
            "sigmaPips": [12.0, 35.0],
            "ratesPerMinute": [[0.0, 0.1], [0.04, 0.0]],
        }
    return cfg


@pytest.mark.parametrize("regime_count", [2, 3])
def test_joint_regime_simulation_and_analytic_pnl_agree(regime_count):
    cfg = small_config(regime_count)
    engine = Engine(cfg)
    solved = engine.solve()
    assert len(solved["regimes"]) == regime_count
    assert len(solved["valueByRegime"]) == regime_count

    initial = 0
    horizon = 60.0
    analytic = engine.statistics(horizon, initial_inventory=2.0, initial_regime=initial)
    assert analytic["stdQuote"] > 0
    assert analytic["meanBase"] == pytest.approx(analytic["meanQuote"] / cfg["spot"])
    assert analytic["stdBase"] == pytest.approx(analytic["stdQuote"] / cfg["spot"])
    if "internalizationTimes" in analytic:
        times = analytic["internalizationTimes"]
        assert times["qGrid"] == solved["qGrid"]
        assert len(times["minutes"]) == len(solved["qGrid"])
        assert times["minutes"][times["qGrid"].index(0.0)] == 0.0
        assert all(math.isfinite(t) and t >= 0 for t in times["minutes"])

    npaths = 12_000
    sampled = engine.simulate(
        horizon_minutes=horizon, paths=npaths, initial_inventory=2.0,
        seed=720, retained_paths=2, sample_points=15, initial_regime=initial,
    )
    assert len(sampled["pnlQuote"]) == npaths
    assert set(sampled["finalRegime"]) == set(range(regime_count))
    assert all(len(path["regimes"]) == len(path["times"]) == len(path["spots"])
               for path in sampled["samplePaths"])
    assert all(set(path["regimes"]) <= set(range(regime_count))
               for path in sampled["samplePaths"])
    assert any(len(set(path["regimes"])) > 1 for path in sampled["samplePaths"])
    # Distributional consistency: the exact finite-horizon first two PnL
    # moments should agree with the independent event-driven Monte Carlo.
    mc_mean = mean(sampled["pnlQuote"])
    mc_std = pstdev(sampled["pnlQuote"])
    mean_tol = 5.0 * analytic["stdQuote"] / math.sqrt(npaths) + 35.0
    assert abs(mc_mean - analytic["meanQuote"]) < mean_tol
    assert abs(mc_std - analytic["stdQuote"]) < 0.12 * analytic["stdQuote"]


@pytest.mark.parametrize("regime_count", [2, 3])
def test_frontier_re_solves_joint_model_and_is_consistent(regime_count):
    cfg = small_config(regime_count, dark=False)
    engine = Engine(cfg)
    engine.solve()
    reference = engine.statistics(horizon_minutes=35, initial_regime=0)
    frontier = engine.frontier([0.06, cfg["gamma"], 0.14], horizon_minutes=35, initial_regime=0)
    assert len(frontier) == 3
    middle = frontier[1]
    assert middle["gamma"] == cfg["gamma"]
    assert middle["mean"] == pytest.approx(reference["meanBase"], abs=1e-6)
    assert middle["std"] == pytest.approx(reference["stdBase"], abs=1e-6)
    assert middle["ratio"] == pytest.approx(middle["mean"] / middle["std"])
    assert all(math.isfinite(point["mean"]) and point["std"] > 0 for point in frontier)


def test_initial_regime_affects_analytics_and_simulation():
    cfg = small_config(2, dark=False)
    engine = Engine(cfg)
    engine.solve()
    low = engine.statistics(horizon_minutes=5, initial_inventory=3, initial_regime=0)
    high = engine.statistics(horizon_minutes=5, initial_inventory=3, initial_regime=1)
    assert high["stdQuote"] > low["stdQuote"]
    for initial in (0, 1):
        mc = engine.simulate(horizon_minutes=0, paths=4, retained_paths=1,
                             sample_points=2, initial_regime=initial)
        assert mc["finalRegime"] == [initial] * 4
        assert mc["samplePaths"][0]["regimes"] == [initial]
    with pytest.raises((ValueError, RuntimeError), match="initial regime out of range"):
        engine.statistics(horizon_minutes=5, initial_regime=2)


def test_json_two_regime_override_drives_solver_and_analytics(monkeypatch):
    import trinity.defaults as defaults
    from pathlib import Path
    pair = "EURUSD"
    model = {"sigmaPips": [10., 29.], "ratesPerMinute": [[0., .12], [.08, 0.]]}
    monkeypatch.setitem(defaults.DEFAULT_FX_PAIR_VOLATILITY, pair, model)
    values = ui_values_from_default()
    values["target-pair"] = pair
    cfg = build_config(values)
    assert cfg["volatilityRegimes"] == model
    assert len(Engine(cfg).solve()["regimes"]) == 2
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text()
    assert '"volatility-model-store"' not in source
    assert '"flow-vol-count"' not in source
    assert 'named_values["target-pair"] = target_pair' in source
