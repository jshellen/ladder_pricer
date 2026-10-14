"""Production q × sigma integration without a markout state."""
from copy import deepcopy
import pytest
from trinity.defaults import default_config
from trinity.engine import Engine


def test_production_solve_is_coupled_volatility_without_x():
    cfg = default_config()
    assert "markoutState" not in cfg
    sol = Engine(cfg).solve()
    assert sol["stateVariables"] == ["inventory", "volatility"]
    assert "markoutState" not in sol
    assert sol["sigmaPips"] == [12.0, 20.0, 35.0]
    assert len(sol["regimes"]) == 3
    assert len(sol["valueByRegime"]) == 3
    assert len(sol["qGrid"]) == 41
    assert sol["evaluationResidual"] < 1e-8
    assert sol["tiers"] == sol["regimes"][1]["tiers"]
    assert sol["expectedMtm"] == sol["regimes"][1]["expectedMtm"]
    assert len(sol["expectedMtm"]) == len(sol["tiers"])


def test_markout_curve_remains_in_native_regime_policy():
    cfg = default_config()
    cfg["tiers"][0]["markout"]["asymptoticEurPerMillion"] = -80.0
    with_markout = Engine(cfg).solve()
    cfg["tiers"][0]["useMarkout"] = False
    no_markout = Engine(cfg).solve()
    idx = with_markout["qGrid"].index(0.0)
    z_idx = 0
    assert with_markout["expectedMtm"][0]["grossBidEur"][idx][z_idx] != 0.0
    assert no_markout["expectedMtm"][0]["grossBidEur"][idx][z_idx] == 0.0


def test_analytics_uses_the_coupled_regime_policy():
    cfg = default_config()
    cfg["grid"]["maxAbs"] = 5.0
    cfg["tiers"][0]["sizes"] = [1, 2]
    cfg["tiers"][1]["sizes"] = [1, 2]
    engine = Engine(cfg)
    engine.solve()
    stats = engine.statistics(horizon_minutes=12.0, initial_regime=0)
    assert stats["stdQuote"] > 0.0
    mc = engine.simulate(horizon_minutes=12.0, paths=10, retained_paths=1, sample_points=3, initial_regime=0)
    assert len(mc["pnlQuote"]) == 10
    assert len(mc["samplePaths"][0]["regimes"]) == len(mc["samplePaths"][0]["times"])
    frontier = engine.frontier([cfg["gamma"]], horizon_minutes=12.0, initial_regime=0)
    assert len(frontier) == 1
    assert frontier[0]["std"] > 0.0


def test_custom_volatility_regimes_are_used_by_regular_solve():
    cfg = default_config()
    cfg["volatilityRegimes"]["sigmaPips"] = [10.0, 18.0, 40.0]
    result = Engine(cfg).solve()
    assert result["sigmaPips"] == [10.0, 18.0, 40.0]
    assert [r["sigmaPips"] for r in result["regimes"]] == [10.0, 18.0, 40.0]
    assert result["evaluationResidual"] < 1e-8


def test_missing_volatility_regimes_is_rejected_instead_of_using_a_second_solver():
    cfg = default_config()
    cfg.pop("volatilityRegimes")
    with pytest.raises((ValueError, RuntimeError), match="volatilityRegimes is required"):
        Engine(cfg).solve()
