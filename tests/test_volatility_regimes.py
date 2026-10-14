"""Joint volatility-switching policy tests (no independent-volatility HJB shortcut)."""
from copy import deepcopy
import math
import pytest
from trinity.defaults import default_config
from trinity.engine import Engine


def test_joint_regime_policy_symmetry_and_risk_effect():
    result = Engine(default_config()).solve_volatility()
    assert result["converged"]
    assert result["evaluationResidual"] < 1e-9
    q = result["qGrid"]
    assert result["sigmaPips"] == [12., 20., 35.]
    idx = q.index(10.)
    reverse = q.index(-10.)
    deltas = []
    for regime in result["regimes"]:
        ecn = regime["passiveEcn"]
        assert ecn["askActive"][idx]
        assert ecn["bidActive"][reverse]
        assert math.isclose(ecn["askDelta"][idx], ecn["bidDelta"][reverse],abs_tol=1e-10)
        assert not ecn["askActive"][q.index(0.)]
        assert not ecn["bidActive"][q.index(0.)]
        deltas.append(ecn["askDelta"][idx])
    assert deltas[0] < deltas[1] < deltas[2]


def test_no_risk_aversion_removes_volatility_dependence():
    config=default_config()
    config["gamma"]=0.0
    result=Engine(config).solve_volatility()
    assert result["converged"]
    for side in ("bid", "ask"):
        key=side+"Delta"
        active_key=side+"Active"
        ref=result["regimes"][0]["passiveEcn"]
        for regime in result["regimes"][1:]:
            ecn=regime["passiveEcn"]
            assert ecn[active_key]==ref[active_key]
            assert ecn[key]==pytest.approx(ref[key],abs=1e-9)


def test_invalid_regime_generator_rejected():
    config=default_config()
    config["volatilityRegimes"]["ratesPerMinute"][0][1]=-0.1
    with pytest.raises(ValueError,match="negative transition"):
        Engine(config).solve_volatility()
