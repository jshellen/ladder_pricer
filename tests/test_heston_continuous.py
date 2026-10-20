"""End-to-end checks for continuous Heston variance and variance interpolation."""
import math
import pytest
from trinity.defaults import default_config
from trinity.engine import Engine


def test_heston_correlated_mc_and_risk_reducing_quotes():
    cfg = default_config()
    cfg["hestonVariance"]["correlation"] = -0.7
    e = Engine(cfg)
    result = e.solve()
    assert result["converged"] and result["evaluationResidual"] < 1e-9
    assert result["stateVariables"] == ["inventory", "continuousVariance"]
    q = result["qGrid"]
    idx = q.index(10.0)
    # Full solution is still optimized separately at the HJB variance nodes.
    assert result["regimes"][0]["passiveEcn"]["askDelta"][idx] < \
           result["regimes"][-1]["passiveEcn"]["askDelta"][idx]
    m = e.simulate(horizon_minutes=8.0, paths=6, initial_inventory=10.0,
                   seed=448, retained_paths=2, sample_points=7)
    assert all(math.isfinite(x) for x in m["pnlQuote"])
    for path in m["samplePaths"]:
        assert len(path["variancePipsSquared"]) == len(path["times"]) == len(path["spots"])
        assert all(v >= 0 and math.isfinite(v) for v in path["variancePipsSquared"])
        assert len({round(v, 5) for v in path["variancePipsSquared"]}) > 5


def test_correlation_is_absent_from_translation_invariant_hjb():
    cfg = default_config()
    cfg["hestonVariance"]["correlation"] = 0.0
    uncorrelated = Engine(cfg).solve()
    cfg["hestonVariance"]["correlation"] = 0.8
    correlated = Engine(cfg).solve()
    # V = cash + q*S + h(q,v), hence V_Sv = 0. Rho is irrelevant to HJB.
    assert correlated["averageReward"] == pytest.approx(uncorrelated["averageReward"],abs=1e-12)
    for a,b in zip(correlated["regimes"], uncorrelated["regimes"]):
        assert a["passiveEcn"]["askDelta"] == pytest.approx(b["passiveEcn"]["askDelta"])


def test_invalid_correlation_is_rejected_at_simulation():
    cfg = default_config()
    cfg["hestonVariance"]["correlation"] = 1.2
    e = Engine(cfg)
    e.solve()
    with pytest.raises(ValueError, match="correlation"):
        e.simulate(horizon_minutes=1, paths=1, retained_paths=0)
