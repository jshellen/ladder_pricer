"""Native HJB + Monte Carlo regression for original knot-only customer RFQ flow.

Requires locally compiled `trinity._native` (python setup.py build_ext --inplace).
"""
import math
import pytest

from trinity.defaults import default_config
_native = pytest.importorskip("trinity._native")
from trinity.engine import Engine


def _engine_for_knots(knots):
    cfg = default_config()
    cfg["tiers"][0]["sizes"] = list(knots)
    for tier in cfg["tiers"][1:]:
        tier["enabled"] = False
    cfg["volatilityModel"]["enabled"] = False
    cfg["passiveEcn"]["enabled"] = False
    cfg["darkPool"]["enabled"] = False
    cfg["grid"]["maxAbs"] = 20.
    return Engine(cfg)


@pytest.mark.parametrize("knots", [[1., 5., 10.], [1., 20.]])
def test_native_policy_and_simulation_only_include_selected_knots(knots):
    engine = _engine_for_knots(knots)
    sol = engine.solve()
    assert sol["tiers"][0]["sizes"] == knots
    assert math.isfinite(sol["averageReward"])
    result = engine.simulate(60., 180, 0., 1234, 0, 11)
    actual = {float(rec["size"]) for rec in result["rfqAggregates"]}
    assert actual <= set(knots), (actual, knots)
    assert 1.0 in actual
    # Very large requests can be too rare to observe in a finite simulation.
    assert all(rec["requests"] > 0 for rec in result["rfqAggregates"])
