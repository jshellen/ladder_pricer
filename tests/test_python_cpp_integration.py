"""End-to-end tests of the Python façade and compiled C++ native engine.

Build extension before running: python setup.py build_ext --inplace
Run: PYTHONPATH=python python -m pytest tests/test_python_cpp_integration.py -q
"""
from copy import deepcopy
import math

import pytest

from trinity.defaults import default_config

_native = pytest.importorskip("trinity._native", reason="compiled C++ extension not installed")
from trinity.engine import Engine


def config_with_sources(mapping):
    config = default_config()
    config["volatilityModel"]["enabled"] = False
    for tier in config["tiers"]:
        if tier["enabled"]:
            tier["flowSources"] = [
                {"name": "source", "flow": deepcopy(tier["flow"]), "mapping": deepcopy(mapping)}
            ]
    ecn = config["passiveEcn"]
    ecn["flowSources"] = [
        {"name": "source", "flow": deepcopy(ecn["flow"]),
         "tradeSizeProbabilities": list(ecn["tradeSizeProbabilities"]),
         "mapping": deepcopy(mapping)}
    ]
    return config


def test_native_version_and_facade():
    assert _native.ECN_PARAMETERIZATION_VERSION == 11
    cfg = default_config()
    cfg["volatilityModel"]["enabled"] = False
    engine = Engine(cfg)
    result = engine.solve()
    assert result["converged"]
    assert len(result["qGrid"]) == len(result["value"])
    assert math.isfinite(result["averageReward"])
    stats = engine.statistics(2.0, 0.0)
    assert math.isfinite(stats["meanBase"])
    seen = []
    simulation = engine.simulate(0.2, 3, 0.0, 1234, 1, 4,
                                 progress_callback=lambda done, total: seen.append((done, total)))
    assert len(simulation["pnlBase"]) == 3
    assert all(math.isfinite(float(v)) for v in simulation["pnlBase"])
    assert seen, "native progress callback must cross back to Python"


@pytest.mark.parametrize("mapping", [
    {"type": "identity"},
    {"type": "crossed", "crossMid": 0.8463},
    {"type": "affine", "deltaScale": 0.82, "sourceSizePerTarget": 0.8463},
])
def test_flow_sources_solve_statistics_and_simulation(mapping):
    engine = Engine(config_with_sources(mapping))
    result = engine.solve()
    assert result["converged"]
    assert math.isfinite(result["averageReward"])
    stats = engine.statistics(1.0, 0.0)
    assert all(math.isfinite(float(stats[k])) for k in ("meanBase", "stdBase"))
    simulation = engine.simulate(0.1, 2, 0.0, 4242, 1, 4)
    assert len(simulation["pnlBase"]) == 2
    assert all(math.isfinite(float(v)) for v in simulation["pnlBase"])


def test_explicit_identity_mapping_matches_direct_flow():
    direct = default_config()
    direct["volatilityModel"]["enabled"] = False
    explicit = config_with_sources({"type": "identity"})
    lhs = Engine(direct).solve()
    rhs = Engine(explicit).solve()
    assert lhs["averageReward"] == pytest.approx(rhs["averageReward"], abs=1e-10)
    assert rhs["value"] == pytest.approx(lhs["value"], abs=1e-9)
