"""Regression checks for the restored parametric, knot-only RFQ model."""
from copy import deepcopy
from math import isclose, isfinite
import pytest

from trinity.defaults import default_config
from trinity.ui_config import _rfq_curve, build_config


def values_from_default():
    cfg = default_config()
    values = {
        "qmax": cfg["grid"]["maxAbs"], "spot": cfg["spot"],
        "spread": cfg["spreadPips"], "drift": cfg["spotDrift"],
        "vol-enabled": ["on"] if cfg["volatilityModel"]["enabled"] else [],
        "vol-initial-state": cfg["volatilityModel"]["initialState"],
        "gamma": cfg["gamma"],
        "tau0": cfg["internalization"]["tau0"],
        "tau1": cfg["internalization"]["tau1"],
        "tau2": cfg["internalization"]["tau2"],
    }
    for i, tier in enumerate(cfg["tiers"]):
        p = f"t{i}"
        values.update({
            f"{p}-enabled": ["on"] if tier["enabled"] else [],
            f"{p}-name": tier["name"],
            f"{p}-sizes": ", ".join(map(str, tier["sizes"])),
            f"{p}-a0": tier["flow"]["A0"],
            f"{p}-theta": tier["flow"]["theta"],
            f"{p}-beta": tier["flow"]["beta"],
            f"{p}-steep": tier["flow"]["steepness"],
            f"{p}-shift": tier["flow"]["shift"],
            f"{p}-vshift": tier["flow"]["volumeShift"],
            f"{p}-fee": tier["feeEurPerEurM"],
            f"{p}-markout-enabled": ["on"] if tier["useMarkout"] else [],
            f"{p}-markout-level": tier["markout"]["asymptoticEurPerEurM"],
            f"{p}-markout-tau": tier["markout"]["tauMinutes"],
            f"{p}-dmin": tier["deltaMin"], f"{p}-dmax": tier["deltaMax"],
        })
    dp = cfg["darkPool"]
    values.update({
        "dp-enabled": ["on"] if dp["enabled"] else [],
        "dp-lambda": dp["lambda"], "dp-mu": dp["mu"],
        "dp-p0": dp["p0"], "dp-fee": dp["feeEurPerEurM"],
        "dp-sizes": ", ".join(map(str, dp["postedSizes"])),
    })
    ecn = cfg["passiveEcn"]
    values.update({
        "ecn-enabled": ["on"] if ecn["enabled"] else [],
        "ecn-A": ecn["flow"]["A"], "ecn-k": ecn["flow"]["k"],
        "ecn-size-probs": list(ecn["tradeSizeProbabilities"]),
        "ecn-dmin": ecn["minDelta"], "ecn-dmax": ecn["maxDelta"],
        "ecn-size": ecn["quoteSize"], "ecn-fee": ecn["makerFeeEurPerEurM"],
    })
    return values


def test_ui_round_trip():
    actual = build_config(values_from_default())
    expected = default_config()
    expected["sigmaPips"] = float(expected["volatilityModel"]["sigmaStatesPips"][expected["volatilityModel"]["initialState"]])
    assert actual == expected


def test_historical_curve_parameters_restored():
    tiers = default_config()["tiers"]
    assert [tuple(t["flow"][k] for k in ("A0", "theta", "beta")) for t in tiers] == [
        (0.0155, 0.144, 0.0857),
        (0.0232, 0.303, 0.122),
        (1.0, 0.144, 0.0857),
    ]


@pytest.mark.parametrize("key, bad", [
    ("a0", -0.1), ("theta", -0.1), ("beta", -0.1),
    ("a0", float("nan")), ("beta", float("inf")),
])
def test_invalid_curve_rejected(key, bad):
    vals = values_from_default()
    vals[f"t0-{key}"] = bad
    with pytest.raises(ValueError):
        build_config(vals)


def test_only_knots_change_not_flow_curve_parameters():
    vals = values_from_default()
    vals["t0-sizes"] = "1, 5, 10"
    cfg = build_config(vals)
    assert cfg["tiers"][0]["sizes"] == [1, 5, 10]
    assert cfg["tiers"][0]["flow"] == default_config()["tiers"][0]["flow"]
    assert not any(k in cfg["tiers"][0]["flow"] for k in ("rfqSizes", "rfqProbabilities", "totalRfqRate"))


def test_single_knot_is_legal():
    vals = values_from_default()
    vals["t0-sizes"] = "1"
    assert build_config(vals)["tiers"][0]["sizes"] == [1.0]
