from trinity.defaults import default_config
from trinity.ui_config import build_config, checked


def ui_values_from_default():
    cfg = default_config()
    values = {
        "grid-mode": cfg["grid"]["mode"], "qmax": cfg["grid"]["maxAbs"], "qstep": cfg["grid"]["step"],
        "qfinehalf": cfg["grid"]["fineHalfWidth"], "qfinestep": cfg["grid"]["fineStep"], "qcoarsestep": cfg["grid"]["coarseStep"],
        "spot": cfg["spot"], "spread": cfg["spreadPips"], "drift": cfg["spotDrift"],
        "sigma": cfg["sigmaPips"], "gamma": cfg["gamma"],
        "tau0": cfg["internalization"]["tau0"], "tau1": cfg["internalization"]["tau1"], "tau2": cfg["internalization"]["tau2"],
    }
    for i, tier in enumerate(cfg["tiers"]):
        p = f"t{i}"
        values.update({
            f"{p}-enabled": ["on"] if tier["enabled"] else [],
            f"{p}-name": tier["name"],
            f"{p}-sizes": ", ".join(str(x) for x in tier["sizes"]),
            f"{p}-A0": tier["flow"]["A0"], f"{p}-theta": tier["flow"]["theta"], f"{p}-beta": tier["flow"]["beta"],
            f"{p}-steep": tier["flow"]["steepness"], f"{p}-shift": tier["flow"]["shift"], f"{p}-vshift": tier["flow"]["volumeShift"],
            f"{p}-markout-enabled": ["on"] if tier["useMarkout"] else [],
            f"{p}-impact": tier["markout"]["impactScalePips"], f"{p}-impact-beta": tier["markout"]["sizeExponent"],
            f"{p}-impact-tau": tier["markout"]["tauMinutes"], f"{p}-dmin": tier["deltaMin"], f"{p}-dmax": tier["deltaMax"],
        })
    dp = cfg["darkPool"]
    values.update({
        "dp-enabled": ["on"] if dp["enabled"] else [], "dp-dist": dp["distribution"],
        "dp-lb": dp["lambdaBid"], "dp-la": dp["lambdaAsk"], "dp-pb": dp["pBid"], "dp-pa": dp["pAsk"],
        "dp-mub": dp["muBid"], "dp-mua": dp["muAsk"], "dp-p0b": dp["p0Bid"], "dp-p0a": dp["p0Ask"],
        "dp-fb": dp["feeBid"], "dp-fa": dp["feeAsk"], "dp-sizes": ", ".join(str(x) for x in dp["postedSizes"]),
        "dp-both": ["on"] if dp["allowBothSides"] else [],
    })
    return values


def test_default_dash_values_round_trip_to_model_config():
    expected = default_config()
    actual = build_config(ui_values_from_default())
    assert actual == expected


def test_markout_checkbox_is_not_parsed_as_impact_scale():
    values = ui_values_from_default()
    values["t0-markout-enabled"] = ["on"]
    values["t0-impact"] = 1.25
    actual = build_config(values)
    assert actual["tiers"][0]["useMarkout"] is True
    assert actual["tiers"][0]["markout"]["impactScalePips"] == 1.25


def test_checked_dash_checklist_values():
    assert checked(["on"]) is True
    assert checked([]) is False
    assert checked(None) is False
