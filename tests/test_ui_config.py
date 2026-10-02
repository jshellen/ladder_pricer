from trinity.defaults import default_config
from trinity.ui_config import build_config, checked


def ui_values_from_default():
    cfg = default_config()
    values = {
        "qmax": cfg["grid"]["maxAbs"],
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
            f"{p}-fee": tier.get("feePips", 0.0),
            f"{p}-markout-enabled": ["on"] if tier["useMarkout"] else [],
            f"{p}-impact": tier["markout"]["impactScalePips"], f"{p}-impact-beta": tier["markout"]["sizeExponent"],
            f"{p}-impact-tau": tier["markout"]["tauMinutes"], f"{p}-dmin": tier["deltaMin"], f"{p}-dmax": tier["deltaMax"],
        })
    dp = cfg["darkPool"]
    values.update({
        "dp-enabled": ["on"] if dp["enabled"] else [],
        "dp-lambda": dp["lambda"], "dp-mu": dp["mu"], "dp-p0": dp["p0"],
        "dp-fee": dp["feePips"], "dp-sizes": ", ".join(str(x) for x in dp["postedSizes"]),
    })
    ecn = cfg["passiveEcn"]
    values.update({
        "ecn-enabled": ["on"] if ecn["enabled"] else [],
        "ecn-A": ecn["flow"]["A"], "ecn-k": ecn["flow"]["k"],
        "ecn-dmin": ecn["minDistancePips"], "ecn-dmax": ecn["maxDistancePips"],
        "ecn-size": ecn["quoteSize"], "ecn-fee": ecn["makerFeePips"],
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


def test_passive_ecn_uses_tenth_pip_distance_grid():
    cfg = build_config(ui_values_from_default())
    ecn = cfg["passiveEcn"]
    deltas = ecn["deltas"]
    assert ecn["minDistancePips"] == 0.0
    assert ecn["maxDistancePips"] == 20.0
    assert len(deltas) == 201
    assert deltas[0] == 0.0
    assert deltas[-1] == 20.0
    assert all(abs((b - a) - 0.1) < 1e-12 for a, b in zip(deltas, deltas[1:]))


def test_passive_ecn_distance_bounds_drive_grid():
    values = ui_values_from_default()
    values["ecn-dmin"] = 2.0
    values["ecn-dmax"] = 5.0
    cfg = build_config(values)
    assert cfg["passiveEcn"]["deltas"] == [round(2.0 + 0.1 * i, 10) for i in range(31)]


def test_inventory_grid_is_fixed_uniform_one_million():
    values = ui_values_from_default()
    values["qmax"] = 25
    cfg = build_config(values)
    assert cfg["grid"] == {"mode": "uniform", "maxAbs": 25.0, "step": 1.0}


def test_tier_fee_round_trips_from_ui():
    values = ui_values_from_default()
    values["t0-fee"] = 1.75
    cfg = build_config(values)
    assert cfg["tiers"][0]["feePips"] == 1.75


def test_tier_theta_accepts_four_decimal_precision():
    values = ui_values_from_default()
    values["t0-theta"] = 0.0144
    cfg = build_config(values)
    assert cfg["tiers"][0]["flow"]["theta"] == 0.0144


def test_continuous_dash_number_inputs_do_not_impose_step_grid():
    """HTML number-input step grids must not reject otherwise-valid model decimals."""
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    allowed_discrete = (
        'number_input("Max |q| [M]",',
        'number_input("Min distance from mid [pips]",',
        'number_input("Max distance from mid [pips]",',
        'number_input("Paths",',
        'number_input("Random seed",',
        'number_input("Points",',
    )
    offenders = []
    for line in app_source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("number_input("):
            continue
        if '"any"' in stripped:
            continue
        if any(token in stripped for token in allowed_discrete):
            continue
        offenders.append(stripped)
    assert offenders == []


def test_continuous_parameters_accept_extra_decimal_precision():
    values = ui_values_from_default()
    values.update({
        "spot": 11.51234567,
        "spread": 19.375,
        "drift": 0.000012345,
        "sigma": 20.125,
        "gamma": 0.0144,
        "tau0": 4.0125,
        "tau1": 0.00703125,
        "tau2": 0.00084375,
        "t0-A0": 0.0155123,
        "t0-theta": 0.0144,
        "t0-beta": 0.00857,
        "t0-steep": 8.42123,
        "t0-shift": 0.52125,
        "t0-vshift": 0.0026125,
        "t0-fee": 0.1375,
        "t0-impact": 1.0125,
        "t0-impact-beta": 0.5125,
        "t0-impact-tau": 0.4875,
        "t0-dmin": -10.125,
        "t0-dmax": 99.875,
        "dp-lambda": 2.0125,
        "dp-mu": 2.0125,
        "dp-p0": 0.1125,
        "dp-fee": 0.3125,
        "ecn-A": 0.15125,
        "ecn-k": 0.24125,
        "ecn-size": 1.125,
        "ecn-fee": 3.125,
    })
    cfg = build_config(values)
    assert cfg["spot"] == 11.51234567
    assert cfg["spreadPips"] == 19.375
    assert cfg["gamma"] == 0.0144
    assert cfg["tiers"][0]["flow"]["A0"] == 0.0155123
    assert cfg["tiers"][0]["flow"]["theta"] == 0.0144
    assert cfg["tiers"][0]["deltaMin"] == -10.125
    assert cfg["darkPool"]["lambda"] == 2.0125
    assert cfg["darkPool"]["mu"] == 2.0125
    assert cfg["darkPool"]["p0"] == 0.1125
    assert cfg["darkPool"]["feePips"] == 0.3125
    assert cfg["passiveEcn"]["flow"]["A"] == 0.15125
    assert cfg["passiveEcn"]["quoteSize"] == 1.125


def test_dark_pool_is_zero_inflated_poisson_only():
    cfg = build_config(ui_values_from_default())
    dp = cfg["darkPool"]
    assert "distribution" not in dp
    assert "pBid" not in dp and "pAsk" not in dp

    from pathlib import Path
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert 'id="dp-dist"' not in app_source
    assert '"dp-dist"' not in app_source
    assert '"dp-pb"' not in app_source
    assert '"dp-pa"' not in app_source
