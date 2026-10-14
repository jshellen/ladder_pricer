import pytest
from trinity.defaults import default_config
from trinity.ui_config import build_config, checked


def ui_values_from_default():
    cfg = default_config()
    values = {
        "qmax": cfg["grid"]["maxAbs"],
        "spot": cfg["spot"], "spread": cfg["spreadPips"],
        "sigma": cfg["sigmaPips"], "gamma": cfg["gamma"],
    }
    for i, tier in enumerate(cfg["tiers"]):
        p = f"t{i}"
        values.update({
            f"{p}-enabled": ["on"] if tier["enabled"] else [],
            f"{p}-name": tier["name"],
            f"{p}-sizes": ", ".join(str(x) for x in tier["sizes"]),
            f"{p}-A0": tier["flow"]["A0"],
            f"{p}-theta": tier["flow"]["theta"], f"{p}-beta": tier["flow"]["beta"],
            f"{p}-steep": tier["flow"]["steepness"], f"{p}-shift": tier["flow"]["shift"], f"{p}-vshift": tier["flow"]["volumeShift"],
            f"{p}-fee": tier.get("feePips", 0.0),
            f"{p}-markout-enabled": ["on"] if tier["useMarkout"] else [],
            f"{p}-markout-asymptote": tier["markout"]["asymptoticEurPerMillion"],
            f"{p}-markout-tau-seconds": tier["markout"]["tauSeconds"],
            f"{p}-dmin": tier["deltaMin"], f"{p}-dmax": tier["deltaMax"],
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
        "ecn-size-probs": list(ecn["tradeSizeProbabilities"]),
        "ecn-dmin": ecn["minDelta"], "ecn-dmax": ecn["maxDelta"],
        "ecn-size": ecn["quoteSize"], "ecn-fee": ecn["makerFeePips"],
    })
    return values


def test_default_dash_values_round_trip_to_model_config():
    expected = default_config()
    actual = build_config(ui_values_from_default())
    assert actual == expected


def test_tier_markout_curves_and_switches_are_independent():
    values = ui_values_from_default()
    values["t0-markout-enabled"] = ["on"]
    values["t1-markout-enabled"] = []
    values["t0-markout-asymptote"] = -60.25
    values["t0-markout-tau-seconds"] = 42.125
    values["t1-markout-asymptote"] = -20.0
    values["t1-markout-tau-seconds"] = 10.0
    actual = build_config(values)
    assert actual["tiers"][0]["useMarkout"] is True
    assert actual["tiers"][1]["useMarkout"] is False
    assert [t["markout"] for t in actual["tiers"]] == [
        {"asymptoticEurPerMillion": -60.25, "tauSeconds": 42.125},
        {"asymptoticEurPerMillion": -20.0, "tauSeconds": 10.0},
    ]
    assert "markout" not in actual  # no longer a global parameter


@pytest.mark.parametrize("key,value", [
    ("t0-markout-asymptote", 10.0),
    ("t1-markout-asymptote", float("nan")),
    ("t0-markout-tau-seconds", 0.0),
    ("t1-markout-tau-seconds", float("inf")),
])
def test_invalid_markout_config_rejected(key, value):
    values = ui_values_from_default()
    values[key] = value
    with pytest.raises(ValueError, match="Markout"):
        build_config(values)


def test_checked_dash_checklist_values():
    assert checked(["on"]) is True
    assert checked([]) is False
    assert checked(None) is False


def test_passive_ecn_uses_one_percent_delta_grid():
    cfg = build_config(ui_values_from_default())
    ecn = cfg["passiveEcn"]
    deltas = ecn["deltas"]
    assert ecn["minDelta"] == 0.0
    assert ecn["maxDelta"] == 0.5
    assert len(deltas) == 51
    assert deltas[0] == 0.0
    assert deltas[-1] == 0.5
    assert all(abs((b - a) - 0.01) < 1e-12 for a, b in zip(deltas, deltas[1:]))


def test_passive_ecn_delta_bounds_drive_grid():
    values = ui_values_from_default()
    values["ecn-dmin"] = 0.20
    values["ecn-dmax"] = 0.35
    cfg = build_config(values)
    assert cfg["passiveEcn"]["deltas"] == [round(0.20 + 0.01 * i, 10) for i in range(16)]


def test_passive_ecn_cannot_cross_through_mid():
    values = ui_values_from_default()
    values["ecn-dmax"] = 0.51
    with pytest.raises(ValueError, match="<= 0.5"):
        build_config(values)


def test_passive_ecn_bounds_must_align_to_one_percent_grid():
    values = ui_values_from_default()
    values["ecn-dmin"] = 0.005
    with pytest.raises(ValueError, match="0.01 grid"):
        build_config(values)


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
        'number_input("X-grid nodes (odd)",',
        'number_input("Native improvement rounds",',
        'number_input("Min delta",',
        'number_input("Max delta",',
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
        "sigma": 20.125,
        "gamma": 0.0144,
        "t0-A0": 0.0155123,
        "t0-theta": 0.0144,
        "t0-beta": 0.00857,
        "t0-steep": 8.42123,
        "t0-shift": 0.52125,
        "t0-vshift": 0.0026125,
        "t0-fee": 0.1375,
        "t0-markout-asymptote": -50.8125,
        "t0-markout-tau-seconds": 36.7125,
        "t0-dmin": -10.125,
        "t0-dmax": 99.875,
        "dp-lambda": 2.0125,
        "dp-mu": 2.0125,
        "dp-p0": 0.1125,
        "dp-fee": 0.3125,
        "ecn-A": 0.15125,
        "ecn-k": 0.24125,
        "ecn-size-probs": [0.30, 0.20, 0.15, 0.10, 0.25],
        "ecn-size": 1.125,
        "ecn-fee": 3.125,
    })
    cfg = build_config(values)
    assert cfg["spot"] == 11.51234567
    assert cfg["spreadPips"] == 19.375
    assert cfg["gamma"] == 0.0144
    assert cfg["tiers"][0]["markout"] == {"asymptoticEurPerMillion": -50.8125, "tauSeconds": 36.7125}
    assert cfg["tiers"][0]["flow"]["A0"] == 0.0155123
    assert cfg["tiers"][0]["flow"]["theta"] == 0.0144
    assert cfg["tiers"][0]["deltaMin"] == -10.125
    assert cfg["darkPool"]["lambda"] == 2.0125
    assert cfg["darkPool"]["mu"] == 2.0125
    assert cfg["darkPool"]["p0"] == 0.1125
    assert cfg["darkPool"]["feePips"] == 0.3125
    assert cfg["passiveEcn"]["flow"]["A"] == 0.15125
    assert cfg["passiveEcn"]["tradeSizeProbabilities"] == [0.30, 0.20, 0.15, 0.10, 0.25]
    assert cfg["passiveEcn"]["quoteSize"] == 1.125


def test_rfq_flow_curve_itself_defines_size_density():
    cfg = build_config(ui_values_from_default())
    flow = cfg["tiers"][0]["flow"]
    assert "A0" in flow
    assert "totalRfqRate" not in flow
    assert "sizeProbabilities" not in flow

    tier = cfg["tiers"][0]
    assert tier["sizes"] == [1, 2, 3, 5, 10, 20]
    assert "rfqSizeStep" not in tier
    rfq_sizes = tier["sizes"]
    weights = [z ** (-flow["theta"] - flow["beta"] * z) for z in rfq_sizes]
    probabilities = [w / sum(weights) for w in weights]
    assert sum(probabilities) == pytest.approx(1.0)

    from pathlib import Path
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert 'A0 · exogenous intensity scale' in app_source
    assert 'intensity curve itself is the density' in app_source
    assert 'RFQ size probabilities' not in app_source
    assert 'RFQ size step [EUR M]' not in app_source
    assert 'RFQ sizes are exactly the pricing knots.' not in app_source

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


def test_default_fx_mid_supports_direct_and_inverse_pairs():
    from trinity.defaults import DEFAULT_FX_MIDS, default_fx_mid

    assert default_fx_mid("EURUSD") == DEFAULT_FX_MIDS["EURUSD"]
    assert default_fx_mid("USDEUR") == pytest.approx(1.0 / DEFAULT_FX_MIDS["EURUSD"])
    assert default_fx_mid(None) is None
    assert default_fx_mid("BAD") is None


def test_flow_source_dynamic_buttons_ignore_mount_events():
    """Dynamic Dash controls mount with n_clicks=0; those are not user actions."""
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert 'if not _remove:\n            return (no_update,) * 14' in app_source
    assert 'if not _add:\n            return (no_update,) * 14' in app_source
    assert 'if not triggered_value:\n            return no_update' in app_source


def test_ecn_flow_source_ui_is_wired_for_crossed_sources():
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert 'id="open-ecn-flow-sources"' in app_source
    assert 'id="ecn-extra-flow-sources-store"' in app_source
    assert 'ecn["flowSources"] = [direct] + deepcopy(ecn_extras)' in app_source
    assert 'if not _remove:\n            return (no_update,) * 5' in app_source
    assert 'if not _add:\n            return (no_update,) * 5' in app_source


def test_fill_by_tier_plot_can_switch_between_trade_count_and_volume():
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert '"fill-tier-metric"' in app_source
    assert '{"label": "Number of trades", "value": "count"}' in app_source
    assert '{"label": "Total volume", "value": "volume"}' in app_source
    assert 'fill_aggregates = list(mc.get("fillAggregates", []))' in app_source
    assert 'value / fill_total if fill_total > 0.0 else 0.0' in app_source
    assert 'yaxis_title="Share of executed volume" if fill_metric == "volume" else "Share of trades"' in app_source
    assert 'title="Fills by tier and side · all simulated paths"' in app_source


def test_realized_rfq_hit_ratio_chart_uses_population_aggregates():
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert 'diagnostic_expander("RFQ hit ratios", graph("rfq-realized-hit-chart", 360))' in app_source
    assert 'list(mc.get("rfqAggregates", []))' in app_source
    assert 'wins = [int(row.get("wins", 0)) for row in rfq_aggregates]' in app_source
    assert 'requests = [int(row.get("requests", 0)) for row in rfq_aggregates]' in app_source
    assert 'rfq_hit.add_trace(go.Bar(' in app_source
    assert 'title="Realized RFQ hit ratio by size · all simulated paths"' in app_source
    assert 'yaxis_title="Won RFQs / total RFQs"' in app_source


def test_mc_rfq_validation_uses_all_path_population_aggregates_and_tier_selector():
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert 'diagnostic_expander("RFQ Monte Carlo validation", [' in app_source
    assert 'select("Tier", "mc-rfq-tier-select", [], None)' in app_source
    assert 'mc.get("rfqTierSizeAggregates", [])' in app_source
    assert 'mc.get("rfqDeltaAggregates", [])' in app_source
    assert 'mc.get("rfqInventoryAggregates", [])' in app_source
    assert 'requests_by_size.get(float(z), 0) / exposure' in app_source
    assert 'int(row.get("admissibleRequests", 0))' in app_source
    assert 'title=f"Implied hit ratios by inventory · {tier_name}"' in app_source


def test_mc_progress_bar_uses_string_html_attributes():
    """Dash html.Progress validates value/max as strings in the deployed UI version."""
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert 'html.Progress(id="mc-progress-bar", value="0", max="100"' in app_source
    assert 'str(percent), "Monte Carlo failed"' in app_source
    assert 'str(percent), message' in app_source
    assert '"100", "Monte Carlo complete"' in app_source


def test_mc_completed_job_is_retained_for_late_progress_polls():
    """Late dcc.Interval polls must not invalidate an already-completed MC result."""
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert '_MC_JOB_RETENTION_SECONDS = 300.0' in app_source
    assert '_prune_finished_mc_jobs_locked()' in app_source
    assert 'finishedAt=time.time()' in app_source
    assert '_MC_JOBS.pop(str(job_id), None)' not in app_source
    assert 'Returning the same completed payload is intentionally idempotent.' in app_source


def test_native_rfq_sampling_matches_pricing_knots():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    simulation = (root / "cpp/src/simulation.cpp").read_text()
    core = (root / "cpp/src/core.cpp").read_text()
    bindings = (root / "cpp/src/python_bindings.cpp").read_text()
    analytics = (root / "cpp/src/analytics.cpp").read_text()
    assert 'source.target_sizes().at(size_idx)' in simulation
    assert 'policy_delta(policy, rfq_side, size_idx, q)' in simulation
    assert 'policy_delta_for_size' not in simulation
    assert 'flow_.configure_target_sizes(sizes_)' in core
    assert 'build_rfq_size_support' not in core
    assert 'const std::vector<double>& target_sizes = pricing_sizes;' in bindings
    assert 'rfqSizeStep' not in bindings
    assert 'tier.sizes()[j]' in analytics
    assert 'interpolate_size_row' not in analytics


def test_rfq_total_rate_increases_with_added_pricing_knot():
    import pytest
    from trinity.defaults import default_config
    tier = default_config()["tiers"][0]
    source = tier["flow"]
    def two_sided_rate(sizes):
        return 2 * source["A0"] * sum(
            z ** (-source["theta"] - source["beta"] * z) for z in sizes)

    before = two_sided_rate(tier["sizes"])
    tier["sizes"] = [1, 2, 3, 4, 5, 10, 20]
    after = two_sided_rate(tier["sizes"])
    assert after > before
    assert after - before == pytest.approx(2 * source["A0"]
                                          * 4 ** (-source["theta"] - source["beta"] * 4))


def test_passive_ecn_trade_size_distribution_is_discrete_and_normalized():
    cfg = build_config(ui_values_from_default())
    probs = cfg["passiveEcn"]["tradeSizeProbabilities"]
    assert len(probs) == 5
    assert sum(probs) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_passive_ecn_trade_size_probabilities_must_sum_to_one():
    values = ui_values_from_default()
    values["ecn-size-probs"] = [0.4, 0.3, 0.2, 0.2, -0.1]
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_config(values)



def test_pair_specific_ecn_trade_size_defaults():
    from trinity.defaults import default_ecn_trade_size_probabilities

    assert default_ecn_trade_size_probabilities("EURSEK") == [0.35, 0.08, 0.22, 0.16, 0.19]
    assert default_ecn_trade_size_probabilities("USDSEK") == [0.34, 0.135, 0.216, 0.13, 0.179]
    assert abs(sum(default_ecn_trade_size_probabilities("EURSEK")) - 1.0) < 1e-15
    assert abs(sum(default_ecn_trade_size_probabilities("USDSEK")) - 1.0) < 1e-15


def test_mc_large_result_stays_server_side():
    """Completed MC payloads should not be serialized wholesale through dcc.Store."""
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert '_MC_RESULTS:' in app_source
    assert '_resolve_mc_store' in app_source
    assert '{"resultId": result_id}' in app_source
    assert 'compact_result=True' in app_source


def test_mc_inventory_quantiles_use_parallel_selection_not_full_sort():
    """Large runs must avoid the old O(sample_points * paths log paths) tail."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "cpp" / "src" / "simulation.cpp").read_text()
    assert 'percentile_select' in source
    assert 'std::nth_element' in source
    assert 'std::atomic<int> next_snapshot' in source
    assert 'std::sort(snapshot.begin(), snapshot.end())' not in source


def test_dashboard_defaults_only_two_tiers():
    cfg = build_config(ui_values_from_default())
    assert [x["name"] for x in cfg["tiers"]] == ["Tier 1", "Tier 2"]


def test_volatility_is_json_only_and_ignores_stale_dashboard_values():
    from trinity.defaults import default_fx_pair_volatility
    values = ui_values_from_default()
    values["volatility-model"] = {
        "sigmaPips": [10., 18., 33.],
        "ratesPerMinute": [[0., .12, .03], [.04, 0., .05], [.06, .07, 0.]],
    }
    cfg = build_config(values)
    assert cfg["volatilityRegimes"] == default_fx_pair_volatility("EURSEK")
    assert cfg["sigmaPips"] == 20.


@pytest.mark.parametrize("model", [
    {"sigmaPips": [21., 18.], "ratesPerMinute": [[0, .12], [.04, 0]]},
    {"sigmaPips": [10., 18.], "ratesPerMinute": [[0, .12], [-.1, 0]]},
    {"sigmaPips": [10., 18., 33.], "ratesPerMinute": [[0., .12, float("nan")], [.04, 0, .05], [.06, .07, 0]]},
    {"sigmaPips": [10., 18.], "ratesPerMinute": [[0, 0], [.1, 0]]},
    {"sigmaPips": [10., 18., 33.], "ratesPerMinute": [[0., .1], [.2, 0.]]},
    {"sigmaPips": [10., 18.], "ratesPerMinute": [[-.1, .1], [.1, 0.]]},
])
def test_invalid_json_volatility_models_rejected(model):
    from trinity.defaults import _validated_volatility_regimes
    with pytest.raises(ValueError, match="Volatility|volatility"):
        _validated_volatility_regimes(model, "TESTPAIR")


def test_pair_volatility_json_override_is_isolated_and_supports_two_regimes(monkeypatch):
    import trinity.defaults as defaults
    from trinity.defaults import default_fx_pair_volatility
    import json
    from pathlib import Path
    raw = json.loads((Path(__file__).resolve().parents[1] / "python/trinity/fx_pair_defaults.json").read_text())
    assert default_fx_pair_volatility("EURSEK") == raw["volatilityRegimes"]
    two = {"sigmaPips": [10., 30.], "ratesPerMinute": [[0., .05], [.1, 0.]]}
    monkeypatch.setitem(defaults.DEFAULT_FX_PAIR_VOLATILITY, "GBPSEK", two)
    values = ui_values_from_default()
    values["target-pair"] = "GBPSEK"
    assert build_config(values)["volatilityRegimes"] == two
    copy = default_fx_pair_volatility("GBPSEK")
    copy["sigmaPips"][0] = -100
    assert default_fx_pair_volatility("GBPSEK") == two
    assert default_fx_pair_volatility("EURSEK") == raw["volatilityRegimes"]


def test_volatility_per_pair_override_is_loaded_from_json(tmp_path):
    from trinity.defaults import _load_pair_volatility_defaults
    from pathlib import Path
    import json
    original = json.loads((Path(__file__).resolve().parents[1] / "python/trinity/fx_pair_defaults.json").read_text())
    original["pairs"]["GBPSEK"]["volatilityRegimes"] = {
        "sigmaPips": [10., 30.],
        "ratesPerMinute": [[0., 0.08], [0.04, 0.]],
    }
    path = tmp_path / "fx_pair_defaults.json"
    path.write_text(json.dumps(original))
    models = _load_pair_volatility_defaults(path)
    assert models["GBPSEK"]["sigmaPips"] == [10., 30.]
    assert models["EURSEK"] == original["volatilityRegimes"]
    assert all(2 <= len(model["sigmaPips"]) <= 3 for model in models.values())


def test_pair_market_defaults_read_json_without_mutating_cached_data():
    from trinity.defaults import default_fx_pair_market_data
    from pathlib import Path
    import json
    path = Path(__file__).resolve().parents[1] / "python/trinity/fx_pair_defaults.json"
    raw = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    assert default_fx_pair_market_data("GBPSEK") == raw["GBPSEK"]
    assert default_fx_pair_market_data("EURUSD") == raw["EURUSD"]
    result = default_fx_pair_market_data("GBPSEK")
    result["spreadPips"] = 999.0
    assert default_fx_pair_market_data("GBPSEK") == raw["GBPSEK"]
    assert default_fx_pair_market_data("MISSING") is None


def test_pricing_knot_edits_propagate_into_solver_config():
    values = ui_values_from_default()
    values["t0-sizes"] = "1, 3, 7, 12"
    values["t1-sizes"] = "2, 4, 8"
    cfg = build_config(values)
    assert cfg["tiers"][0]["sizes"] == [1.0, 3.0, 7.0, 12.0]
    assert cfg["tiers"][1]["sizes"] == [2.0, 4.0, 8.0]
