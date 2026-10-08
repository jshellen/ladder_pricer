import pytest
from trinity.defaults import default_config
from trinity.ui_config import build_config, checked


def ui_values_from_default():
    cfg = default_config()
    values = {
        "qmax": cfg["grid"]["maxAbs"],
        "spot": cfg["spot"], "spread": cfg["spreadPips"], "drift": cfg["spotDrift"],
        "vol-enabled": ["on"] if cfg["volatilityModel"]["enabled"] else [],
        "vol-initial-state": cfg["volatilityModel"]["initialState"],
        "gamma": cfg["gamma"],
        "tau0": cfg["internalization"]["tau0"], "tau1": cfg["internalization"]["tau1"], "tau2": cfg["internalization"]["tau2"],
    }
    for i, tier in enumerate(cfg["tiers"]):
        p = f"t{i}"
        values.update({
            f"{p}-enabled": ["on"] if tier["enabled"] else [],
            f"{p}-name": tier["name"],
            f"{p}-sizes": ", ".join(str(x) for x in tier["sizes"]),
            f"{p}-rfq-size-step": tier.get("rfqSizeStep", 1.0),
            f"{p}-A0": tier["flow"]["A0"],
            f"{p}-theta": tier["flow"]["theta"], f"{p}-beta": tier["flow"]["beta"],
            f"{p}-steep": tier["flow"]["steepness"], f"{p}-shift": tier["flow"]["shift"], f"{p}-vshift": tier["flow"]["volumeShift"],
            f"{p}-fee": tier.get("feeEurPerEurM", 0.0),
            f"{p}-markout-enabled": ["on"] if tier["useMarkout"] else [],
            f"{p}-markout-level": tier["markout"]["asymptoticEurPerEurM"],
            f"{p}-markout-tau": tier["markout"]["tauMinutes"], f"{p}-dmin": tier["deltaMin"], f"{p}-dmax": tier["deltaMax"],
        })
    dp = cfg["darkPool"]
    values.update({
        "dp-enabled": ["on"] if dp["enabled"] else [],
        "dp-lambda": dp["lambda"], "dp-mu": dp["mu"], "dp-p0": dp["p0"],
        "dp-fee": dp["feeEurPerEurM"], "dp-sizes": ", ".join(str(x) for x in dp["postedSizes"]),
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


def test_default_dash_values_round_trip_to_model_config():
    expected = default_config()
    actual = build_config(ui_values_from_default())
    assert actual == expected


def test_volatility_calibration_round_trips_from_ui():
    values = ui_values_from_default()
    cfg = build_config(values)
    vol = cfg["volatilityModel"]
    assert vol["enabled"] is True
    assert vol["initialState"] == 2
    assert len(vol["sigmaStatesPips"]) == 5
    assert len(vol["generatorPerMinute"]) == 5
    assert cfg["sigmaPips"] == pytest.approx(vol["sigmaStatesPips"][2])


def test_disabling_stochastic_vol_keeps_selected_sigma_as_legacy_scalar():
    values = ui_values_from_default()
    values["vol-enabled"] = []
    values["vol-initial-state"] = 4
    cfg = build_config(values)
    assert cfg["volatilityModel"]["enabled"] is False
    assert cfg["sigmaPips"] == pytest.approx(cfg["volatilityModel"]["sigmaStatesPips"][4])


def test_markout_checkbox_is_not_parsed_as_markout_level():
    values = ui_values_from_default()
    values["t0-markout-enabled"] = ["on"]
    values["t0-markout-level"] = 12.5
    actual = build_config(values)
    assert actual["tiers"][0]["useMarkout"] is True
    assert actual["tiers"][0]["markout"]["asymptoticEurPerEurM"] == 12.5


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
    assert cfg["tiers"][0]["feeEurPerEurM"] == 1.75


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
        "drift": 0.000012345,
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
        "t0-markout-level": 9.0125,
        "t0-markout-tau": 0.4875,
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
    assert cfg["tiers"][0]["flow"]["A0"] == 0.0155123
    assert cfg["tiers"][0]["flow"]["theta"] == 0.0144
    assert cfg["tiers"][0]["deltaMin"] == -10.125
    assert cfg["darkPool"]["lambda"] == 2.0125
    assert cfg["darkPool"]["mu"] == 2.0125
    assert cfg["darkPool"]["p0"] == 0.1125
    assert cfg["darkPool"]["feeEurPerEurM"] == 0.3125
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
    assert tier["rfqSizeStep"] == 1.0
    rfq_sizes = list(range(1, 21))
    weights = [z ** (-flow["theta"] - flow["beta"] * z) for z in rfq_sizes]
    probabilities = [w / sum(weights) for w in weights]
    assert sum(probabilities) == pytest.approx(1.0)

    from pathlib import Path
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert 'A0 · exogenous intensity scale' in app_source
    assert 'intensity curve itself is the density' in app_source
    assert 'RFQ size probabilities' not in app_source
    assert 'RFQ size step [EUR M]' in app_source
    assert 'prices between knots are linearly interpolated' in app_source

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
    assert 'if not _remove:\n            return (no_update,) * 20' in app_source
    assert 'if not _add:\n            return (no_update,) * 20' in app_source
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


def test_native_rfq_sampling_is_independent_of_pricing_knots():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    simulation = (root / "cpp/src/simulation.cpp").read_text()
    core = (root / "cpp/src/core.cpp").read_text()
    bindings = (root / "cpp/src/python_bindings.cpp").read_text()
    assert 'source.target_sizes().at(size_idx)' in simulation
    assert 'policy_delta_for_size(policy, rfq_side, rfq_size, q)' in simulation
    assert 'linear interpolation of delta is exactly linear interpolation of price' in simulation
    assert 'build_rfq_size_support' in core
    assert 'optional_number(tier, "rfqSizeStep", 1.0)' in bindings


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


def test_default_ecn_fee_is_three_eur_per_eurm():
    cfg = default_config()
    fee = cfg["passiveEcn"]["makerFeeEurPerEurM"]
    assert fee == pytest.approx(3.0)

def test_default_tier_fees_are_three_eur_per_eurm():
    cfg = default_config()
    assert [tier["feeEurPerEurM"] for tier in cfg["tiers"]] == pytest.approx([3.0, 3.0, 3.0])



def test_crossed_fx_sources_inherit_target_delta_and_do_not_configure_source_spread():
    """Crossed SEK sources are theoretical crosses of the target quote, so delta is 1:1."""
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    bindings_source = (Path(__file__).resolve().parents[1] / "cpp" / "src" / "python_bindings.cpp").read_text()
    assert 'number_input("Source spread [pips]"' not in app_source
    assert 'sourceSpreadPips' not in app_source
    assert 'delta_scale = 1.0;' in bindings_source
    assert 'crossed flow mapping requires positive crossMid' in bindings_source
    assert 'crossed ECN flow mapping requires positive crossMid' in bindings_source


def test_tier_markout_is_native_eur_per_eurm_and_size_independent():
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    defaults_source = (Path(__file__).resolve().parents[1] / "python" / "trinity" / "defaults.py").read_text()
    assert 'Show as EUR / EURm traded' not in app_source
    assert 'Size exponent' not in app_source
    assert 'impactScalePips' not in app_source
    assert 'sizeExponent' not in app_source
    assert 'Asymptotic markout [EUR / EURm traded]' in app_source
    assert 'Adverse markout [EUR / EURm traded]' in app_source
    assert 'asymptoticEurPerEurM' in defaults_source


def test_native_markout_boundary_uses_eur_per_eurm_and_no_size_scaling():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bindings = (root / "cpp" / "src" / "python_bindings.cpp").read_text()
    core_hpp = (root / "cpp" / "include" / "ladder_pricer" / "core.hpp").read_text()
    core_cpp = (root / "cpp" / "src" / "core.cpp").read_text()
    assert 'number(markout, "asymptoticEurPerEurM")' in bindings
    assert 'eur_per_eurm_to_price_units' in bindings
    assert 'impactScalePips' not in bindings
    assert 'sizeExponent' not in bindings
    assert 'double asymptotic() const noexcept' in core_hpp
    assert 'size_exponent' not in core_hpp
    assert 'markout().asymptotic(z)' not in core_cpp
    assert 'size_exponent' not in core_cpp
