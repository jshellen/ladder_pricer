"""Check that policy diagnostics select sigma locally, without touching the HJB solver."""
from __future__ import annotations

import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path

import pytest


class Component:
    def __init__(self, *children, **attrs):
        self.children = children[0] if children else attrs.pop("children", None)
        for key, val in attrs.items():
            setattr(self, key, val)


class ComponentFactory:
    def __getattr__(self, name):
        def create(*children, **attrs):
            node = Component(*children, **attrs)
            node.type = name
            return node
        return create


class Binding:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Input(Binding):
    pass


class Output(Binding):
    pass


class State(Binding):
    pass


class MockDash:
    def __init__(self, *args, **kwargs):
        self.callbacks = []
        self.server = None

    def callback(self, *args, **kwargs):
        def register(fn):
            self.callbacks.append((args, kwargs, fn))
            return fn
        return register


def walk(value):
    if isinstance(value, Component):
        yield value
        yield from walk(value.children)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from walk(item)


@pytest.fixture(scope="module")
def dashboard():
    """Construct the complete layout/callback set even without Dash installed."""
    mock_dash = types.ModuleType("dash")
    mock_dash.ALL = object()
    mock_dash.Dash = MockDash
    mock_dash.Input, mock_dash.Output, mock_dash.State = Input, Output, State
    mock_dash.ctx = types.SimpleNamespace(triggered_id=None, triggered=[])
    mock_dash.dcc = ComponentFactory()
    mock_dash.html = ComponentFactory()
    mock_dash.no_update = object()
    actual_dash = sys.modules.get("dash")
    sys.modules["dash"] = mock_dash
    try:
        path = Path(__file__).resolve().parents[1] / "app.py"
        spec = importlib.util.spec_from_file_location("_ladder_dashboard_regime_test_app", path)
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        yield app
    finally:
        if actual_dash is None:
            sys.modules.pop("dash", None)
        else:
            sys.modules["dash"] = actual_dash


CHART_REGIMES = {
    "value-chart": "overview-value-regime",
    "implied-hit-chart": "tier-implied-hit-regime",
    "quote-inventory-chart": "tier-quote-inventory-regime",
    "ladder-chart": "tier-ladder-regime",
    "dp-policy-chart": "dark-policy-regime",
    "ecn-implied-hit-chart": "ecn-implied-hit-regime",
    "ecn-quote-inventory-chart": "ecn-quote-inventory-regime",
    "ecn-policy-chart": "ecn-policy-regime",
}

REGIME_INDEPENDENT = (
    "rfq-arrival-chart", "flow-chart", "hit-ratio-chart", "markout-chart",
    "dp-arrival-chart", "dp-full-fill-chart", "ecn-fill-chart",
)


def test_each_regime_dependent_expander_has_its_own_selector(dashboard):
    ids = [node.id if isinstance(node.id, str) else repr(node.id)
           for node in walk(dashboard.app.layout) if hasattr(node, "id")]
    assert all(count == 1 for count in Counter(ids).values())
    assert set(dashboard.POLICY_REGIME_SELECTOR_IDS) == set(CHART_REGIMES.values())
    assert set(CHART_REGIMES) <= set(ids)

    policy_expanders = [node for node in walk(dashboard.policy_tab) if node.type == "Details"]
    for graph_id, selector_id in CHART_REGIMES.items():
        matching = [node for node in policy_expanders if graph_id in {
            getattr(c, "id", None) for c in walk(node)
        }]
        assert len(matching) == 1, graph_id
        local_ids = {getattr(c, "id", None) for c in walk(matching[0])}
        assert local_ids & set(dashboard.POLICY_REGIME_SELECTOR_IDS) == {selector_id}

    for graph_id in REGIME_INDEPENDENT:
        matching = [node for node in policy_expanders if graph_id in {
            getattr(c, "id", None) for c in walk(node)
        }]
        assert len(matching) == 1, graph_id
        local_ids = {getattr(c, "id", None) for c in walk(matching[0])}
        assert not (local_ids & set(dashboard.POLICY_REGIME_SELECTOR_IDS)), graph_id

    assert all(x not in ids for x in ("m-residual", "residual-chart", "value-change-chart"))
    assert all(x not in ids for x in ("overview-regime", "tier-regime", "ecn-regime", "dark-regime"))
    assert len(dashboard.regime_dropdown_options(None)) == 2 * len(CHART_REGIMES)
    assert "expected-mtm-chart" not in ids
    assert "tier-expected-mtm-regime" not in ids
    assert not hasattr(dashboard, "expected_mtm_figure")


def mock_solution():
    return {
        "qGrid": [-1.0, 0.0, 1.0],
        "valueByRegime": [[i, i+1, i+2] for i in (12, 20, 35)],
        "regimes": [
            {"sigmaPips": sigma, "tiers": [{"name": f"Tier-{sigma}"}],
             "passiveEcn": {}, "darkPool": {}, "expectedMtm": []}
            for sigma in (12, 20, 35)
        ],
        "tiers": [{"name": "Tier 1"}],
        "converged": True, "iterations": 3, "averageReward": 123.4,
    }


def test_regime_options_and_overview_are_regime_specific(dashboard):
    solution = mock_solution()
    count = len(CHART_REGIMES)
    results = dashboard.regime_dropdown_options(solution)
    assert len(results) == 2 * count
    assert results[count:] == (1,) * count
    assert all(len(options) == 3 for options in results[:count])
    assert all("σ=35" in options[2]["label"] for options in results[:count])
    assert list(dashboard.render_continuation_value(solution, 0).data[0].y) == [12, 13, 14]
    assert list(dashboard.render_continuation_value(solution, 2).data[0].y) == [35, 36, 37]
    summary = dashboard.render_solution_summary(solution)
    assert summary[:3] == ("Yes", "3", "123.4")
    assert len(summary) == 7


def test_tier_policy_expanders_are_independent(dashboard, monkeypatch):
    solution = mock_solution()
    monkeypatch.setattr(dashboard, "_tier_cfg", lambda *_: {"name": "Tier 1"})
    static_funcs = ("exogenous_arrival_figure", "flow_curve_figure", "hit_ratio_figure",
                    "markout_figure", "flow_parameter_table")
    for name in static_funcs:
        monkeypatch.setattr(dashboard, name, lambda *args: "static")
    policy_funcs = ("implied_hit_ratio_figure", "quote_inventory_figure",
                    "ladder_figure", "ladder_table")
    for name in policy_funcs:
        monkeypatch.setattr(dashboard, name, lambda selected, *args: selected["selectedSigmaPips"])
    output = dashboard.render_tier_diagnostics(
        solution, {"spreadPips": 20.0}, 0, 0.0, 0, 2, 2,
    )
    assert [output[i] for i in (3, 5, 6, 8)] == [12, 35, 35, 35]
    assert [output[i] for i in (0, 1, 2, 4, 7)] == ["static"] * 5


def test_ecn_and_darkpool_expanders_are_independent(dashboard, monkeypatch):
    solution = mock_solution()
    for name in ("passive_ecn_fill_figure", "passive_ecn_parameter_table"):
        monkeypatch.setattr(dashboard, name, lambda *args: "static")
    for name in ("passive_ecn_implied_hit_ratio_figure", "passive_ecn_quote_inventory_figure",
                 "passive_ecn_policy_figure", "passive_ecn_table"):
        monkeypatch.setattr(dashboard, name, lambda selected, *args: selected["selectedSigmaPips"])
    ecn = dashboard.render_passive_ecn_diagnostics(solution, {"passiveEcn": {"enabled": True}}, 2, 0, 1)
    assert ecn == ("static", 35, 12, 20, "static", 20)

    for name in ("dark_pool_arrival_figure", "dark_pool_full_fill_figure"):
        monkeypatch.setattr(dashboard, name, lambda *args: "static")
    for name in ("dark_pool_policy_figure", "dark_pool_table"):
        monkeypatch.setattr(dashboard, name, lambda selected: selected["selectedSigmaPips"])
    dark = dashboard.render_dark_pool_diagnostics(solution, {"darkPool": {"enabled": True}}, 2)
    assert dark == ("static", "static", 35, 35)


def test_each_callback_output_has_a_unique_target(dashboard):
    outputs = [arg.args for args, _, _ in dashboard.app.callbacks for arg in args if isinstance(arg, Output)]
    # The pre-existing dashboard has intentional duplicate outputs in modals and
    # Monte Carlo (with allow_duplicate). Our new policy controls must not add any.
    policy_ids = {"m-converged", "m-iterations", "m-rho", "value-chart",
                  "tier-select", "ladder-q"} | set(CHART_REGIMES) | set(CHART_REGIMES.values())
    duplicates = [val for val, count in Counter(outputs).items()
                  if count > 1 and val[0] in policy_ids]
    assert duplicates == []
    # Every local selector participates in the shared option-population callback
    # and at least one figure callback; no dangling control is left after removal.
    inputs = {arg.args for args, _, _ in dashboard.app.callbacks for arg in args
              if isinstance(arg, Input) and isinstance(arg.args[0], str)}
    for selector_id in dashboard.POLICY_REGIME_SELECTOR_IDS:
        assert (selector_id, "value") in inputs
        assert (selector_id, "options") in outputs
        assert (selector_id, "value") in outputs


def test_channel_enabled_labels_are_scoped_white(dashboard):
    """Only visible Tier/Dark pool/ECN main enable toggles get white text."""
    ids = ["t0-enabled", "t1-enabled", "dp-enabled", "ecn-enabled"]
    for toggle_id in ids:
        toggles = [n for n in walk(dashboard.app.layout) if getattr(n, "id", None) == toggle_id]
        assert len(toggles) == 1, toggle_id
        toggle = toggles[0]
        assert "channel-enable-checkbox" in toggle.className
        assert toggle.labelStyle["color"] == "#ffffff"
        assert len(toggle.options) == 1
        option_label = toggle.options[0]["label"]
        assert isinstance(option_label, Component)
        assert option_label.type == "Span"
        assert option_label.children == "Enabled"
        assert option_label.className == "channel-enable-label"
        assert option_label.style["color"] == "#ffffff"
        assert option_label.style["WebkitTextFillColor"] == "#ffffff"

    # Keep independent modal and hidden-markout enable controls untouched.
    for toggle_id in ["t0-markout-enabled", "t1-markout-enabled", "flow-edit-markout-enabled"]:
        toggles = [n for n in walk(dashboard.app.layout) if getattr(n, "id", None) == toggle_id]
        if toggles:
            assert "channel-enable-checkbox" not in toggles[0].className


def test_policy_selector_text_color_scoped_to_regimes_and_tiers(dashboard):
    """Check all 11 regime selectors and both Tier selectors use dark text."""
    control_ids = (*dashboard.POLICY_REGIME_SELECTOR_IDS, "tier-select", "mc-rfq-tier-select")
    for cid in control_ids:
        controls = [n for n in walk(dashboard.app.layout) if getattr(n, "id", None) == cid]
        assert len(controls) == 1, cid
        assert "policy-selector-dropdown" in controls[0].className
        assert controls[0].placeholder
    # The ladder inventory selection is another white-surface dropdown and
    # must therefore share the same dark-text styling.
    ladder = [n for n in walk(dashboard.app.layout) if getattr(n, "id", None) == "ladder-q"]
    assert len(ladder) == 1
    assert "policy-selector-dropdown" in ladder[0].className
    for other_id in ("path-select", "path-size-select", "global-target-pair"):
        controls = [n for n in walk(dashboard.app.layout) if getattr(n, "id", None) == other_id]
        assert len(controls) == 1, other_id
        assert "policy-selector-dropdown" not in controls[0].className

    css = (Path(__file__).resolve().parents[1] / "assets" / "style.css").read_text()
    assert ".channel-enable-checkbox label" in css
    assert "color: #ffffff !important" in css
    assert ".policy-selector-field button.dash-dropdown .dash-dropdown-value-item" in css
    assert ".policy-selector-field .dash-dropdown-option" in css
    assert "Dash 4 renders className on the BUTTON itself" in css
    assert "-webkit-text-fill-color: #111821 !important" in css


def test_json_only_volatility_matrix_for_solve_and_analytics(dashboard):
    from trinity.defaults import default_fx_pair_volatility
    from trinity.ui_config import build_config
    from test_ui_config import ui_values_from_default

    ids = [getattr(node, "id", None) for node in walk(dashboard.app.layout)]
    assert "volatility-model-store" not in ids
    assert "flow-vol-count" not in ids
    assert not any(isinstance(x, str) and x.startswith("flow-vol-") for x in ids)
    keys = [key for key, _ in dashboard.CONFIG_FIELDS]
    assert "volatility-model" not in keys

    values = ui_values_from_default()
    for pair in ("EURSEK", "GBPSEK", "EURUSD"):
        values["target-pair"] = pair
        # A stale, forged UI volatility model must have no effect on Solve.
        values["volatility-model"] = {"sigmaPips": [999, 1000], "ratesPerMinute": [[0, .1], [.1, 0]]}
        cfg = build_config(values)
        assert cfg["volatilityRegimes"] == default_fx_pair_volatility(pair)
        assert cfg["targetPair"] == pair
        solution = {"regimes": [{"sigmaPips": x} for x in cfg["volatilityRegimes"]["sigmaPips"]],
                    "sigmaPips": cfg["volatilityRegimes"]["sigmaPips"]}
        choices, index = dashboard.mc_volatility_regime_options(solution)
        assert len(choices) == len(cfg["volatilityRegimes"]["sigmaPips"])
        assert 0 <= index < len(choices)


def test_modal_shows_readonly_json_volatility_values_and_starts_collapsed(dashboard):
    from trinity.defaults import default_fx_pair_volatility
    source = dashboard._direct_flow_source(dashboard.DEFAULT["tiers"][0]["flow"], "EURSEK")
    editor = dashboard.flow_source_settings_editor("direct", source, "EURSEK")
    sections = [n for n in walk(editor) if getattr(n, "type", None) == "Details"]
    assert len(sections) == 3
    assert all(n.open is False for n in sections)
    assert [next(n for n in walk(section) if getattr(n, "type", None) == "Summary").children
            for section in sections] == [
        "RFQ flow intensities", "Volatility", "Markouts"]
    volatility_section = sections[1]
    assert not any(getattr(n, "type", None) in ("Input", "Dropdown") for n in walk(volatility_section))
    tables = [n for n in walk(volatility_section) if getattr(n, "type", None) == "Table"]
    assert len(tables) == 2
    generator = dashboard.volatility_generator_matrix(**{
        "sigma": default_fx_pair_volatility("EURSEK")["sigmaPips"],
        "rates": default_fx_pair_volatility("EURSEK")["ratesPerMinute"],
    })
    assert generator.shape == (3, 3)
    assert len([n for n in walk(tables[1]) if getattr(n, "type", None) == "Tr"]) == 4


def example_ecn_regimes(sigmas=(12, 35)):
    """Tiny solved shape with real active/inactive ECN quotes in each regime."""
    regimes = []
    for i, sigma in enumerate(sigmas):
        regimes.append({
            "sigmaPips": sigma,
            "tiers": [], "darkPool": {}, "expectedMtm": [],
            "passiveEcn": {
                "qGrid": [-1.0, 0.0, 1.0],
                "bidActive": [True, False, False],
                "askActive": [False, False, True],
                "bidDelta": [0.2 + i * 0.1, 0.0, 0.0],
                "askDelta": [0.0, 0.0, 0.3 + i * 0.1],
            },
        })
    return {"regimes": regimes, "valueByRegime": [[0.0, 0.0, 0.0] for _ in regimes]}


@pytest.mark.parametrize("sigmas", [(12.0, 35.0), (10.0, 20.0, 30.0)])
@pytest.mark.parametrize("metric", ["delta", "quotes", "intensity"])
def test_ecn_heatmap_uses_every_regime_and_masks_inactive_quotes(dashboard, monkeypatch, sigmas, metric):
    import math

    monkeypatch.setattr(dashboard, "_ecn_arrival_rate", lambda cfg, delta: delta * 10.0)
    result = dashboard.passive_ecn_regime_heatmap_figure(
        example_ecn_regimes(sigmas), {"spreadPips": 20.0, "passiveEcn": {"minDelta": 0.0, "maxDelta": 0.5}}, metric,
    )
    assert len(result.data) == 2
    assert all(trace.type == "heatmap" and trace.coloraxis == "coloraxis" for trace in result.data)
    bid, ask = result.data
    assert bid.z.shape == ask.z.shape == (len(sigmas), 3)
    assert list(bid.x) == [-1, 0, 1]
    assert all(f"σ={sigma:g}" in " ".join(bid.y) for sigma in sigmas)
    for i in range(len(sigmas)):
        assert math.isnan(float(bid.z[i, 1]))
        assert math.isnan(float(bid.z[i, 2]))
        assert math.isnan(float(ask.z[i, 0]))
        assert math.isnan(float(ask.z[i, 1]))
        if metric == "delta":
            assert bid.z[i, 0] == pytest.approx(0.2 + i * 0.1)
            assert ask.z[i, 2] == pytest.approx(0.3 + i * 0.1)
        elif metric == "quotes":
            assert bid.z[i, 0] == pytest.approx(-20 * (0.5 - (0.2 + i * 0.1)))
            assert ask.z[i, 2] == pytest.approx(20 * (0.5 - (0.3 + i * 0.1)))
        else:
            assert bid.z[i, 0] == pytest.approx((0.2 + i * 0.1) * 10)
            assert ask.z[i, 2] == pytest.approx((0.3 + i * 0.1) * 10)
    assert "all volatility regimes" in result.layout.title.text
    assert result.layout.height == 610


def test_ecn_heatmap_toggles_hide_only_their_own_regime_selector(dashboard):
    ids = {getattr(node, "id", None): node for node in walk(dashboard.policy_tab)
           if isinstance(getattr(node, "id", None), str)}
    for prefix in ("ecn-implied-hit", "ecn-quote-inventory", "ecn-policy"):
        toggle = ids[f"{prefix}-heatmap"]
        assert toggle.options[0]["value"] == "heatmap"
        assert toggle.value == []
        assert ids[f"{prefix}-regime-wrapper"]

    active = dashboard.toggle_ecn_regime_heatmaps(["heatmap"], [], ["heatmap"])
    assert active[:3] == ({"display": "none"}, {}, {"display": "none"})
    assert [d["height"] for d in active[3:6]] == ["610px", "420px", "610px"]
    assert active[6] == {"display": "none"}
    off = dashboard.toggle_ecn_regime_heatmaps([], [], [])
    assert off[:3] == ({}, {}, {})
    assert off[6] == {}


def test_ecn_policy_heatmap_does_not_depend_on_regime_selector(dashboard, monkeypatch):
    solution = example_ecn_regimes()
    monkeypatch.setattr(dashboard, "passive_ecn_fill_figure", lambda cfg: "static")
    monkeypatch.setattr(dashboard, "passive_ecn_parameter_table", lambda cfg: "static")
    for name in ("passive_ecn_implied_hit_ratio_figure", "passive_ecn_quote_inventory_figure", "passive_ecn_policy_figure"):
        monkeypatch.setattr(dashboard, name, lambda selected, *args: selected["selectedSigmaPips"])
    monkeypatch.setattr(dashboard, "passive_ecn_table", lambda selected: selected["selectedSigmaPips"])
    monkeypatch.setattr(dashboard, "passive_ecn_regime_heatmap_figure", lambda solved, cfg, metric: f"heatmap {metric}")
    cfg = {"passiveEcn": {"enabled": True}}
    result = dashboard.render_passive_ecn_diagnostics(solution, cfg, 1, 0, 1, [], [], ["heatmap"])
    assert result == ("static", 35, 12, "heatmap delta", "static", 35)
    result = dashboard.render_passive_ecn_diagnostics(solution, cfg, 1, 0, 1, ["heatmap"], ["heatmap"], [])
    assert result == ("static", "heatmap intensity", "heatmap quotes", 35, "static", 35)


@pytest.mark.parametrize("regime_count", [2, 3])
def test_ecn_heatmap_accepts_actual_joint_solver_output(dashboard, regime_count):
    from trinity.defaults import default_config
    from trinity.engine import Engine
    cfg = default_config()
    cfg["grid"]["maxAbs"] = 4.0
    for tier in cfg["tiers"]:
        tier["sizes"] = [1.0, 2.0]
    if regime_count == 2:
        cfg["volatilityRegimes"] = {
            "sigmaPips": [12.0, 35.0],
            "ratesPerMinute": [[0.0, 0.12], [0.08, 0.0]],
        }
    solved = Engine(cfg).solve()
    for metric in ("delta", "quotes", "intensity"):
        fig = dashboard.passive_ecn_regime_heatmap_figure(solved, cfg, metric)
        assert len(fig.data) == 2
        assert len(fig.data[0].z) == regime_count
        assert len(fig.data[0].z[0]) == len(solved["qGrid"])
        # In the hedge-only ECN, both sides are off at flat inventory.
        q_zero = solved["qGrid"].index(0.0)
        assert all(__import__("math").isnan(float(trace.z[regime][q_zero]))
                   for trace in fig.data for regime in range(regime_count))


def test_ecn_heatmap_toggle_has_short_label_and_checkbox_spacing(dashboard):
    for toggle_id in ("ecn-implied-hit-heatmap", "ecn-quote-inventory-heatmap", "ecn-policy-heatmap"):
        matches = [n for n in walk(dashboard.app.layout) if getattr(n, "id", None) == toggle_id]
        assert len(matches) == 1
        toggle = matches[0]
        assert toggle.options[0]["value"] == "heatmap"
        assert toggle.options[0]["label"].children == "Heatmap"
        assert toggle.inputStyle["marginRight"] == "12px"
    css = (Path(__file__).resolve().parents[1] / "assets" / "style.css").read_text()
    assert ".ecn-heatmap-toggle input" in css
    assert "margin: 0 12px 0 0 !important" in css


def test_ladder_inventory_picker_uses_readable_light_dropdown(dashboard):
    """The inventory value (e.g. 0M) should not be white on white in Dash 4."""
    picker = [node for node in walk(dashboard.policy_tab)
              if getattr(node, "id", None) == "ladder-q"]
    assert len(picker) == 1
    assert "policy-selector-dropdown" in picker[0].className.split()
    # Parent field owns the CSS scope for the Dash 4 selected-value and popover.
    parent_fields = [node for node in walk(dashboard.policy_tab)
                     if getattr(node, "className", None)
                     and "policy-selector-field" in node.className.split()
                     and picker[0] in list(walk(node))]
    assert len(parent_fields) == 1
    css = (Path(__file__).resolve().parents[1] / "assets" / "style.css").read_text()
    assert ".policy-selector-field button.dash-dropdown .dash-dropdown-value" in css
    assert "color: #111821 !important;" in css


def test_spot_and_spread_live_only_in_target_pair_flow_configuration(dashboard):
    """Target market data moved without introducing duplicate or hidden fields."""
    modal = next(node for node in walk(dashboard.app.layout)
                 if getattr(node, "id", None) == "flow-source-modal")
    market_data = next(node for node in walk(modal)
                       if getattr(node, "className", "") == "flow-target-market-data")
    in_modal = {getattr(node, "id", None) for node in walk(market_data)}
    in_sidebar = {node.id for node in walk(dashboard.sidebar)
                  if isinstance(getattr(node, "id", None), str)}
    ids = [getattr(node, "id", None) for node in walk(dashboard.app.layout)]

    assert {"spot", "spread"} <= in_modal
    assert "flow-target-spot-label" not in in_modal
    assert {"spot", "spread"}.isdisjoint(in_sidebar)
    assert ids.count("spot") == 1
    assert ids.count("spread") == 1

    controls = {node.id: node for node in walk(market_data)
                if getattr(node, "id", None) in {"spot", "spread"}}
    assert controls["spot"].value == dashboard.DEFAULT["spot"]
    assert controls["spread"].value == dashboard.DEFAULT["spreadPips"]
    assert controls["spot"].step == "any"
    assert controls["spread"].step == "any"

    # The existing CONFIG_FIELDS chain feeds Solve from these same component IDs.
    states = {name: (binding.args[0], binding.args[1])
              for name, binding in dashboard.CONFIG_FIELDS}
    assert states["spot"] == ("spot", "value")
    assert states["spread"] == ("spread", "value")


def test_target_pair_selection_reloads_spot_and_spread_defaults(dashboard):
    from trinity.defaults import default_fx_pair_market_data

    assert dashboard.update_target_market_defaults("EURSEK") == (11.5, 20.0)
    assert dashboard.update_target_market_defaults("GBPSEK") == (13.0682, 25.0)
    assert dashboard.update_target_market_defaults("EURUSD") == (1.18, 1.5)
    spot, spread = dashboard.update_target_market_defaults("USDSEK")
    assert {"spot": spot, "spreadPips": spread} == default_fx_pair_market_data("USDSEK")
    spot, spread = dashboard.update_target_market_defaults(None)
    assert spot is dashboard.no_update and spread is dashboard.no_update


def test_pricing_knots_are_in_target_pair_modal_and_not_sidebar(dashboard):
    modal = next(node for node in walk(dashboard.app.layout)
                 if getattr(node, "id", None) == "flow-source-modal")
    knots = next(node for node in walk(modal)
                 if getattr(node, "className", "") == "flow-target-pricing-knots")
    modal_ids = [getattr(node, "id", None) for node in walk(knots)]
    sidebar_ids = [getattr(node, "id", None) for node in walk(dashboard.sidebar)]
    all_ids = [getattr(node, "id", None) for node in walk(dashboard.app.layout)]
    assert set(modal_ids) >= {"t0-sizes", "t1-sizes"}
    for tier in (0, 1):
        field = f"t{tier}-sizes"
        assert field not in sidebar_ids
        assert all_ids.count(field) == 1
        node = next(node for node in walk(knots) if getattr(node, "id", None) == field)
        assert node.value == ", ".join(str(size) for size in dashboard.DEFAULT["tiers"][tier]["sizes"])
    # The same CONFIG_FIELDS IDs continue to feed both the solver and analytics.
    states = {name: (binding.args[0], binding.args[1])
              for name, binding in dashboard.CONFIG_FIELDS}
    assert states["t0-sizes"] == ("t0-sizes", "value")
    assert states["t1-sizes"] == ("t1-sizes", "value")


def test_all_target_pair_options_have_json_market_defaults(dashboard):
    from trinity.defaults import default_fx_pair_market_data
    assert all(default_fx_pair_market_data(pair) for pair in dashboard.FX_PAIRS)


def test_target_market_data_is_first_flow_expander_and_uses_closed_default(dashboard):
    modal = next(node for node in walk(dashboard.app.layout)
                 if getattr(node, "id", None) == "flow-source-modal")
    target_controls = next(node for node in walk(modal)
                           if getattr(node, "id", None) == "flow-target-direct-settings")
    sections = [node for node in walk(target_controls)
                if getattr(node, "type", "") == "Details"]
    assert len(sections) == 2
    assert [next(n for n in walk(section) if getattr(n, "type", "") == "Summary").children
            for section in sections] == [
        "Market Data", "Pricing Rungs",
    ]
    assert all(section.open is False for section in sections)
    assert target_controls.style == {"display": "none"}

    editor = next(node for node in walk(modal)
                  if getattr(node, "id", None) == "flow-source-editor")
    assert [getattr(n, "id", "") for n in editor.children] == [
        "flow-source-cards", "flow-source-active-card",
    ]
    # Both input expanders are now *inside* the same bordered target-pair
    # card as RFQ flow, volatility and markout; they appear after its header.
    active_card = editor.children[1]
    assert active_card.className == "flow-source-settings"
    assert [getattr(n, "id", "") for n in active_card.children] == [
        "flow-source-heading", "flow-target-picker-container",
        "flow-target-direct-settings", "flow-source-settings",
    ]
    direct = dashboard._direct_flow_source(dashboard.DEFAULT["tiers"][0]["flow"], "EURSEK")
    editor_contents = dashboard.flow_source_settings_editor("direct", direct, "EURSEK")
    header, dynamic_sections = editor_contents.children
    assert any(getattr(n, "className", "") == "flow-source-row-head" for n in walk(header))
    dynamic_titles = [next(n for n in walk(expander)
                           if getattr(n, "type", "") == "Summary").children
                      for expander in walk(dynamic_sections) if getattr(expander, "type", "") == "Details"]
    assert ["Market Data", "Pricing Rungs"] + dynamic_titles == [
        "Market Data", "Pricing Rungs", "RFQ flow intensities", "Volatility", "Markouts",
    ]
    assert all(n.open is False for n in walk(dynamic_sections) if getattr(n, "type", "") == "Details")
    assert "Each Tier has its own discrete RFQ sizes" not in str(dashboard.app.layout.children)


def test_only_active_tier_knots_are_visible_but_both_values_stay_mounted(dashboard):
    modal = next(node for node in walk(dashboard.app.layout)
                 if getattr(node, "id", None) == "flow-source-modal")
    ids = [getattr(n, "id", None) for n in walk(dashboard.app.layout)]
    assert ids.count("t0-sizes") == 1 and ids.count("t1-sizes") == 1
    tier_0 = next(n for n in walk(modal)
                  if getattr(n, "id", None) == "flow-target-tier-knots-0")
    tier_1 = next(n for n in walk(modal)
                  if getattr(n, "id", None) == "flow-target-tier-knots-1")
    assert tier_0.style == tier_1.style == {"display": "none"}

    assert dashboard.show_target_pair_settings_for_tier(0, "direct") == (
        {"display": "block"}, {"display": "block"}, {"display": "none"})
    assert dashboard.show_target_pair_settings_for_tier(1, "direct") == (
        {"display": "block"}, {"display": "none"}, {"display": "block"})
    # Crossed flow sources have their own settings, not the target's pricing knots.
    assert dashboard.show_target_pair_settings_for_tier(0, "extra-0") == (
        {"display": "none"}, {"display": "none"}, {"display": "none"})
    assert dashboard.show_target_pair_settings_for_tier(None, "direct") == (
        {"display": "none"}, {"display": "none"}, {"display": "none"})
    # The presentation callback controls only style, not input value or children.
    states = {name: (binding.args[0], binding.args[1])
              for name, binding in dashboard.CONFIG_FIELDS}
    assert states["t0-sizes"] == ("t0-sizes", "value")
    assert states["t1-sizes"] == ("t1-sizes", "value")


def test_unified_flow_card_renders_header_then_tier_inputs_then_source_expanders(dashboard):
    """Source changes must not duplicate or remount spot/spread/Tier knot inputs."""
    tiers = dashboard.DEFAULT["tiers"]
    flows = [item["flow"] for item in tiers]
    direct_values = [
        value
        for flow in flows
        for value in (flow["A0"], flow["theta"], flow["beta"],
                      flow["steepness"], flow["shift"], flow["volumeShift"])
    ] + [
        value
        for tier in tiers
        for value in (["on"] if tier["useMarkout"] else [],
                      tier["markout"]["asymptoticEurPerMillion"],
                      tier["markout"]["tauSeconds"])
    ]

    for tier_index in (0, 1):
        header, dynamic = dashboard.render_flow_source_settings(
            tier_index, "EURSEK", "direct", {"0": [], "1": []}, *direct_values,
        )
        assert any(getattr(node, "className", None) == "flow-source-row-head"
                   for node in walk(header))
        headings = [node.children for node in walk(dynamic)
                    if getattr(node, "type", None) == "Summary"]
        assert headings == ["RFQ flow intensities", "Volatility", "Markouts"]
        assert all(node.open is False for node in walk(dynamic)
                   if getattr(node, "type", None) == "Details")

    # With a crossed source selected the target-only sections are hidden, but
    # both Tier input IDs remain mounted under the stable active-card wrapper.
    cross_source = {
        "pair": "USDSEK",
        "crossPair": "EURUSD",
        "flow": flows[0].copy(),
        "mapping": {"crossMid": 1.1, "sourceSpreadPips": 20.0},
    }
    header, dynamic = dashboard.render_flow_source_settings(
        0, "EURSEK", "extra-0", {"0": [cross_source], "1": []}, *direct_values,
    )
    # No extra crossed-source heading or removal button takes vertical space.
    assert not any(getattr(node, "className", None) == "flow-source-row-head"
                   for node in walk(header))
    headings = [node.children for node in walk(dynamic)
                if getattr(node, "type", None) == "Summary"]
    assert headings == ["Market Data", "RFQ flow intensities", "Markouts"]
    assert dashboard.show_target_pair_settings_for_tier(0, "extra-0")[0] == {"display": "none"}

    outputs = [
        binding.args[0] for args, _kwargs, callback in dashboard.app.callbacks
        if callback is dashboard.render_flow_source_settings
        for binding in args if isinstance(binding, Output)
    ]
    assert outputs == ["flow-source-heading", "flow-source-settings"]


def test_flow_dialog_has_compact_labels_and_no_repeated_target_text(dashboard):
    modal = next(n for n in walk(dashboard.app.layout)
                 if getattr(n, "id", None) == "flow-source-modal")
    spans = [n.children for n in walk(modal)
             if getattr(n, "type", "") == "Span" and isinstance(n.children, str)]
    assert "Mid-rate" in spans
    assert "Base M" in spans
    assert "Target / priced FX pair" not in spans
    assert not any(getattr(n, "id", None) == "flow-source-tier-label" for n in walk(modal))
    assert not any(getattr(n, "id", None) == "flow-target-spot-label" for n in walk(modal))
    assert [n.id for n in walk(modal) if getattr(n, "id", None) == "flow-direct-target-display"] == ["flow-direct-target-display"]
    assert dashboard.render_flow_source_cards(0, "EURSEK", "direct", {"0": [], "1": []})
    assert dashboard.render_flow_source_cards(None, "EURSEK", "direct", {"0": [], "1": []}) == []


def test_flow_source_markout_enabled_is_white_and_explanatory_bloat_removed(dashboard):
    tiers = dashboard.DEFAULT["tiers"]
    source = dashboard._direct_flow_source(tiers[0]["flow"], "EURSEK")
    editor = dashboard.flow_source_settings_editor("direct", source, "EURSEK")
    assert "Direct target-pair flow" not in [n.children for n in walk(editor)
                                               if isinstance(n.children, str)]
    markout_checkbox = next(n for n in walk(editor)
                            if getattr(n, "id", None) == "flow-edit-markout-enabled")
    label = markout_checkbox.options[0]["label"]
    assert label.children == "Enabled"
    assert label.style["color"] == "#ffffff"
    assert label.style["WebkitTextFillColor"] == "#ffffff"
    assert markout_checkbox.labelStyle["color"] == "#ffffff"
    all_text = [n.children for n in walk(editor) if isinstance(n.children, str)]
    assert not any("RFQ sizes are exactly the pricing knots" in t for t in all_text)
    assert not any("Read-only values from fx_pair_defaults.json" in t for t in all_text)


def test_flow_markout_preview_has_short_axis_labels_and_no_title(dashboard):
    figure = dashboard.flow_markout_figure(True, -50.8, 36.7, "EURSEK")
    assert figure.layout.title.text in (None, "")
    assert figure.layout.xaxis.title.text == "seconds"
    assert figure.layout.yaxis.title.text == "EUR / EUR m"
    assert figure.layout.margin.t == 25
    assert len(figure.data) == 1
    assert float(figure.data[0].y[0]) == 0.0
    assert float(figure.data[0].y[-1]) < -50.0


def test_target_pair_dropdown_only_visible_on_direct_source(dashboard):
    """Target selection is a global property, not a crossed-source input."""
    modal = next(node for node in walk(dashboard.app.layout)
                 if getattr(node, "id", None) == "flow-source-modal")
    containers = [node for node in walk(modal)
                  if getattr(node, "id", None) == "flow-target-picker-container"]
    assert len(containers) == 1
    picker = containers[0]
    assert [node.id for node in walk(picker)
            if getattr(node, "id", None) == "flow-direct-target-display"] == ["flow-direct-target-display"]

    assert dashboard.show_target_pair_picker("direct") == {"display": "flex"}
    assert dashboard.show_target_pair_picker(None) == {"display": "flex"}
    assert dashboard.show_target_pair_picker("extra-0") == {"display": "none"}
    assert dashboard.show_target_pair_picker("extra-1") == {"display": "none"}

    callbacks = [(args, fn) for args, _kw, fn in dashboard.app.callbacks
                 if fn is dashboard.show_target_pair_picker]
    assert len(callbacks) == 1
    bindings, _ = callbacks[0]
    assert [(type(b).__name__, b.args[:2]) for b in bindings] == [
        ("Output", ("flow-target-picker-container", "style")),
        ("Input", ("flow-source-selected-store", "data")),
    ]
    # Hiding rather than removing the dropdown preserves the selected target
    # for crossed-pair calculations, Solve and return to direct-source editing.
    target = next(n for n in walk(picker) if getattr(n, "id", None) == "flow-direct-target-display")
    assert target.value == dashboard.DEFAULT["targetPair"]



def test_direct_and_crossed_pair_selectors_have_consistent_layout(dashboard):
    """The selectors follow the source cards, and only a crossed source can pick a crossing pair."""
    modal = next(n for n in walk(dashboard.app.layout) if getattr(n, "id", None) == "flow-source-modal")
    editor = next(n for n in walk(modal) if getattr(n, "id", None) == "flow-source-editor")
    cards, active = editor.children
    assert cards.id == "flow-source-cards"
    ids = [getattr(n, "id", None) for n in active.children]
    assert ids.index("flow-source-heading") < ids.index("flow-target-picker-container") < ids.index("flow-target-direct-settings")

    direct = next(n for n in walk(active) if getattr(n, "id", None) == "flow-target-picker-container")
    assert "flow-pair-selectors" in direct.className
    direct_controls = [n for n in walk(direct) if getattr(n, "id", None) in (
        "flow-direct-target-display", "flow-direct-crossing-placeholder"
    )]
    assert [n.id for n in direct_controls] == ["flow-direct-target-display", "flow-direct-crossing-placeholder"]
    assert direct_controls[0].value == dashboard.DEFAULT["targetPair"]
    assert direct_controls[0].disabled is True
    assert direct_controls[1].disabled is True
    assert direct_controls[1].placeholder == "Select FX pair"

    crossed = {
        "pair": "USDSEK", "crossPair": "EURUSD",
        "flow": dashboard.DEFAULT["tiers"][0]["flow"],
        "mapping": {"type": "crossed", "crossMid": 1.13, "sourceSpreadPips": 20},
    }
    header, _ = dashboard.flow_source_settings_editor("extra-0", crossed, "EURSEK").children
    selectors = next(n for n in walk(header) if getattr(n, "className", "") == "flow-pair-selectors")
    crossed_controls = [n for n in walk(selectors) if getattr(n, "id", None) in (
        "flow-edit-pair", "flow-edit-cross-pair"
    )]
    assert [n.id for n in crossed_controls] == ["flow-edit-pair", "flow-edit-cross-pair"]
    assert all(getattr(n, "disabled", False) is False for n in crossed_controls)
    assert crossed_controls[0].value == "USDSEK"
    assert crossed_controls[1].value == "EURUSD"



def test_add_fx_pair_activates_crossing_selector(dashboard):
    """Add selects a new crossed source without changing the target FX pair."""
    direct_values = [0.01, 0.144, 0.0857, 8.42, 0.52, 0.026] * 2
    old_trigger = dashboard.ctx.triggered_id
    try:
        dashboard.ctx.triggered_id = "flow-source-add"
        result = dashboard.update_flow_source_editor(
            1, 0, None, None, None, None,
            None, None, None, None, None, None,
            0, "EURSEK", "direct", {"0": [], "1": []}, *direct_values,
        )
    finally:
        dashboard.ctx.triggered_id = old_trigger
    stored, selected = result[:2]
    assert selected == "extra-0"
    assert len(stored["0"]) == 1
    assert stored["0"][0]["mapping"]["type"] == "crossed"
    assert dashboard.show_target_pair_picker(selected) == {"display": "none"}

    header, _ = dashboard.flow_source_settings_editor(selected, stored["0"][0], "EURSEK").children
    crossing = next(n for n in walk(header) if getattr(n, "id", None) == "flow-edit-cross-pair")
    assert not getattr(crossing, "disabled", False)
    assert crossing.placeholder == "Select FX pair"


def test_crossed_source_editor_has_no_redundant_pair_heading(dashboard):
    """Only the editable source pair selector identifies a crossed RFQ flow."""
    direct = dashboard._direct_flow_source(dashboard.DEFAULT["tiers"][0]["flow"], "EURSEK")
    direct_header, _ = dashboard.flow_source_settings_editor("direct", direct, "EURSEK").children
    assert any(getattr(node, "className", None) == "flow-source-title"
               for node in walk(direct_header))

    crossed = {
        "pair": "USDSEK", "crossPair": "EURUSD", "flow": direct["flow"],
        "mapping": {"crossMid": 1.15, "sourceSpreadPips": 20.0},
    }
    crossed_header, sections = dashboard.flow_source_settings_editor(
        "extra-0", crossed, "EURSEK",
    ).children
    assert not any(getattr(node, "className", None) in ("flow-source-title", "source-kind-note")
                   for node in walk(crossed_header))
    assert not any(getattr(node, "children", None) in ("EURSEK", "USDSEK", "Crossed source")
                   for node in walk(crossed_header) if isinstance(getattr(node, "children", None), str))
    # Removal is a persistent footer action rather than part of the editor.
    assert not any(getattr(node, "id", None) == "flow-source-remove-selected"
                   for node in walk(crossed_header))
    assert any(getattr(node, "id", None) == "flow-edit-pair" for node in walk(crossed_header))
    assert any(getattr(node, "id", None) == "flow-edit-cross-pair" for node in walk(crossed_header))
    # Both dropdowns share the same placeholder without changing selected values.
    input_by_id = {getattr(node, "id", None): node for node in walk(crossed_header)
                   if getattr(node, "id", None) in ("flow-edit-pair", "flow-edit-cross-pair")}
    assert input_by_id["flow-edit-pair"].placeholder == "Select FX pair"
    assert input_by_id["flow-edit-cross-pair"].placeholder == "Select FX pair"
    assert input_by_id["flow-edit-pair"].value == "USDSEK"
    assert input_by_id["flow-edit-cross-pair"].value == "EURUSD"
    field_labels = [node.children for node in walk(crossed_header)
                    if getattr(node, "type", None) == "Span"]
    assert "Crossed pair" in field_labels
    assert "Crossing pair" in field_labels
    assert "Source pair" not in field_labels
    assert "Cross / hedge pair" not in field_labels


def test_crossed_market_data_inputs_are_inside_closed_expander(dashboard):
    """Crossing-rate inputs are only shown inside crossed-source Market Data."""
    source = {
        "pair": "USDSEK",
        "crossPair": "EURUSD",
        "mapping": {"type": "crossed", "crossMid": 1.1234, "sourceSpreadPips": 14.0},
        "flow": dashboard.DEFAULT["tiers"][0]["flow"].copy(),
    }
    editor = dashboard.flow_source_settings_editor("extra-0", source, "EURSEK")
    header, sections = editor.children
    # Pair selectors remain visible above the expanders.
    assert [n.id for n in walk(header) if getattr(n, "id", None) in (
        "flow-edit-pair", "flow-edit-cross-pair"
    )] == ["flow-edit-pair", "flow-edit-cross-pair"]
    assert not any(getattr(n, "id", None) in ("flow-edit-cross-mid", "flow-edit-spread")
                   for n in walk(header))

    expanders = [n for n in walk(sections) if getattr(n, "type", None) == "Details"]
    assert [next(n for n in walk(e) if getattr(n, "type", None) == "Summary").children
            for e in expanders] == ["Market Data", "RFQ flow intensities", "Markouts"]
    assert all(e.open is False for e in expanders)
    market = expanders[0]
    fields = {n.id: n for n in walk(market) if getattr(n, "id", None) in (
        "flow-edit-cross-mid", "flow-edit-spread"
    )}
    assert set(fields) == {"flow-edit-cross-mid", "flow-edit-spread"}
    assert fields["flow-edit-cross-mid"].value == 1.1234
    assert fields["flow-edit-spread"].value == 14.0
    captions = [n.children for n in walk(market) if getattr(n, "type", None) == "Span"]
    assert "Crossing Mid-Rate" in captions
    assert "Crossing Spread" in captions
    assert "Cross mid" not in captions
    assert "Source spread [pips]" not in captions

    direct = dashboard.flow_source_settings_editor(
        "direct", dashboard._direct_flow_source(source["flow"], "EURSEK"), "EURSEK"
    )
    assert [n.id for n in walk(direct) if getattr(n, "id", None) in (
        "flow-edit-cross-mid", "flow-edit-spread"
    )] == ["flow-edit-cross-mid", "flow-edit-spread"]
    assert not any(
        next((n.children for n in walk(e) if getattr(n, "type", None) == "Summary"), None)
        == "Market Data"
        for e in walk(direct) if getattr(e, "type", None) == "Details"
    )


def test_flow_modal_action_buttons_have_equal_sizing_and_danger_colours(dashboard):
    """RFQ and ECN footer actions share sizing, with red Close/remove buttons."""
    buttons = {
        getattr(node, "id", None): node
        for node in walk(dashboard.app.layout)
        if getattr(node, "type", None) == "Button"
        and isinstance(getattr(node, "id", None), str)
    }
    for prefix in ("flow-source", "ecn-flow-source"):
        close = buttons[f"{prefix}-cancel"]
        apply = buttons[f"{prefix}-apply"]
        assert "modal-action-button" in close.className.split()
        assert "modal-action-button" in apply.className.split()
        assert "modal-action-danger" in close.className.split()
        assert "modal-action-apply" in apply.className.split()
        assert "primary-button" not in apply.className.split()

    target = dashboard.DEFAULT["targetPair"]
    rfq_crossed = dashboard.flow_source_settings_editor(
        "extra-0", {"pair": "USDSEK", "crossPair": "EURUSD", "flow": {}, "mapping": {}}, target,
    )
    ecn_crossed = dashboard.ecn_flow_source_settings_editor(
        "extra-0", {"pair": "USDSEK", "crossPair": "EURUSD", "flow": {}, "mapping": {}}, target,
    )
    for settings, remove_id in (
        (rfq_crossed, "flow-source-remove-selected"),
        (ecn_crossed, "ecn-flow-source-remove-selected"),
    ):
        assert not any(getattr(node, "id", None) == remove_id for node in walk(settings))
        remove = buttons[remove_id]
        assert "modal-remove-button" in remove.className.split()
        assert remove.disabled is True  # Direct source is selected initially.

    css = (Path(__file__).resolve().parents[1] / "assets" / "style.css").read_text()
    assert ".modal-actions-right .modal-action-button" in css
    assert "flex:0 0 112px; width:112px; height:44px;" in css
    assert ".modal-action-danger, .modal-remove-button" in css
    assert ".modal-actions-left .modal-remove-button" in css
    assert ".modal-actions-left .secondary-button" in css


def test_ecn_direct_pair_dropdowns_match_tier_layout(dashboard):
    """ECN Direct presents the same compact pair picker as RFQ Direct."""
    modal = next(n for n in walk(dashboard.app.layout)
                 if getattr(n, "id", None) == "ecn-flow-source-modal")
    editor = next(n for n in walk(modal)
                  if getattr(n, "className", None) == "flow-source-editor")
    cards, picker, settings = editor.children
    assert cards.id == "ecn-flow-source-cards"
    assert picker.id == "ecn-target-picker-container"
    assert settings.id == "ecn-flow-source-settings"
    assert "flow-pair-selectors" in picker.className.split()
    direct_dropdowns = [n for n in walk(picker) if getattr(n, "id", None) in (
        "ecn-direct-target-display", "ecn-direct-crossing-placeholder")]
    assert [n.id for n in direct_dropdowns] == [
        "ecn-direct-target-display", "ecn-direct-crossing-placeholder"]
    assert direct_dropdowns[0].value == dashboard.DEFAULT["targetPair"]
    assert direct_dropdowns[0].disabled is True
    assert direct_dropdowns[1].disabled is True
    assert direct_dropdowns[1].placeholder == "Select FX pair"
    assert dashboard.show_ecn_target_pair_picker("direct") == {"display": "flex"}
    assert dashboard.show_ecn_target_pair_picker("extra-0") == {"display": "none"}
    assert not any(getattr(n, "id", None) == "ecn-flow-source-target-label" for n in walk(modal))


def test_ecn_crossed_pair_dropdowns_match_tier_layout(dashboard):
    src = {"pair": "USDSEK", "crossPair": "EURUSD", "flow": {"A": 0.2, "k": 8},
           "mapping": {"type": "crossed", "crossMid": 1.12, "sourceSpreadPips": 20}}
    editor = dashboard.ecn_flow_source_settings_editor("extra-0", src, "EURSEK")
    selector_group = next(n for n in walk(editor)
                          if getattr(n, "className", None) == "flow-pair-selectors")
    fields = [n for n in walk(selector_group) if getattr(n, "id", None) in (
        "ecn-flow-edit-pair", "ecn-flow-edit-cross-pair")]
    assert [n.id for n in fields] == ["ecn-flow-edit-pair", "ecn-flow-edit-cross-pair"]
    assert [n.value for n in fields] == ["USDSEK", "EURUSD"]
    assert all(n.placeholder == "Select FX pair" for n in fields)
    assert all(not getattr(n, "disabled", False) for n in fields)
    assert [n.children for n in walk(selector_group) if getattr(n, "type", None) == "Span"] == [
        "Crossed pair", "Crossing pair"]
    assert not any(getattr(n, "className", None) == "source-kind-note" for n in walk(editor))
    assert dashboard.render_ecn_flow_source_cards("EURSEK", "extra-0", [src])


def test_global_target_is_the_only_editable_direct_pair(dashboard):
    """Global owns the pair; ECN/Tiers only display a disabled copy."""
    layout = dashboard.app.layout
    global_selectors = [n for n in walk(layout) if getattr(n, "id", None) == "global-target-pair"]
    assert len(global_selectors) == 1
    assert not getattr(global_selectors[0], "disabled", False)
    assert global_selectors[0].value == dashboard.DEFAULT["targetPair"]
    for key in ("flow-direct-target-display", "ecn-direct-target-display"):
        copies = [n for n in walk(layout) if getattr(n, "id", None) == key]
        assert len(copies) == 1 and copies[0].disabled is True
    assert dashboard.show_global_target_in_flow_editors("GBPSEK") == ("GBPSEK", "GBPSEK")
    callbacks = [(bindings, fn) for bindings, _kw, fn in dashboard.app.callbacks
                 if fn is dashboard.show_global_target_in_flow_editors]
    assert len(callbacks) == 1
    binds, _ = callbacks[0]
    assert [(type(b).__name__, b.args[:2]) for b in binds] == [
        ("Output", ("flow-direct-target-display", "value")),
        ("Output", ("ecn-direct-target-display", "value")),
        ("Input", ("global-target-pair", "value")),
    ]
    assert dashboard.clear_crossed_sources_on_target_change("GBPSEK") == ({"0": [], "1": []}, [])


def test_target_is_globally_shared_by_solve_and_market_defaults(dashboard):
    """No stale modal state may decide the target after Global changes."""
    from trinity.defaults import default_fx_pair_market_data, default_fx_pair_volatility
    from trinity.ui_config import build_config
    from test_ui_config import ui_values_from_default
    cfg_fields = [b.args[:2] for args, _kw, fn in dashboard.app.callbacks
                  if fn is dashboard.solve_model for b in args
                  if isinstance(b, State)]
    assert ("global-target-pair", "value") in cfg_fields
    assert ("flow-direct-target-display", "value") not in cfg_fields
    assert ("ecn-direct-target-display", "value") not in cfg_fields
    assert dashboard.update_target_market_defaults("GBPSEK") == (
        default_fx_pair_market_data("GBPSEK")["spot"],
        default_fx_pair_market_data("GBPSEK")["spreadPips"],
    )
    values = ui_values_from_default()
    values["target-pair"] = "GBPSEK"
    cfg = build_config(values)
    assert cfg["targetPair"] == "GBPSEK"
    assert cfg["volatilityRegimes"] == default_fx_pair_volatility("GBPSEK")


def test_ecn_add_fx_pair_selects_crossed_dropdowns(dashboard):
    saved_trigger = dashboard.ctx.triggered_id
    try:
        dashboard.ctx.triggered_id = "ecn-flow-source-add"
        out = dashboard.update_ecn_flow_source_editor(
            1, 0, None, None, None, None, None, None,
            None, None, None, None, "EURSEK", "direct", [], 0.1, 8.4,
            dashboard.ECN_TRADE_SIZE_PROBABILITIES,
        )
    finally:
        dashboard.ctx.triggered_id = saved_trigger
    sources, selected = out[:2]
    assert selected == "extra-0" and len(sources) == 1
    assert sources[0]["mapping"]["type"] == "crossed"
    assert dashboard.show_ecn_target_pair_picker(selected) == {"display": "none"}
    editor = dashboard.ecn_flow_source_settings_editor(selected, sources[0], "EURSEK")
    dropdowns = [n for n in walk(editor) if getattr(n, "id", None) in (
        "ecn-flow-edit-pair", "ecn-flow-edit-cross-pair")]
    assert [n.id for n in dropdowns] == ["ecn-flow-edit-pair", "ecn-flow-edit-cross-pair"]
    assert all(not getattr(n, "disabled", False) for n in dropdowns)
    assert [n.placeholder for n in dropdowns] == ["Select FX pair", "Select FX pair"]


def test_remove_pair_is_next_to_add_in_both_modal_footers(dashboard):
    for prefix in ("flow-source", "ecn-flow-source"):
        modal = next(node for node in walk(dashboard.app.layout)
                     if getattr(node, "id", None) == f"{prefix}-modal")
        left = next(node for node in walk(modal)
                    if getattr(node, "className", None) == "modal-actions-left")
        buttons = [node for node in left.children if getattr(node, "type", None) == "Button"]
        assert [button.id for button in buttons] == [
            f"{prefix}-add", f"{prefix}-remove-selected"
        ]
        assert "modal-remove-button" in buttons[1].className.split()
        assert buttons[1].style == {"display": "none"}
        assert len([node for node in walk(modal)
                    if getattr(node, "id", None) == f"{prefix}-remove-selected"]) == 1


def test_remove_pair_visible_only_for_existing_crossed_source(dashboard):
    rfq = {"0": [{"pair": "USDSEK"}], "1": []}
    assert dashboard.show_flow_source_remove_button("direct", 0, rfq) == ({"display": "none"}, True)
    assert dashboard.show_flow_source_remove_button("extra-0", 0, rfq) == ({"display": "inline-flex"}, False)
    assert dashboard.show_flow_source_remove_button("extra-0", 1, rfq) == ({"display": "none"}, True)
    assert dashboard.show_flow_source_remove_button("extra-1", 0, rfq) == ({"display": "none"}, True)
    assert dashboard.show_flow_source_remove_button("extra-bad", 0, rfq) == ({"display": "none"}, True)
    assert dashboard.show_ecn_flow_source_remove_button("extra-0", [{"pair": "USDSEK"}]) == ({"display": "inline-flex"}, False)
    assert dashboard.show_ecn_flow_source_remove_button("direct", [{"pair": "USDSEK"}]) == ({"display": "none"}, True)
    assert dashboard.show_ecn_flow_source_remove_button("extra-0", []) == ({"display": "none"}, True)


def test_remove_pair_action_removes_selected_not_direct(dashboard):
    prior = dashboard.ctx.triggered_id
    try:
        dashboard.ctx.triggered_id = "flow-source-remove-selected"
        vals = [0.01, 0.144, 0.0857, 8.42, 0.52, 0.026] * 2
        result = dashboard.update_flow_source_editor(
            0, 1, None, None, None, None, None, None, None, None, None, None,
            0, "EURSEK", "extra-0", {"0": [{"pair": "USDSEK"}, {"pair": "GBPSEK"}], "1": []}, *vals,
        )
        assert [item["pair"] for item in result[0]["0"]] == ["GBPSEK"]
        assert result[1] == "direct"
        dashboard.ctx.triggered_id = "ecn-flow-source-remove-selected"
        ecn = dashboard.update_ecn_flow_source_editor(
            0, 1, None, None, None, None, None, None, None, None, None, None,
            "EURSEK", "extra-0", [{"pair": "USDSEK"}, {"pair": "GBPSEK"}],
            0.1, 8.4, dashboard.ECN_TRADE_SIZE_PROBABILITIES,
        )
        assert [item["pair"] for item in ecn[0]] == ["GBPSEK"]
        assert ecn[1] == "direct"
    finally:
        dashboard.ctx.triggered_id = prior


def test_ecn_optimizer_legacy_explainer_removed(dashboard):
    modal = next(node for node in walk(dashboard.app.layout)
                 if getattr(node, "id", None) == "ecn-flow-source-modal")
    assert not any("The ECN optimizer chooses one master delta" in getattr(node, "children", "")
                   for node in walk(modal) if isinstance(getattr(node, "children", None), str))
