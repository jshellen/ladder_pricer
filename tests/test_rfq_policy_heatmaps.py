"""RFQ knot-specific heatmaps with local line/heatmap controls."""
from __future__ import annotations

import numpy as np
import pytest

from test_policy_dashboard_regime_selectors import dashboard, walk


PREFIXES = (
    "tier-implied-hit", "tier-quote-inventory",
)


def example_solution(count: int = 2):
    sigmas = [12, 35] if count == 2 else [12, 20, 35]
    regimes = []
    for k, sigma in enumerate(sigmas):
        # q-index rows, pricing-knot-size columns; distinct across both axes.
        bid = [[0.10 + .01 * i + .02 * j + .03 * k for j in range(3)] for i in range(3)]
        ask = [[0.20 + .01 * i + .02 * j + .02 * k for j in range(3)] for i in range(3)]
        regimes.append({"sigmaPips": sigma, "tiers": [{
            "sizes": [1., 2., 5.], "bid": bid, "ask": ask,
        }], "passiveEcn": {}, "darkPool": {}, "expectedMtm": []})
    return {"qGrid": [-1., 0., 1.],
            "tiers": [{"name": "Tier 1", "sizes": [1., 2., 5.]}],
            "regimes": regimes, "valueByRegime": [[0., 0., 0.] for _ in regimes]}


def test_controls_are_local_to_each_rfq_policy_expander(dashboard):
    ids = {getattr(node, "id", None): node for node in walk(dashboard.policy_tab)
           if isinstance(getattr(node, "id", None), str)}
    mapping = {
        "tier-implied-hit": "implied-hit-chart",
        "tier-quote-inventory": "quote-inventory-chart",
    }
    expanders = [node for node in walk(dashboard.policy_tab) if node.type == "Details"]
    for prefix, chart in mapping.items():
        matching = [node for node in expanders if chart in {
            getattr(c, "id", None) for c in walk(node)}]
        assert len(matching) == 1
        expander_ids = {getattr(c, "id", None) for c in walk(matching[0])}
        assert set((f"{prefix}-heatmap", f"{prefix}-rfq-size",
                    f"{prefix}-rfq-size-wrapper", f"{prefix}-regime-wrapper")) <= expander_ids
        assert len([p for p in PREFIXES if f"{p}-heatmap" in expander_ids]) == 1
        toggle = ids[f"{prefix}-heatmap"]
        assert toggle.options[0]["label"].children == "Heatmap"
        assert toggle.inputStyle["marginRight"] == "12px"
        size = ids[f"{prefix}-rfq-size"]
        assert "policy-selector-dropdown" in size.className
        assert "Select RFQ size" == size.placeholder


def test_knot_dropdowns_are_tier_specific_and_keep_valid_user_choices(dashboard):
    count = len(PREFIXES)
    assert dashboard.update_tier_heatmap_sizes(None, None, *([None]*count)) == (
        *([[]]*count), *([None]*count))
    solution = {"tiers": [
        {"sizes": [1., 2., 5.]}, {"sizes": [3., 7., 10.]}
    ]}
    values = dashboard.update_tier_heatmap_sizes(solution, 0, None, 2.)
    assert all([opt["value"] for opt in opts] == [1., 2., 5.] for opts in values[:count])
    assert values[count:] == (1., 2.)
    changed = dashboard.update_tier_heatmap_sizes(solution, 1, *values[count:])
    assert all([opt["value"] for opt in opts] == [3., 7., 10.] for opts in changed[:count])
    assert changed[count:] == (3.,) * count


def test_toggles_hide_regime_and_reveal_rfq_size_independently(dashboard):
    selections = (["heatmap"], [])
    out = dashboard.toggle_tier_regime_heatmaps(*selections)
    assert out[:2] == ({"display": "none"}, {})
    assert out[2:4] == ({}, {"display": "none"})
    assert [x["height"] for x in out[4:]] == ["610px", "420px"]
    off = dashboard.toggle_tier_regime_heatmaps([], [])
    assert off[:2] == ({},) * 2
    assert off[2:4] == ({"display": "none"},) * 2


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("metric", ["quotes", "win_probability"])
def test_rfq_heatmap_uses_exact_size_across_regimes(dashboard, monkeypatch, count, metric):
    solved = example_solution(count)
    monkeypatch.setattr(dashboard, "_tier_cfg", lambda *_: {"name": "Tier 1"})
    monkeypatch.setattr(dashboard, "_hit_ratio", lambda tier, delta, z, spread: np.asarray(delta) * .5)
    fig = dashboard.rfq_regime_heatmap_figure(solved, {"spreadPips": 20.}, 0, 5., metric=metric)
    assert len(fig.data) == 2
    bid, ask = fig.data
    assert all(trace.type == "heatmap" and trace.coloraxis == "coloraxis" for trace in fig.data)
    assert bid.z.shape == ask.z.shape == (count, 3)
    assert list(bid.x) == [-1., 0., 1.]
    assert all(str(sigma) in name for sigma, name in zip(
        ([12, 35] if count == 2 else [12, 20, 35]), bid.y))
    for k in range(count):
        for i in range(3):
            bid_delta = 0.10 + .01*i + .02*2 + .03*k
            ask_delta = 0.20 + .01*i + .02*2 + .02*k
            if metric == "quotes":
                assert bid.z[k, i] == pytest.approx(20 * (bid_delta - .5))
                assert ask.z[k, i] == pytest.approx(20 * (.5 - ask_delta))
            else:
                assert bid.z[k, i] == pytest.approx(bid_delta*.5)
                assert ask.z[k, i] == pytest.approx(ask_delta*.5)
    assert "5M" in fig.layout.title.text
    assert fig.layout.height == 610
    assert fig.layout.coloraxis.colorscale


def test_unknown_knot_is_rejected_rather_than_interpolated(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard, "_tier_cfg", lambda *_: {"name": "Tier 1"})
    with pytest.raises(ValueError, match="not a pricing knot"):
        dashboard.rfq_regime_heatmap_figure(example_solution(), {"spreadPips": 20.}, 0, 3.)


def test_figure_callback_switches_between_independent_rfq_views(dashboard, monkeypatch):
    solved = example_solution()
    monkeypatch.setattr(dashboard, "_tier_cfg", lambda *_: {"name": "Tier 1"})
    static = ("exogenous_arrival_figure", "flow_curve_figure", "hit_ratio_figure",
              "markout_figure", "flow_parameter_table")
    for name in static:
        monkeypatch.setattr(dashboard, name, lambda *args: "static")
    for name in ("implied_hit_ratio_figure", "quote_inventory_figure",
                 "ladder_figure", "ladder_table"):
        monkeypatch.setattr(dashboard, name, lambda current, *args: current["selectedSigmaPips"])
    monkeypatch.setattr(dashboard, "rfq_regime_heatmap_figure",
                        lambda solution, cfg, tier, size, *, metric="quotes":
                        f"{metric}:{size:g}M")
    output = dashboard.render_tier_diagnostics(
        solved, {"spreadPips": 20.}, 0, 0., 0, 0, 1,
        ["heatmap"], [], 2., 1.,
    )
    assert output[3] == "win_probability:2M"
    assert output[5] == 12  # quote line chart still uses regime 0
    assert output[6] == 35  # volume premium remains a line chart


@pytest.mark.parametrize("count", [2, 3])
def test_heatmaps_accept_native_joint_solver_policies(dashboard, count):
    from trinity.defaults import default_config
    from trinity.engine import Engine
    cfg = default_config()
    cfg["grid"]["maxAbs"] = 4.0
    for tier in cfg["tiers"]:
        tier["sizes"] = [1., 2.]
    if count == 2:
        cfg["volatilityRegimes"] = {
            "sigmaPips": [12., 35.], "ratesPerMinute": [[0., .12], [.08, 0.]],
        }
    solved = Engine(cfg).solve()
    for metric in ("quotes", "win_probability"):
        fig = dashboard.rfq_regime_heatmap_figure(solved, cfg, 0, 2., metric=metric)
        assert len(fig.data) == 2
        assert all(trace.z.shape == (count, len(solved["qGrid"])) for trace in fig.data)


def test_full_tier_diagnostics_callback_with_native_solution_and_heatmaps(dashboard):
    from trinity.defaults import default_config
    from trinity.engine import Engine
    cfg = default_config()
    cfg["grid"]["maxAbs"] = 4.0
    for tier in cfg["tiers"]:
        tier["sizes"] = [1., 2.]
    solved = Engine(cfg).solve()
    results = dashboard.render_tier_diagnostics(
        solved, cfg, 0, 0.0, 0, 2, 1,
        ["heatmap"], ["heatmap"], 1., 2.,
    )
    assert len(results) == 9
    assert len(results[3].data) == 2
    assert len(results[5].data) == 2
    assert [len(trace.z) for trace in results[5].data] == [3, 3]
    assert not results[6].layout.annotations  # Ladder chart still works


def test_heatmap_dropdowns_have_unique_callback_outputs_and_inputs(dashboard):
    from test_policy_dashboard_regime_selectors import Input, Output
    from collections import Counter
    callback_outputs = Counter(
        (arg.args[0], arg.args[1]) for args, _, _ in dashboard.app.callbacks
        for arg in args if isinstance(arg, Output) and isinstance(arg.args[0], str)
    )
    callback_inputs = {
        (arg.args[0], arg.args[1]) for args, _, _ in dashboard.app.callbacks
        for arg in args if isinstance(arg, Input) and isinstance(arg.args[0], str)
    }
    for prefix in PREFIXES:
        assert callback_outputs[(f"{prefix}-rfq-size", "value")] == 1
        assert callback_outputs[(f"{prefix}-rfq-size", "options")] == 1
        assert callback_outputs[(f"{prefix}-regime-wrapper", "style")] == 1
        assert callback_outputs[(f"{prefix}-rfq-size-wrapper", "style")] == 1
        assert (f"{prefix}-rfq-size", "value") in callback_inputs
        assert (f"{prefix}-heatmap", "value") in callback_inputs


def test_old_quote_surface_sections_and_callbacks_are_removed(dashboard):
    from test_policy_dashboard_regime_selectors import Input, Output, walk
    old_ids = {"bid-surface-chart", "ask-surface-chart"}
    old_ids |= {f"tier-{side}-surface-{suffix}" for side in ("bid", "ask")
                for suffix in ("heatmap", "regime", "rfq-size", "regime-wrapper", "rfq-size-wrapper")}
    layout_ids = {node.id for node in walk(dashboard.app.layout)
                  if isinstance(getattr(node, "id", None), str)}
    assert old_ids.isdisjoint(layout_ids)
    callback_ids = {binding.args[0] for args, _, _ in dashboard.app.callbacks
                    for binding in args if isinstance(binding, (Input, Output))
                    and isinstance(binding.args[0], str)}
    assert old_ids.isdisjoint(callback_ids)
    assert not hasattr(dashboard, "quote_surface_figure")
    assert dashboard.RFQ_HEATMAP_PREFIXES == PREFIXES
