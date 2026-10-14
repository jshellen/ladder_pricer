"""ECN dashboard intensity labels, total-rate scaling, and removed UI."""
from __future__ import annotations

import math

import pytest

from test_policy_dashboard_regime_selectors import dashboard, walk


def two_source_cfg(max_delta=0.5):
    return {
        "targetPair": "EURSEK", "spreadPips": 20.0,
        "passiveEcn": {
            "enabled": True, "minDelta": 0.0, "maxDelta": max_delta,
            "deltas": [0.0, 0.1, max_delta],
            "flowSources": [
                {"name": "EURSEK", "flow": {"A": 3.0, "k": 10.0},
                 "mapping": {"type": "identity"}},
                {"name": "USDSEK", "flow": {"A": 2.0, "k": 6.0},
                 "mapping": {"type": "affine", "deltaScale": 0.8,
                             "sourceSizePerTarget": 1.0}},
            ],
        },
    }


def sample_ecn_policy(bid_delta=0.15, ask_delta=0.20):
    return {"passiveEcn": {
        "qGrid": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "bidDelta": [bid_delta, bid_delta, 0.0, 0.0, 0.0],
        "askDelta": [0.0, 0.0, 0.0, ask_delta, ask_delta],
        "bidActive": [True, True, False, False, False],
        "askActive": [False, False, False, True, True],
    }}


def test_trade_intensity_curve_labels_and_total_scatter(dashboard):
    cfg = two_source_cfg()
    fig = dashboard.passive_ecn_fill_figure(cfg)
    assert fig.layout.title.text == "Trade intensity curves"
    assert fig.layout.xaxis.title.text == "Target pair price improvement"
    assert fig.layout.yaxis.title.text == "trades / min"
    assert [t.name for t in fig.data] == ["EURSEK", "USDSEK", "Total", "Total"]
    assert fig.data[-1].mode == "markers"
    assert fig.data[-1].showlegend is False  # avoid two 'Total' legend entries
    assert float(fig.data[-1].y[-1]) == pytest.approx(5.0)
    assert list(fig.layout.xaxis.range) == [0.0, 0.5]


def test_targeted_trade_intensity_axis_is_full_total_rate(dashboard):
    cfg = two_source_cfg()
    max_total = dashboard._ecn_max_total_intensity(cfg)
    assert max_total == pytest.approx(5.0)  # A_EURSEK + A_USDSEK at mid
    for bid, ask in [(0.0, 0.0), (0.15, 0.45)]:
        fig = dashboard.passive_ecn_implied_hit_ratio_figure(
            sample_ecn_policy(bid_delta=bid, ask_delta=ask), cfg,
        )
        assert fig.layout.title.text == "Targeted trade intensity"
        assert fig.layout.xaxis.title.text == "Inventory EURm"
        assert fig.layout.yaxis.title.text == "trades / min"
        assert tuple(fig.layout.yaxis.range) == pytest.approx((0.0, 5.0))
        assert fig.layout.yaxis.autorange is False
        for trace in fig.data:
            if trace.mode == "markers":
                assert all(float(y) <= max_total for y in trace.y if y is not None and math.isfinite(y))


def test_heatmap_color_scale_matches_total_intensity(dashboard):
    cfg = two_source_cfg()
    sample = sample_ecn_policy()
    solved = {"regimes": [
        {"sigmaPips": sigma, "passiveEcn": sample["passiveEcn"]}
        for sigma in (12.0, 35.0)
    ]}
    fig = dashboard.passive_ecn_regime_heatmap_figure(solved, cfg, "intensity")
    assert fig.layout.title.text == "Targeted trade intensity · all volatility regimes"
    assert fig.layout.coloraxis.cmin == 0.0
    assert fig.layout.coloraxis.cmax == pytest.approx(5.0)
    assert fig.layout.coloraxis.colorbar.title.text == "trades / min"
    assert fig.layout.xaxis2.title.text == "Inventory EURm"


def test_old_relative_intensity_section_and_callback_removed(dashboard):
    expanders = [node for node in walk(dashboard.policy_tab) if node.type == "Details"]
    headings = [node.children for section in expanders for node in walk(section)
                if node.type == "Summary"]
    assert "Relative intensity" not in headings
    assert not hasattr(dashboard, "passive_ecn_hit_ratio_figure")
    ids = {node.id for node in walk(dashboard.app.layout)
           if isinstance(getattr(node, "id", None), str)}
    assert "ecn-hit-ratio-chart" not in ids
    callback_outputs = {
        arg.args[0] for args, _, _ in dashboard.app.callbacks for arg in args
        if getattr(arg, "__class__", None).__name__ == "Output"
    }
    assert "ecn-hit-ratio-chart" not in callback_outputs
    assert "Trade intensity curves" in headings
    assert "Targeted trade intensity" in headings


def test_unavailable_solution_produces_six_outputs(dashboard):
    assert len(dashboard.render_passive_ecn_diagnostics(None, two_source_cfg(), 0, 0, 0)) == 6
    cfg = two_source_cfg()
    cfg["passiveEcn"]["enabled"] = False
    assert len(dashboard.render_passive_ecn_diagnostics({"regimes": []}, cfg, 0, 0, 0)) == 6
