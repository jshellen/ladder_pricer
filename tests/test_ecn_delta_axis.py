"""Optimal passive ECN delta line charts always use the full 0–0.5 axis."""
from __future__ import annotations

import pytest

from test_policy_dashboard_regime_selectors import dashboard


@pytest.mark.parametrize("delta", [0.01, 0.20, 0.50])
def test_ecn_delta_chart_is_fixed_to_half(dashboard, delta):
    fig = dashboard.passive_ecn_policy_figure({
        "passiveEcn": {
            "qGrid": [-1.0, 0.0, 1.0],
            "bidDelta": [delta, delta, delta],
            "askDelta": [delta, delta, delta],
            "bidActive": [True, False, False],
            "askActive": [False, False, True],
        }
    })
    assert tuple(fig.layout.yaxis.range) == (0.0, 0.5)
    assert fig.layout.yaxis.autorange is False
    assert len(fig.data) == 4


def test_ecn_delta_chart_has_fixed_axis_before_solving(dashboard):
    fig = dashboard.passive_ecn_policy_figure({})
    assert tuple(fig.layout.yaxis.range) == (0.0, 0.5)
