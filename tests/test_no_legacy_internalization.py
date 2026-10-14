"""The inactive internalization-time proxy must not re-enter the model."""
from pathlib import Path

from trinity.defaults import default_config
from trinity.ui_config import build_config
from test_ui_config import ui_values_from_default


def test_removed_proxy_not_in_native_config_or_dash_values():
    cfg = default_config()
    assert "internalization" not in cfg
    values = ui_values_from_default()
    assert not any(key in values for key in ("tau0", "tau1", "tau2"))
    actual = build_config(values)
    assert "internalization" not in actual
    assert actual == cfg


def test_retained_rfq_markout_tau_is_distinct_from_removed_proxy():
    actual = build_config(ui_values_from_default())
    assert actual["tiers"][0]["markout"]["tauSeconds"] > 0.0
    assert "volatilityRegimes" in actual


def test_dashboard_has_no_orphaned_legacy_proxy_components():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    for forbidden in (
        '"internalization-chart"',
        '"tau0"', '"tau1"', '"tau2"',
        'internalization_figure(',
        'Legacy internalization proxy',
    ):
        assert forbidden not in source
