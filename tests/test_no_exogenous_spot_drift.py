"""Guard against accidentally restoring the old exogenous spot drift input."""
from pathlib import Path

from trinity.defaults import default_config
from trinity.ui_config import build_config
from test_ui_config import ui_values_from_default

ROOT = Path(__file__).resolve().parents[1]


def test_spot_drift_is_not_part_of_model_configuration():
    assert 'spotDrift' not in default_config()
    cfg = build_config(ui_values_from_default())
    assert 'spotDrift' not in cfg
    assert cfg == default_config()


def test_exogenous_spot_drift_removed_from_native_solver_and_analytics():
    checked = (
        'cpp/include/ladder_pricer/core.hpp',
        'cpp/src/core.cpp',
        'cpp/src/policy_operator.cpp',
        'cpp/src/solver.cpp',
        'cpp/src/analytics.cpp',
        'cpp/src/simulation.cpp',
        'cpp/src/python_bindings.cpp',
    )
    for name in checked:
        source = (ROOT / name).read_text()
        assert 'spot_drift' not in source, name
        assert 'base_drift' not in source, name
        assert 'spotDrift' not in source, name


def test_gamma_moved_to_global_and_old_panel_deleted():
    source = (ROOT / 'app.py').read_text()
    global_panel = source.split('panel("Global", [', 1)[1].split('panel("', 1)[0]
    assert '"qmax"' in global_panel
    assert '"gamma"' in global_panel
    assert 'panel("Risk & drift"' not in source
    assert '"drift", State(' not in source
    assert '"drift", DEFAULT[' not in source
