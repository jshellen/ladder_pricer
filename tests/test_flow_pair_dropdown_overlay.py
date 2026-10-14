"""Keep Tier/ECN pair selectors aligned and on top of later expanders."""
from pathlib import Path
from test_policy_dashboard_regime_selectors import dashboard, walk

ROOT = Path(__file__).resolve().parents[1]


def test_no_redundant_direct_heading_in_either_popup(dashboard):
    direct_tier = dashboard._direct_flow_source(dashboard.DEFAULT['tiers'][0]['flow'], 'EURSEK')
    direct_ecn = dashboard._direct_ecn_flow_source({'A': 0.15, 'k': 8.4}, 'EURSEK')
    for editor in (dashboard.flow_source_settings_editor('direct', direct_tier, 'EURSEK'),
                   dashboard.ecn_flow_source_settings_editor('direct', direct_ecn, 'EURSEK')):
        assert not any(getattr(x, 'className', None) in ('flow-source-row-head', 'flow-source-title')
                       for x in walk(editor))
        assert not any(getattr(x, 'children', None) == 'EURSEK'
                       for x in walk(editor) if isinstance(getattr(x, 'children', None), str))


def test_source_pair_fields_and_target_picker_keep_their_ids(dashboard):
    tier = dashboard._default_cross_flow_source(0, 'EURSEK')
    ecn = dashboard._default_cross_ecn_flow_source(0, 'EURSEK')
    for editor, pair_id, cross_id in (
        (dashboard.flow_source_settings_editor('extra-0', tier, 'EURSEK'), 'flow-edit-pair', 'flow-edit-cross-pair'),
        (dashboard.ecn_flow_source_settings_editor('extra-0', ecn, 'EURSEK'), 'ecn-flow-edit-pair', 'ecn-flow-edit-cross-pair'),
    ):
        ids = [getattr(x, 'id', None) for x in walk(editor)]
        assert pair_id in ids and cross_id in ids
    layout_ids = [getattr(x, 'id', None) for x in walk(dashboard.app.layout)]
    assert 'flow-direct-target-display' in layout_ids
    assert 'ecn-direct-target-display' in layout_ids


def test_pair_menu_layers_stack_above_expanders():
    css = (ROOT / 'assets/style.css').read_text()
    assert '.flow-source-modal .flow-source-header-fields {' in css
    assert '.flow-source-modal .flow-target-picker,' in css
    assert '.flow-source-modal .flow-source-dynamic-expanders,' in css
    assert '.flow-source-modal .flow-pair-selectors > .field:focus-within' in css
    assert '.flow-source-modal .flow-pair-selectors .dash-dropdown-content' in css
