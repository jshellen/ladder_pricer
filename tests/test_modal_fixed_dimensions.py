"""Direct/Crossed updates must not change the size of either flow-source dialog.

Pin dimensions on the real card nodes, not only in cached CSS, because dynamic
Dash callbacks replace the editor contents while the dialog is open.
"""
from test_policy_dashboard_regime_selectors import dashboard, walk


def test_both_modal_cards_have_inline_fixed_height(dashboard):
    for modal_id in ('flow-source-modal', 'ecn-flow-source-modal'):
        modal = next(node for node in walk(dashboard.app.layout)
                     if getattr(node, 'id', None) == modal_id)
        cards = [node for node in walk(modal)
                 if getattr(node, 'className', None) == 'flow-source-modal-card']
        assert len(cards) == 1
        style = cards[0].style
        assert style['height'] == style['minHeight'] == style['maxHeight']
        assert '100vh' in style['height']
        assert style['flexShrink'] == 0
        assert style['overflow'] == 'hidden'
        assert style['display'] == 'flex'
        assert style['flexDirection'] == 'column'


def test_both_modal_editors_are_scroll_containers(dashboard):
    for editor_id in ('flow-source-editor', 'ecn-flow-source-editor'):
        editor = next(node for node in walk(dashboard.app.layout)
                      if getattr(node, 'id', None) == editor_id)
        assert editor.style['minHeight'] == 0
        assert editor.style['overflowY'] == 'auto'
        assert editor.style['flex'] == '1 1 auto'


def test_opening_and_closing_do_not_recenter_modal(dashboard):
    for callback in (dashboard.render_flow_source_modal_visibility,
                     dashboard.render_ecn_flow_source_modal_visibility):
        assert callback(True) == {'display': 'flex', 'alignItems': 'flex-start'}
        assert callback(False) == {'display': 'none'}
