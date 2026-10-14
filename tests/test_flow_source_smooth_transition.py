"""Both flow-source popups retain a fixed outer size across source switches."""
from pathlib import Path

from test_policy_dashboard_regime_selectors import dashboard, walk


ROOT = Path(__file__).resolve().parents[1]


def test_both_editors_have_stable_ids(dashboard):
    ids = {getattr(node, "id", None) for node in walk(dashboard.app.layout)
           if isinstance(getattr(node, "id", None), str)}
    assert "flow-source-editor" in ids
    assert "ecn-flow-source-editor" in ids
    for name in ("flow-source-cards", "ecn-flow-source-cards", "flow-source-settings", "ecn-flow-source-settings"):
        assert name in ids


def test_fixed_height_popup_and_scrollable_editor():
    css = (ROOT / "assets/style.css").read_text()
    assert "height: min(800px, calc(100dvh - 56px))" in css
    assert "max-height: none" in css
    assert ".flow-source-modal-card > .flow-source-editor" in css
    assert "overflow-y: auto" in css
    assert "flex: 1 1 auto" in css
    assert "min-height: 0" in css
    assert ".flow-source-modal-card > .modal-actions" in css
    assert "flex: 0 0 auto" in css


def test_only_subtree_fade_no_resizing_script():
    css = (ROOT / "assets/style.css").read_text()
    assert "flow-editor-content-enter" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "flow-source-header-fields" in css
    assert "align-items: flex-start" in css
    # A previous script animated the editor height and made the popup shrink.
    assert not (ROOT / "assets/flow_source_transitions.js").exists()
    assert "transform: translateY(6px)" not in css
