"""Prevent mixing compiled engines from the global- and per-tier-markout models."""
from types import SimpleNamespace
import pytest


def test_native_schema_guard_rejects_previous_binary():
    from trinity.engine import _check_pricing_schema
    with pytest.raises(RuntimeError, match="does not support tier-specific RFQ markout"):
        _check_pricing_schema(SimpleNamespace(ECN_PARAMETERIZATION_VERSION=10))


def test_native_schema_guard_accepts_current_binary():
    from trinity import _native
    from trinity.engine import _check_pricing_schema
    assert _native.PRICING_CONFIG_SCHEMA_VERSION == 2
    _check_pricing_schema(_native)
