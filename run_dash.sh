#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
# Prefer this checkout over any previously installed editable Ladder Pricing.
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

# Reject older native modules that may still use the markout X-state solver.
if ! python3 -c 'from trinity import _native; assert getattr(_native, "PRICING_CONFIG_SCHEMA_VERSION", None) == 11' >/dev/null 2>&1; then
    echo "Outdated or missing native extension: rebuilding production C++ (inventory, stochastic volatility) HJB..."
    # pip's isolated build installs pybind11 as needed, unlike directly running
    # setup.py, and works when the user changed to a freshly extracted ZIP.
    python3 -m pip install -e . --no-deps --force-reinstall
fi

exec python3 app.py
