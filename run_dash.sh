#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
# Prefer this checkout over any previously installed editable Ladder Pricing.
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

# Older compiled binaries expect a global config["markout"] even though the
# current Python config has per-tier markout. Schema 2 explicitly recognizes
# per-tier markout; the ECN parameterization version alone cannot detect it.
if ! python3 -c 'from trinity import _native; assert getattr(_native, "PRICING_CONFIG_SCHEMA_VERSION", None) == 2' >/dev/null 2>&1; then
    echo "Outdated or missing native extension: rebuilding for per-tier RFQ markout..."
    # pip's isolated build installs pybind11 as needed, unlike directly running
    # setup.py, and works when the user changed to a freshly extracted ZIP.
    python3 -m pip install -e . --no-deps --force-reinstall
fi

exec python3 app.py
