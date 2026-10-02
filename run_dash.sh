#!/usr/bin/env bash
set -euo pipefail

# Keep the pybind11 module synchronized with the C++ pricing sources.
# setuptools skips recompilation when nothing has changed.
python3 setup.py build_ext --inplace
exec python3 app.py
