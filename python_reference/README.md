# Pure-Python numerical reference

This directory preserves the last trusted pure-Python Howard implementation from before the production C++ rewrite.

It is **not** imported by the Dash application and it is intentionally outside the normal `python/` package tree. The production path is:

```text
C++ Howard engine -> pybind11 -> trinity.Engine -> Dash
```

The reference path exists only as a readable implementation, regression oracle, and emergency fallback:

```text
python_reference/ladder_pricer/__init__.py
```

It contains the same major numerical features that existed in the final Python model:

- stationary ergodic HJB;
- Howard policy iteration;
- analytical logistic/Lambert-W quote improvement;
- operational + hidden inventory grids;
- ladder level/gap constraints with Jacobi-style cross-inventory propagation;
- saturating markout model;
- dark-pool controls;
- event-driven Monte Carlo;
- impact-aware closed-form terminal PnL mean and standard deviation.

## Run the reference tests

From the repository root:

```bash
PYTHONPATH=python_reference python -m pytest -q python_reference/tests
```

`NumPy` is the only numerical dependency of the reference solver itself.

## Import it explicitly

The main application never modifies `sys.path` to include this folder. For ad-hoc use:

```python
import sys
sys.path.insert(0, "python_reference")
import ladder_pricer as reference
```

Do not install `python_reference` as a normal site package in the same environment. Keeping it isolated prevents it from being confused with the production C++/pybind backend.

## Compare against C++

After installing the main project (`python -m pip install -e .`), run:

```bash
python python_reference/compare_with_cpp.py
```

The comparison uses the Tier-3 EURSEK stress calibration and reports the Howard average reward plus analytical PnL mean/stdev from both implementations.
