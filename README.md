# Trinity 2.0 — C++ Howard engine + Python + Dash

This is the Python/Dash application architecture for the Trinity ladder pricer:

```text
C++ pricing engine
      ↓ pybind11
thin Python façade
      ↓
Dash + Plotly UI
```

All pricing mathematics remain in C++. Python owns application state/caching and Dash owns presentation.

The repository also preserves the last trusted pure-Python numerical implementation under `python_reference/`. It is isolated from the production import path and is kept only as a readable reference/regression oracle and emergency fallback.

## Numerical core

The C++17 library contains:

- stationary continuous-time HJB;
- Howard policy iteration;
- analytical logistic/Lambert-W quote improvement;
- inventory ladder level/gap constraints;
- hidden inventory solve buffer;
- saturating adverse-selection markout;
- dark-pool controls;
- event-driven fills with an independent continuous market-price process;
- corrected impact-aware closed-form terminal PnL mean and standard deviation.

The Dash layer does **not** implement pricing mathematics.

## Python API

The public Python boundary is intentionally small:

```python
from trinity import Engine, default_config

config = default_config()
engine = Engine(config)

solution = engine.solve()
stats = engine.statistics(570, initial_inventory=0)
mc = engine.simulate(570, paths=1000, initial_inventory=0, seed=12345)
frontier = engine.frontier([0.01, 0.03, 0.1, 0.3], 570, initial_inventory=0)
```

`Engine` is a thin façade over the native pybind11 extension. Heavy calls release the Python GIL. A small Dash-side cache keeps solved engines alive across UI callbacks.

## Dash application

The application has three main work areas:

### Policy

- convergence diagnostics;
- continuation value `h(q)` and internalization-time overview;
- Streamlit-style Tier Diagnostics presented as a single vertical stack of collapsible sections;
- diagnostic order: tier parameters → flow curves → hit ratios → implied hit ratios → markouts → quotes vs inventory → bid surface → ask surface → volume premium / ladder table;
- consistent side colors throughout diagnostics: **bid = blue**, **ask = red**.

### Monte Carlo

- MC mean and standard deviation;
- exact closed-form mean and standard deviation;
- PnL histogram;
- inventory median + 95% interval;
- retained path inventory and spot/fill plots;
- retained spot paths evolve on an internal 15-second market clock even between fills;
- separate event/spot RNG streams keep fills and PnL invariant to UI sampling density;
- fill tape.

## Monte Carlo market clock

RFQ and dark-pool fills remain asynchronous continuous-time events, but the spot
process is not tied to those events. The production C++ simulator advances spot
on an internal 15-second market clock using the configured Brownian volatility,
exogenous drift and outstanding exponential markout impulses. Fill events can
occur between those market ticks and are inserted at their exact event times.

The event-arrival RNG and spot RNG are independent. The regular inventory/output
sampling grid therefore affects only what is returned to the UI; changing
`sample_points` cannot change the fill sequence, terminal inventory, terminal
spot or path PnL for a fixed seed.

## Efficient frontier

- closed-form PnL standard deviation vs expected PnL;
- risk parameter `φ` / `γ` vs `E[PnL] / Std[PnL]`;
- underlying frontier table and Howard iteration counts.

## Install

A normal Python environment with a C++17 compiler is enough. No Emscripten, Node, React, Boost, Eigen or SciPy are required.

On macOS, install Xcode command-line tools if needed:

```bash
xcode-select --install
```

Then from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`pip install -e .` installs Dash/Plotly/NumPy, installs pybind11 as a build dependency, and compiles the native C++ extension for your machine.

## Run

```bash
source .venv/bin/activate
./run_dash.sh
```

or:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8050
```

## Native C++ regression tests

The numerical core can also be tested without Python:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/test_howard
./build/test_analytics
./build/test_simulation
```

Expected reference outputs for the Tier-3 EURSEK stress calibration are approximately:

```text
rho        = 5.01762e-05
mean PnL   = 4603.94 EUR
PnL stdev  = 12451.4 EUR
```

## Project layout

```text
cpp/
  include/ladder_pricer/     # clean C++ domain/numerics API
  src/                       # implementation + pybind11 adapter
  tests/                     # native regression tests
python/trinity/
  engine.py                  # tiny Python façade
  service.py                 # server-side engine cache
  defaults.py                # UI defaults
assets/
  style.css                  # Dash application styling
app.py                       # Dash layout and callbacks
setup.py                     # native extension build
pyproject.toml               # build dependencies
```

## Pure-Python reference solver

The production backend remains C++, but the previous numerical Python implementation is retained deliberately:

```text
python_reference/
├── ladder_pricer/__init__.py   # complete pure-Python Howard engine
├── tests/                      # original numerical regression tests
├── compare_with_cpp.py         # side-by-side C++ vs Python check
└── README.md
```

It is outside the normal `python/` package tree, so installing/running Trinity does not import it.

To run its tests explicitly:

```bash
PYTHONPATH=python_reference python -m pytest -q python_reference/tests
```

After the main C++ extension is installed, compare the two implementations with:

```bash
python python_reference/compare_with_cpp.py
```

The reference implementation should not be used by the Dash application; C++ remains the canonical production numerical backend.


## Visualization parity with the Streamlit research app

The Dash application now carries the full diagnostic plot set from the previous Streamlit version:

- continuation value `h(q)`
- Howard Bellman-residual and value-change convergence histories
- internalization time vs inventory
- tier flow curves and logistic hit-ratio curves
- implied hit ratios under the solved policy
- saturating markout curves by size
- quotes vs inventory across all rungs
- bid and ask 3D quote surfaces
- volume-premium ladder at a selected inventory
- flow/markout parameter and ladder tables
- dark-pool fill-size density, full-fill probability and solved posted-size policy
- Monte Carlo PnL distribution and inventory confidence band
- exact event-time retained inventory path
- spot + tier quotes + executed fills for a selected ticket size
- fill counts by tier and side, plus the fill tape
- efficient frontier and risk-adjusted PnL vs phi

C++ remains the production numerical backend. The diagnostic plots are computed in Python from the solved C++ policy and the same model configuration; no pricing mathematics has been duplicated into Dash callbacks.

## Dash configuration mapping

Dash control values are converted to the numerical model by component ID in
`python/trinity/ui_config.py`. The model configuration no longer depends on the
positional ordering of callback states, so adding or reordering UI controls does
not silently shift tier parameters. The regression tests in
`tests/test_ui_config.py` cover the markout checkbox / impact-parameter mapping.

### Dark-pool diagnostics layout

The Policy → Dark pool tab mirrors the Streamlit-style diagnostic layout: plots are stacked vertically in independent collapsible sections for arrival-size density, full-fill probability, and the solved dark-pool policy (with its table).
