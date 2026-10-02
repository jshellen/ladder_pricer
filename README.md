# FX Ladder Pricer — C++ Howard engine + Python + Dash

This is the Python/Dash application architecture for the FX ladder pricer:

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
- per-tier trading fees that enter quote optimization, analytical PnL and Monte Carlo PnL;
- dark-pool controls;
- optional Crisafi-style passive ECN hedge control: `NONE` or a quote distance from the moving mid/reference in pips. The default action grid runs from 0 to 20 pips in 0.5-pip increments, and ECN fills are strictly risk-reducing;
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
- diagnostic order: tier parameters → exogenous RFQ arrival intensity → won-trade intensity → win probabilities → implied win probabilities → markouts → quotes vs inventory → bid surface → ask surface → volume premium / ladder table;
- consistent side colors throughout diagnostics: **bid = blue**, **ask = red**;
- each tier exposes a configurable **Fee [pips]**. The fill edge is `size * (spread * (0.5 - delta) - fee)`, so fees directly influence the solved quote ladder as well as realized/analytical PnL.

### Passive ECN hedge

The optional passive ECN module is deliberately reduced-form and does not use `xfills`.
ECN quotes use their own coordinate system: `delta` is the absolute distance from the moving
mid/reference, measured directly in pips. Buy and sell ECN market trades are modeled as two
independent Poisson processes, each with exogenous rate `A`. When a market trade arrives, its
distance `D` from mid is sampled from `Exp(k)`. An active quote at depth `delta` fills iff
`D >= delta`, so the implied quote-fill intensity is exactly
`lambda_fill(delta) = A exp(-k delta)`. Defaults are `A=0.15` trades/min **per side** and
`k=0.24 1/pip`. The optimizer evaluates `NONE` plus a 0.5-pip grid between configurable
`minDistancePips` and `maxDistancePips`, which default to 0 and 20 pips.
At every inventory state the Howard improvement compares `NONE` with every allowed distance.
ECN quotes are hedge-only: positive inventory may only post an ask, negative inventory may
only post a bid, flat inventory posts nothing, and a fill is never allowed to cross through
zero and create risk on the opposite side. A passive ECN fill at distance `delta` earns
`size * (delta pips - maker fee)` before continuation-value effects.

### Monte Carlo

- MC mean and standard deviation;
- exact closed-form mean and standard deviation;
- PnL histogram;
- inventory median + 95% interval;
- retained path inventory and spot/fill plots;
- retained spot paths evolve on an internal one-second market clock even between fills;
- separate event/spot RNG streams keep fills and PnL invariant to UI sampling density;
- fill tape.

## Monte Carlo market clock

RFQ and dark-pool fills remain asynchronous continuous-time events, but the spot
process is not tied to those events. Retained paths shown in the UI advance spot
on an internal one-second market clock using the configured Brownian volatility,
exogenous drift and outstanding exponential markout impulses. Fill events can
occur between those one-second ticks and are inserted at their exact event times.

For performance, non-retained Monte Carlo paths propagate spot directly from one
fill event to the next. Under the current Brownian + deterministic drift/markout
dynamics this is mathematically equivalent in distribution to subdividing every
interval into one-second steps, while avoiding a large runtime penalty.

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

It is outside the normal `python/` package tree, so installing/running the application does not import it.

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
`tests/test_ui_config.py` cover the markout checkbox / impact-parameter mapping and tier-fee mapping.

### Dark-pool diagnostics layout

The Policy → Dark pool tab mirrors the Streamlit-style diagnostic layout: plots are stacked vertically in independent collapsible sections for arrival-size density, full-fill probability, and the solved dark-pool policy (with its table).


### UI diagnostics

- **Passive ECN diagnostics** show exponential-flow parameters, the flow curve, relative intensity, optimal trade intensity vs inventory, quotes vs inventory, and the discrete optimal distance policy. The ECN action grid is configurable from min to max distance from mid in fixed 0.5-pip increments (plus `NONE`).
- **Inventory grid** is fixed to uniform 1M spacing; only the maximum absolute inventory is configurable.


- Inventory grid is fixed to a uniform 1M spacing from `-maxAbs` to `+maxAbs`; the nonuniform/piecewise grid has been removed from the production and reference comparison paths.


### Passive ECN exponential flow

Passive ECN market trades arrive independently on the buy and sell sides at rate `A`. Their distance from mid is `D ~ Exp(k)`, so an active quote at depth `delta` has fill intensity `lambda_fill(delta)=A exp(-k delta)`. Defaults: A=0.15 trades/min per side, k=0.24 1/pip, and a 0-to-20-pip quote grid in 0.5-pip increments.


## Dark pool model

The dark pool uses an exogenous incoming-order process on each side. Incoming sell orders
arrive at `lambda` and can hit our posted bid; incoming buy orders arrive at
`lambda` and can hit our posted ask. At each arrival, incoming size `X` is sampled
from the zero-inflated model: size zero with probability `p0`, otherwise a
zero-truncated Poisson with mean parameter `mu`. If our posted size is `u`, the
realized fill is `min(X, u)`. Thus an incoming 3M order fills 3M of a 5M post, while
an incoming 5M order fills all of a 1M post. Dark-pool executions occur at mid.
The HJB/analytics use the equivalent transition rates implied by this same capped-size
mechanism. The previous geometric fill-size model has been removed.


## Customer RFQs in Monte Carlo

Customer tiers are simulated as a two-stage RFQ process.  For each tier, side,
and requested size `z`, RFQs arrive exogenously at
`lambda_RFQ(z) = A0 * z^(-theta - beta*z)`.  Once an RFQ arrives, the current
quote determines the conditional win probability through the logistic curve.
Only a won RFQ changes inventory and cash.  The RFQ markout shock is applied on
every RFQ, including lost RFQs; for a won RFQ, inventory/cash are updated before
the markout is applied.  The product `lambda_RFQ(z) * p_win(delta,z)` remains
the won-trade intensity used by the current HJB/closed-form policy machinery.
