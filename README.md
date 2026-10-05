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
- optional Crisafi-style passive ECN hedge control: `NONE` or a Tier-style price-improvement delta `d`. Here `d=0` is the same-side touch and `d=0.5` is mid; the default action grid is 0.00 to 0.50 in 0.01 increments, and ECN fills are strictly risk-reducing;
- event-driven fills with an independent continuous market-price process;
- corrected impact-aware closed-form terminal PnL mean and standard deviation.

The Dash layer does **not** implement pricing mathematics.


### Aggregating customer flow into one target FX risk

A customer tier can now be driven by several FX-pair flow sources while the HJB keeps a
single target-pair quote control and inventory state. Existing configs that provide only
`tier["flow"]` are unchanged. To combine EURSEK and crossed USDSEK flow, use
`tier["flowSources"]` instead:

```python
config["tiers"][0]["flowSources"] = [
    {
        "name": "EURSEK",
        "flow": config["tiers"][0]["flow"],
        "mapping": {"type": "identity"},
    },
    {
        "name": "USDSEK",
        "flow": {
            # One-sided exogenous RFQ intensity curve
            # lambda_RFQ(z) = A0 * z^(-theta-beta*z).
            "A0": 0.02,
            "theta": 0.20,
            "beta": 0.08,
            "steepness": 7.0,
            "shift": 0.50,
            "volumeShift": 0.02,
        },
        "mapping": {
            "type": "crossed",
            "crossMid": 1.18,          # EURUSD: USD per EUR
            "sourceSpreadPips": 18.0, # USDSEK full customer spread
        },
    },
]
```

For a crossed source the target EURSEK delta is converted to the source-pair delta by

```text
source_delta = 0.5 + delta_scale * (target_delta - 0.5)
delta_scale  = target_spread / (cross_mid * source_spread)
```

and a target EUR-equivalent size `z` is evaluated on the source hit-ratio curve at
`source_size = cross_mid * z`. The calibrated exogenous intensity curve itself defines the
RFQ-size distribution. Customer RFQ sizes are independent of the pricing-control knots:
by default Monte Carlo evaluates the curve on a 1M grid from the smallest to the largest
pricing size, normalizes those values into probabilities, and samples the requested size
from that full support.  Quotes for customer sizes between pricing knots are obtained by
linear interpolation of the neighboring prices.

Because several transformed logistic curves no longer have the single-curve Lambert-W
optimum, aggregated tiers use a bounded one-dimensional numerical quote optimization.
Single-source legacy tiers retain the original analytical fast path.

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
ECN quotes use the same normalized price-improvement convention as customer tiers: `d=0` is
the same-side touch and `d=0.5` is mid. If `S` is the full spread, the quote depth from mid
is `S * (0.5-d)`.

The ECN can now aggregate several streamed FX pairs into one target-risk quote control. The
direct target pair and every crossed ECN source have their own exponential price-reach curve
`lambda_s(d_s)=A_s exp(-k_s(0.5-d_s))` and their own exponential parent-trade size distribution.
`meanTradeSize` is the mean parent size in millions of the source pair's base currency. The
optimizer still chooses only one master target-pair delta `d`. For a crossed source,

```text
d_source   = 0.5 + alpha * (d_target - 0.5)
alpha      = target_spread / (cross_mid * source_spread)
sourceSize = cross_mid * targetSize
```

so EURSEK can be the master quote while USDSEK and GBPSEK inherit economically equivalent
quotes derived through EURUSD and EURGBP. If `V_s` is the source-pair parent trade size and
`c_s = sourceSizePerTarget`, the target-equivalent executed size is

```text
V_s ~ Exp(mean = meanTradeSize_s)
fill_target = min(V_s, c_s * quoteSize_target) / c_s
```

so a posted 1M target quote may fill by any smaller positive amount but can never fill above
1M. The HJB integrates each source's capped size distribution when evaluating continuation
value, spread capture and fees. Monte Carlo samples the same distribution directly. Source
`s` still has parent trade intensity `A_s`, independent price reach `X_s ~ Exp(k_s)`, and the
quote is hit when `X_s >= alpha_s * (0.5-d_target)`. Retained paths record the source parent
trade size and realized target fill size for ECN arrivals.

The optimizer evaluates `NONE` plus a 0.01 grid between configurable `minDelta` and `maxDelta`,
defaulting to 0.00 and 0.50. One 0.01 step is one percentage point of spread. At every
inventory state the Howard improvement compares `NONE` with every allowed delta. ECN quotes
are hedge-only: positive inventory may only post an ask, negative inventory may only post a
bid, flat inventory posts nothing, and the posted size itself must be risk reducing so no
realized partial or full fill can cross through zero and create opposite-side risk. A realized
passive ECN fill of size `v` at target delta `d` earns
`v * (spread * (0.5-d) - maker fee)` before continuation-value effects.

### Monte Carlo

- MC mean and standard deviation;
- exact closed-form mean and standard deviation;
- PnL histogram;
- inventory median + 95% interval;
- retained path inventory and spot/fill plots;
- retained spot paths evolve on an internal one-second market clock even between fills;
- separate event/spot RNG streams keep fills and PnL invariant to UI sampling density;
- a modal progress bar reports genuinely completed native paths during long runs;
- fill composition is aggregated over every simulated path and shown as percentage
  shares of all trades or all executed volume by tier/venue and side;
- realized RFQ hit ratios pool wins and RFQ requests over every simulated path for
  each RFQ size, rather than relying on the small retained-path sample;
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

- **Passive ECN diagnostics** show every direct/crossed exponential-flow source, its FX mapping, source-specific and total fill curves, optimal trade intensity vs inventory, quotes vs inventory, and the discrete master-delta policy. The ECN action grid is configurable from `minDelta` to `maxDelta` in fixed 0.01 increments (one percentage point of spread), plus `NONE`.
- **Inventory grid** is fixed to uniform 1M spacing; only the maximum absolute inventory is configurable.


- Inventory grid is fixed to a uniform 1M spacing from `-maxAbs` to `+maxAbs`; the nonuniform/piecewise grid has been removed from the production and reference comparison paths.


### Passive ECN exponential flow

Each passive ECN source has its own parent trade intensity `A_s` and reach distribution `X_s ~ Exp(k_s)`. Direct sources use the master delta unchanged. Crossed sources use the same affine FX mapping as Tier flow sources, so their target-delta fill contribution is `A_s exp(-k_s alpha_s (0.5-d))`. The total ECN fill intensity is the sum across sources. The default direct EURSEK source remains A=4.0 trades/min per side, k=8.4, with a 0.00-to-0.50 master-delta grid in 0.01 increments.


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

The **exogenous RFQ-arrival intensity curve itself defines the RFQ-size density**. For each
customer flow source `s`, the calibrated one-sided curve is

```text
lambda_RFQ,s(z) = A0_s * z^(-theta_s - beta_s*z).
```

Pricing sizes and customer RFQ sizes are deliberately separate.  A tier can, for example,
optimize prices only at

```text
pricing knots = {1, 2, 3, 5, 10, 20} M
```

while customer RFQs are sampled on the full grid

```text
RFQ support = {1, 2, 3, 4, ..., 20} M
```

when `rfqSizeStep = 1.0`.  The support runs from the smallest to the largest pricing knot;
pricing knots are always included exactly.  `rfqSizeStep` is configurable per tier.

For each source, evaluate the exogenous curve on that full RFQ support. The Monte Carlo
size PMF is

```text
p_s,j = lambda_RFQ,s(z_j) / sum_k lambda_RFQ,s(z_k),
```

and the pooled two-sided source clock is

```text
Lambda_s = 2 * sum_j lambda_RFQ,s(z_j).
```

There is therefore **no independent RFQ-size probability calibration**. Each customer RFQ
is generated as

```text
source RFQ clock:          Lambda_s
customer side:             Bid / Ask with probability 1/2 each
requested size:            z_j with probability p_s,j
quoted price:              linear interpolation between neighboring pricing knots
conditional trade win:     Bernoulli(p_win,s(interpolated_price, z_j))
```

For example, if 3M and 5M are pricing knots and a 4M RFQ arrives, the 4M bid or ask price is
exactly halfway between the current 3M and 5M prices. In the code the equivalent `delta` is
interpolated; because execution price is affine in `delta` for a fixed side and reference
spot, this is mathematically identical to interpolating the price itself.

The sampled **actual RFQ size** drives win probability, cash, inventory, markout, retained
RFQ events, and the population hit-ratio aggregates.  Hence the realized hit-ratio chart can
show 4M, 6M, 7M, etc. even though those sizes are not explicit pricing knots.

The HJB continues to store the control policy at the pricing knots. Its existing pricing-knot
optimization is unchanged; Monte Carlo applies the resulting ladder to intermediate customer
sizes via the same linear price interpolation used in production-style quoting.

Crossed sources evaluate the exogenous curve and hit-ratio model at their mapped source-currency
size before normalization / win sampling. For compatibility, configs produced by the short-lived
`totalRfqRate` version are converted back to an equivalent `A0` using the full RFQ support.

Once an RFQ arrives, only a won RFQ changes inventory and cash. The RFQ markout shock is
applied on every RFQ, including lost RFQs; for a won RFQ, inventory/cash are updated before
the markout is applied.


### Temporary FX reference mids

The Dash flow-source editor currently obtains crossed-source reference mids from
`trinity.defaults.DEFAULT_FX_MIDS`. `default_fx_mid(pair)` also resolves inverse
orientations (for example `USDEUR` from `EURUSD`). This is intentionally a
temporary market-data boundary: a production deployment should replace the
default lookup with the bank's/database market-data source while leaving the
flow-source and HJB interfaces unchanged.
