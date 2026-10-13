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
- independent two-parameter adverse-selection markout curve for each RFQ tier (EUR / traded EUR 1M);
- per-tier trading fees that enter quote optimization, analytical PnL and Monte Carlo PnL;
- dark-pool controls;
- optional Crisafi-style passive ECN hedge control: `NONE` or a Tier-style price-improvement delta `d`. Here `d=0` is the same-side touch and `d=0.5` is mid; the default action grid is 0.00 to 0.50 in 0.01 increments, and ECN fills are strictly risk-reducing;
- event-driven fills with an independent continuous market-price process;
- corrected impact-aware closed-form terminal PnL mean and standard deviation.

The Dash layer does **not** implement pricing mathematics.

### RFQ markout: simplified curve, independently calibrated by tier

Every RFQ tier now has its **own** two-parameter markout curve, configured
inside `config["tiers"][i]["markout"]`:

`m_i(t_seconds) = m_inf_i * (1 - exp(-t_seconds / tau_seconds_i))`

`m_inf_i` is a **negative** adverse markout in EUR per traded EUR 1M, and
`tau_seconds_i > 0` is the curve's time constant in seconds. Each tier's
curve is **independent of RFQ size within that tier**. Its existing `useMarkout`
toggle still enables/disables that tier's markout.

All three tiers initially use the same empirical defaults (`-50.8 EUR/M`,
`36.7 seconds`) for a common starting calibration. The optimized policies
will differ from the previous fixed-internalization-time reward model. Configure different
values in **Pricing tiers → Tier N → RFQ markout**. The tier diagnostics chart
plots the selected tier's curve and overlays other enabled tiers as dashed lines.

The native builder converts each tier's magnitude into quote/base price units:
`-m_inf_i * spot / 1e6` (e.g. `50 EUR/M * 11.5 SEK/EUR / 1e6 =
0.000575 SEK/EUR`). A customer buy generates an upward price impact and a
customer sell a downward price impact, as before. Each native Tier owns its
own `SaturatingMarkout`, and the existing HJB, Monte Carlo and analytical PnL
code use each tier's curve. `tau_seconds_i` is converted into minutes for C++.

### Policy-dependent expected MTM (active C++ solver)

The HJB no longer uses `tau0/tau1/tau2` as a proxy for the post-trade
markout holding period. Those input fields remain visible only as a clearly
labelled **legacy/inactive** diagnostic; they have no effect on the HJB policy.

For every fixed policy π the combined RFQ, ECN and dark-pool inventory
transition generator is G^π. For each distinct tier markout time constant τ,
we solve the resolvent:

```text
(I - τ G^π) u_τ = q_grid.
```

Here `u_τ(q) = E_q[Q_E]` for an independent exponentially distributed time
E with mean τ. In particular, this incorporates *all future inventory fills*
without estimating an exogenous average internalization time or adding a new
HJB state variable.

Using dealer inventory direction `d = +1` for a customer SELL (dealer BID)
and `d = -1` for a customer BUY (dealer ASK), the gross expected MTM of a
won RFQ after changing inventory q -> q+d*z is

```text
MTM_gross = -d * M_inf_tier * u_τ(q+d*z)
```

where M_inf_tier >= 0 is the native, positive adverse price-change magnitude.
Gross MTM is in *millions of quote currency*. It can be positive when the
client's directional markout helps an existing position that remains after
a risk-reducing fill. The public solve output converts this to EUR/event
using 1e6/spot.

**Critical comparison: the price moves even when we LOSE an RFQ.** As with
Monte Carlo, an unfilled customer RFQ affects the market price but does not
change inventory. The native model assumes exogenous buys/sells arrive with
identical intensities, so the two unfilled baseline price shifts cancel in
expectation. Therefore the reward improvement from *winning instead of
losing* a specific RFQ is

```text
MTM_increment = -d * M_inf_tier * [u_τ(q+d*z) - u_τ(q)].
```

That is the term entering the quote Hamiltonian and expected HJB reward;
`MTM_gross` is **not** subtracted once again or multiplied by z. Otherwise
the model would double-charge the same information shock or ignore lost
RFQs. In `solution["expectedMtm"]`, both arrays appear for every active tier,
RFQ pricing size, inventory state, and bid/ask direction: `grossBidEur`,
`grossAskEur`, `incrementalBidEur`, and `incrementalAskEur`.
The dashboard plots gross/incremental effects for the available pricing size
nearest 5M, in the tier diagnostics.

**Policy-dependent optimization:** For a FIXED policy the resolvent and
average reward are exact for this model. Since the incremental MTM also
depends on future policies, ordinary Howard improvement is NOT the exact
global optimizer of the policy-dependent objective. The C++ solver now uses
a damped, actual-reward-checked best-response iteration: after a proposed
policy it recomputes the entire generator, MTM and average reward; if the
proposal worsens actual average reward it also tests conservative RFQ ladder
interpolations, retaining only improving candidates. `converged` means this
local improvement search settled or policy changes reached tolerance, **not**
that a global mathematical optimum has been proven. Do not treat the old
Bellman residual as an optimality bound in this new policy-dependent model.

Monte Carlo already simulated information shocks from all requests. The
analytical PnL moments have been made consistent: both won AND lost RFQs
now contribute impact shocks, and only won RFQs alter cash/inventory.
The HJB still uses the configured pricing knots rather than the complete
interpolated RFQ size support (a previously known approximation).

The `python_reference/` directory is a legacy independent implementation and
is not the active C++ solver; its historical markout representation and
comparison helper have not been migrated to this interface.


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
`lambda_s(d_s)=A_s exp(-k_s(0.5-d_s))` and their own discrete parent-trade size distribution.
The size pillars are fixed at `1.00M, 0.75M, 0.50M, 0.25M, 0.10M` in the source pair's base currency.
The first four probabilities are configured explicitly and the 100k probability is the residual `1 - sum(other probabilities)`. The
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
V_s in {1.00, 0.75, 0.50, 0.25, 0.10} source-base millions
P(V_s = 0.10) = 1 - P(1.00) - P(0.75) - P(0.50) - P(0.25)
fill_target = min(V_s, c_s * quoteSize_target) / c_s
```

so a posted 1M target quote can fill only at the mapped/capped discrete pillar sizes and can never fill above
1M. The HJB sums each source's capped discrete size distribution when evaluating continuation
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

## Fix for `Solve failed · 'markout'` after upgrading

This error occurs when an old compiled C++ extension (expecting a global
`config["markout"]`) is loaded alongside newer Python code (which puts a markout
curve inside each RFQ tier). **This is a stale extension, not HJB non-convergence.**

Start the app using `./run_dash.sh`: it now detects this mismatch and rebuilds
an outdated native extension automatically (using pip's isolated build so
pybind11 need not be installed separately). If you prefer `python3 app.py`, run
this once from the extracted project directory, using the same Python
interpreter/virtual environment as the dashboard:

```bash
python3 -m pip install -e . --no-deps --force-reinstall
python3 app.py
```

The native engine now also checks `PRICING_CONFIG_SCHEMA_VERSION` at import, so
an outdated extension reports explicit rebuild instructions instead of failing
inside the solve callback with a misleading `markout` key error.

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
- per-tier saturating markout curves (same curve for all RFQ sizes within each tier)
- quotes vs inventory across all rungs
- bid and ask 3D quote surfaces
- volume-premium ladder at a selected inventory
- flow parameters, selected tier markout at 60 seconds and ladder tables
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
`tests/test_ui_config.py` cover independent tier markout parameters, tier markout switches and tier-fee mapping.

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

The sampled **actual RFQ size** drives win probability, cash, inventory, retained
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

## Policy-improvement stability fix (2026-10-13)

The constrained RFQ ladder is optimized by sweeping outward from zero inventory.
`PolicyBuilder::improve()` now uses the **already updated** inner-neighbour ladder
from the current iteration (`out`), rather than the lagged previous-iteration
ladder (`old`), when constructing cross-inventory monotonicity/gap bounds.
This is a two-line algorithmic fix; the per-state Howard safeguard still
compares candidates to the previous policy at that same inventory state.

Regression: `cpp/tests/test_policy_improvement_regression.cpp` reproduces the
large q=0 curvature spike at an asymptotic markout of -50 EUR per EUR 1M,
ECN disabled, gamma=1. The test fails before the fix and passes afterwards,
and also checks both sides' cross-inventory ladder constraints.

**Historical note:** At the time of this stability fix the HJB still used
the old markout reward; the policy-dependent expected-MTM model was added
subsequently. The original ladder neighbor update remains in place.

## Per-tier markout configuration (2026-10-13)

- Removed the obsolete top-level `config["markout"]` and the global markout sidebar panel.
- Each tier now has `markout: {asymptoticEurPerMillion: ..., tauSeconds: ...}`.
- Updated Dash callbacks, native C++ binding conversion, the selected-tier chart,
  and configuration/regression tests. Expected MTM was added subsequently.
- The earlier Gauss–Seidel-style policy-improvement stability fix remains intact.

## Validation of the expected-MTM update (2026-10-13)

- New `cpp/tests/test_expected_mtm.cpp` checks the resolvent against an exact
  Markov-chain solution, both trade directions, gross-vs-incremental MTM, and
  independence from the legacy `tau0/tau1/tau2` inputs.
- Existing Howard/ladder/analytics/simulation tests still pass in debug mode.
- Closed-form PnL and Monte Carlo use matching RFQ-impact conventions. In
  the included simulation regression case, their expected PnL outputs are
  approximately 6,821 EUR and 6,850 EUR, respectively.
- 42 Python tests pass with the current native extension; Python/native schema
  is bumped to 3 and `run_dash.sh` rebuilds incompatible binaries.
- Dash syntax is checked, but a live Dash server was not run in the build
  environment, where Dash was unavailable.
