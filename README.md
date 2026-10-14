
## One global target FX pair (Tier, dark pool and ECN)

Select **Target FX pair** under the sidebar **Global** expander. It is now the
only editable direct / priced pair selector, used by **Tier 1, Tier 2, dark
pool, ECN, HJB Solve, Monte Carlo and the frontier**. In both Configure flow
sources popups the Direct pair is displayed with a **disabled** dropdown;
Crossed pair and Crossing pair remain editable on a crossed source card.

Changing the Global pair reloads its spot and spread from
`python/trinity/fx_pair_defaults.json`, and the stochastic-volatility matrix
is read from the same JSON through the existing solver configuration. The
previously configured RFQ and ECN **crossed sources are cleared** whenever the
global target pair changes: their crossing mapping and spread/mid assumptions
may be invalid for the new FX risk. Add them again after changing the target.
Manual changes to Mid-rate and Spread remain possible for the selected pair.

### Flow source pair selectors (October 2026)

The **Tier** and **ECN** Configure flow sources dialogs use the same pair-selection layout. The Direct/Crossed source cards precede the pair selectors. The direct source displays the **globally selected Target pair as disabled**, with a disabled **Crossing pair** field underneath. Selecting **+ Add FX pair** creates a crossed source and opens enabled **Crossed pair** and **Crossing pair** selectors. There is no editable direct pair in either modal; the sole editable direct pair is in Global. ECN-specific calibration and RFQ-specific expander contents remain separate.

# Ladder Pricing — Coupled inventory and stochastic volatility

This version removes the separate **outstanding markout state X** from the C++ pricing engine. The production HJB solved by the ordinary Dash **Solve** button uses the joint state **(inventory q, volatility regime σ)**. The old 3D markout-state solvers, research models, chart callbacks and their Eigen requirement have been removed.

## Model

- Inventory q evolves through RFQ wins, ECN fills and optional dark-pool executions; existing quoted-price and size constraints are retained.
- σ is an exogenous finite-state continuous-time Markov chain with off-diagonal rates `volatilityRegimes.ratesPerMinute` in 1/minute. Its own jump does not change inventory. Configured defaults are 12, 20 and 35 pips/√minute, with rates configured in `python/trinity/fx_pair_defaults.json`.
- Inventory risk penalty in regime r is **γ σ_r² q²** (per-minute reward convention of the existing solver).
- Customer RFQ markout is still supported, with each tier's existing exponential markout curve (asymptote and time constant). It does **not** create an additional state. For a fixed joint `(q,σ)` policy, the solver computes the expected post-event inventory using the resolvent

  `(I - τ G_π) u_τ = q`,

  where `G_π` is the **joint inventory/volatility generator** and τ is the tier's exponential markout time constant in minutes. The incremental RFQ markout reward is calculated from the **won-versus-lost difference** using `u_τ`. Markout is recomputed during policy evaluation and improvement.
- RFQ and ECN quote decisions depend on `(q,σ)`; dark pool posted-size decisions also depend on `(q,σ)`. **No control depends on X**.
- The engine solves the **stationary ergodic** Poisson/HJB equation and uses damped policy improvement with policy-dependent expected MTM. Convergence/stalling is not a guarantee of global optimality.

For the default 41 operational inventory nodes and three volatility regimes, the output contains **123 operational (q,σ) combinations** (plus internal inventory buffer states used numerically). The **Policy** tab now has **individual regime selectors inside each regime-dependent diagnostics expander** in Overview, Tier diagnostics, Dark pool and Passive ECN. Each visualization uses its own chosen volatility cross-section independently of other expanders. The ordinary Solve button reads the regime parameters from `fx_pair_defaults.json`, displayed read-only in **Tier → Configure flow sources → direct target pair → Stochastic volatility**.

## Run

```bash
python3 -m pip install -e .
./run_dash.sh
```

`pybind11` is installed by the Python build workflow; no Eigen headers are needed. To run the native regression suite:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTRINITY_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Internalization-time cleanup

The previous inactive legacy proxy `tau(q) = tau0 + tau1*abs(q) + tau2*q*q`
was removed completely from the native PricingProblem API, Python defaults/configuration,
Dash controls/diagnostics and tests. It was not part of the HJB reward or transition
operator, so removing it does not alter policies or the coupled volatility model.

The separate **per-tier RFQ markout time constant** (`markout.tauSeconds`) is
still a model input. The dashboard now computes the **policy-implied expected time to hit zero inventory** using
the same joint (inventory, volatility) transition generator, with zero inventory absorbing.
No legacy time proxy is used.

## Single production solver

`Solver` (`cpp/include/ladder_pricer/solver.hpp`, `cpp/src/solver.cpp`) is the **only** HJB optimizer.
It requires `volatilityRegimes` and optimizes joint `(inventory, volatility regime)` policies.
The old `HowardSolver` and its q-only solve path were deleted. Its reusable fixed-policy
RFQ/ECN/dark-pool operator was moved without changing the equations into
`policy_operator.hpp/.cpp`. The results of accepted candidate evaluation are now reused,
so one redundant joint policy evaluation per accepted iteration is eliminated.

The previous single-volatility analytics and Monte Carlo were generalized to use the
**same joint (q, sigma) pricing policy** as the one production `Solver`. The C++
analytics and simulation implementations are now linked into the native extension.
No old HJB/Howard solver or q-only fallback is used. Native schema version is **10**;
rebuild the Python extension when upgrading this project.

The markout **curve** is deliberately retained. Removing the **markout state** must not be interpreted as removing markout economics from RFQ price optimization. Other features, including crossed RFQ/ECN flow sources, size distributions, fees and operational inventory constraints, are carried over from the original PricingProblem.

## Discrete customer RFQ sizes (pricing knots)

For each tier, `sizes` is the **only** RFQ size support and the set of optimized
pricing rungs. The former `rfqSizeStep` config and dashboard field are gone;
quotes are not interpolated in the RFQ size dimension. We still interpolate
**inventory** when evaluating a policy between inventory grid points.

Each RFQ flow source maps a target size `z` to its own base-currency size
`z_s` and provides one-sided exogenous arrivals
`lambda_s(z) = A0_s * z_s ** (-theta_s-beta_s*z_s)`. The current Monte Carlo samples
only `sizes` with probabilities `lambda_s(z) / sum(lambda_s(z))` and a
pooled two-sided source rate `2 * sum(lambda_s(z))`. The active fixed-policy
generator and HJB policy improvement both enumerate those same rungs. **Adding or removing a knot changes total modeled RFQ flow**
by design; it is not a density quadrature. ECN trade size distribution and dark
pool sizes are separate mechanisms and remain unchanged.

## Read-only volatility regimes from FX pair JSON

The **only volatility-regime source** is `python/trinity/fx_pair_defaults.json`.
The top-level `volatilityRegimes` contains a 2×2 or 3×3 model shared by all
FX pairs. A pair may override this by adding its own `"volatilityRegimes"`
object inside `"pairs": {"GBPSEK": { ... }}`. All entries must supply
`"sigmaPips"` (increasing levels) and `"ratesPerMinute"` (nonnegative,
zero diagonal, positive exit rate). Changing the JSON requires restarting
Dash to reload the offline defaults. All shipped pair values inherit the
existing three-regime 12/20/35 example until calibrated per pair.

The modal displays the selected pair's volatility levels and the **full
continuous-time generator Q** in numerical read-only tables. Q's diagonal
is −sum of the off-diagonal jump intensities. No Dash inputs, mutable stores,
or callbacks can override the volatility model. Solve gets the pair directly
from the target-pair selector; Monte Carlo and efficient frontier use this
solved configuration. The Monte Carlo starting regime selector remains an
initial-condition choice, not a different model.

Every diagnostic expander **inside Configure flow sources** starts closed
(exogenous RFQ flow, volatility and RFQ markout). This does not change the
open/closed defaults of the main Policy/Monte Carlo/frontier tabs.

## Dashboard flow-source and policy interface

The **Configure flow sources** modal shows **Market Data** and **Pricing Rungs**
inside the **same bordered target-pair flow card** as **RFQ flow intensities**,
**Volatility**, and **Markouts**, in that order. All five expanders start closed.
The header for the selected FX pair remains above them. Both target-only
sections are hidden when editing a crossed flow source.

The spot/spread inputs and both Tier pricing-knot fields remain mounted at a
stable position in the modal; changes to the selected source or Tier do not
remount or reset them. Each Tier shows only **its own** pricing-knot field,
and both Tier size lists remain separate inputs to Solve. The redundant
explanatory sentence above the knot inputs was removed.

- The default pricing setup has **Tier 1 and Tier 2 only**; Tier 3 and its configuration controls are removed from the dashboard and default Python config. The C++ pricing engine remains generic over the number of supplied pricing tiers.
- In **Configure flow sources**, each pair's exogenous RFQ flow inputs are grouped in an expander. Existing direct and crossed source size, intensity and win-probability mapping semantics are unchanged.
- The **direct target pair** flow-source view displays the read-only volatility levels and generator matrix immediately after the RFQ flow expander, loaded from the target FX pair's JSON defaults. No matrix inputs remain in the dashboard. The process is shared by all tiers and crossed sources of that target risk.
- The **RFQ markout** section in the same modal provides an enabled switch, an asymptote and a time constant with an immediate time-curve preview. The solver still supports **one markout curve per Tier shared across all its sources**; the crossed source card displays/edits that Tier's common curve. No source-specific markout economics are invented by the dashboard.
- The **Policy** tab has **8 local volatility regime dropdowns**, placed inside each expander whose displayed result depends on the solved `(q, σ)` policy: continuation value; Tier implied hit ratios, quotes and volume premium; dark-pool policy; ECN implied fill intensity, quotes and optimal delta. A dropdown selects only its expander’s plotted/table policy slice; selectors are independent and default to Normal after each Solve. Exogenous arrival curves, fitted flow parameters, calibration win probabilities, direct markout curves and dark-pool size models have no redundant regime dropdown because they do not depend on the selected volatility regime.
- The standalone **Stochastic volatility** tab is removed. Stochastic volatility itself is **retained in the C++ HJB**.
- **Monte Carlo, closed-form PnL analytics and efficient frontier work for both two and three volatility regimes.** The Monte Carlo tab lets you choose the initial volatility regime; the simulated regime follows its own continuous-time Markov chain and determines the instantaneous spot volatility and RFQ/ECN/dark-pool policies. Retained time series display the regime-specific bid/ask quotes.
- The expected inventory-to-zero diagnostic uses the joint (q,σ) fixed-policy generator, rather than an empirical internalization-time curve.

### UI verification

The Python and native tests validate the configuration and solver. Browser-level Dash testing requires Dash dependencies (`pip install -e .`) in your environment. This build environment did not have Dash available, so the UI was verified through a static/mocked component and callback-construction check rather than a running browser.

## Policy diagnostic cleanup

Removed the misleading **Frozen best-response residual** metric card and its empty convergence-history graph, plus the empty **Value-function change** graph. The compact header displays **Converged**, **Best-response iterations** and **Average reward**. Numerical evaluation residuals and the fixed-policy solver checks remain available internally in C++ and regression tests. This dashboard-only cleanup does not change the HJB equations, risk penalty, stochastic-volatility generator, or pricing decisions.

The layout/callback tests verify that each regime-dependent expander owns a distinct selector and that Tier, dark-pool and ECN diagnostics route each selector to its matching figure/table. Browser-level Dash verification was not available in the development environment.

## Dashboard control-text colours

- The visible **Enabled** controls for Tier 1, Tier 2, Dark pool and ECN use
  `channel-enable-checkbox` with an explicit white `html.Span` option label and white inline label styling, so Dash checklist defaults cannot leave the text dark.
- The **Volatility regime** dropdown in every policy expander, and the
  **Tier** selectors in Policy/Monte Carlo, use `policy-selector-dropdown`.
  Their selected text, placeholder, and menu options are dark on a white
  surface in both legacy React-Select and Dash 4 dropdowns. In Dash 4 the
  `policy-selector-dropdown` class is applied to the dropdown button itself;
  the `policy-selector-field` wrapper scopes the fix to its button, label and
  popup, without changing unrelated dropdowns. Before Solve a legible
  placeholder explains why no regime value is available.
- These are CSS/layout-only changes. HJB, native C++, mathematical parameters,
  callbacks and solution values are unchanged.

## Joint-regime analytics (2 or 3 regimes)

`Engine.statistics(horizon_minutes, initial_inventory, initial_regime)` calculates
finite-horizon **mean and standard deviation of mark-to-market PnL** from the
regime-aware Markov generator, including RFQ markout and its time decay, spot
Brownian volatility per regime, RFQ win/loss events, ECN passive fills, dark-pool
fills, and exogenous regime switches. It uses augmented moment equations, not a
second HJB optimizer. `initial_regime=-1` selects the middle configured regime.

`Engine.simulate(...)` uses the same solved policy, with regime switches as
competing Poisson events. `samplePaths[*].regimes` records the regime at every
retained timestamp. `finalRegime` contains one ending regime per Monte Carlo path.
The dashboard shows regime-specific quotes along each retained path. The existing
progress dialog and aggregate RFQ/ECN/fill diagnostics are retained.

`Engine.frontier(gamma_values, ...)` re-runs the **single** joint C++ Solver for
each risk parameter, then evaluates the corresponding regime-aware PnL moments.
It does not use the old one-volatility `HowardSolver`.

For any regime-aware analytic PnL statistic, base-currency (EUR) PnL uses the
**reference spot** to convert from quote currency; Monte Carlo converts each path
at its realized terminal spot. The analytic moment calculation also uses the
solver's discretized inventory generator. Partial ECN fills create fractional
inventories in Monte Carlo, so comparisons may exhibit grid approximation errors.
These conventions should be considered when comparing EUR statistics from very
long or volatile sessions.

### Verification

- Native tests and full Python regression suite include two- and three-regime
  analytics, Monte Carlo regime switches, policy-dependent expected hitting
  times, and efficient frontier consistency.
- Dedicated regression tests compare the analytic PnL moments against many-path
  event-driven Monte Carlo, including Tier markout, ECN and dark-pool channels.
- The default 570-minute, three-regime production configuration was exercised
  for Solve, PnL moments, 500 Monte Carlo paths, and a three-point frontier.
- Dash was not installed in the offline test environment; static/mock component
  and callback construction were checked, but live browser behaviour was not
  verified.

## ECN regime heatmap view (dashboard)

Under Policy → Passive ECN, the regime-dependent **Fill intensity vs inventory**,
**Quotes vs inventory**, and **Optimal quote delta** expanders each have their own
**Heatmap · all volatility regimes** toggle. The default is the existing line
plot and its local volatility-regime selector. When enabled, the selector is
hidden and the graph shows two aligned panels (bid/ask): horizontal axis is
inventory in EUR M, and vertical axis lists all configured volatility regimes.

The **Optimal quote delta** line chart always uses a fixed vertical range of 0 to 0.5,
regardless of the displayed volatility regime or active delta values.

Inactive ECN quotes are masked (slate background), not shown as zero values.
The bid/ask panels share a color scale: Viridis for deltas/intensities and a
zero-centered red/blue scale for quotes relative to mid. The optimal-delta
single-regime table is hidden in heatmap mode, returning when the toggle is
turned off. This is a dashboard-only change; HJB, Monte Carlo and calibration
are unchanged.

## RFQ pricing-knot policy heatmaps

The **Policy → Tier diagnostics** panel now has independent **Heatmap** toggles
for *Implied hit ratios* and *Quotes vs inventory*.
A toggle switches only its own chart between the original single-regime
line view and a matrix view over **all configured volatility regimes** and
**inventory q**. In heatmap mode, the local volatility-regime selector is hidden
and replaced by an **RFQ size [EUR M]** dropdown containing only the selected
Tier's actual pricing knots; no RFQ-size interpolation occurs. Switching off
heatmap restores the original view and the previous single-regime selection.

Bid and ask appear in separate heatmap panels, sharing a colour scale.
Quote heatmaps show the price relative to mid in pips with a diverging red/blue
colour map; implied win probabilities use Viridis. The redundant Bid/Ask quote
surface sections have been removed from Tier diagnostics. Every plot reads existing
`solution.regimes[*].tiers[*]` data and does **not** change the C++ HJB solver,
pricing knots, policy or analytics.


## UI fix: ladder inventory dropdown text (2026-10-14)

The `Inventory level for ladder` selector now uses the same white-background, dark-text dropdown styles as the policy regime and Tier selectors. The selected value and option menu use `#111821` for contrast. No changes to the HJB solver, pricing mathematics or analytics.

## Tier diagnostics cleanup (2026-10-14)

Removed the **Expected whole-inventory MTM · 5M RFQ** expander from the Policy → Tier diagnostics dashboard, including its chart, regime selection, plotting helper and callback connections. The HJB's expected markout / MTM calculations and their native solver outputs are unchanged. The separate RFQ **Markouts** visualization remains available.

## ECN intensity dashboard (October 2026)

The **Passive ECN** policy tab shows **Trade intensity curves** with target-pair
price improvement on the horizontal axis and one-sided trades/minute on the
vertical axis. The aggregated grid markers and total curve are both called
**Total** (a single legend entry is shown). The former **Relative intensity**
expander and its callback have been removed.

**Targeted trade intensity** shows bid and ask policy-implied trade rates against
**Inventory EURm**. Its vertical axis is fixed from zero to the **maximum
aggregated ECN flow rate over the configured delta range**, calculated from
all ECN flow sources after their mappings to the target pair. The intensity
heatmap uses that same maximum for its colour scale. This is a per-side rate,
not the sum of bid-side and ask-side events. These changes affect only Dash
visualization, not the HJB solver, policy, or Monte Carlo.

### Target-pair spot and spread in the flow-source editor

The `EURSEK spot` (renamed for the currently selected target FX pair) and
`Spread [pips]` inputs now live under **Configure flow sources → Target pair
market data**, directly below the target FX pair selector. They no longer
appear in the sidebar. Both retain their existing
`spot` and `spread` input IDs, and are consumed by the same Solve/analytics
config builder as before. The inputs stay mounted while the modal is closed
and while users switch between direct and crossed flow sources.

### Removed exogenous spot drift

The model assumes **zero exogenous spot drift**. The `spotDrift` parameter and
its dashboard input were removed from the HJB reward, native `PricingProblem`,
Monte Carlo spot process, PnL moment equations, and Python configuration.
Transient trade markout is unchanged. The volatility-regime Markov generator
(and any volatility-process mean reversion used in offline research) is unrelated
to spot drift and remains available. The sidebar `Global` expander now contains
`Max |q| [M]` and `Gamma / φ`; the old `Risk & drift` expander is gone.
Compiled native schema version 11 is required.

### Regression check for zero-drift equivalence

The refactor was checked against the previous compiled solver at zero drift
using the default three-regime config and a two-regime config with ECN and
darkpool enabled. Across policy solutions, PnL moments, and seeded Monte Carlo
results, all 49,913 numerical values matched exactly. C++: 3/3 tests; Python:
119/119 tests. Browser-based Dash rendering was not exercised.


### Target FX pair JSON defaults and pricing knots in Configure flow sources

Changing **Global → Target FX pair** now loads
both its spot and its reference spread from
`python/trinity/fx_pair_defaults.json`. The same JSON provides the temporary
mids used for crossed flow sources (including reciprocal cross rates). All
29 supported target FX pairs have explicit entries. The *spread* values for
pairs other than EURSEK are **illustrative offline placeholders, not real
market measurements or validated trading costs**; replace them with your own
pair-specific references before trusting the resulting quotes. Manually changing
either field remains possible, and a normal Solve does not reset these values.
Selecting another target pair reloads both values from the JSON.

Under **Configure flow sources → Pricing Rungs**, the
Tier 1 and Tier 2 pricing knots can now be edited side by side, separately per
Tier (sizes are in millions of the selected target pair's base currency).
The input IDs remain `t0-sizes` and `t1-sizes`; there are no duplicate hidden
inputs in the sidebar. The normal HJB Solve, simulation and diagnostics use
these edited discrete RFQ size supports without interpolation between sizes.

The JSON is included as package data in `setup.py`, so these defaults remain
available when the project is installed rather than run from its checkout.
This update changes only offline input defaults and dashboard placement; no
HJB equation or C++ source file was modified.

## Compact flow-source dialog labels

The Tier flow-source dialog displays the globally selected target pair as read-only and retains the configuration inputs,
but removes duplicate target-risk and direct-flow headings. Under **Market Data**,
the spot input is labeled **Mid-rate**; under **Pricing Rungs**, the visible
Tier-specific size field is labeled **Base M**. The RFQ and volatility explainer
sentences have been removed. **Markouts → Enabled** uses the same white label
style as the other channel toggles. These changes affect only the UI: spot,
spread, pricing-rung values and all solver configuration remain unchanged.

### Target FX pair selector in Configure flow sources

The **global** target FX pair is displayed as a disabled dropdown in the
Direct card. When editing a crossed source, the read-only Direct selector is
hidden and only crossed pair selectors are editable. Target FX pair changes
must be made in **Global**, and all flow channels share that target.


### Crossed RFQ source market data

In **Configure flow sources**, a crossed RFQ source shows its **Crossed pair**
 and **Crossing pair** selectors directly, followed by a collapsed **Market Data**
expander. **Crossing Mid-Rate** and **Crossing Spread** are edited inside this
expander. The original `flow-edit-cross-mid` and `flow-edit-spread` controls
retain their IDs and value wiring, so changing the selected source or invoking
Solve does not change the underlying FX cross mapping logic. The direct target
pair retains its separate **Market Data** expander with Mid-rate and Spread.

### Flow source popup layout (Oct 2026)

In both RFQ Tier and ECN **Configure flow sources** popups, the red **Remove FX pair**
button appears immediately to the right of **+ Add FX pair** in the footer,
only when a valid crossed source is selected. The Direct source cannot be
removed. The ECN popup's old descriptive paragraph has been removed.
These are dashboard-only changes; native pricing and simulation are unchanged.
