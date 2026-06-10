# HJB Ladder Pricer

A stationary HJB solver for FX market-making with multi-tier size-ladder quotes and an optional dark-pool venue.
The solver is written in C++23, exposed to Python via pybind11, and comes with an interactive Streamlit UI.

---

## Getting started

**Requirements:** Python ≥ 3.10, a C++23-capable compiler, CMake ≥ 3.5, pybind11.

```bash
# Install in editable mode (builds the C++ extension automatically)
pip install -e .

# Run the Streamlit UI
streamlit run app.py

# Run the test suite
python -m unittest discover -s tests -v
```

---

## Table of contents

1. [Purpose](#1-purpose)
2. [Notation](#2-notation)
3. [Flow model](#3-flow-model)
4. [Markout and inventory penalty](#4-markout-and-inventory-penalty)
5. [Stationary HJB derivation](#5-stationary-hjb-derivation)
6. [MDP tier](#6-mdp-tier)
7. [MDP ladder monotonicity and bounds](#7-mdp-ladder-monotonicity-and-bounds)
8. [Dark-pool venue](#8-dark-pool-venue)
9. [Global Bellman RHS](#9-global-bellman-rhs)
10. [Numerical algorithm](#10-numerical-algorithm)
11. [Convergence diagnostics](#11-convergence-diagnostics)
12. [File architecture](#12-file-architecture)
13. [Parameter guide](#13-parameter-guide)
14. [Summary of core formulas](#14-summary-of-core-formulas)
15. [Units reference](#15-units-reference)


---

## 1. Purpose

### The business problem

As an FX market maker we earn money by quoting two prices to clients — a bid (we buy their euros) and an ask (we sell euros to them). The gap between the two is the spread, and every trade earns us a slice of it.

The catch is **inventory**. Every trade shifts our position. If we buy euros all morning and the euro then weakens, we lose money on the position we built up. The longer we hold it and the larger it is, the more it costs — so there is a real tension between capturing spread income and controlling the inventory we accumulate.

### What the model does

**This model resolves that tension by telling us exactly how aggressively to quote, for every trade size and every inventory level, all at once.**

The core intuition is straightforward:

- When we are **long** euros we should *discourage* clients from selling to us (widen the bid) and *encourage* clients to buy from us (tighten the ask) — steering flow in the direction that reduces the position.
- When we are **short** euros, the opposite applies.
- When we are **flat**, we quote symmetrically to maximise spread income.

The model makes this precise. It accounts for how sensitive clients are to our prices, how long it typically takes to work off a given position, and how much the market can move against us while we hold it.

### How this differs from the previous model

The previous generation model optimised a single quote — the 1M EUR price — and applied that decision uniformly across all trade sizes. This new model is a significant step forward in three respects.

**Full size ladder.** Rather than pricing one size and scaling the rest manually, this model optimises every rung of the ladder simultaneously. The 1M, 2M, 3M, 5M, 10M, and 20M quotes are each solved for their own optimal delta, with the constraint that larger sizes are never quoted tighter than smaller ones. The volume premium curve emerges naturally from the optimisation rather than being set by hand.

**Inventory-aware across all sizes.** The previous model's single-price output meant that inventory skewing had to be applied as a post-hoc adjustment. Here, inventory feeds directly into the optimisation for every size at once — so the ladder tilts and reshapes coherently as position changes, rather than receiving a uniform shift on top of a static curve.

**Dark pool integration.** The previous model had no mechanism for passive inventory management. The new model incorporates a dark pool channel and optimises posting sizes alongside the client-facing quotes, giving us an additional lever to reduce inventory at mid without moving the public ladder.

### Built on empirical customer flow data

A pricing model is only as good as its understanding of how clients actually behave. Rather than relying on assumed or generic price sensitivity, this model is driven by **hit ratio curves calibrated individually for each customer and each FX pair** from our own RFQ history.

For every client we estimate, statistically, how their likelihood of trading changes as our quote moves tighter or wider. These individual curves are then aggregated across the client base to produce a realistic, up-to-date picture of the flow dynamics we face at the portfolio level — capturing not just the average client but the full mix of flow types, sizes, and price sensitivities we actually see.

The result is that the model's view of "how clients respond to our prices" is not a rough assumption — it is a continuously refreshed, data-driven input grounded in the actual behaviour of our specific counterparties.

The same empirical discipline extends to the dark pool. Dark pool flow operates through a fundamentally different trading mechanism — fills arrive anonymously, there is no negotiation, and the size we actually receive depends on what counterparty flow happens to cross our resting order. This calls for a different model structure: rather than a price-sensitivity curve, we calibrate the arrival intensity and the distribution of fill sizes from historical dark pool activity. Crucially, because both the MDP and the dark pool are calibrated from real data and solved jointly, the model can account for the interaction between them — the value of a dark pool fill at a given inventory level is priced consistently with what we would otherwise earn by adjusting our client-facing quotes. This means the MDP ladder already reflects the presence of the dark pool as an inventory management alternative, and vice versa.

### Inventory skews the volume premium curve

The output is a complete **quote ladder** — a bid and an ask for each trade size (1M, 2M, 3M, 5M, 10M, 20M EUR) that updates coherently as inventory changes. Importantly, it is not just the overall level of quotes that shifts — the **shape of the volume premium curve** changes too.

When we are long, the ladder steepens on the bid side: the extra spread we charge for larger sizes increases, because a 20M fill would make a bad position significantly worse. On the ask side the ladder flattens: we become more willing to deal in size to get the position down faster. The opposite pattern holds when we are short. Inventory therefore skews both the level and the steepness of the size-vs-spread relationship simultaneously.

### Dark pool

Alongside the client-facing ladder, the model supports an optional **dark pool** channel.

A dark pool is an anonymous electronic venue where participants post orders without revealing their identity or intent. There is no spread to earn or pay — trades execute at the prevailing mid price. The attraction for us is purely inventory management: if we are long euros and a counterparty is looking to buy, we can offload position at mid rather than waiting for a client RFQ or moving our quotes aggressively and signalling distress to the market.

The challenge is that dark pool fills are uncertain. We post a size, but the actual fill depends on what counterparty flow arrives — we may get a partial fill, a full fill, or nothing at all. Posting too large a size ties up capacity without guarantee of execution; posting too small leaves inventory reduction on the table.

The model handles this explicitly. At each inventory level it evaluates all candidate posting sizes, computes the expected value of each taking into account the probability distribution of fill sizes and the cost of any execution fee, and selects the size that offers the best expected outcome. Posting is also directional by default: we only post to buy when we are short and only post to sell when we are long, so the dark pool always works in the same direction as the inventory signal from the quote ladder.

### Automated calibration pipeline

The quality of the model's inputs is maintained through a fully automated calibration process that runs daily across all three flow channels.

**MDP customers** — the bilateral RFQ clients — are calibrated using logistic regression inspired techniques. For each customer and currency pair we fit a price-sensitivity curve to our historical quote and fill data, capturing how their probability of trading changes as we move our quote tighter or wider. This gives us a statistically grounded view of each client's behaviour rather than a desk estimate.

**ECN flows** — trades executed on Electronic Communication Networks, the anonymous electronic venues where we stream prices to a broader market — present a different calibration challenge. On an ECN we do not observe every trade that takes place, only the fills that come to us. Calibration therefore has to be performed purely from our own execution history, without visibility into what competitors are quoting or what the rest of the market is doing. Despite this constraint, we can still extract a robust signal about arrival rates and price sensitivity from the pattern of our own fills over time.

**Dark pool flows** are modelled as a zero-inflated Poisson process, reflecting the reality that arrivals are sporadic and fill sizes cluster around certain levels with a meaningful probability of receiving nothing at all. This too is calibrated from our historical dark pool activity.

The entire estimation process — data extraction, model fitting, validation, and parameter publishing — is orchestrated by **Apache Airflow**. Airflow is a widely used open-source platform for scheduling and monitoring automated data workflows: think of it as a sophisticated job scheduler that runs a defined sequence of steps at a set time each day, tracks whether each step succeeded, and alerts the team if something goes wrong. In our case it ensures that freshly estimated flow parameters are available to the pricing model every morning before trading begins, without any manual intervention.

The practical consequence is that the model's view of customer behaviour is never more than one day old. Flow dynamics that shift — a client becoming more price-sensitive, a change in dark pool liquidity, a new pattern emerging on ECNs — are reflected in the next day's calibration run automatically.

### Adaptive to changing market conditions

A pricing model that cannot keep up with the market is of limited use. The solver is implemented in C++ and typically converges in **under a second**, which means the entire quote ladder — all sizes, all inventory levels, bid and ask — can be recomputed from scratch as conditions evolve.

In practice this matters most when the market moves quickly. If realised volatility spikes, the cost of holding inventory rises and the model will respond by quoting more defensively across the board. If we observe a sustained flow imbalance — more clients selling than buying, for instance — the model can be re-calibrated with updated flow parameters so that the ladder reflects what clients are actually doing today, not last month. Similarly, if the internalization environment changes or spreads widen, the solver can be rerun immediately to incorporate the new reality.

The result is a system that stays aligned with live market dynamics rather than drifting on stale assumptions.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $q$ | inventory (EUR lots) |
| $z$ | trade size (EUR lots) |
| $m$ | EUR/SEK spot rate |
| $\delta$ | quote control in normalized spread units |
| $s$ | reference spread (SEK/EUR) |
| $h(q)$ | stationary inventory value function (SEK) |
| $\lambda(\delta,z)$ | fill / arrival rate (fills/min) |
| $\mu(z,t)$ | expected markout (pips) — positive means price moves in your favour, negative means adverse selection |
| $t(q)$ | internalization time (min) — expected minutes to offload inventory $q$ |
| $\Pi(q)$ | inventory penalty (SEK/min) |

### Delta convention

The project uses a single normalized delta for both sides, where $s(0.5 - \delta)$ is the distance from mid:

$$p^{\text{bid}} = m - s(0.5 - \delta), \qquad p^{\text{ask}} = m + s(0.5 - \delta)$$

For passive quoting $\delta < 0.5$, so $s(0.5-\delta) > 0$: the bid is below mid and the ask is above mid. The spread income per lot is $s(0.5-\delta)$ on both sides.

| Value | Meaning |
|---|---|
| $\delta < 0.5$ | passive quote: bid below mid / ask above mid |
| $\delta = 0.5$ | quote exactly at mid, zero spread income |
| $\delta > 0.5$ | aggressive quote: crosses through mid |

---

## 3. Flow model

Each tier uses a logistic fill curve.

### 3.1 Arrival scale

$$A(z) = A_0\, z^{-\theta - \beta z}$$

- $A_0$: overall flow intensity — expected RFQ arrivals per minute at $\delta = 0.5$, $z = 1$
- $\theta$, $\beta$: size-decay exponents

### 3.2 Hit ratio

$$\operatorname{HR}(\delta,z) = \frac{1}{1+e^{-y(\delta,z)}}, \qquad y(\delta,z) = \text{steepness}\cdot\bigl(\delta - \text{shift} + \text{volume\_shift}(z-1)\bigr)$$

### 3.3 Arrival rate

$$\lambda(\delta,z) = A(z)\,\operatorname{HR}(\delta,z)$$

A larger $\delta$ is a more aggressive quote: higher fill probability, lower margin per fill.
`volume_shift` shifts the 50 % fill point for larger sizes.

---

## 4. Markout and inventory penalty

### 4.1 Markout models

Two markout models are available. Both accept trade size $z$ and internalization time $t$ (pre-computed by the solver from inventory $q$ via §4.2).

**`SqrtMarkoutModel`** — time-independent:

$$\mu(z, t) = \mu_0 + c\sqrt{z}$$

Larger trades carry higher expected adverse selection; the cost does not vary with time.

**`SqrtTimeMarkoutModel`** — time-dependent:

$$\mu(z, t) = \alpha(z)\sqrt{t} + \beta(z)\,t$$

$$\alpha(z) = a_0 + a_1 z + a_2 z^2, \qquad \beta(z) = b_0 + b_1 z + b_2 z^2$$

The $\sqrt{t}$ term captures diffusive adverse selection and the $t$ term captures drift-like adverse selection. The solver computes $t = t(q)$ from §4.2 and passes it in; the markout model itself is inventory-agnostic.

### 4.2 Internalization time

$$t(q) = \tau_0 + \tau_1|q| + \tau_2|q|^2 \quad \text{(minutes)}$$

Implemented by `PolynomialInternalizationTime`. Represents the expected number of minutes needed to offload inventory $q$: a positive constant floor $\tau_0$ plus terms that grow with $|q|$.

### 4.3 Inventory penalty

$$\Pi(q) = \underbrace{\gamma\sigma^2}_{\text{CarryCost}} \cdot q^2 \cdot t(q)$$

- $\gamma$ (`risk_aversion`): risk-aversion coefficient
- $\sigma$: one-minute volatility (pips / √min); carry cost per unit time = $\gamma\sigma^2$

`PolynomialInventoryPenalty` composes `CarryCost` and `PolynomialInternalizationTime`.

#### Variance interpretation

The carry cost penalty has a natural interpretation as the mark-to-market variance of inventory under linearly-decaying execution. 

**Setup:** Suppose inventory decays linearly from initial position $q$ to zero over the internalization horizon $t(q)$:

$$q(s) = q\left(1 - \frac{s}{t(q)}\right), \quad s \in [0, t(q)]$$

Under an arithmetic Brownian motion price model $dP_s = \sigma \, dW_s$, the mark-to-market P&L variance is:

$$\text{Var}(\text{MtM}) = \sigma^2 \int_0^{t(q)} q(s)^2 \, ds = \sigma^2 q^2 \int_0^{t(q)} \left(1 - \frac{s}{t(q)}\right)^2 ds = \frac{\sigma^2 q^2 \cdot t(q)}{3}$$

The carry cost penalty is proportional to this realized variance:

$$\Pi(q) = \gamma\sigma^2 q^2 t(q) = 3\gamma \cdot \text{Var}(\text{MtM})$$

Thus $\gamma$ acts as a risk-aversion multiplier above the realized variance. When $\gamma=1$, the penalty is three times the variance of linearly-decaying inventory, which is conservative but reasonable for controlling risk. The longer inventory takes to internalize (larger $t(q)$), the more severe the quadratic penalty becomes, which discourages holding large positions.

---

## 5. Stationary HJB derivation

We use the reduced stationary ansatz

$$V(x,q,s) = x + qs + h(q)$$

where $x$ is cash, $s$ is the spot / mid price, and $h(q)$ captures the continuation value of holding inventory $q$.
The code solves for $h$ on a 1D symmetric inventory grid.

### 5.1 One-step expected contribution

A quote with control $\delta$ and size $z$ generates fills at rate $\lambda(\delta,z)$.
The stationary Bellman contribution per unit time (after dropping the common $dt$ factor) is

$$\lambda(\delta,z)\cdot\Bigl(\text{execution value} - \text{expected markout} + \underbrace{h(q') - h(q)}_{\Delta h}\Bigr)$$

For a bid fill $q' = q+z$; for an ask fill $q' = q-z$.

---

## 6. MDP tier

MDP tiers are customer-facing two-sided ladders. For each size $z$, the market maker posts a bid delta $\delta^b(q,z)$ and an ask delta $\delta^a(q,z)$.

### 6.1 Payoff derivation

The value function ansatz is $V(x,q,m) = x + qm + h(q)$, where $x$ is cash and $m$ is the mid/spot price.

**Bid fill** — client sells $z$ to us at $p^{\text{bid}} = m - s(0.5-\delta)$, which is below mid for passive quotes.

$$V_{\text{before}} = x + qm + h(q)$$

$$V_{\text{after}} = \bigl(x - z\,p^{\text{bid}}\bigr) + (q+z)m + h(q+z)$$

$$= x - z\bigl(m - s(0.5-\delta)\bigr) + (q+z)m + h(q+z)$$

$$= x - zm + zs(0.5-\delta) + qm + zm + h(q+z)$$

$$\Delta V_{\text{instant}} = V_{\text{after}} - V_{\text{before}} = zs(0.5-\delta) + h(q+z) - h(q)$$

After the fill the inventory is $q+z$. The market maker must internalize this position over an expected time $t(q+z)$. The expected markout $\mu(z,\,t(q+z))$ is the anticipated price move per lot over that period — negative when the flow is adverse (price moves against the position), positive when it is favourable. Adding it gives the net per-fill payoff:

$$P^{\text{bid}}(q,z,\delta) = zs(0.5-\delta) + z\mu(z,\,t(q+z)) + h(q+z) - h(q)$$

**Ask fill** — client buys $z$ from us at $p^{\text{ask}} = m + s(0.5-\delta)$, which is above mid for passive quotes.

$$V_{\text{before}} = x + qm + h(q)$$

$$V_{\text{after}} = \bigl(x + z\,p^{\text{ask}}\bigr) + (q-z)m + h(q-z)$$

$$= x + z\bigl(m + s(0.5-\delta)\bigr) + (q-z)m + h(q-z)$$

$$= x + zm + zs(0.5-\delta) + qm - zm + h(q-z)$$

$$\Delta V_{\text{instant}} = V_{\text{after}} - V_{\text{before}} = zs(0.5-\delta) + h(q-z) - h(q)$$

Post-fill inventory is $q-z$, so:

$$P^{\text{ask}}(q,z,\delta) = zs(0.5-\delta) + z\mu(z,\,t(q-z)) + h(q-z) - h(q)$$

The spread income $zs(0.5-\delta)$ is identical on both sides. The markout $t$ is evaluated at the **post-fill** inventory because it is the resulting position that must be internalized.

### 6.2 Bellman contributions

$$H^{\text{bid}}(q,z,\delta) = \lambda(\delta,z)\,P^{\text{bid}}(q,z,\delta)$$

$$H^{\text{ask}}(q,z,\delta) = \lambda(\delta,z)\,P^{\text{ask}}(q,z,\delta)$$

### 6.3 Ladder objective

$$H^{\text{MDP}}(q) = \sum_{z\in\mathcal{Z}} H^{\text{bid}}(q,z,\delta^b(q,z)) + \sum_{z\in\mathcal{Z}} H^{\text{ask}}(q,z,\delta^a(q,z))$$

Each rung is optimized independently by golden-section search within the admissible bounds described in §7.

---

## 7. MDP ladder monotonicity and bounds

The policy is built rung by rung from the center of the inventory grid outward.

### 7.1 Cross-size monotonicity

Larger sizes must not be more competitive than smaller sizes:

$$\delta_j \le \delta_{j-1}$$

### 7.2 Cross-inventory consistency

As inventory moves away from zero, the ladder evolves in two regimes:

- **Shrink side** (bid at $q<0$, ask at $q>0$): inventory needs to be reduced — deltas rise (quotes tighten) as $|q|$ grows.
- **Expand side** (bid at $q>0$, ask at $q<0$): inventory is on the wrong side — deltas fall (quotes widen) as $|q|$ grows.

The gap between consecutive rungs is preserved across inventory rows so the ladder shape evolves smoothly rather than collapsing or inverting. See `mdp_bounds_for_rung` in [solver.hpp](cpp/ladder_pricer/solver.hpp) for the three-case implementation.

---

## 8. Dark-pool venue

The dark pool is an optional additional flow source. The market maker posts a size $u$ and receives fills drawn from a pluggable arrival distribution (`ArrivalDistribution` interface in `arrival_distributions.hpp`).

### 8.1 Fill distributions

Two concrete distributions are available:

**`GeometricArrivalDist`** — geometric fill size with cap $u$:

$$P(\text{fill} = k) = p(1-p)^{k-1}, \quad k = 1,\ldots,u-1$$
$$P(\text{fill} = u) = (1-p)^{u-1}$$

Parameters: `lambda` (arrival intensity), `p` (success probability per unit).

**`ZeroInflatedPoissonArrivalDist`** — zero-inflated truncated Poisson:

$$P(\text{fill} = 0) = p_0$$
$$P(\text{fill} = k) = (1-p_0)\,\frac{\tilde{P}(k;\mu)}{\sum_{j=1}^{u}\tilde{P}(j;\mu)}, \quad k = 1,\ldots,u$$

where $\tilde{P}(k;\mu) = e^{-\mu}\mu^k/k!$ is the unnormalized Poisson PMF. With probability $p_0$ no fill occurs; otherwise the fill size is drawn from a Poisson($\mu$) distribution truncated to $\{1,\ldots,u\}$.

Parameters: `lambda_arr` (arrival intensity), `mu` (Poisson rate parameter — note: the mean of the truncated distribution differs from `mu` for small `mu` or small `u`), `p0` (zero-inflation mass).

### 8.2 Fill value

For a bid posting of size $u$ at inventory $q$ (direction $d = +1$):

$$H^{\text{DP,bid}}(q,u) = \mathtt{dist\_bid.expected\_fill\_value}\!\left(h,\,q,\,h(q),\,u,\,f^{\text{bid}},\,+1\right)$$

where the concrete expansion depends on the chosen distribution. For **GeometricArrivalDist**:

$$H^{\text{DP,bid}}(q,u) = \lambda \left[\sum_{k=1}^{u-1} p(1-p)^{k-1}\bigl(h(q+k)-h(q)-f^{\text{bid}}k\bigr) + (1-p)^{u-1}\bigl(h(q+u)-h(q)-f^{\text{bid}}u\bigr)\right]$$

For **ZeroInflatedPoissonArrivalDist**:

$$H^{\text{DP,bid}}(q,u) = \lambda_{\text{arr}} \sum_{k=1}^{u} (1-p_0)\,\frac{\tilde{P}(k;\mu)}{\sum_{j=1}^{u}\tilde{P}(j;\mu)}\bigl(h(q+k)-h(q)-f^{\text{bid}}k\bigr)$$

The ask side is symmetric with $d = -1$ and the ask distribution and fee.

### 8.3 Activation policy

For each inventory level, the solver evaluates all candidate sizes and posts the one with the highest fill value — provided that value exceeds `min_fill_value` (default `0`).
`min_fill_value` can be set above zero to guard against activation noise from floating-point interpolation asymmetries on non-uniform grids.

### 8.4 Admissibility

By default (`allow_both_sides = False`):

- Bid posting is only active at $q < 0$ (short inventory — use dark pool to buy back).
- Ask posting is only active at $q > 0$ (long inventory — use dark pool to sell).

---

## 9. Global Bellman RHS

The full stationary Bellman RHS at inventory $q$ is

$$\operatorname{RHS}(q) = -\Pi(q) + \operatorname{spot\_drift}\cdot q + \sum_{k} H_k^{\text{MDP}}(q) + H^{\text{DP}}(q)$$

The pseudo-time fixed-point iteration is

$$h^{n+1}(q) = h^n(q) + dt\cdot\operatorname{RHS}^n(q)$$

After each step the center value is subtracted to fix the ergodic free constant:

$$h^{n+1}(q) \leftarrow h^{n+1}(q) - h^{n+1}(0)$$

---

## 10. Numerical algorithm

### 10.1 Solve loop

```text
initialize h = 0
repeat until convergence or iteration cap:
    update MDP policies from current h
    update dark-pool policy from current h   (if present)
    compute Bellman RHS at every grid point
    euler_step:     h_next = h + dt * rhs
    normalize:      h_next -= h_next[q0];  record max |Δh|
    check convergence; swap h ↔ h_next
rebuild policies one final time on the converged h
```

### 10.2 MDP policy construction

```text
for each MDP tier:
    build center row (q = 0) with no cross-inventory constraint
    walk positive inventories: each row constrained by the row below it
    walk negative inventories: each row constrained by the row above it
    for each row:
        for each rung j:
            compute admissible [lower, upper] (§7)
            maximize H^{bid/ask} on [lower, upper] via golden-section search
```

---

## 11. Convergence diagnostics

The solver records per-iteration:

- **max |Δh|**: largest pointwise update between iterations
- **max |rhs|**: largest absolute Bellman RHS value

Stopping requires both to fall below their respective tolerances (`tol_h`, `tol_rhs`) for `consecutive_passes_required` consecutive iterations, after at least `min_iter` iterations.

---

## 12. File architecture

```
cpp/ladder_pricer/
├── hjb_ladder.hpp           # aggregator — single include for downstream code
├── common.hpp               # Matrix/OptionalDeltaRow types, grid utilities, Side enum
├── flow_curve.hpp           # LogisticFlowCurve
├── carry_cost.hpp           # CarryCost (γσ²)
├── internalization_time.hpp # PolynomialInternalizationTime  t(q) = τ₀ + τ₁|q| + τ₂|q|²
├── markout.hpp              # SqrtMarkoutModel, SqrtTimeMarkoutModel, MarkoutModel variant
├── penalty.hpp              # PolynomialInventoryPenalty (composes CarryCost × InternalizationTime)
├── quote_analytics.hpp      # QuoteSummary, QuoteMetrics
├── policy.hpp               # QuotePolicy, DarkPoolPolicy
├── solver_config.hpp        # SolverConfig, GoldenSectionSearch, SolverDiagnostics
├── tiers.hpp                # MDPTier
├── arrival_distributions.hpp # ArrivalDistribution (abstract), GeometricArrivalDist, ZeroInflatedPoissonArrivalDist
├── dark_pool.hpp            # DarkPoolVenue — depends on arrival_distributions.hpp
├── solver.hpp               # HJBLadderSolver, HJBSolution
└── bindings.cpp             # pybind11 Python bindings

app.py                       # Streamlit UI
tests/
└── test_symmetry.py         # Numerical symmetry tests (zero drift → h and policies are even)
```

The Streamlit app lets you configure tiers, choose a uniform or piecewise grid, solve the HJB, and inspect flow curves, quote profiles, the value function, and the dark-pool posted-size policy.

---

## 13. Parameter guide

### Solver

| Parameter | Default | Effect |
|---|---|---|
| `spread` | 20 bps | Reference half-spread used in payoff calculations |
| `spot_drift` | 0 | Breaks bid/ask symmetry when non-zero |
| `dt` | 0.002 | Pseudo-time step — smaller is safer but slower |
| `n_iter` | 140 | Maximum iterations |
| `tol_h` / `tol_rhs` | 1e-5 / 1e-4 | Convergence tolerances |
| `min_iter` | 5 | Minimum iterations before early stopping is allowed |
| `consecutive_passes_required` | 3 | How many back-to-back iterations must both tolerances be met before stopping |
| `early_stop` | true | Whether to stop before `n_iter` once convergence criteria are satisfied |

### MDP tier

| Parameter | Effect |
|---|---|
| `A0` | Flow intensity: RFQ arrivals per minute at $\delta = 0.5$, $z = 1$ |
| `theta`, `beta` | Size-decay exponents; $A(z) = A_0\,z^{-\theta - \beta z}$ |
| `shift` | Delta at which the fill curve is at 50 % |
| `steepness` | Sensitivity of fill rate to delta |
| `volume_shift` | Shifts the fill curve for larger sizes |
| `delta_min`, `delta_max` | Hard bounds on the quote control |
| `use_markout` | Whether to include the markout term in the per-fill payoff; set to `false` to ignore adverse selection |

**`SqrtMarkoutModel`** parameters:

| Parameter | Effect |
|---|---|
| `base` $\mu_0$ | Constant adverse-selection cost (pips) |
| `coeff` $c$ | Size-dependent coefficient (pips); total = $\mu_0 + c\sqrt{z}$ pips |

**`SqrtTimeMarkoutModel`** parameters:

| Parameter | Effect |
|---|---|
| $a_0, a_1, a_2$ | Coefficients of $\alpha(z) = a_0 + a_1 z + a_2 z^2$ (pips / √min) |
| $b_0, b_1, b_2$ | Coefficients of $\beta(z) = b_0 + b_1 z + b_2 z^2$ (pips / min) |

**Note on Streamlit UI:** The Streamlit app includes a `trading_cost` parameter (EUR/million) for convenience. This is not a C++ parameter; it is converted and added to the `base` parameter during configuration.

### Dark pool (`DarkPoolVenue`)

| Parameter | Effect |
|---|---|
| `dist_bid` / `dist_ask` | `ArrivalDistribution` objects for each side (set via constructor) |
| `fee_per_unit_bid/ask` | Per-unit execution cost |
| `posted_sizes` | Candidate posting sizes to evaluate |
| `min_fill_value` | Activation threshold; raise above 0 to suppress noise on non-uniform grids |

The legacy positional constructor `DarkPoolVenue(lambda_bid, lambda_ask, p_bid, p_ask, ...)` creates a `GeometricArrivalDist` on each side automatically.

**`GeometricArrivalDist`** parameters:

| Parameter | Effect |
|---|---|
| `lambda` | Arrival intensity (fills/min) |
| `p` | Success probability per unit; controls the tail of the fill-size distribution |

**`ZeroInflatedPoissonArrivalDist`** parameters:

| Parameter | Effect |
|---|---|
| `lambda_arr` | Arrival intensity (fills/min) |
| `mu` | Poisson rate parameter — controls the shape of the truncated fill-size distribution (not equal to the mean for small `mu` or small cap `u`) |
| `p0` | Zero-inflation probability — chance of no fill regardless of arrival |

### Carry cost (`CarryCost`)

| Parameter | Effect |
|---|---|
| `risk_aversion` $\gamma$ | Scales the penalty; higher values penalise inventory more strongly |
| `sigma` $\sigma$ | One-minute volatility (pips / √min); carry cost per unit time = $\gamma\sigma^2$ |

### Internalization time (global)

| Parameter | Effect |
|---|---|
| `tau0` | Constant floor — minimum time even at zero inventory |
| `tau1` | Linear growth with $|q|$ |
| `tau2` | Quadratic growth with $|q|$ |

---

## 14. Summary of core formulas

**Flow**
$$A(z)=A_0 z^{-\theta-\beta z}, \qquad \lambda(\delta,z)=A(z)\cdot\frac{1}{1+e^{-y(\delta,z)}}$$

**Internalization time**
$$t(q)=\tau_0+\tau_1|q|+\tau_2|q|^2$$

**Markout — `SqrtMarkoutModel`**
$$\mu(z,t)=\mu_0+c\sqrt{z}$$

**Markout — `SqrtTimeMarkoutModel`**
$$\mu(z,t)=\alpha(z)\sqrt{t}+\beta(z)\,t, \qquad \alpha(z)=a_0+a_1 z+a_2 z^2, \quad \beta(z)=b_0+b_1 z+b_2 z^2$$

**Inventory penalty**
$$\Pi(q)=\gamma\sigma^2\cdot q^2\cdot t(q)$$

**MDP bid / ask payoff per fill**
$$zs(0.5-\delta) + z\mu(z,\,t(q\pm z)) + h(q\pm z) - h(q)$$

**Bellman RHS**
$$\operatorname{RHS}(q) = -\Pi(q) + \operatorname{spot\_drift}\cdot q + \sum_k H_k^{\text{MDP}}(q) + H^{\text{DP}}(q)$$

**Pseudo-time iteration**
$$h^{n+1}(q) = h^n(q) + dt\cdot\operatorname{RHS}^n(q) - h^{n+1}(0)$$

---

## 15. Units reference

All quantities in the HJB are in **SEK/min** (the P&L currency per unit pseudo-time). The value function $h(q)$ is in **SEK**.

| Quantity | Units |
|---|---|
| $h(q)$ | SEK |
| $\Pi(q)$ | SEK/min |
| $\lambda(\delta,z)$ | fills/min |
| $\mu(z,t)$ | pips (= $10^{-4}$ SEK/EUR, converted in payoff) |
| $\sigma$ | pips/√min |
| $t(q)$ | min |
| $m$ | SEK/EUR (EUR/SEK spot) |

**Currency note:** inventory $q$ is in EUR lots; P&L and value function are in SEK. To compare with real-world EUR P&L, divide by the spot $m$.
