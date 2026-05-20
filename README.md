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

---

## 1. Purpose

This project solves a **stationary inventory-control problem** for FX market making.
The market maker posts two-sided size-ladder quotes (MDP tiers) and optionally participates in a dark pool.
Quotes are optimized rung by rung via golden-section search under inventory-consistent monotonicity bounds.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $q$ | inventory |
| $z$ | trade size |
| $\delta$ | quote control in normalized spread units |
| $s$ | reference spread |
| $h(q)$ | stationary inventory value function |
| $\lambda(\delta,z)$ | fill / arrival rate |
| $\mu(z,t)$ | expected markout — positive means price moves in your favour, negative means adverse selection |
| $t(q)$ | internalization time — expected minutes to offload inventory $q$ |
| $\Pi(q)$ | inventory penalty |

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

The penalty decomposes into a **carry cost** and the internalization time:

$$\Pi(q) = \underbrace{\gamma\sigma^2}_{\text{CarryCost}} \cdot\, q^2\, t(q)$$

- $\gamma$ (`risk_aversion`): risk-aversion coefficient
- $\sigma$: one-minute volatility (pips / √min); carry cost per unit time = $\gamma\sigma^2$
- $\tau_0, \tau_1, \tau_2$: internalization time coefficients (minutes)

`PolynomialInventoryPenalty` composes `CarryCost` and `PolynomialInternalizationTime`.

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

The dark pool is an optional additional flow source. The market maker posts a size $u$ and receives fills drawn from a geometric distribution.

### 8.1 Fill distribution

Arrival sizes are geometric with success probability $p$:

$$P(\text{fill} = k) = p(1-p)^{k-1}, \quad k < u$$

The posted size $u$ acts as a cap: if the underlying draw would exceed $u$, the fill is truncated to $u$ with residual probability $(1-p)^{u-1}$.

### 8.2 Fill value

For a bid posting of size $u$ at inventory $q$ (direction $= +1$):

$$H^{\text{DP,bid}}(q,u) = \lambda^{\text{bid}} \left[\sum_{k=1}^{u-1} p(1-p)^{k-1}\bigl(h(q+k)-h(q)-f^{\text{bid}}k\bigr) + (1-p)^{u-1}\bigl(h(q+u)-h(q)-f^{\text{bid}}u\bigr)\right]$$

The ask side is symmetric with direction $= -1$, $\lambda^{\text{ask}}$, and $f^{\text{ask}}$.

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

$$\operatorname{RHS}(q) = -\Pi(q) + \text{spot\_drift}\cdot q + \sum_{k} H_k^{\text{MDP}}(q) + H^{\text{DP}}(q)$$

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
├── dark_pool.hpp            # DarkPoolVenue
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

### MDP tier

| Parameter | Effect |
|---|---|
| `A0` | Flow intensity: RFQ arrivals per minute at $\delta = 0.5$, $z = 1$ |
| `theta`, `beta` | Size-decay exponents; $A(z) = A_0\,z^{-\theta - \beta z}$ |
| `shift` | Delta at which the fill curve is at 50 % |
| `steepness` | Sensitivity of fill rate to delta |
| `volume_shift` | Shifts the fill curve for larger sizes |
| `delta_min`, `delta_max` | Hard bounds on the quote control |

**`SqrtMarkoutModel`** parameters:

| Parameter | Effect |
|---|---|
| `markout_base` $\mu_0$ | Constant adverse-selection cost (pips) |
| `markout_coeff` $c$ | Size-dependent coefficient (pips); total = $\mu_0 + c\sqrt{z}$ pips |
| `trading_cost` | Flat execution cost (EUR/million); converted via spot and added to base |

**`SqrtTimeMarkoutModel`** parameters:

| Parameter | Effect |
|---|---|
| $a_0, a_1, a_2$ | Coefficients of $\alpha(z) = a_0 + a_1 z + a_2 z^2$ (pips / √min) |
| $b_0, b_1, b_2$ | Coefficients of $\beta(z) = b_0 + b_1 z + b_2 z^2$ (pips / min) |

### Dark pool

| Parameter | Effect |
|---|---|
| `lambda_bid` / `lambda_ask` | Arrival intensity on each side |
| `p_bid` / `p_ask` | Geometric fill probability |
| `fee_per_unit_bid/ask` | Per-unit execution cost |
| `posted_sizes` | Candidate posting sizes to evaluate |
| `min_fill_value` | Activation threshold; raise above 0 to suppress noise on non-uniform grids |

### Carry cost

| Parameter | Effect |
|---|---|
| `risk_aversion` $\gamma$ | Scales the penalty; higher values penalise inventory more strongly |
| `sigma` $\sigma$ | One-minute volatility (pips / √min); carry cost per minute = $\gamma\sigma^2$ |

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
$$\operatorname{RHS}(q) = -\Pi(q) + \text{spot\_drift}\cdot q + \sum_k H_k^{\text{MDP}}(q) + H^{\text{DP}}(q)$$

**Pseudo-time iteration**
$$h^{n+1}(q) = h^n(q) + dt\cdot\operatorname{RHS}^n(q) - h^{n+1}(0)$$
