# Ladder Pricer — pure Python Howard solver

This repository implements a stationary continuous-time HJB / average-reward inventory-control model for RFQ ladder pricing, with an optional dark-pool channel.

The implementation is now **pure Python + NumPy**. There is no C++, pybind11, CMake, Boost, SciPy, or compiled extension.

## Solver

The only solver is exact Howard policy iteration:

1. Hold the current quote/dark-pool policy fixed.
2. Build the fixed-policy affine Bellman operator

   $$
   R^\pi(h)=c^\pi+L^\pi h.
   $$

3. Solve the ergodic Poisson equation

   $$
   \rho^\pi \mathbf 1=c^\pi+L^\pi h^\pi,
   \qquad h^\pi(0)=0.
   $$

4. Improve RFQ quotes analytically using the logistic-flow Lambert-$W$ solution.
5. Apply the within-row size-ladder rule: larger sizes cannot quote tighter than the preceding rung.
6. Enforce trader-shaped cross-inventory rules on both the **level** and **slope** of each ladder. Moving away from zero inventory, every rung becomes less aggressive on the inventory-increasing side and more aggressive on the inventory-reducing side; simultaneously, volume-premium gaps widen on the inventory-increasing side and flatten on the inventory-reducing side.
7. Cross-inventory bounds are taken from the previous complete policy (Jacobi-style), avoiding order-dependent cascading within one Howard improvement pass.
8. A greedy row proposal is accepted only if it improves that row's fixed-$h$ Hamiltonian relative to a feasible row under the shape constraints.
9. Repeat until both the Bellman residual and the policy change are at floating-point scale.

There is no pseudo-time step, discount rate, damping factor, optimizer tolerance, or user-selected iteration count.


## Inventory domain and boundary

`SolverConfig.q_grid` is the **operational pricing grid**: the inventory range returned in the solution and shown in Streamlit. It is not used as an extrapolation boundary.

Let

$$
z_{\max}=\max\{\text{RFQ sizes, dark-pool posted sizes}\}.
$$

The Howard solver automatically constructs a hidden symmetric solve grid with hard limit

$$
Q_{\mathrm{hard}}=Q_{\mathrm{operational}}+z_{\max}.
$$

Therefore every allowed fill starting from the operational range lands inside a state for which $h(q)$ is explicitly solved. Internal interpolation is bounded: attempting to evaluate $h$ outside the hidden solve domain is an error, not a linear extrapolation.

At the hidden hard edge, an RFQ or dark-pool action that could increase inventory beyond $Q_{\mathrm{hard}}$ is inadmissible. Inventory-reducing actions remain available. The trader level/gap shape constraints are enforced on the operational pricing surface; the hidden buffer exists only to provide economically defined continuation values.

The returned `HJBSolution` exposes both views:

- `solution.q_grid`, `solution.h`: operational/display grid;
- `solution.solve_q_grid`, `solution.h_solve`: hidden Howard solve grid;
- `solution.hard_inventory_limit`: hard absolute inventory limit.

## Inventory penalty

The running inventory penalty is the standard quadratic form

$$
C(q)=\gamma\sigma^2q^2.
$$

The polynomial internalization-time model is separate from the inventory penalty. It provides the horizon at which the saturating adverse markout is evaluated; it is not multiplied into the running inventory penalty.

## RFQ flow and analytical quote

For size $z$,

$$
\lambda(\delta,z)=A(z)L(\delta,z),
$$

with

$$
A(z)=A_0z^{-\theta-\beta z},
$$

and logistic hit ratio

$$
L(\delta,z)=\frac{1}{1+\exp[-\kappa(\delta-c_z)]}.
$$

For a fixed continuation value, each unconstrained rung optimum is available in closed form via the principal Lambert-$W$ branch. The code evaluates $W_0(e^x)$ with a small self-contained Newton solve in log space, so no special-function dependency is required.

## Markout model

Markout is represented as a positive adverse-selection cost:

$$
m(z,t)=a z^{\beta}\left(1-e^{-t/\tau}\right).
$$

Here $a$ is the eventual markout of a unit-size trade, $\beta$ controls how eventual impact grows with trade size, and $\tau$ controls how quickly the impact is realized. The curve starts at zero, rises fastest immediately after the trade, and saturates at

$$
M(z)=a z^{\beta}.
$$

The RFQ fill payoff therefore uses

$$
\lambda(\delta,z)\left[z s(0.5-\delta)-z m(z,t)+h(q')-h(q)\right].
$$

A larger positive markout always makes the trade less attractive. In the Streamlit app, $a$ is entered in pips and $\tau$ in minutes. Markout time and internalization time therefore use the same time unit throughout the model.

## Ladder constraints

Within every inventory row, larger sizes cannot quote tighter:

$$
\delta_{z_1}(q)\ge\delta_{z_2}(q)\ge\cdots,
\qquad z_1<z_2<\cdots.
$$

Define the adjacent volume premium

$$
g_j(q)=\delta_{z_j}(q)-\delta_{z_{j+1}}(q)\ge0.
$$

The cross-inventory rule is symmetric and controls both the **level** of each
rung and the **slope** of the ladder.

For a long position ($q>0$), bids increase inventory while asks reduce it.
Moving from $q_i$ to the next larger inventory $q_{i+1}$:

$$
\delta_z^{bid}(q_{i+1})\le\delta_z^{bid}(q_i),
\qquad
g_j^{bid}(q_{i+1})\ge g_j^{bid}(q_i),
$$

$$
\delta_z^{ask}(q_{i+1})\ge\delta_z^{ask}(q_i),
\qquad
g_j^{ask}(q_{i+1})\le g_j^{ask}(q_i).
$$

Thus every bid price moves lower and the bid ladder fans out; every ask price
also moves lower (larger ask delta) and the ask ladder flattens.

For a short position the rules mirror:

$$
\delta_z^{ask}(q_{i-1})\le\delta_z^{ask}(q_i),
\qquad
g_j^{ask}(q_{i-1})\ge g_j^{ask}(q_i),
$$

$$
\delta_z^{bid}(q_{i-1})\ge\delta_z^{bid}(q_i),
\qquad
g_j^{bid}(q_{i-1})\le g_j^{bid}(q_i),
\qquad q_{i-1}<q_i\le0.
$$

So, as inventory becomes more extreme, the **wrong-way ladder shifts away and
steepens**, while the **right-way ladder shifts toward the market and
flattens**. Cross-inventory bounds are applied from the previous complete
policy, not from rows updated earlier in the same Howard pass.

## Install

From the repository root:

```bash
python -m pip install -e .
```

The only package dependency of `ladder_pricer` itself is NumPy. The Streamlit app additionally requires its normal UI/plotting dependencies (`streamlit`, `pandas`, `plotly`).

Run the app with:

```bash
streamlit run app.py
```

## Minimal example

```python
import ladder_pricer as lp

config = lp.SolverConfig(
    q_grid=list(range(-20, 21)),
    spot=1.0,
    spot_drift=0.0,
    spread=20 / 10_000,
)

flow = lp.LogisticFlowCurve(
    A0=0.0155,
    theta=0.144,
    beta=0.0857,
    shift=0.52,
    steepness=8.42,
    volume_shift=0.026,
)

markout = lp.SaturatingMarkoutModel(
    impact_scale=1.0 / 10_000,  # 1 pip eventual impact at 1M
    size_exponent=0.5,
    tau=0.5,                    # minutes
)

tier = lp.MDPTier(
    name="Tier 1",
    sizes=[1, 2, 3, 5, 10, 20],
    flow_curve=flow,
    markout_model=markout,
    delta_min=-10,
    delta_max=100,
)

penalty = lp.QuadraticInventoryPenalty(
    lp.CarryCost(risk_aversion=0.01, sigma=20 / 10_000)
)

internalization_time = lp.PolynomialInternalizationTime(4.0, 0.070, 0.0084)

solution = lp.HJBLadderSolver(
    config,
    penalty,
    internalization_time,
    [tier],
).solve()

print(solution.average_reward)
print(solution.diagnostics.iterations_used)
```

## Tests

```bash
python -m unittest discover -s tests -v
```
