# Penalty Calibration from Data

The inventory penalty is

$$\Pi(q) = \bigl(c\cdot|q| + \rho\cdot q^2\bigr)\cdot t(q)$$

This document describes how to estimate $c$, $\rho$, and $t(q)$ from trade and position data.

---

## Data requirements

Per time step $i$ (uniform interval $\Delta t$, e.g. one minute):

| Column | Description |
|---|---|
| $q_i$ | Inventory at the start of the interval (EUR lots) |
| $m_i$ | Mid-price at the start of the interval (SEK/EUR) |
| $\Delta m_i = m_{i+1} - m_i$ | Spot move over the interval |

Mark-to-market P&L over the interval:

$$\Delta P_i = q_i \cdot \Delta m_i$$

This is the only quantity needed for the regressions below. Do not include spread income or fee P&L — those are already captured by the flow model.

---

## Estimating $c$ — linear carry rate

$c$ is the expected mark-to-market loss per EUR of inventory per minute. Under the model

$$dm = -c\cdot\operatorname{sign}(q)\,dt + \sigma\,dW$$

the conditional mean is

$$E\!\left[\frac{\Delta P_i}{\Delta t}\,\Big|\,q_i\right] = -c\cdot|q_i|$$

### Regression

Run OLS without intercept:

$$\frac{\Delta P_i}{\Delta t} = -c\cdot|q_i| + \varepsilon_i$$

The estimate is

$$\hat{c} = -\frac{\sum_i |q_i|\,(\Delta P_i/\Delta t)}{\sum_i q_i^2}$$

A negative slope confirms the adverse carry; a positive slope means the inventory is on average profitable, which would suggest reconsidering the sign convention.

### Sanity check

Bin observations by $|q|$ and plot the mean of $\Delta P_i / \Delta t$ against $|q|$. The relationship should be approximately linear through the origin with negative slope $-c$.

---

## Estimating $\rho$ — quadratic variance rate

$\rho$ is the variance of mark-to-market P&L per EUR² of inventory per minute. From the diffusion term alone $\rho = \sigma^2$, but estimating it directly from your P&L is preferable because it captures any q-dependent microstructure effects.

### Residuals from the mean

Let $\hat{c}$ be the estimate from above. Form the mean-adjusted P&L:

$$r_i = \frac{\Delta P_i}{\Delta t} + \hat{c}\cdot|q_i|$$

These residuals isolate the unpredictable component.

### Regression on squared inventory

Run OLS without intercept on the squared residuals:

$$r_i^2 = \rho\cdot q_i^2 + \eta_i$$

The estimate is

$$\hat{\rho} = \frac{\sum_i q_i^2\, r_i^2}{\sum_i q_i^4}$$

### Alternative: direct spot variance

If inventory does not vary much in your sample, $\hat{\rho}$ may be noisy. A fallback is

$$\hat{\rho} = \hat{\sigma}^2 = \frac{1}{N}\sum_i \left(\frac{\Delta m_i}{\sqrt{\Delta t}}\right)^2$$

This is the realised variance of the spot, which equals $\rho$ under the model. Using spot variance directly is less data-hungry but ignores any q-dependent heteroskedasticity.

### Sanity check

Bin by $q^2$ and plot the mean of $r_i^2 / \Delta t$ against $q_i^2$. The slope should be approximately $\hat{\rho}$.

---

## Estimating $t(q)$ — internalization time

$t(q)$ is the expected time in minutes to reach zero inventory starting from $q$.

### From inventory trajectories

For each time $i$ where $|q_i| > 0$, find the first subsequent time $j > i$ at which $q_j = 0$ (or changes sign). Define

$$\tau_i = (j - i)\cdot\Delta t$$

Pool observations by $|q_i|$ and compute the conditional mean:

$$\hat{t}(|q|) = E[\tau_i \mid |q_i| = |q|]$$

### Polynomial fit

Fit the polynomial model

$$t(q) = \tau_0 + \tau_1|q| + \tau_2|q|^2$$

by regressing $\hat{t}(|q_i|)$ on $1$, $|q_i|$, $q_i^2$ via OLS. This directly gives the $(\tau_0, \tau_1, \tau_2)$ coefficients used in `PolynomialInternalizationTime`.

### Practical note

Censoring is common: if the inventory never reaches zero within the sample window, the trajectory is right-censored. Use a survival-regression estimator (e.g. Kaplan-Meier or a parametric accelerated-failure-time model) if censored observations are frequent.

---

## Joint model and identification

The full conditional distribution of $\Delta P_i / \Delta t$ given $q_i$ is

$$\frac{\Delta P_i}{\Delta t} \sim \mathcal{N}\!\left(-c\cdot|q_i|,\; \rho\cdot q_i^2\right)$$

Both parameters can be estimated jointly by maximum likelihood:

$$\log L = -\frac{1}{2}\sum_i \left[\log(2\pi\rho q_i^2\Delta t) + \frac{(\Delta P_i + c|q_i|\Delta t)^2}{\rho q_i^2 \Delta t}\right]$$

This is a heteroskedastic normal regression — it weights observations at large $|q|$ less than the two-stage OLS approach does, which is appropriate because those observations have higher noise.

The MLE closed forms do not exist in general; use numerical optimisation over $(c, \rho)$.

---

## Putting it together

Once $\hat{c}$, $\hat{\rho}$, and $t(q)$ are estimated, the penalty is

$$\Pi(q) = \bigl(\hat{c}\cdot|q| + \hat{\rho}\cdot q^2\bigr)\cdot t(q)$$

Plug $\hat{\rho}$ directly in place of $\gamma\sigma^2$ in `CarryCost`. The $\hat{c}$ term requires a separate linear component; currently the model only has the quadratic term, so adding $\hat{c}$ means extending `PolynomialInventoryPenalty` to include a linear coefficient.

### Expected magnitudes (EUR/SEK, 1-minute bars)

| Parameter | Typical order | Unit |
|---|---|---|
| $c$ | $10^{-5}$ – $10^{-4}$ | SEK / EUR / min |
| $\rho$ | $10^{-7}$ – $10^{-6}$ | SEK / EUR² / min |
| $\tau_0$ | 2 – 10 | min |

If $\hat{c} \approx 0$ within statistical error, the linear term is not supported by your data and you can revert to the pure quadratic penalty.
