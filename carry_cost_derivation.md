# Carry Cost Derivation

## Setup

At time $s = 0$ the market-maker holds inventory $q$ EUR.
Empirical estimation gives an expected time-to-zero of $t(q)$ minutes.
We model inventory as decaying smoothly to zero:

$$q(s), \quad s \in [0,\, t(q)], \quad q(0) = q, \quad q(t(q)) = 0.$$

## Carry Rate

At any instant $s$ the carry accrues at a rate proportional to the current
absolute inventory:

$$\text{carry rate}(s) = c \cdot |q(s)| \quad [\text{EUR/min}]$$

where $c > 0$ is a carry coefficient (positive — it represents a cost).

Converting to the value function's currency (SEK) via spot rate $m$
[SEK/EUR]:

$$\text{carry rate}(s) \text{ in SEK/min} = c \cdot m \cdot |q(s)|.$$

## Total Carry Over the Internalization Period

Integrate over the full internalization horizon:

$$\mathcal{C}(q) = c \cdot m \int_0^{t(q)} |q(s)|\, ds \quad [\text{SEK}].$$

This is the expected total carry cost incurred while working out of inventory
$q$.

## Linear Decay Assumption

Assume inventory depletes at a constant rate:

$$q(s) = q\!\left(1 - \frac{s}{t(q)}\right).$$

Then $|q(s)| = |q|\!\left(1 - \dfrac{s}{t(q)}\right)$ and the integral becomes:

$$\int_0^{t(q)} |q(s)|\, ds
  = |q| \int_0^{t(q)} \left(1 - \frac{s}{t(q)}\right) ds
  = |q| \cdot \frac{t(q)}{2}.$$

So under linear decay:

$$\mathcal{C}(q) = \frac{c \cdot m}{2} \cdot |q| \cdot t(q).$$

This gives an **$|q| \cdot t(q)$** penalty — linear in inventory magnitude.

## Quadratic Penalty Form

If the per-unit carry rate itself scales with inventory size
(e.g. because larger positions face wider bid-offer on the hedge leg):

$$\text{carry rate}(s) = c \cdot |q(s)|^2,$$

the integral becomes:

$$\int_0^{t(q)} |q(s)|^2\, ds
  = q^2 \int_0^{t(q)} \left(1 - \frac{s}{t(q)}\right)^2 ds
  = q^2 \cdot \frac{t(q)}{3}.$$

Total carry cost:

$$\mathcal{C}(q) = \frac{c \cdot m}{3} \cdot q^2 \cdot t(q).$$

This gives a **$q^2 \cdot t(q)$** penalty — quadratic in inventory.

## Connection to the HJB Penalty

The HJB value function uses the inventory penalty:

$$\Pi(q) = \gamma \sigma^2 \cdot q^2 \cdot t(q).$$

Matching the quadratic derivation above:

$$\gamma \sigma^2 \;\longleftrightarrow\; \frac{c \cdot m}{3}.$$

So the risk-aversion-weighted volatility term $\gamma\sigma^2$ can be
interpreted as a rescaled carry coefficient:

$$c = \frac{3\,\gamma\sigma^2}{m}.$$

For $\gamma = 2$, $\sigma = 0.25$, $m = 11.5$:

$$c = \frac{3 \times 2 \times 0.0625}{11.5} \approx 0.033 \;\text{EUR}^{-1}\text{min}^{-1}.$$

## Internalization Time

The polynomial model for $t(q)$ is:

$$t(q) = \tau_0 + \tau_1 |q| + \tau_2 q^2 \quad [\text{min}].$$

Substituting into the quadratic penalty:

$$\Pi(q) = \gamma\sigma^2 \cdot q^2 \cdot (\tau_0 + \tau_1|q| + \tau_2 q^2).$$

This gives terms of order $q^2$, $|q|^3$, and $q^4$ — capturing the
nonlinear cost of holding large inventory.

## Summary

| Carry rate model | Penalty form | Integral factor |
|---|---|---|
| $c \cdot \|q(s)\|$ (linear) | $\tfrac{cm}{2}\|q\|t(q)$ | $1/2$ |
| $c \cdot \|q(s)\|^2$ (quadratic) | $\tfrac{cm}{3}q^2 t(q)$ | $1/3$ |

The current implementation uses the **quadratic** form, which is consistent
with the risk-aversion framework where penalty scales as variance ($\sigma^2$)
times squared inventory.
