"""Construct a finite-state Markov chain approximating the CIR/Heston variance factor.

Time is measured in minutes; sigma is in pips/sqrt(minute); v=sigma**2.
The innovation coefficient is `vol_of_vol * sqrt(v)` in the same minute
convention, and the boundary scheme is one-sided (reflecting/no flux).

This module only constructs exogenous variance-regime transitions; it does not
add a spot/variance Brownian correlation or make risk flows volatility-driven.
"""

from __future__ import annotations
import math


def cir_volatility_regimes(
    *,
    levels: int = 11,
    sigma_min_pips: float = 8.0,
    sigma_max_pips: float = 42.0,
    long_run_sigma_pips: float = 20.0,
    mean_reversion_per_minute: float = 0.04,
    vol_of_vol: float = 2.0,
) -> dict:
    """Return `{sigmaPips, ratesPerMinute}` for a positive-rate CIR approximation.

    Variance states are equally spaced in `v = sigma**2`, rather than in sigma.
    For interior states the local CTMC generator matches the CIR drift exactly
    and overapproximates the infinitesimal variance by the upwind drift term
    `|drift| * dv`.  Refining the variance grid reduces this diffusion bias.
    At boundaries we truncate outward jumps, implementing a finite-state
    reflecting approximation rather than an exact CIR boundary condition.
    """
    if not isinstance(levels, int) or not 2 <= levels <= 12:
        raise ValueError("levels must be an integer from 2 to 12")
    if not (0 < sigma_min_pips < long_run_sigma_pips < sigma_max_pips):
        raise ValueError("require 0 < sigma_min < long_run_sigma < sigma_max")
    if not (mean_reversion_per_minute > 0 and vol_of_vol > 0):
        raise ValueError("mean reversion and vol of vol must be positive")
    v_min, v_max = sigma_min_pips ** 2, sigma_max_pips ** 2
    theta = long_run_sigma_pips ** 2
    dv = (v_max - v_min) / (levels - 1)
    variance = [v_min + i * dv for i in range(levels)]
    sigma = [math.sqrt(v) for v in variance]
    rates = [[0.0 for _ in range(levels)] for _ in range(levels)]
    for i, v in enumerate(variance):
        b = mean_reversion_per_minute * (theta - v)
        diffusion = (vol_of_vol ** 2) * v / (2.0 * dv ** 2)
        if i < levels - 1:
            rates[i][i + 1] = diffusion + max(b, 0.0) / dv
        if i > 0:
            rates[i][i - 1] = diffusion + max(-b, 0.0) / dv
    return {"sigmaPips": sigma, "ratesPerMinute": rates}
