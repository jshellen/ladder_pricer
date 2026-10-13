"""RFQ tier markout curve expressed in EUR per traded EUR 1M.

Negative values are adverse for the RFQ taker's opposite-side dealer position.
This module describes the *observed time curve only*: it does not model the
future inventory path or the policy-dependent MTM exposure.
"""
from __future__ import annotations

import numpy as np


def markout_eur_per_million(t_seconds, asymptotic_eur_per_million: float, tau_seconds: float):
    """m(t) = m_inf * (1 - exp(-t/tau)), for t >= 0 seconds."""
    t = np.asarray(t_seconds, dtype=float)
    if not (np.isfinite(asymptotic_eur_per_million) and asymptotic_eur_per_million <= 0.0):
        raise ValueError("Markout asymptote must be nonpositive and finite")
    if not (np.isfinite(tau_seconds) and tau_seconds > 0.0):
        raise ValueError("Markout tau must be finite and positive")
    result = asymptotic_eur_per_million * (-np.expm1(-np.maximum(t, 0.0) / tau_seconds))
    return float(result) if result.ndim == 0 else result
