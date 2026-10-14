from __future__ import annotations

import math

from typing import Any, Mapping

from .defaults import default_config, default_fx_pair_volatility, ecn_delta_grid


def checked(value: Any) -> bool:
    return "on" in (value or [])


def parse_numbers(raw: Any) -> list[float]:
    values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def _number(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(f"{key} must be a scalar number, got {type(value).__name__}")
    if value is None:
        raise ValueError(f"{key} is required")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric, got {value!r}") from exc


def _ecn_trade_size_probabilities(values: Mapping[str, Any], fallback: list[float]) -> list[float]:
    raw = values.get("ecn-size-probs")
    if raw is None:
        probs = [float(x) for x in fallback]
    else:
        if not isinstance(raw, (list, tuple)) or len(raw) != 5:
            raise ValueError("ECN trade-size probabilities must contain five entries")
        probs = [float(x) for x in raw]
    if any((not 0.0 <= p <= 1.0) for p in probs):
        raise ValueError("ECN trade-size probabilities must lie between 0 and 1")
    if abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError("ECN trade-size probabilities must sum to 1")
    return probs


def build_config(values: Mapping[str, Any]) -> dict[str, Any]:
    """Build the native model config from named Dash component values.

    The mapping is keyed by Dash component ID rather than callback position. This
    intentionally makes UI control reordering harmless to the numerical config.
    """
    cfg = default_config()

    cfg["grid"] = {
        "mode": "uniform",
        "maxAbs": _number(values, "qmax"),
        "step": 1.0,
    }
    cfg["spot"] = _number(values, "spot")
    cfg["spreadPips"] = _number(values, "spread")
    cfg["gamma"] = _number(values, "gamma")
    # One read-only target-pair volatility model, loaded from JSON rather than
    # from any editable dashboard state. The same config reaches the C++ solver
    # and is subsequently reused by Monte Carlo and the efficient frontier.
    cfg["targetPair"] = str(values.get("target-pair") or cfg["targetPair"]).upper()
    cfg["volatilityRegimes"] = default_fx_pair_volatility(cfg["targetPair"])
    cfg["sigmaPips"] = cfg["volatilityRegimes"]["sigmaPips"][len(cfg["volatilityRegimes"]["sigmaPips"]) // 2]
    tiers: list[dict[str, Any]] = []
    for i in range(len(cfg["tiers"])):
        p = f"t{i}"
        sizes = parse_numbers(values.get(f"{p}-sizes", ""))
        flow = {
            "A0": _number(values, f"{p}-A0"),
            "theta": _number(values, f"{p}-theta"),
            "beta": _number(values, f"{p}-beta"),
            "steepness": _number(values, f"{p}-steep"),
            "shift": _number(values, f"{p}-shift"),
            "volumeShift": _number(values, f"{p}-vshift"),
        }
        m_inf = _number(values, f"{p}-markout-asymptote")
        tau_s = _number(values, f"{p}-markout-tau-seconds")
        if not (float("-inf") < m_inf <= 0.0):
            raise ValueError(f"Markout asymptote for Tier {i + 1} must be finite and <= 0 EUR/M")
        if not (0.0 < tau_s < float("inf")):
            raise ValueError(f"Markout tau for Tier {i + 1} must be finite and > 0 seconds")
        tiers.append({
            "enabled": checked(values.get(f"{p}-enabled")),
            "name": str(values.get(f"{p}-name", f"Tier {i + 1}")),
            "sizes": sizes,
            "flow": flow,
            "useMarkout": checked(values.get(f"{p}-markout-enabled")),
            "markout": {"asymptoticEurPerMillion": m_inf, "tauSeconds": tau_s},
            "feePips": _number(values, f"{p}-fee") if values.get(f"{p}-fee") is not None else 0.0,
            "deltaMin": _number(values, f"{p}-dmin"),
            "deltaMax": _number(values, f"{p}-dmax"),
        })
    cfg["tiers"] = tiers

    cfg["darkPool"] = {
        "enabled": checked(values.get("dp-enabled")),
        "lambda": _number(values, "dp-lambda"),
        "mu": _number(values, "dp-mu"),
        "p0": _number(values, "dp-p0"),
        "feePips": _number(values, "dp-fee"),
        "postedSizes": parse_numbers(values.get("dp-sizes", "")),
    }
    ecn_default = cfg["passiveEcn"]
    min_delta = _number(values, "ecn-dmin") if values.get("ecn-dmin") is not None else ecn_default["minDelta"]
    max_delta = _number(values, "ecn-dmax") if values.get("ecn-dmax") is not None else ecn_default["maxDelta"]
    cfg["passiveEcn"] = {
        "enabled": checked(values.get("ecn-enabled")),
        "minDelta": min_delta,
        "maxDelta": max_delta,
        "deltas": ecn_delta_grid(min_delta, max_delta),
        "flow": {
            "A": _number(values, "ecn-A") if values.get("ecn-A") is not None else ecn_default["flow"]["A"],
            "k": _number(values, "ecn-k") if values.get("ecn-k") is not None else ecn_default["flow"]["k"],
        },
        "tradeSizeProbabilities": _ecn_trade_size_probabilities(
            values, ecn_default["tradeSizeProbabilities"]
        ),
        "quoteSize": _number(values, "ecn-size") if values.get("ecn-size") is not None else ecn_default["quoteSize"],
        "makerFeePips": _number(values, "ecn-fee") if values.get("ecn-fee") is not None else ecn_default["makerFeePips"],
    }
    return cfg
