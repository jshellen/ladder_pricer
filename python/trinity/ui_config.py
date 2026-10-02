from __future__ import annotations

from typing import Any, Mapping

from .defaults import default_config, ecn_delta_grid


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
    cfg["spotDrift"] = _number(values, "drift")
    cfg["sigmaPips"] = _number(values, "sigma")
    cfg["gamma"] = _number(values, "gamma")
    cfg["internalization"] = {
        "tau0": _number(values, "tau0"),
        "tau1": _number(values, "tau1"),
        "tau2": _number(values, "tau2"),
    }

    tiers: list[dict[str, Any]] = []
    for i in range(3):
        p = f"t{i}"
        tiers.append({
            "enabled": checked(values.get(f"{p}-enabled")),
            "name": str(values.get(f"{p}-name", f"Tier {i + 1}")),
            "sizes": parse_numbers(values.get(f"{p}-sizes", "")),
            "flow": {
                "A0": _number(values, f"{p}-A0"),
                "theta": _number(values, f"{p}-theta"),
                "beta": _number(values, f"{p}-beta"),
                "steepness": _number(values, f"{p}-steep"),
                "shift": _number(values, f"{p}-shift"),
                "volumeShift": _number(values, f"{p}-vshift"),
            },
            "markout": {
                "impactScalePips": _number(values, f"{p}-impact"),
                "sizeExponent": _number(values, f"{p}-impact-beta"),
                "tauMinutes": _number(values, f"{p}-impact-tau"),
            },
            "useMarkout": checked(values.get(f"{p}-markout-enabled")),
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
        "quoteSize": _number(values, "ecn-size") if values.get("ecn-size") is not None else ecn_default["quoteSize"],
        "makerFeePips": _number(values, "ecn-fee") if values.get("ecn-fee") is not None else ecn_default["makerFeePips"],
    }
    return cfg
