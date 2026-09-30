from __future__ import annotations

from typing import Any, Mapping

from .defaults import default_config


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
        "mode": str(values.get("grid-mode", cfg["grid"]["mode"])),
        "maxAbs": _number(values, "qmax"),
        "step": _number(values, "qstep"),
        "fineHalfWidth": _number(values, "qfinehalf"),
        "fineStep": _number(values, "qfinestep"),
        "coarseStep": _number(values, "qcoarsestep"),
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
            "deltaMin": _number(values, f"{p}-dmin"),
            "deltaMax": _number(values, f"{p}-dmax"),
        })
    cfg["tiers"] = tiers

    cfg["darkPool"] = {
        "enabled": checked(values.get("dp-enabled")),
        "distribution": str(values.get("dp-dist", "geometric")),
        "lambdaBid": _number(values, "dp-lb"),
        "lambdaAsk": _number(values, "dp-la"),
        "pBid": _number(values, "dp-pb"),
        "pAsk": _number(values, "dp-pa"),
        "muBid": _number(values, "dp-mub"),
        "muAsk": _number(values, "dp-mua"),
        "p0Bid": _number(values, "dp-p0b"),
        "p0Ask": _number(values, "dp-p0a"),
        "feeBid": _number(values, "dp-fb"),
        "feeAsk": _number(values, "dp-fa"),
        "postedSizes": parse_numbers(values.get("dp-sizes", "")),
        "allowBothSides": checked(values.get("dp-both")),
    }
    ecn_default = cfg["passiveEcn"]
    cfg["passiveEcn"] = {
        "enabled": checked(values.get("ecn-enabled")),
        "deltas": list(ecn_default["deltas"]),
        "flow": {
            "A0": _number(values, "ecn-A0") if values.get("ecn-A0") is not None else ecn_default["flow"]["A0"],
            "theta": _number(values, "ecn-theta") if values.get("ecn-theta") is not None else ecn_default["flow"]["theta"],
            "beta": _number(values, "ecn-beta") if values.get("ecn-beta") is not None else ecn_default["flow"]["beta"],
            "steepness": _number(values, "ecn-steep") if values.get("ecn-steep") is not None else ecn_default["flow"]["steepness"],
            "shift": _number(values, "ecn-shift") if values.get("ecn-shift") is not None else ecn_default["flow"]["shift"],
            "volumeShift": _number(values, "ecn-vshift") if values.get("ecn-vshift") is not None else ecn_default["flow"]["volumeShift"],
        },
        "quoteSize": _number(values, "ecn-size") if values.get("ecn-size") is not None else ecn_default["quoteSize"],
        "makerFeePips": _number(values, "ecn-fee") if values.get("ecn-fee") is not None else ecn_default["makerFeePips"],
    }
    return cfg
