from __future__ import annotations

from copy import deepcopy

_DEFAULT_CONFIG = {
    "spot": 11.5,
    "spreadPips": 20.0,
    "spotDrift": 0.0,
    "sigmaPips": 20.0,
    "gamma": 0.1,
    "grid": {
        "mode": "piecewise",
        "maxAbs": 20.0,
        "step": 1.0,
        "fineHalfWidth": 3.0,
        "fineStep": 0.25,
        "coarseStep": 1.0,
    },
    "internalization": {"tau0": 4.0, "tau1": 0.070, "tau2": 0.0084},
    "tiers": [
        {
            "enabled": True,
            "name": "Tier 1",
            "sizes": [1, 2, 3, 5, 10, 20],
            "flow": {"A0": 0.0155, "theta": 0.144, "beta": 0.0857, "steepness": 8.42, "shift": 0.52, "volumeShift": 0.026},
            "markout": {"impactScalePips": 1.0, "sizeExponent": 0.5, "tauMinutes": 0.5},
            "useMarkout": True,
            "deltaMin": -10.0,
            "deltaMax": 100.0,
        },
        {
            "enabled": True,
            "name": "Tier 2",
            "sizes": [1, 2, 3, 5, 10, 20],
            "flow": {"A0": 0.0232, "theta": 0.303, "beta": 0.122, "steepness": 2.86, "shift": 0.48, "volumeShift": 0.020},
            "markout": {"impactScalePips": 1.0, "sizeExponent": 0.5, "tauMinutes": 0.5},
            "useMarkout": True,
            "deltaMin": -10.0,
            "deltaMax": 100.0,
        },
        {
            "enabled": False,
            "name": "Tier 3",
            "sizes": [1],
            "flow": {"A0": 1.0, "theta": 0.144, "beta": 0.0857, "steepness": 20.0, "shift": 0.52, "volumeShift": 0.026},
            "markout": {"impactScalePips": 4.0, "sizeExponent": 0.5, "tauMinutes": 0.10},
            "useMarkout": True,
            "deltaMin": -10.0,
            "deltaMax": 100.0,
        },
    ],
    "darkPool": {
        "enabled": False,
        "distribution": "geometric",
        "lambdaBid": 2.0,
        "lambdaAsk": 2.0,
        "pBid": 0.5,
        "pAsk": 0.5,
        "muBid": 2.0,
        "muAsk": 2.0,
        "p0Bid": 0.1,
        "p0Ask": 0.1,
        "feeBid": 0.0,
        "feeAsk": 0.0,
        "postedSizes": [1, 2, 3, 5],
        "allowBothSides": False,
    },
}


def default_config() -> dict:
    return deepcopy(_DEFAULT_CONFIG)
