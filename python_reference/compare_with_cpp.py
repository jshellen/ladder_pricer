"""Compare the preserved pure-Python Howard engine with the production C++ backend.

Run from the repository root after installing the main package:

    python python_reference/compare_with_cpp.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "python_reference"
sys.path.insert(0, str(REFERENCE_ROOT))

import ladder_pricer as ref  # noqa: E402
from trinity import Engine, default_config  # noqa: E402


def uniform_grid(max_abs: float, step: float = 1.0) -> list[float]:
    if abs(step - 1.0) > 1e-12:
        raise ValueError("Reference comparison expects the fixed 1M inventory step.")
    n = int(round(max_abs))
    if abs(max_abs - n) > 1e-12:
        raise ValueError("maxAbs must be an integer on the uniform 1M grid.")
    return [float(q) for q in range(-n, n + 1)]


def reference_solver_from_config(cfg: dict) -> ref.HJBLadderSolver:
    grid_cfg = cfg["grid"]
    if grid_cfg["mode"] != "uniform":
        raise NotImplementedError("Comparison helper expects the production uniform inventory grid.")

    solver_config = ref.SolverConfig(
        q_grid=uniform_grid(float(grid_cfg["maxAbs"]), float(grid_cfg["step"])),
        spot=float(cfg["spot"]),
        spot_drift=float(cfg["spotDrift"]),
        spread=float(cfg["spreadPips"]) / 10_000.0,
    )

    tiers: list[ref.MDPTier] = []
    for spec in cfg["tiers"]:
        if not spec["enabled"]:
            continue
        flow = spec["flow"]
        markout = spec["markout"]
        tiers.append(
            ref.MDPTier(
                name=str(spec["name"]),
                sizes=[float(x) for x in spec["sizes"]],
                flow_curve=ref.LogisticFlowCurve(
                    A0=float(flow["A0"]),
                    theta=float(flow["theta"]),
                    beta=float(flow["beta"]),
                    shift=float(flow["shift"]),
                    steepness=float(flow["steepness"]),
                    volume_shift=float(flow["volumeShift"]),
                ),
                markout_model=ref.SaturatingMarkoutModel(
                    asymptotic_price_move=(
                        float(markout["asymptoticEurPerEurM"]) * float(cfg["spot"]) / 1_000_000.0
                    ),
                    tau=float(markout["tauMinutes"]),
                ),
                delta_min=float(spec["deltaMin"]),
                delta_max=float(spec["deltaMax"]),
                use_markout=bool(spec["useMarkout"]),
            )
        )

    penalty = ref.QuadraticInventoryPenalty(
        carry_cost=ref.CarryCost(
            risk_aversion=float(cfg["gamma"]),
            sigma=float(cfg["sigmaPips"]) / 10_000.0,
        )
    )
    t = cfg["internalization"]
    internalization = ref.PolynomialInternalizationTime(
        tau0=float(t["tau0"]), tau1=float(t["tau1"]), tau2=float(t["tau2"])
    )

    # The standard comparison intentionally leaves the dark pool disabled,
    # matching the current default/stress configuration.
    if cfg["darkPool"]["enabled"]:
        raise NotImplementedError("Comparison helper currently expects dark pool disabled.")

    return ref.HJBLadderSolver(
        config=solver_config,
        penalty=penalty,
        internalization_time=internalization,
        mdp_tiers=tiers,
        dark_pool=None,
    )


def main() -> None:
    cfg = default_config()
    cfg["tiers"][2]["enabled"] = True

    cpp = Engine(cfg)
    cpp_solution = cpp.solve()
    cpp_stats = cpp.statistics(570.0, 0.0)

    py_solver = reference_solver_from_config(cfg)
    py_solution = py_solver.solve()
    py_stats = py_solver.closed_form_pnl_statistics(
        horizon_minutes=570.0,
        sigma=float(cfg["sigmaPips"]) / 10_000.0,
        initial_inventory=0.0,
    )

    py_mean_eur = py_stats.expected_pnl_base_ccy_at_reference_spot
    py_std_eur = py_stats.std_pnl_base_ccy_at_reference_spot

    print("Tier-3 EURSEK regression comparison")
    print("-----------------------------------")
    print(f"Average reward   C++={cpp_solution['averageReward']:.10g}  Python={py_solution.average_reward:.10g}")
    print(f"Mean PnL [EUR]   C++={cpp_stats['meanBase']:.4f}  Python={py_mean_eur:.4f}")
    print(f"Stdev [EUR]      C++={cpp_stats['stdBase']:.4f}  Python={py_std_eur:.4f}")
    print()
    print(f"|Δ reward| = {abs(cpp_solution['averageReward'] - py_solution.average_reward):.3g}")
    print(f"|Δ mean|   = {abs(cpp_stats['meanBase'] - py_mean_eur):.3g} EUR")
    print(f"|Δ stdev|  = {abs(cpp_stats['stdBase'] - py_std_eur):.3g} EUR")


if __name__ == "__main__":
    main()
