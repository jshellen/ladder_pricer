from __future__ import annotations

from copy import deepcopy
import threading
from typing import Any, Callable, Iterable

from . import _native


_EXPECTED_ECN_PARAMETERIZATION_VERSION = 8
if getattr(_native, "ECN_PARAMETERIZATION_VERSION", None) != _EXPECTED_ECN_PARAMETERIZATION_VERSION:
    raise RuntimeError(
        "The compiled trinity._native extension is stale. Rebuild it with "
        "`python3 setup.py build_ext --inplace` (or reinstall with `pip install -e .`) "
        "before running the dashboard."
    )


class Engine:
    """Small Python façade around the C++ pricing engine.

    The model is configured once. Heavy numerical work remains entirely in C++.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = deepcopy(config)
        self._native = _native.Engine(self.config)
        self._lock = threading.RLock()

    def solve(self) -> dict[str, Any]:
        with self._lock:
            return self._native.solve()

    def statistics(self, horizon_minutes: float = 570.0, initial_inventory: float = 0.0) -> dict[str, Any]:
        with self._lock:
            return self._native.statistics(float(horizon_minutes), float(initial_inventory))

    def simulate(
        self,
        horizon_minutes: float = 570.0,
        paths: int = 1000,
        initial_inventory: float = 0.0,
        seed: int = 12345,
        retained_paths: int = 6,
        sample_points: int = 191,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            return self._native.simulate(
                float(horizon_minutes), int(paths), float(initial_inventory), int(seed),
                int(retained_paths), int(sample_points), progress_callback,
            )

    def frontier(
        self,
        gamma_values: Iterable[float],
        horizon_minutes: float = 570.0,
        initial_inventory: float = 0.0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            return self._native.frontier(
                [float(x) for x in gamma_values], float(horizon_minutes), float(initial_inventory)
            )
