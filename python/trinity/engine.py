from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import threading
from typing import Any, Callable, Iterable

from . import _native


_EXPECTED_ECN_PARAMETERIZATION_VERSION = 10
if getattr(_native, "ECN_PARAMETERIZATION_VERSION", None) != _EXPECTED_ECN_PARAMETERIZATION_VERSION:
    raise RuntimeError(
        "The compiled trinity._native extension is stale. Rebuild it with "
        "`python3 setup.py build_ext --inplace` (or reinstall with `pip install -e .`) "
        "before running the dashboard."
    )


class Engine:
    """Small Python façade around the C++ pricing engine.

    The solved native policy is immutable for statistics/simulation calls.
    A small reader/writer gate therefore allows multiple read-only numerical
    jobs (notably Monte Carlo + closed-form analytics) to run concurrently,
    while ``solve`` remains exclusive.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = deepcopy(config)
        self._native = _native.Engine(self.config)
        self._gate = threading.Condition(threading.RLock())
        self._readers = 0
        self._writer = False
        self._solved = False

    @contextmanager
    def _read_access(self):
        with self._gate:
            while self._writer:
                self._gate.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._gate:
                self._readers -= 1
                if self._readers == 0:
                    self._gate.notify_all()

    @contextmanager
    def _write_access(self):
        with self._gate:
            while self._writer or self._readers:
                self._gate.wait()
            self._writer = True
        try:
            yield
        finally:
            with self._gate:
                self._writer = False
                self._gate.notify_all()

    def _ensure_solved(self) -> None:
        if self._solved:
            return
        with self._write_access():
            if not self._solved:
                # This path is mainly defensive; the dashboard normally calls
                # solve() before exposing Monte Carlo controls.
                self._native.solve()
                self._solved = True

    def solve(self) -> dict[str, Any]:
        with self._write_access():
            result = self._native.solve()
            self._solved = True
            return result

    def statistics(self, horizon_minutes: float = 570.0, initial_inventory: float = 0.0) -> dict[str, Any]:
        self._ensure_solved()
        with self._read_access():
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
        compact_result: bool = False,
    ) -> dict[str, Any]:
        self._ensure_solved()
        with self._read_access():
            return self._native.simulate(
                float(horizon_minutes), int(paths), float(initial_inventory), int(seed),
                int(retained_paths), int(sample_points), progress_callback, bool(compact_result),
            )

    def frontier(
        self,
        gamma_values: Iterable[float],
        horizon_minutes: float = 570.0,
        initial_inventory: float = 0.0,
    ) -> list[dict[str, Any]]:
        self._ensure_solved()
        with self._read_access():
            return self._native.frontier(
                [float(x) for x in gamma_values], float(horizon_minutes), float(initial_inventory)
            )

