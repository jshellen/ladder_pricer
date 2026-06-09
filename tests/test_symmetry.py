
import sys
import os
import unittest
import numpy as np
import ladder_pricer as lp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import build_uniform_centered_q_grid, build_piecewise_centered_q_grid

UNIFORM_GRID = build_uniform_centered_q_grid(q_abs_max=20.0, q_step=1.0).tolist()
PIECEWISE_GRID = build_piecewise_centered_q_grid(
    q_abs_max=20.0, fine_half_width=3.0, fine_step=0.25, coarse_step=1.0
).tolist()
SIZES = [1, 2, 3, 5, 10, 20]


def build_solution(
    *,
    q_grid: list[float] = UNIFORM_GRID,
    sizes: list[float] = SIZES,
    dark_pool: lp.DarkPoolVenue | None = None,
    dt: float = 0.002,
    n_iter: int = 140,
) -> tuple[lp.HJBSolution, lp.HJBLadderSolver]:
    config = lp.SolverConfig()
    config.q_grid = q_grid
    config.spread = 20.0 / 10000.0
    config.spot_drift = 0.0
    config.dt = dt
    config.n_iter = n_iter
    config.tol_h = 1e-5
    config.tol_rhs = 1e-4
    config.min_iter = 5
    config.consecutive_passes_required = 3

    flow = lp.LogisticFlowCurve(
        A0=1.0,
        theta=0.0,
        beta=0.0,
        shift=0.5,
        steepness=10.0,
        volume_shift=0.015,
    )
    markout = lp.SqrtMarkoutModel(base=0.0, coeff=0.0)
    tier = lp.MDPTier(
        name="test",
        sizes=sizes,
        flow_curve=flow,
        markout_model=markout,
        delta_min=-100.0,
        delta_max=100.0,
    )

    penalty = lp.PolynomialInventoryPenalty(
        carry_cost=lp.CarryCost(risk_aversion=10.0, sigma=20.0 / 10_000.0),
        internalization_time=lp.PolynomialInternalizationTime(tau0=5.0, tau1=0.1, tau2=0.0015),
    )

    solver = lp.HJBLadderSolver(
        config=config,
        penalty=penalty,
        mdp_tiers=[tier],
        dark_pool=dark_pool,
    )
    solution = solver.solve()
    return solution, solver


class _ValueFunctionSymmetryMixin:
    """Shared symmetry assertions; mixed into grid-specific test classes."""

    TOL = 1e-5  # loose enough for pseudo-time convergence noise
    solution: lp.HJBSolution

    def test_h_is_even(self):
        """h(-q) == h(q) for all q."""
        h = list(self.solution.h)
        n = len(h)
        for i in range(n):
            j = n - 1 - i
            self.assertAlmostEqual(h[i], h[j], delta=self.TOL, msg=f"h[{i}] != h[{j}]")

    def test_h_zero_at_center(self):
        """h is normalised so h(0) == 0."""
        h = list(self.solution.h)
        mid = len(h) // 2
        self.assertAlmostEqual(h[mid], 0.0, delta=self.TOL)


class TestValueFunctionSymmetryUniform(_ValueFunctionSymmetryMixin, unittest.TestCase):
    """Uniform grid, step = 1.0."""

    def setUp(self):
        self.solution, _ = build_solution(q_grid=UNIFORM_GRID, dt=0.002, n_iter=140)


class TestValueFunctionSymmetryPiecewise(_ValueFunctionSymmetryMixin, unittest.TestCase):
    """Piecewise grid: fine half-width = 3, fine step = 0.25, coarse step = 1.0."""

    def setUp(self):
        self.solution, _ = build_solution(q_grid=PIECEWISE_GRID, dt=0.002, n_iter=140)


class TestMDPPolicySymmetry(unittest.TestCase):

    TOL = 1e-4  # golden-section tolerance dominates here

    def setUp(self):
        self.solution, _ = build_solution()
        self.tier = self.solution.mdp_tiers[0]

    def test_bid_at_negative_q_equals_ask_at_positive_q(self):
        """bid delta at -q == ask delta at +q for all rungs."""
        policy = self.tier.policy
        q_grid = list(policy.q_grid)
        n = len(q_grid)
        nz = len(SIZES)

        for i in range(n):
            j = n - 1 - i
            for k in range(nz):
                bid = policy.bid[i][k]
                ask = policy.ask[j][k]
                self.assertAlmostEqual(
                    bid, ask, delta=self.TOL,
                    msg=f"bid[q={q_grid[i]}, size={SIZES[k]}]={bid:.6f} "
                        f"!= ask[q={q_grid[j]}, size={SIZES[k]}]={ask:.6f}"
                )


class TestDarkPoolPolicySymmetry(unittest.TestCase):

    TOL = 1e-10

    def setUp(self):
        dark_pool = lp.DarkPoolVenue(
            lambda_bid=2.0,
            lambda_ask=2.0,
            p_bid=0.5,
            p_ask=0.5,
            fee_per_unit_bid=0.0,
            fee_per_unit_ask=0.0,
            posted_sizes=[1, 2, 3, 5, 10, 20],
            allow_both_sides=False,
        )
        self.solution, _ = build_solution(dark_pool=dark_pool)
        self.venue = self.solution.dark_pool

    def test_bid_posted_size_mirrors_ask(self):
        """bid active/size at -q == ask active/size at +q."""
        policy = self.venue.policy
        q_grid = list(policy.q_grid)
        n = len(q_grid)

        for i in range(n):
            j = n - 1 - i
            bid_active = policy.is_active(i, "bid")
            ask_active = policy.is_active(j, "ask")
            self.assertEqual(
                bid_active, ask_active,
                msg=f"active mismatch: bid at q={q_grid[i]} active={bid_active}, "
                    f"ask at q={q_grid[j]} active={ask_active}"
            )
            if bid_active:
                bid_size = policy.bid_size[i]
                ask_size = policy.ask_size[j]
                print(f"bid at q={q_grid[i]}: size={bid_size}, ask at q={q_grid[j]}: size={ask_size}")
                self.assertAlmostEqual(
                    bid_size, ask_size, delta=self.TOL,
                    msg=f"size mismatch: bid at q={q_grid[i]}={bid_size}, "
                        f"ask at q={q_grid[j]}={ask_size}"
                )


class TestDarkPoolZIPPolicySymmetry(unittest.TestCase):

    TOL = 1e-10

    def setUp(self):
        dark_pool = lp.DarkPoolVenue(
            dist_bid=lp.ZeroInflatedPoissonArrivalDist(lambda_arr=2.0, mu=2.0, p0=0.1),
            dist_ask=lp.ZeroInflatedPoissonArrivalDist(lambda_arr=2.0, mu=2.0, p0=0.1),
            fee_per_unit_bid=0.0,
            fee_per_unit_ask=0.0,
            posted_sizes=[1, 2, 3, 5, 10, 20],
            allow_both_sides=False,
        )
        self.solution, _ = build_solution(dark_pool=dark_pool)
        self.venue = self.solution.dark_pool

    def test_bid_posted_size_mirrors_ask(self):
        """bid active/size at -q == ask active/size at +q (ZIP distribution)."""
        policy = self.venue.policy
        q_grid = list(policy.q_grid)
        n = len(q_grid)

        for i in range(n):
            j = n - 1 - i
            bid_active = policy.is_active(i, "bid")
            ask_active = policy.is_active(j, "ask")
            self.assertEqual(
                bid_active, ask_active,
                msg=f"active mismatch: bid at q={q_grid[i]} active={bid_active}, "
                    f"ask at q={q_grid[j]} active={ask_active}"
            )
            if bid_active:
                bid_size = policy.bid_size[i]
                ask_size = policy.ask_size[j]
                self.assertAlmostEqual(
                    bid_size, ask_size, delta=self.TOL,
                    msg=f"size mismatch: bid at q={q_grid[i]}={bid_size}, "
                        f"ask at q={q_grid[j]}={ask_size}"
                )


if __name__ == "__main__":
    unittest.main()
