
import unittest
import numpy as np
import ladder_pricer as lp

def build_uniform_centered_q_grid(q_abs_max: float, q_step: float) -> np.ndarray:
    pos = np.arange(0.0, q_abs_max + 0.5 * q_step, q_step, dtype=float)
    pos = pos[pos <= q_abs_max + 1e-12]
    if not np.isclose(pos[-1], q_abs_max, atol=1e-10, rtol=0.0):
        pos = np.append(pos, q_abs_max)
    pos = np.unique(np.round(pos, 12))
    return np.concatenate((-pos[:0:-1], pos))



UNIFORM_GRID = build_uniform_centered_q_grid(q_abs_max=20.0, q_step=1.0).tolist()
SIZES = [1, 2, 3, 5, 10, 20]


def build_solution(
    *,
    q_grid: list[float] = UNIFORM_GRID,
    sizes: list[float] = SIZES,
    dark_pool: lp.DarkPoolVenue | None = None,
) -> tuple[lp.HJBSolution, lp.HJBLadderSolver]:
    config = lp.SolverConfig()
    config.q_grid = q_grid
    config.spread = 20.0 / 10000.0
    config.spot_drift = 0.0

    flow = lp.LogisticFlowCurve(
        A0=1.0,
        theta=0.0,
        beta=0.0,
        shift=0.5,
        steepness=10.0,
        volume_shift=0.015,
    )
    markout = lp.SaturatingMarkoutModel(asymptotic_price_move=0.0, tau=0.5)
    tier = lp.MDPTier(
        name="test",
        sizes=sizes,
        flow_curve=flow,
        markout_model=markout,
        delta_min=-10.0,
        delta_max=100.0,
    )

    penalty = lp.QuadraticInventoryPenalty(
        carry_cost=lp.CarryCost(risk_aversion=10.0, sigma=20.0 / 10_000.0),
    )
    internalization_time = lp.PolynomialInternalizationTime(tau0=5.0, tau1=0.1, tau2=0.0015)

    solver = lp.HJBLadderSolver(
        config=config,
        penalty=penalty,
        internalization_time=internalization_time,
        mdp_tiers=[tier],
        dark_pool=dark_pool,
    )
    solution = solver.solve()
    return solution, solver


class _ValueFunctionSymmetryMixin:
    """Shared symmetry assertions; mixed into grid-specific test classes."""

    TOL = 1e-9
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
        self.solution, _ = build_solution(q_grid=UNIFORM_GRID)


class TestMDPPolicySymmetry(unittest.TestCase):

    TOL = 1e-9

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
            mu_bid=2.0,
            mu_ask=2.0,
            p0_bid=0.1,
            p0_ask=0.1,
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


class TestSaturatingMarkoutModel(unittest.TestCase):

    def setUp(self):
        self.model = lp.SaturatingMarkoutModel(asymptotic_price_move=1.0 / 10_000.0, tau=0.5
        )

    def test_zero_at_trade_time_and_increases_then_saturates(self):
        self.assertEqual(self.model.expected_markout(0.0), 0.0)
        vals = [self.model.expected_markout(t) for t in [0.1, 0.5, 1.0, 3.0]]
        self.assertTrue(all(vals[i + 1] > vals[i] for i in range(len(vals) - 1)))
        asymptote = self.model.asymptotic_markout()
        self.assertLess(vals[-1], asymptote)
        self.assertAlmostEqual(
            self.model.expected_markout(20.0), asymptote, delta=1e-12
        )

    def test_normalized_markout_is_size_independent(self):
        t = 1.0
        expected = self.model.expected_markout(t)
        tier = lp.MDPTier(
            name="T", sizes=[1.0, 2.0, 5.0, 10.0, 20.0],
            markout_model=self.model,
        )
        impacts = [tier.expected_markout(z, t) for z in tier.sizes()]
        self.assertTrue(all(abs(x - expected) < 1e-15 for x in impacts))


class TestQuadraticInventoryPenalty(unittest.TestCase):

    def test_standard_penalty_is_gamma_sigma_squared_q_squared(self):
        penalty = lp.QuadraticInventoryPenalty(lp.CarryCost(risk_aversion=3.0, sigma=0.002))
        self.assertAlmostEqual(penalty.value(5.0), 3.0 * 0.002**2 * 25.0, delta=1e-15)


class TestAnalyticalQuoteOptimizer(unittest.TestCase):

    def test_lambert_w_solution_satisfies_first_order_condition(self):
        flow = lp.LogisticFlowCurve(
            A0=1.0,
            theta=0.0,
            beta=0.0,
            shift=0.5,
            steepness=10.0,
            volume_shift=0.015,
        )
        spread = 20.0 / 10_000.0

        for z in [1.0, 2.0, 5.0, 20.0]:
            for additive_value in [-0.002, 0.0, 0.003, 0.02]:
                delta = flow.optimal_delta(z, spread, additive_value)
                hit = flow.hit_ratio(delta, z)
                bracket = z * spread * (0.5 - delta) + additive_value
                foc = flow.steepness * (1.0 - hit) * bracket - z * spread
                self.assertAlmostEqual(foc, 0.0, delta=1e-11)


class TestHowardSolverUniform(unittest.TestCase):

    def test_howard_converges_and_preserves_symmetry(self):
        solution, _ = build_solution(q_grid=UNIFORM_GRID)

        self.assertTrue(solution.diagnostics.converged)
        self.assertLess(solution.diagnostics.final_max_rhs, 1e-7)
        self.assertGreater(solution.average_reward, 0.0)

        h = list(solution.h)
        for i in range(len(h)):
            self.assertAlmostEqual(h[i], h[-1 - i], delta=1e-10)


class TestProductionLikeHowardSolve(unittest.TestCase):

    def test_uniform_two_tier_calibration_converges_without_solver_knobs(self):
        config = lp.SolverConfig()
        config.q_grid = UNIFORM_GRID
        config.spot = 11.5
        config.spot_drift = 0.0
        config.spread = 20.0 / 10_000.0

        tier_params = [
            (0.0155, 0.144, 0.0857, 0.52, 8.42, 0.026),
            (0.0232, 0.303, 0.122, 0.48, 2.86, 0.020),
        ]
        tiers = []
        for idx, (A0, theta, beta, shift, steepness, volume_shift) in enumerate(tier_params):
            flow = lp.LogisticFlowCurve(
                A0=A0, theta=theta, beta=beta, shift=shift,
                steepness=steepness, volume_shift=volume_shift,
            )
            markout = lp.SaturatingMarkoutModel(asymptotic_price_move=1.0 / 10_000.0, tau=0.5
            )
            tiers.append(lp.MDPTier(
                name=f"Tier {idx + 1}", sizes=SIZES, flow_curve=flow,
                markout_model=markout, delta_min=-10.0, delta_max=100.0,
            ))

        penalty = lp.QuadraticInventoryPenalty(
            carry_cost=lp.CarryCost(risk_aversion=0.01, sigma=20.0 / 10_000.0),
        )
        internalization_time = lp.PolynomialInternalizationTime(4.0, 0.070, 0.0084)
        solution = lp.HJBLadderSolver(config, penalty, internalization_time, tiers, None).solve()

        self.assertTrue(solution.diagnostics.converged)
        self.assertLess(solution.diagnostics.iterations_used, 160)
        self.assertTrue(np.isfinite(solution.average_reward))
        self.assertTrue(np.all(np.isfinite(np.asarray(solution.h, dtype=float))))


class TestLowPenaltyHoward(unittest.TestCase):

    def _solve(self, gamma: float):
        config = lp.SolverConfig()
        config.q_grid = UNIFORM_GRID
        config.spot = 11.5
        config.spot_drift = 0.0
        config.spread = 20.0 / 10_000.0

        tier_params = [
            (0.0155, 0.144, 0.0857, 0.52, 8.42, 0.026),
            (0.0232, 0.303, 0.122, 0.48, 2.86, 0.020),
        ]
        tiers = []
        for idx, (A0, theta, beta, shift, steepness, volume_shift) in enumerate(tier_params):
            tiers.append(lp.MDPTier(
                name=f"Tier {idx + 1}",
                sizes=SIZES,
                flow_curve=lp.LogisticFlowCurve(A0, theta, beta, shift, steepness, volume_shift),
                markout_model=lp.SaturatingMarkoutModel(asymptotic_price_move=1.0 / 10_000.0, tau=0.5
                ),
                delta_min=-10.0,
                delta_max=100.0,
            ))

        penalty = lp.QuadraticInventoryPenalty(lp.CarryCost(gamma, 20.0 / 10_000.0))
        internalization_time = lp.PolynomialInternalizationTime(4.0, 0.070, 0.0084)
        return lp.HJBLadderSolver(config, penalty, internalization_time, tiers, None).solve()

    def test_low_penalties_converge_and_respect_trader_gap_constraints(self):
        rewards = []
        for gamma in [0.20, 0.10, 0.01]:
            solution = self._solve(gamma)
            self.assertTrue(solution.diagnostics.converged, msg=f"gamma={gamma}")
            self.assertLess(solution.diagnostics.iterations_used, 180, msg=f"gamma={gamma}")
            self.assertTrue(np.isfinite(solution.average_reward), msg=f"gamma={gamma}")
            rewards.append(solution.average_reward)

            # Within each inventory row, larger sizes cannot be quoted tighter.
            for tier in solution.mdp_tiers:
                for row in tier.policy.bid + tier.policy.ask:
                    self.assertTrue(all(row[j] <= row[j - 1] + 1e-8 for j in range(1, len(row))))

                q_grid = np.asarray(tier.policy.q_grid, dtype=float)
                q0 = int(np.where(np.isclose(q_grid, 0.0))[0][0])

                # Long inventory: every bid rung becomes no more aggressive
                # (delta falls) while every ask rung becomes more aggressive
                # (delta rises) as q moves farther above zero.
                prev_bid = np.asarray(tier.policy.bid[q0], dtype=float)
                prev_ask = np.asarray(tier.policy.ask[q0], dtype=float)
                for i in range(q0 + 1, len(q_grid)):
                    bid = np.asarray(tier.policy.bid[i], dtype=float)
                    ask = np.asarray(tier.policy.ask[i], dtype=float)
                    self.assertTrue(np.all(bid <= prev_bid + 1e-8), msg=f"bid level rose at q={q_grid[i]} gamma={gamma}")
                    self.assertTrue(np.all(ask >= prev_ask - 1e-8), msg=f"ask level fell at q={q_grid[i]} gamma={gamma}")
                    prev_bid, prev_ask = bid, ask

                # Short inventory mirrors the level rule.
                prev_bid = np.asarray(tier.policy.bid[q0], dtype=float)
                prev_ask = np.asarray(tier.policy.ask[q0], dtype=float)
                for i in range(q0 - 1, -1, -1):
                    bid = np.asarray(tier.policy.bid[i], dtype=float)
                    ask = np.asarray(tier.policy.ask[i], dtype=float)
                    self.assertTrue(np.all(bid >= prev_bid - 1e-8), msg=f"bid level fell at q={q_grid[i]} gamma={gamma}")
                    self.assertTrue(np.all(ask <= prev_ask + 1e-8), msg=f"ask level rose at q={q_grid[i]} gamma={gamma}")
                    prev_bid, prev_ask = bid, ask

                # Long inventory: bid is inventory-increasing. Adjacent volume
                # premia must not shrink as q increases away from zero.
                prev_gaps = np.diff(-np.asarray(tier.policy.bid[q0], dtype=float))
                for i in range(q0 + 1, len(q_grid)):
                    gaps = np.diff(-np.asarray(tier.policy.bid[i], dtype=float))
                    self.assertTrue(
                        np.all(gaps >= prev_gaps - 1e-8),
                        msg=f"bid gap shrink at q={q_grid[i]} for gamma={gamma}: {gaps} < {prev_gaps}",
                    )
                    prev_gaps = gaps

                # Long inventory: ask is inventory-reducing. Its adjacent
                # volume premia may stay equal or flatten, but may not widen.
                prev_gaps = np.diff(-np.asarray(tier.policy.ask[q0], dtype=float))
                for i in range(q0 + 1, len(q_grid)):
                    gaps = np.diff(-np.asarray(tier.policy.ask[i], dtype=float))
                    self.assertTrue(
                        np.all(gaps <= prev_gaps + 1e-8),
                        msg=f"ask gap widened at q={q_grid[i]} for gamma={gamma}: {gaps} > {prev_gaps}",
                    )
                    prev_gaps = gaps

                # Short inventory: ask is inventory-increasing. Walk outward
                # from zero toward more-negative q and impose the steepening rule.
                prev_gaps = np.diff(-np.asarray(tier.policy.ask[q0], dtype=float))
                for i in range(q0 - 1, -1, -1):
                    gaps = np.diff(-np.asarray(tier.policy.ask[i], dtype=float))
                    self.assertTrue(
                        np.all(gaps >= prev_gaps - 1e-8),
                        msg=f"ask gap shrink at q={q_grid[i]} for gamma={gamma}: {gaps} < {prev_gaps}",
                    )
                    prev_gaps = gaps

                # Short inventory: bid is inventory-reducing. Its adjacent
                # volume premia may stay equal or flatten, but may not widen.
                prev_gaps = np.diff(-np.asarray(tier.policy.bid[q0], dtype=float))
                for i in range(q0 - 1, -1, -1):
                    gaps = np.diff(-np.asarray(tier.policy.bid[i], dtype=float))
                    self.assertTrue(
                        np.all(gaps <= prev_gaps + 1e-8),
                        msg=f"bid gap widened at q={q_grid[i]} for gamma={gamma}: {gaps} > {prev_gaps}",
                    )
                    prev_gaps = gaps

                # Because the q=0 policy is symmetric, the two monotonicity
                # rules imply that the wrong-way ladder is never flatter than
                # the right-way ladder at the same nonzero inventory.
                for i in range(q0 + 1, len(q_grid)):
                    bid_gaps = np.diff(-np.asarray(tier.policy.bid[i], dtype=float))
                    ask_gaps = np.diff(-np.asarray(tier.policy.ask[i], dtype=float))
                    self.assertTrue(np.all(bid_gaps >= ask_gaps - 1e-8))
                for i in range(q0 - 1, -1, -1):
                    bid_gaps = np.diff(-np.asarray(tier.policy.bid[i], dtype=float))
                    ask_gaps = np.diff(-np.asarray(tier.policy.ask[i], dtype=float))
                    self.assertTrue(np.all(ask_gaps >= bid_gaps - 1e-8))

                # Explicitly protect the edge case that motivated the change.
                i19 = int(np.where(np.isclose(q_grid, 19.0))[0][0])
                i20 = int(np.where(np.isclose(q_grid, 20.0))[0][0])
                bid19 = np.diff(-np.asarray(tier.policy.bid[i19], dtype=float))
                bid20 = np.diff(-np.asarray(tier.policy.bid[i20], dtype=float))
                self.assertTrue(np.all(bid20 >= bid19 - 1e-8))

                im19 = int(np.where(np.isclose(q_grid, -19.0))[0][0])
                im20 = int(np.where(np.isclose(q_grid, -20.0))[0][0])
                ask19 = np.diff(-np.asarray(tier.policy.ask[im19], dtype=float))
                ask20 = np.diff(-np.asarray(tier.policy.ask[im20], dtype=float))
                self.assertTrue(np.all(ask20 >= ask19 - 1e-8))

                # Symmetric reducing-side edge checks.
                ask_p19 = np.diff(-np.asarray(tier.policy.ask[i19], dtype=float))
                ask_p20 = np.diff(-np.asarray(tier.policy.ask[i20], dtype=float))
                self.assertTrue(np.all(ask_p20 <= ask_p19 + 1e-8))

                bid_m19 = np.diff(-np.asarray(tier.policy.bid[im19], dtype=float))
                bid_m20 = np.diff(-np.asarray(tier.policy.bid[im20], dtype=float))
                self.assertTrue(np.all(bid_m20 <= bid_m19 + 1e-8))

        # All solved policies have finite positive long-run reward in this calibration.
        self.assertTrue(all(np.isfinite(r) and r > 0.0 for r in rewards))


if __name__ == "__main__":
    unittest.main()


class TestInventoryBoundaryPadding(unittest.TestCase):

    def test_operational_grid_is_padded_by_largest_trade_size(self):
        solution, solver = build_solution(q_grid=UNIFORM_GRID, sizes=SIZES)

        self.assertAlmostEqual(solution.q_grid[0], -20.0)
        self.assertAlmostEqual(solution.q_grid[-1], 20.0)
        self.assertAlmostEqual(solution.solve_q_grid[0], -40.0)
        self.assertAlmostEqual(solution.solve_q_grid[-1], 40.0)
        self.assertAlmostEqual(solution.hard_inventory_limit, 40.0)
        self.assertEqual(len(solution.h_solve), len(solution.solve_q_grid))

        # A 20M bid at the displayed +20 inventory state lands at +40 and is
        # therefore valued on a solved continuation state, not extrapolated.
        self.assertTrue(solver._mdp_transition_admissible(20.0, 20.0, lp.Side.Bid))

        # Once the hidden hard bound would be breached, wrong-way fills are
        # inadmissible. Inventory-reducing fills remain available.
        self.assertFalse(solver._mdp_transition_admissible(21.0, 20.0, lp.Side.Bid))
        self.assertFalse(solver._mdp_transition_admissible(40.0, 1.0, lp.Side.Bid))
        self.assertTrue(solver._mdp_transition_admissible(40.0, 20.0, lp.Side.Ask))

        # The returned quote policy is intentionally only defined on the
        # operational pricing range; public quote lookup does not extrapolate.
        with self.assertRaises(ValueError):
            solution.mdp_tiers[0].quote(21.0, 1.0, lp.Side.Bid)

    def test_fixed_policy_operator_never_needs_extrapolation(self):
        _, solver = build_solution(q_grid=UNIFORM_GRID, sizes=SIZES)
        c, L = solver.fixed_policy_affine_operator()
        self.assertEqual(L.shape, (len(solver.solve_q_grid_), len(solver.solve_q_grid_)))
        self.assertTrue(np.all(np.isfinite(c)))
        self.assertTrue(np.all(np.isfinite(L)))
        # Generator rows sum to zero because every allowed transition is
        # represented inside the solved domain.
        self.assertTrue(np.allclose(L.sum(axis=1), 0.0, atol=1e-12))

class TestMonteCarloPnL(unittest.TestCase):

    def _build_small_solver(self):
        config = lp.SolverConfig(
            q_grid=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            spot=1.1,
            spot_drift=0.0,
            spread=10.0 / 10_000.0,
        )
        tier = lp.MDPTier(
            name="mc",
            sizes=[1.0, 2.0],
            flow_curve=lp.LogisticFlowCurve(
                A0=0.15, theta=0.0, beta=0.0, shift=0.2,
                steepness=8.0, volume_shift=0.01,
            ),
            markout_model=lp.SaturatingMarkoutModel(asymptotic_price_move=0.5 / 10_000.0, tau=0.5),
            delta_min=-5.0,
            delta_max=5.0,
        )
        solver = lp.HJBLadderSolver(
            config=config,
            penalty=lp.QuadraticInventoryPenalty(
                lp.CarryCost(risk_aversion=0.1, sigma=10.0 / 10_000.0)
            ),
            internalization_time=lp.PolynomialInternalizationTime(4.0, 0.07, 0.0084),
            mdp_tiers=[tier],
        )
        solver.solve()
        return solver

    def _build_tier3_regression_solver(self):
        config = lp.SolverConfig(
            q_grid=UNIFORM_GRID, spot=11.5, spot_drift=0.0, spread=20.0 / 10_000.0
        )

        def tier(name, sizes, A0, theta, beta, steepness, shift, volume_shift, impact_pips, tau):
            return lp.MDPTier(
                name=name, sizes=sizes,
                flow_curve=lp.LogisticFlowCurve(
                    A0=A0, theta=theta, beta=beta, shift=shift,
                    steepness=steepness, volume_shift=volume_shift,
                ),
                markout_model=lp.SaturatingMarkoutModel(asymptotic_price_move=impact_pips / 10_000.0, tau=tau),
                delta_min=-10.0, delta_max=100.0, use_markout=True,
            )

        tiers = [
            tier("Tier 1", [1, 2, 3, 5, 10, 20], 0.0155, 0.144, 0.0857, 8.42, 0.52, 0.026, 1.0, 0.5),
            tier("Tier 2", [1, 2, 3, 5, 10, 20], 0.0232, 0.303, 0.122, 2.86, 0.48, 0.020, 1.0, 0.5),
            tier("Tier 3", [1], 1.0, 0.144, 0.0857, 20.0, 0.52, 0.026, 4.0, 0.10),
        ]
        solver = lp.HJBLadderSolver(
            config=config,
            penalty=lp.QuadraticInventoryPenalty(
                lp.CarryCost(risk_aversion=0.1, sigma=20.0 / 10_000.0)
            ),
            internalization_time=lp.PolynomialInternalizationTime(4.0, 0.070, 0.0084),
            mdp_tiers=tiers,
        )
        solver.solve()
        return solver

    def test_closed_form_is_finite(self):
        solver = self._build_small_solver()
        benchmark = solver.closed_form_expected_pnl(10.0, 0.0)
        self.assertTrue(np.isfinite(benchmark.expected_pnl_quote_ccy))

    def test_full_session_closed_form_is_real_and_finite(self):
        solver = self._build_small_solver()
        benchmark = solver.closed_form_expected_pnl(570.0, 0.0)
        self.assertTrue(np.isfinite(benchmark.expected_pnl_quote_ccy))
        self.assertTrue(np.isfinite(benchmark.expected_pnl_rate_quote_ccy_per_min))

    def test_tier3_full_session_uses_impact_aware_closed_form(self):
        solver = self._build_tier3_regression_solver()
        benchmark = solver.closed_form_expected_pnl(570.0, 0.0)
        # Regression for the exact EURSEK settings that previously displayed
        # ~EUR 2.68k from the obsolete per-fill markout approximation. The
        # impact-aware first moment is ~EUR 4.60k.
        self.assertAlmostEqual(
            benchmark.expected_pnl_base_ccy_at_reference_spot, 4603.94, delta=5.0
        )

    def test_tier3_closed_form_terminal_pnl_standard_deviation(self):
        solver = self._build_tier3_regression_solver()
        stats = solver.closed_form_pnl_statistics(570.0, sigma=20.0 / 10_000.0, initial_inventory=0.0)
        # Exact second-moment regression for the EURSEK/Tier-3 setup.
        self.assertAlmostEqual(
            stats.expected_pnl_base_ccy_at_reference_spot, 4603.94, delta=5.0
        )
        self.assertAlmostEqual(
            stats.std_pnl_base_ccy_at_reference_spot, 12451.39, delta=10.0
        )
        self.assertGreater(stats.variance_pnl_quote_ccy, 0.0)

    def test_closed_form_second_moment_matches_monte_carlo_standard_deviation(self):
        solver = self._build_small_solver()
        horizon = 30.0
        sigma = 10.0 / 10_000.0
        stats = solver.closed_form_pnl_statistics(horizon, sigma=sigma, initial_inventory=0.0)
        mean_only = solver.closed_form_expected_pnl(horizon, 0.0)
        self.assertAlmostEqual(
            stats.expected_pnl_quote_ccy, mean_only.expected_pnl_quote_ccy, delta=1e-3
        )
        mc = solver.simulate_pnl(
            horizon, 2000, sigma=sigma, initial_inventory=0.0,
            initial_spot=solver.config.spot, seed=77, sample_paths=0, sample_points=3,
        )
        mc_std = float(np.std(np.asarray(mc.pnl_quote_ccy, dtype=float), ddof=1))
        self.assertLess(
            abs(mc_std - stats.std_pnl_quote_ccy),
            0.08 * stats.std_pnl_quote_ccy,
        )

    def test_closed_form_mean_matches_zero_brownian_monte_carlo_with_markout(self):
        solver = self._build_small_solver()
        horizon = 60.0
        benchmark = solver.closed_form_expected_pnl(horizon, 0.0)
        mc = solver.simulate_pnl(
            horizon, 4000, sigma=0.0, initial_inventory=0.0,
            initial_spot=solver.config.spot, seed=314159, sample_paths=0, sample_points=3,
        )
        pnl = np.asarray(mc.pnl_quote_ccy, dtype=float)
        se = float(np.std(pnl, ddof=1) / np.sqrt(len(pnl)))
        # Closed form is the exact first moment of the MC dynamics. Allow a
        # few Monte Carlo standard errors for sampling noise.
        self.assertLess(abs(float(np.mean(pnl)) - benchmark.expected_pnl_quote_ccy), 5.0 * se + 1e-6)

    def test_monte_carlo_is_seed_reproducible(self):
        solver = self._build_small_solver()
        a = solver.simulate_pnl(5.0, 25, sigma=10.0 / 10_000.0, seed=99, sample_paths=2, sample_points=10)
        b = solver.simulate_pnl(5.0, 25, sigma=10.0 / 10_000.0, seed=99, sample_paths=2, sample_points=10)
        self.assertEqual(a.pnl_quote_ccy, b.pnl_quote_ccy)
        self.assertEqual(a.pnl_base_ccy, b.pnl_base_ccy)
        self.assertEqual(a.final_inventory, b.final_inventory)
        self.assertEqual(len(a.sample_times), 2)
        self.assertEqual(len(a.sample_times[0]), 10)


    def test_monte_carlo_base_currency_pnl_uses_final_spot(self):
        solver = self._build_small_solver()
        result = solver.simulate_pnl(8.0, 20, sigma=10.0 / 10_000.0, seed=123, sample_paths=1, sample_points=9)
        self.assertEqual(len(result.pnl_base_ccy), len(result.pnl_quote_ccy))
        for pnl_quote, pnl_base, final_spot in zip(result.pnl_quote_ccy, result.pnl_base_ccy, result.final_spot):
            self.assertGreater(final_spot, 0.0)
            self.assertAlmostEqual(pnl_base, pnl_quote / final_spot, delta=1e-9)

    def test_closed_form_base_currency_benchmark_uses_reference_spot(self):
        solver = self._build_small_solver()
        benchmark = solver.closed_form_expected_pnl(30.0, 0.0)
        self.assertGreater(benchmark.reference_spot, 0.0)
        self.assertAlmostEqual(
            benchmark.expected_pnl_base_ccy_at_reference_spot,
            benchmark.expected_pnl_quote_ccy / benchmark.reference_spot,
            delta=1e-9,
        )

    def test_inventory_confidence_band_uses_all_paths(self):
        solver = self._build_small_solver()
        result = solver.simulate_pnl(8.0, 100, sigma=10.0 / 10_000.0, seed=17, sample_paths=2, sample_points=17)
        self.assertEqual(len(result.inventory_sample_times), 17)
        self.assertEqual(len(result.inventory_ci_lower), 17)
        self.assertEqual(len(result.inventory_median), 17)
        self.assertEqual(len(result.inventory_ci_upper), 17)
        lower = np.asarray(result.inventory_ci_lower)
        median = np.asarray(result.inventory_median)
        upper = np.asarray(result.inventory_ci_upper)
        self.assertTrue(np.all(lower <= median + 1e-12))
        self.assertTrue(np.all(median <= upper + 1e-12))
        self.assertAlmostEqual(lower[0], 0.0, delta=1e-12)
        self.assertAlmostEqual(median[0], 0.0, delta=1e-12)
        self.assertAlmostEqual(upper[0], 0.0, delta=1e-12)
        # Only two detailed paths are retained, so the percentile arrays cannot
        # simply be an envelope over the path-explorer subset.
        self.assertEqual(len(result.sample_paths), 2)

    def test_detailed_sample_path_records_inventory_quotes_and_fill_tape(self):
        solver = self._build_small_solver()
        result = solver.simulate_pnl(8.0, 10, sigma=10.0 / 10_000.0, seed=7, sample_paths=1, sample_points=17)
        self.assertEqual(len(result.sample_paths), 1)
        path = result.sample_paths[0]
        self.assertEqual(len(path.times), 17)
        self.assertEqual(len(path.spots), 17)
        self.assertEqual(len(path.inventories), 17)
        self.assertEqual(len(path.quote_series), 4)  # 1 tier x 2 sides x 2 sizes
        self.assertTrue(all(len(series.prices) == 17 for series in path.quote_series))
        self.assertEqual(path.inventory_event_times[0], 0.0)
        self.assertEqual(path.inventory_event_values[0], 0.0)
        self.assertEqual(len(path.inventory_event_times), len(path.fills) + 1)
        self.assertEqual(len(path.inventory_event_values), len(path.fills) + 1)
        for fill in path.fills:
            self.assertTrue(0.0 <= fill.time_minutes <= 8.0)
            self.assertIn(fill.side, {"bid", "ask"})
            self.assertTrue(np.isfinite(fill.execution_price))
            self.assertAlmostEqual(
                fill.inventory_after - fill.inventory_before,
                fill.size if fill.side == "bid" else -fill.size,
                delta=1e-12,
            )


def test_pnl_benchmark_version_is_exported():
    assert hasattr(lp, "PNL_BENCHMARK_VERSION")
    assert lp.PNL_BENCHMARK_VERSION == "impact-aware-v3-second-moment"
