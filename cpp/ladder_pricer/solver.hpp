#pragma once

#include "common.hpp"
#include "dark_pool.hpp"
#include "penalty.hpp"
#include "policy.hpp"
#include "solver_config.hpp"
#include "tiers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace hjb {

struct HJBSolution {
    std::vector<double> h;
    std::vector<double> q_grid;
    std::vector<MDPTier> mdp_tiers;
    std::optional<DarkPoolVenue> dark_pool;
    SolverDiagnostics diagnostics;
};

struct HJBLadderSolver {
    SolverConfig config;
    PolynomialInventoryPenalty penalty;

    std::vector<MDPTier> mdp_tiers;
    std::optional<DarkPoolVenue> dark_pool;

    GoldenSectionSearch optimizer;
    SolverGridMeta grid_meta_;

    HJBLadderSolver(
        SolverConfig config_,
        PolynomialInventoryPenalty penalty_,
        std::vector<MDPTier> mdp_tiers_ = {},
        std::optional<DarkPoolVenue> dark_pool_ = std::nullopt
    )
        : config(std::move(config_)),
          penalty(std::move(penalty_)),
          mdp_tiers(std::move(mdp_tiers_)),
          dark_pool(std::move(dark_pool_)),
          optimizer(config.golden_tol, config.golden_max_iter) {
        config.validate();
    }

    static void repair_or_throw_bounds(
        double& lower,
        double& upper,
        const std::string& tier_name,
        const char* side_name
    ) {
        // Accumulated bound arithmetic can leave lower > upper by a tiny floating-point
        // residual even when the interval is mathematically a point — snap it rather than throw.
        constexpr double tol = 1e-12;

        if (upper < lower) {
            if (lower - upper > tol) {
                throw std::runtime_error(
                    "Infeasible ladder bounds for MDP tier '" + tier_name +
                    "', side '" + std::string(side_name) + "'."
                );
            }
            upper = lower;
        }
    }

    void validate_problem_definition() const {
        config.validate();

        for (const auto& tier : mdp_tiers) {
            tier.validate();
        }
        if (dark_pool.has_value()) {
            dark_pool->validate();
        }
    }

    void initialize_policy_shapes() {
        for (auto& tier : mdp_tiers) {
            tier.reset_policy_shape(config.q_grid);
        }
        if (dark_pool.has_value()) {
            dark_pool->reset_policy_shape(config.q_grid);
        }
    }

    void prepare_solve_context() {
        validate_problem_definition();
        grid_meta_ = build_solver_grid_meta(config.q_grid);
        initialize_policy_shapes();
    }

    // --------------------------------------------------------
    // MDP fill payoff
    // --------------------------------------------------------

    double mdp_fill_payoff(double lam, double z, double d, double mu, double dh) const {
        return lam * (z * config.spread * (0.5 - d) + z * mu + dh);
    }

    // --------------------------------------------------------
    // MDP policy construction
    // --------------------------------------------------------

    std::pair<double, double> mdp_bounds_for_rung(
        const MDPTier& tier,
        std::size_t rung_idx,
        const std::vector<double>& current_delta,
        double q,
        Side side,
        OptionalDeltaRow prev_delta
    ) const {
        double lower = tier.delta_min;
        double upper = tier.delta_max;

        // shrink: inventory is on the side that needs to be reduced — quotes tighten
        //         (delta rises) as we move further from zero.
        // expand: inventory is on the opposite side — quotes widen (delta falls).
        const bool shrink = (side == Side::Bid && q < 0.0) || (side == Side::Ask && q > 0.0);
        const bool expand = (side == Side::Bid && q > 0.0) || (side == Side::Ask && q < 0.0);

        // Intra-row monotonicity: larger sizes always quote at least as wide.
        if (rung_idx > 0) {
            upper = std::min(upper, current_delta[rung_idx - 1]);
        }

        if (!prev_delta.has_value()) {
            // Case 1: no previous inventory row — only intra-row monotonicity applies.

        } else if (rung_idx == 0) {
            // Case 2: first rung of a new row — anchor delta[0] relative to prev_row[0].
            const auto& prev_row = prev_delta->get();
            if (shrink)      lower = std::max(lower, prev_row[0]);
            else if (expand) upper = std::min(upper, prev_row[0]);

        } else {
            // Case 3: inner rung with a previous row — preserve ladder gap shape.
            const auto& prev_row = prev_delta->get();

            if (expand) {
                const double required_tail_span = prev_row[rung_idx] - prev_row.back();
                lower = std::max(lower, tier.delta_min + required_tail_span);
            }

            // Preserve the gap between consecutive rungs as seen in the previous inventory row:
            // current[j] must stay within current[j-1] ± prev_gap so the ladder shape evolves
            // smoothly rather than collapsing or inverting across inventory steps.
            const double prev_curr    = current_delta[rung_idx - 1];
            const double prev_row_gap = prev_row[rung_idx - 1] - prev_row[rung_idx];

            if (shrink)      lower = std::max(lower, prev_curr - prev_row_gap);
            else if (expand) upper = std::min(upper, prev_curr - prev_row_gap);

            repair_or_throw_bounds(lower, upper, tier.name, side == Side::Bid ? "bid" : "ask");
        }

        return {lower, upper};
    }

    void build_mdp_ladder_bid(
        const MDPTier& tier,
        const LinearInterpolator1D& h,
        std::vector<double>& delta_row,
        double q,
        OptionalDeltaRow prev_delta = std::nullopt
    ) const {
        const std::size_t nz = tier.sizes_.size();
        delta_row.assign(nz, tier.delta_min);

        for (std::size_t j = 0; j < nz; ++j) {
            const double z = tier.sizes_[j];
            auto [lower, upper] = mdp_bounds_for_rung(tier, j, delta_row, q, Side::Bid, prev_delta);
            const double tau = penalty.internalization_time.value(q + z);
            const double mu = expected_markout(tier.markout_model, z, tau);
            const auto obj = [&](double d) {
                return mdp_fill_payoff(tier.flow_curve.arrival_rate(d, z), z, d, mu, h(q + z) - h(q));
            };

            delta_row[j] = clamp(optimizer.maximize(obj, lower, upper), lower, upper);
        }
    }

    void build_mdp_ladder_ask(
        const MDPTier& tier,
        const LinearInterpolator1D& h,
        std::vector<double>& delta_row,
        double q,
        OptionalDeltaRow prev_delta = std::nullopt
    ) const {
        const std::size_t nz = tier.sizes_.size();
        delta_row.assign(nz, tier.delta_min);

        for (std::size_t j = 0; j < nz; ++j) {
            const double z = tier.sizes_[j];
            auto [lower, upper] = mdp_bounds_for_rung(tier, j, delta_row, q, Side::Ask, prev_delta);
            const double tau = penalty.internalization_time.value(q - z);
            const double mu = expected_markout(tier.markout_model, z, tau);
            const auto obj = [&](double d) {
                return mdp_fill_payoff(tier.flow_curve.arrival_rate(d, z), z, d, mu, h(q - z) - h(q));
            };

            delta_row[j] = clamp(optimizer.maximize(obj, lower, upper), lower, upper);
        }
    }

    void build_mdp_bid_policy(MDPTier& tier, const LinearInterpolator1D& h) const {
        const auto& q_grid = config.q_grid;
        Matrix& mat = tier.policy.matrix_mut(Side::Bid);
        const std::size_t nq = grid_meta_.nq;
        const std::size_t q0_idx = grid_meta_.q0_idx;
        
        // solve 0 inventory case
        build_mdp_ladder_bid(tier, h, mat[q0_idx], q_grid[q0_idx]);
        
        // solve positive inventories
        for (std::size_t i = q0_idx + 1; i < nq; ++i) {
            build_mdp_ladder_bid(tier, h, mat[i], q_grid[i], std::cref(mat[i - 1]));
        }

        // solve negative inventories
        for (std::size_t i = q0_idx; i-- > 0;) {
            build_mdp_ladder_bid(tier, h, mat[i], q_grid[i], std::cref(mat[i + 1]));
        }
    }

    void build_mdp_ask_policy(MDPTier& tier, const LinearInterpolator1D& h) const {
        const auto& q_grid = config.q_grid;
        Matrix& mat = tier.policy.matrix_mut(Side::Ask);
        const std::size_t nq = grid_meta_.nq;
        const std::size_t q0_idx = grid_meta_.q0_idx;

        build_mdp_ladder_ask(tier, h, mat[q0_idx], q_grid[q0_idx]);
        for (std::size_t i = q0_idx + 1; i < nq; ++i) {
            build_mdp_ladder_ask(tier, h, mat[i], q_grid[i], std::cref(mat[i - 1]));
        }
        for (std::size_t i = q0_idx; i-- > 0;) {
            build_mdp_ladder_ask(tier, h, mat[i], q_grid[i], std::cref(mat[i + 1]));
        }
    }

    // --------------------------------------------------------
    // Dark-pool geometric fill value
    // --------------------------------------------------------

    // Expected value of posting size u at inventory q in the dark pool.
    // direction = +1.0 for bid (inventory increases), -1.0 for ask (inventory decreases).
    // Arrival sizes are geometric: P(fill = k) = p*(1-p)^{k-1} for k < u,
    // with the final term absorbing the tail P(fill >= u) = (1-p)^{u-1} — the posted size
    // acts as a cap so the last term has no leading p factor.
    double dark_pool_fill_value(
        const LinearInterpolator1D& h,
        double q,
        double hq,
        int u,
        double lambda,
        double p,
        double fee,
        double direction
    ) const {
        const double r = 1.0 - p;
        double expected = 0.0;
        for (int k = 1; k < u; ++k) {
            const double kd = static_cast<double>(k);
            expected += p * std::pow(r, kd - 1.0) * (h(q + direction * kd) - hq - fee * kd);
        }
        const double ud = static_cast<double>(u);
        expected += std::pow(r, ud - 1.0) * (h(q + direction * ud) - hq - fee * ud);
        return lambda * expected;
    }

    // --------------------------------------------------------
    // Dark-pool policy construction
    // --------------------------------------------------------

    void build_dark_pool_bid_policy(DarkPoolVenue& venue, const LinearInterpolator1D& h) const {
        const double threshold = venue.min_fill_value;
        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            const double q = config.q_grid[i];
            if (!venue.is_admissible(q, Side::Bid)) {
                venue.policy.set_posted_size(i, Side::Bid, 0.0, false);
                continue;
            }
            const double hq = h(q);
            double best_size = 0.0;
            double best_value = std::numeric_limits<double>::lowest();
            for (double u_raw : venue.posted_sizes) {
                if (venue.lambda_bid <= 0.0 || u_raw <= 0.0) continue;
                const double value = dark_pool_fill_value(
                    h, q, hq, static_cast<int>(std::llround(u_raw)),
                    venue.lambda_bid, venue.p_bid, venue.fee_per_unit_bid, +1.0
                );
                if (value > best_value) { best_value = value; best_size = u_raw; }
            }
            const bool active = best_value > threshold;
            venue.policy.set_posted_size(i, Side::Bid, active ? best_size : 0.0, active);
        }
    }

    void build_dark_pool_ask_policy(DarkPoolVenue& venue, const LinearInterpolator1D& h) const {
        const double threshold = venue.min_fill_value;
        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            const double q = config.q_grid[i];
            if (!venue.is_admissible(q, Side::Ask)) {
                venue.policy.set_posted_size(i, Side::Ask, 0.0, false);
                continue;
            }
            const double hq = h(q);
            double best_size = 0.0;
            double best_value = std::numeric_limits<double>::lowest();
            for (double u_raw : venue.posted_sizes) {
                if (venue.lambda_ask <= 0.0 || u_raw <= 0.0) continue;
                const double value = dark_pool_fill_value(
                    h, q, hq, static_cast<int>(std::llround(u_raw)),
                    venue.lambda_ask, venue.p_ask, venue.fee_per_unit_ask, -1.0
                );
                if (value > best_value) { best_value = value; best_size = u_raw; }
            }
            const bool active = best_value > threshold;
            venue.policy.set_posted_size(i, Side::Ask, active ? best_size : 0.0, active);
        }
    }

    void build_dark_pool_policy(DarkPoolVenue& venue, const LinearInterpolator1D& h) const {
        build_dark_pool_bid_policy(venue, h);
        build_dark_pool_ask_policy(venue, h);
    }

    // --------------------------------------------------------
    // Bellman contributions
    // --------------------------------------------------------

    double mdp_bellman_contribution_side(
        const MDPTier& tier,
        double q,
        std::size_t q_index,
        const LinearInterpolator1D& h,
        Side side
    ) const {
        double total = 0.0;
        const auto& row = tier.policy.matrix(side)[q_index];
        const std::size_t nz = tier.sizes_.size();
        const double hq = h(q);
        const double direction = side == Side::Bid ? +1.0 : -1.0;

        for (std::size_t j = 0; j < nz; ++j) {
            const double z = tier.sizes_[j];
            const double d = row[j];
            total += mdp_fill_payoff(
                tier.flow_curve.arrival_rate(d, z),
                z, d,
                expected_markout(tier.markout_model, z, penalty.internalization_time.value(q + direction * z)),
                h(q + direction * z) - hq
            );
        }
        return total;
    }

    double dark_pool_bellman_contribution(
        const DarkPoolVenue& venue,
        double q,
        std::size_t q_index,
        const LinearInterpolator1D& h
    ) const {
        const double hq = h(q);
        double total = 0.0;

        if (venue.policy.is_active(q_index, Side::Bid) && venue.lambda_bid > 0.0) {
            const double u_raw = venue.policy.bid_size[q_index];
            if (u_raw > 0.0) {
                total += dark_pool_fill_value(
                    h, q, hq, static_cast<int>(std::llround(u_raw)),
                    venue.lambda_bid, venue.p_bid, venue.fee_per_unit_bid, +1.0
                );
            }
        }

        if (venue.policy.is_active(q_index, Side::Ask) && venue.lambda_ask > 0.0) {
            const double u_raw = venue.policy.ask_size[q_index];
            if (u_raw > 0.0) {
                total += dark_pool_fill_value(
                    h, q, hq, static_cast<int>(std::llround(u_raw)),
                    venue.lambda_ask, venue.p_ask, venue.fee_per_unit_ask, -1.0
                );
            }
        }

        return total;
    }

    // --------------------------------------------------------
    // Public-ish internal solver steps
    // --------------------------------------------------------

    void euler_step(
        const std::vector<double>& h,
        const std::vector<double>& rhs,
        std::vector<double>& h_next
    ) const {
        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            h_next[i] = h[i] + config.dt * rhs[i];
        }
    }

    // Pins the ergodic free constant by subtracting h(0), then returns the max change vs h_prev.
    // Without this, h drifts to ±∞ since the HJB solution is unique only up to an additive constant.
    double normalize_and_max_change(
        const std::vector<double>& h_prev,
        std::vector<double>& h_next
    ) const {
        const double h0 = h_next[grid_meta_.q0_idx];
        double max_change = 0.0;
        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            h_next[i] -= h0;
            max_change = std::max(max_change, std::abs(h_next[i] - h_prev[i]));
        }
        return max_change;
    }

    void update_policies(const LinearInterpolator1D& h) {
        for (auto& tier : mdp_tiers) {
            build_mdp_bid_policy(tier, h);
            build_mdp_ask_policy(tier, h);
        }
        if (dark_pool.has_value()) {
            build_dark_pool_policy(*dark_pool, h);
        }
    }

    double bellman_rhs(const LinearInterpolator1D& h, std::vector<double>& rhs) const {
        double max_rhs_abs = 0.0;

        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            const double q = config.q_grid[i];
            double val = -penalty.value(q) + config.spot_drift * q;

            for (const auto& tier : mdp_tiers) {
                val += mdp_bellman_contribution_side(tier, q, i, h, Side::Bid);
                val += mdp_bellman_contribution_side(tier, q, i, h, Side::Ask);
            }
            if (dark_pool.has_value()) {
                val += dark_pool_bellman_contribution(*dark_pool, q, i, h);
            }

            rhs[i] = val;
            max_rhs_abs = std::max(max_rhs_abs, std::abs(val));
        }

        return max_rhs_abs;
    }

    std::vector<double> bellman_rhs_from_policies(const std::vector<double>& h_vec) {
        if (h_vec.size() != config.q_grid.size()) {
            throw std::invalid_argument(
                "bellman_rhs_from_policies: h_vec size must match config.q_grid size."
            );
        }

        prepare_solve_context();

        const LinearInterpolator1D h{config.q_grid, h_vec};
        update_policies(h);

        std::vector<double> rhs(config.q_grid.size(), 0.0);
        bellman_rhs(h, rhs);
        return rhs;
    }

    HJBSolution solve() {
        prepare_solve_context();

        std::vector<double> h(grid_meta_.nq, 0.0);
        std::vector<double> h_next(grid_meta_.nq, 0.0);
        std::vector<double> rhs(grid_meta_.nq, 0.0);

        // h_view holds a reference to h; swap() below replaces h's contents in-place,
        // so h_view always sees the current iterate without being re-created each step.
        const LinearInterpolator1D h_view{config.q_grid, h};
        SolverDiagnostics diagnostics(config.n_iter);

        for (int it = 1; it <= config.n_iter; ++it) {
            update_policies(h_view);
            const double max_rhs_now = bellman_rhs(h_view, rhs);

            euler_step(h, rhs, h_next);
            const double max_h_change = normalize_and_max_change(h, h_next);

            diagnostics.record_iteration(it, max_h_change, max_rhs_now);

            const bool passes_now =
                it >= config.min_iter &&
                max_h_change <= config.tol_h &&
                max_rhs_now <= config.tol_rhs;

            h.swap(h_next);

            if (diagnostics.update_stopping_state(
                    passes_now,
                    config.early_stop,
                    config.consecutive_passes_required
                )) {
                break;
            }
        }

        // Rebuild policies on the final h; the last swap advanced h past the last policy build.
        update_policies(h_view);

        HJBSolution out;
        out.h = std::move(h);
        out.q_grid = config.q_grid;
        out.mdp_tiers = mdp_tiers;
        out.dark_pool = dark_pool;
        out.diagnostics = std::move(diagnostics);
        return out;
    }
};

} // namespace hjb
