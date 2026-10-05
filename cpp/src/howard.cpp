#include "ladder_pricer/howard.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ladder_pricer {
namespace {

constexpr double kTolerance = 1e-10;

double direction(Side side) noexcept { return side == Side::Bid ? 1.0 : -1.0; }

const std::vector<double>& row(const LadderPolicy& policy, Side side, std::size_t i) {
    return side == Side::Bid ? policy.bid[i] : policy.ask[i];
}

struct Weights {
    std::size_t left;
    std::size_t right;
    double wl;
    double wr;
};

Weights interpolation_weights(const std::vector<double>& grid, double q) {
    if (q <= grid.front() + kTolerance) return {0, 0, 1.0, 0.0};
    if (q >= grid.back() - kTolerance) {
        const auto i = grid.size() - 1;
        return {i, i, 1.0, 0.0};
    }
    auto upper = std::lower_bound(grid.begin(), grid.end(), q);
    const auto right = static_cast<std::size_t>(std::distance(grid.begin(), upper));
    if (std::abs(grid[right] - q) <= kTolerance) return {right, right, 1.0, 0.0};
    const auto left = right - 1;
    const double width = grid[right] - grid[left];
    const double wr = (q - grid[left]) / width;
    return {left, right, 1.0 - wr, wr};
}

void add_transition(DenseMatrix& L, const std::vector<double>& grid, std::size_t from,
                    double target, double rate) {
    if (rate == 0.0) return;
    const auto w = interpolation_weights(grid, target);
    L(from, w.left) += rate * w.wl;
    L(from, w.right) += rate * w.wr;
    L(from, from) -= rate;
}

std::vector<double> fill_breakpoints(const std::vector<double>& grid, std::size_t i,
                                     double dir, double posted_size) {
    std::vector<double> out;
    const double q = grid[i];
    if (dir > 0.0) {
        for (std::size_t j = i + 1; j < grid.size(); ++j) {
            const double x = grid[j] - q;
            if (x >= posted_size - kTolerance) break;
            if (x > kTolerance) out.push_back(x);
        }
    } else {
        for (std::size_t j = i; j-- > 0;) {
            const double x = q - grid[j];
            if (x >= posted_size - kTolerance) break;
            if (x > kTolerance) out.push_back(x);
        }
    }
    return out;
}

double max_abs(const std::vector<double>& x) {
    double out = 0.0;
    for (double v : x) out = std::max(out, std::abs(v));
    return out;
}

double max_abs_policy(const Policy& policy) {
    double out = 1.0;
    for (const auto& tier : policy.tiers) {
        for (const auto& r : tier.bid) out = std::max(out, max_abs(r));
        for (const auto& r : tier.ask) out = std::max(out, max_abs(r));
    }
    if (policy.dark_pool) {
        out = std::max(out, max_abs(policy.dark_pool->bid_size));
        out = std::max(out, max_abs(policy.dark_pool->ask_size));
    }
    if (policy.passive_ecn) {
        out = std::max(out, max_abs(policy.passive_ecn->bid_delta));
        out = std::max(out, max_abs(policy.passive_ecn->ask_delta));
    }
    return out;
}

std::vector<double> slice_values(const std::vector<double>& values,
                                 const std::vector<std::size_t>& indices) {
    std::vector<double> out;
    out.reserve(indices.size());
    for (auto i : indices) out.push_back(values[i]);
    return out;
}

LadderPolicy slice_policy(const LadderPolicy& policy,
                          const std::vector<std::size_t>& indices,
                          const std::vector<double>& q_grid) {
    LadderPolicy out;
    out.q_grid = q_grid;
    out.sizes = policy.sizes;
    out.bid.reserve(indices.size());
    out.ask.reserve(indices.size());
    for (auto i : indices) {
        out.bid.push_back(policy.bid[i]);
        out.ask.push_back(policy.ask[i]);
    }
    return out;
}

DarkPoolPolicy slice_policy(const DarkPoolPolicy& policy,
                            const std::vector<std::size_t>& indices,
                            const std::vector<double>& q_grid) {
    DarkPoolPolicy out;
    out.q_grid = q_grid;
    for (auto i : indices) {
        out.bid_size.push_back(policy.bid_size[i]);
        out.ask_size.push_back(policy.ask_size[i]);
        out.bid_active.push_back(policy.bid_active[i]);
        out.ask_active.push_back(policy.ask_active[i]);
    }
    return out;
}

PassiveECNPolicy slice_policy(const PassiveECNPolicy& policy,
                              const std::vector<std::size_t>& indices,
                              const std::vector<double>& q_grid) {
    PassiveECNPolicy out;
    out.q_grid = q_grid;
    for (auto i : indices) {
        out.bid_delta.push_back(policy.bid_delta[i]);
        out.ask_delta.push_back(policy.ask_delta[i]);
        out.bid_active.push_back(policy.bid_active[i]);
        out.ask_active.push_back(policy.ask_active[i]);
    }
    return out;
}

}  // namespace

HowardSolver::AffineOperator HowardSolver::fixed_policy_operator(const Policy& policy) const {
    const auto& grid = problem_.grid();
    const auto& states = grid.states();
    const std::size_t n = states.size();
    AffineOperator op{std::vector<double>(n, 0.0), DenseMatrix(n, n)};

    for (std::size_t i = 0; i < n; ++i) {
        const double q = states[i];
        op.reward[i] = -problem_.penalty().value(q) + problem_.spot_drift() * q;

        for (std::size_t k = 0; k < problem_.tiers().size(); ++k) {
            const auto& tier = problem_.tiers()[k];
            const auto& p = policy.tiers[k];
            for (Side side : {Side::Bid, Side::Ask}) {
                const double dir = direction(side);
                const auto& deltas = row(p, side, i);
                for (std::size_t j = 0; j < tier.sizes().size(); ++j) {
                    const double z = tier.sizes()[j];
                    const double dq = dir * z;
                    if (!grid.admissible(q, dq)) continue;
                    const double delta = deltas[j];
                    const double rate = tier.flow().arrival_rate(delta, z);
                    const double q_next = q + dq;
                    const double markout = tier.use_markout()
                        ? tier.markout().expected(z, problem_.internalization_time().value(q_next))
                        : 0.0;
                    op.reward[i] += rate * (z * (problem_.spread() * (0.5 - delta) - tier.fee()) - z * markout);
                    add_transition(op.generator, states, i, q_next, rate);
                }
            }
        }

        if (problem_.dark_pool() && policy.dark_pool) {
            const auto& venue = *problem_.dark_pool();
            const auto& p = *policy.dark_pool;
            for (Side side : {Side::Bid, Side::Ask}) {
                const bool active = side == Side::Bid ? p.bid_active[i] : p.ask_active[i];
                const double posted = side == Side::Bid ? p.bid_size[i] : p.ask_size[i];
                if (!active || posted <= 0.0 || !venue.risk_reducing(q, side, posted)) continue;
                const int u = static_cast<int>(std::llround(posted));
                const double dir = direction(side);
                for (const auto& [fill, rate] : venue.arrivals(side).fill_rates(u)) {
                    op.reward[i] -= rate * venue.fee(side) * fill;
                    add_transition(op.generator, states, i, q + dir * fill, rate);
                }
            }
        }

        if (problem_.passive_ecn() && policy.passive_ecn) {
            const auto& venue = *problem_.passive_ecn();
            const auto& p = *policy.passive_ecn;
            const double z = venue.quote_size();
            for (Side side : {Side::Bid, Side::Ask}) {
                const bool active = side == Side::Bid ? p.bid_active[i] : p.ask_active[i];
                const double delta = side == Side::Bid ? p.bid_delta[i] : p.ask_delta[i];
                if (!active) continue;
                const double dir = direction(side);
                const auto breaks = fill_breakpoints(states, i, dir, z);
                for (const auto& source : venue.flow().sources()) {
                    const double hit_rate = source.arrival_rate(delta);
                    if (hit_rate <= 0.0) continue;
                    for (const auto& [fill, probability] : source.fill_components(z, breaks)) {
                        const double rate = hit_rate * probability;
                        if (rate <= 0.0) continue;
                        op.reward[i] += rate * fill *
                            (problem_.spread() * (0.5 - delta) - venue.maker_fee());
                        add_transition(op.generator, states, i, q + dir * fill, rate);
                    }
                }
            }
        }
    }
    return op;
}

std::pair<std::vector<double>, double> HowardSolver::evaluate_policy(const Policy& policy) const {
    const auto op = fixed_policy_operator(policy);
    const std::size_t n = op.reward.size();
    DenseMatrix A(n + 1, n + 1);
    std::vector<double> b(n + 1, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) A(i, j) = op.generator(i, j);
        A(i, n) = -1.0;
        b[i] = -op.reward[i];
    }
    A(n, problem_.grid().zero_index()) = 1.0;

    const auto x = A.solve(b);
    std::vector<double> h(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n));
    return {std::move(h), x[n]};
}

double HowardSolver::policy_change(const Policy& lhs, const Policy& rhs) {
    double out = 0.0;
    for (std::size_t k = 0; k < lhs.tiers.size(); ++k) {
        for (std::size_t i = 0; i < lhs.tiers[k].bid.size(); ++i) {
            for (std::size_t j = 0; j < lhs.tiers[k].bid[i].size(); ++j) {
                out = std::max(out, std::abs(lhs.tiers[k].bid[i][j] - rhs.tiers[k].bid[i][j]));
                out = std::max(out, std::abs(lhs.tiers[k].ask[i][j] - rhs.tiers[k].ask[i][j]));
            }
        }
    }
    if (lhs.dark_pool && rhs.dark_pool) {
        for (std::size_t i = 0; i < lhs.dark_pool->bid_size.size(); ++i) {
            out = std::max(out, std::abs(lhs.dark_pool->bid_size[i] - rhs.dark_pool->bid_size[i]));
            out = std::max(out, std::abs(lhs.dark_pool->ask_size[i] - rhs.dark_pool->ask_size[i]));
            if (lhs.dark_pool->bid_active[i] != rhs.dark_pool->bid_active[i] ||
                lhs.dark_pool->ask_active[i] != rhs.dark_pool->ask_active[i]) out = std::max(out, 1.0);
        }
    }
    if (lhs.passive_ecn && rhs.passive_ecn) {
        for (std::size_t i = 0; i < lhs.passive_ecn->bid_delta.size(); ++i) {
            out = std::max(out, std::abs(lhs.passive_ecn->bid_delta[i] - rhs.passive_ecn->bid_delta[i]));
            out = std::max(out, std::abs(lhs.passive_ecn->ask_delta[i] - rhs.passive_ecn->ask_delta[i]));
            if (lhs.passive_ecn->bid_active[i] != rhs.passive_ecn->bid_active[i] ||
                lhs.passive_ecn->ask_active[i] != rhs.passive_ecn->ask_active[i]) out = std::max(out, 1.0);
        }
    }
    return out;
}

double HowardSolver::policy_scale(const Policy& policy) { return max_abs_policy(policy); }

Solution HowardSolver::solve() const {
    const auto& grid = problem_.grid();
    const auto& states = grid.states();
    const auto& op_indices = grid.operational_indices();
    const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());

    PolicyBuilder builder(problem_);
    BellmanModel bellman(problem_);
    Policy policy = builder.initial_policy();
    std::vector<double> h(states.size(), 0.0);
    double rho = 0.0;

    SolverDiagnostics diagnostics;
    constexpr int kMaximumIterations = 256;
    for (int iteration = 1; iteration <= kMaximumIterations; ++iteration) {
        const auto h_previous = h;
        std::tie(h, rho) = evaluate_policy(policy);
        const Policy improved = builder.improve(h, policy);
        const double p_change = policy_change(improved, policy);
        policy = improved;

        const auto rhs = bellman.rhs(h, policy);
        double residual = 0.0;
        for (double x : rhs) residual = std::max(residual, std::abs(x - rho));
        double h_change = 0.0;
        for (std::size_t i = 0; i < h.size(); ++i) h_change = std::max(h_change, std::abs(h[i] - h_previous[i]));
        const double scale = std::max({1.0, std::abs(rho), max_abs(rhs)});
        const double residual_tol = sqrt_eps * scale;
        const double policy_tol = sqrt_eps * policy_scale(policy);

        diagnostics.iterations = iteration;
        diagnostics.average_reward = rho;
        diagnostics.reward_lower_bound = rho - residual;
        diagnostics.reward_upper_bound = rho + residual;
        diagnostics.value_change.push_back(h_change);
        diagnostics.bellman_residual.push_back(residual);

        if (residual <= residual_tol && p_change <= policy_tol) {
            diagnostics.converged = true;
            break;
        }
    }

    policy = builder.snap_operational_constraints(policy);
    const auto final_rhs = bellman.rhs(h, policy);
    double residual = 0.0;
    for (double x : final_rhs) residual = std::max(residual, std::abs(x - rho));
    diagnostics.reward_lower_bound = rho - residual;
    diagnostics.reward_upper_bound = rho + residual;

    Solution solution;
    solution.q_grid = grid.operational_states();
    solution.value = slice_values(h, op_indices);
    solution.solve_q_grid = states;
    solution.value_solve = h;
    solution.average_reward = rho;
    solution.hard_inventory_limit = grid.hard_limit();
    solution.diagnostics = diagnostics;
    solution.solve_tier_policies = policy.tiers;
    solution.tier_policies.reserve(policy.tiers.size());
    for (const auto& p : policy.tiers) {
        solution.tier_policies.push_back(slice_policy(p, op_indices, grid.operational_states()));
    }
    if (policy.dark_pool) {
        solution.solve_dark_pool_policy = *policy.dark_pool;
        solution.dark_pool_policy = slice_policy(*policy.dark_pool, op_indices, grid.operational_states());
    }
    if (policy.passive_ecn) {
        solution.solve_passive_ecn_policy = *policy.passive_ecn;
        solution.passive_ecn_policy = slice_policy(*policy.passive_ecn, op_indices, grid.operational_states());
    }
    return solution;
}

}  // namespace ladder_pricer
