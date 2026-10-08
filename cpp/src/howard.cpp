#include "ladder_pricer/howard.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ladder_pricer {
namespace {

constexpr double kTolerance = 1e-10;

double direction(Side side) noexcept { return side == Side::Bid ? 1.0 : -1.0; }

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

std::size_t joint_index(std::size_t vol, std::size_t qi, std::size_t nq) {
    return vol * nq + qi;
}

void add_inventory_transition(DenseMatrix& L, const std::vector<double>& grid,
                              std::size_t vol, std::size_t qi,
                              double target, double rate) {
    if (rate == 0.0) return;
    const auto w = interpolation_weights(grid, target);
    const std::size_t nq = grid.size();
    const std::size_t from = joint_index(vol, qi, nq);
    L(from, joint_index(vol, w.left, nq)) += rate * w.wl;
    L(from, joint_index(vol, w.right, nq)) += rate * w.wr;
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

const std::vector<double>& row(const LadderPolicy& policy, Side side, std::size_t i) {
    return side == Side::Bid ? policy.bid[i] : policy.ask[i];
}

double interpolate_joint(const std::vector<double>& values,
                         const std::vector<double>& grid,
                         std::size_t vol, double q) {
    const std::size_t nq = grid.size();
    if (values.size() % nq != 0) throw std::runtime_error("joint value has inconsistent dimensions");
    const auto w = interpolation_weights(grid, q);
    return values[joint_index(vol, w.left, nq)] * w.wl
         + values[joint_index(vol, w.right, nq)] * w.wr;
}

double rfq_markout_inventory_pnl(const PricingProblem& problem, const Tier& tier,
                                 Side side, std::size_t vol,
                                 double inventory_after_branch,
                                 const MarkoutResolvent* markout) {
    if (!tier.use_markout() || markout == nullptr) return 0.0;
    const auto* exposure = markout->effective_inventory(tier.markout().tau_minutes());
    if (exposure == nullptr) {
        throw std::runtime_error("missing policy markout resolvent for tier '" + tier.name() + "'");
    }
    const double effective_q = interpolate_joint(*exposure, problem.grid().states(), vol,
                                                 inventory_after_branch);
    return (-direction(side)) * tier.markout().asymptotic() * effective_q;
}

double max_abs(const std::vector<double>& x) {
    double out = 0.0;
    for (double v : x) out = std::max(out, std::abs(v));
    return out;
}

double max_abs_policy_slice(const Policy& policy) {
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
                                 std::size_t vol, std::size_t nq) {
    const auto begin = values.begin() + static_cast<std::ptrdiff_t>(vol * nq);
    return std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(nq));
}

std::vector<double> slice_operational(const std::vector<double>& values,
                                      std::size_t vol, std::size_t nq,
                                      const std::vector<std::size_t>& indices) {
    std::vector<double> out;
    out.reserve(indices.size());
    for (auto i : indices) out.push_back(values[joint_index(vol, i, nq)]);
    return out;
}

MarkoutResolvent slice_markout(const MarkoutResolvent& joint,
                               std::size_t vol, std::size_t nq) {
    MarkoutResolvent out;
    for (const auto& curve : joint.curves) {
        MarkoutExposureCurve sliced;
        sliced.tau_minutes = curve.tau_minutes;
        sliced.effective_inventory = slice_values(curve.effective_inventory, vol, nq);
        out.curves.push_back(std::move(sliced));
    }
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

double policy_slice_change(const Policy& lhs, const Policy& rhs) {
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

}  // namespace

HowardSolver::AffineOperator HowardSolver::fixed_policy_operator(
        const JointPolicy& policy, const MarkoutResolvent* markout) const {
    const auto& grid = problem_.grid();
    const auto& states = grid.states();
    const auto& volatility = problem_.volatility();
    const auto& q_sigma = volatility.generator();
    const std::size_t nq = states.size();
    const std::size_t nv = volatility.state_count();
    if (policy.size() != nv) throw std::invalid_argument("joint policy volatility dimension mismatch");
    const std::size_t n = nq * nv;
    AffineOperator op{std::vector<double>(n, 0.0), DenseMatrix(n, n)};

    for (std::size_t v = 0; v < nv; ++v) {
        const auto& p_slice = policy[v];
        for (std::size_t i = 0; i < nq; ++i) {
            const std::size_t from = joint_index(v, i, nq);
            const double q = states[i];
            op.reward[from] = -problem_.penalty().value(q, volatility.sigma(v))
                            + problem_.spot_drift() * q;

            for (std::size_t k = 0; k < problem_.tiers().size(); ++k) {
                const auto& tier = problem_.tiers()[k];
                const auto& p = p_slice.tiers[k];
                for (Side side : {Side::Bid, Side::Ask}) {
                    const double dir = direction(side);
                    const auto& deltas = row(p, side, i);
                    for (std::size_t j = 0; j < tier.sizes().size(); ++j) {
                        const double z = tier.sizes()[j];
                        const double dq = dir * z;
                        const double rfq_rate = tier.flow().rfq_arrival_rate(z);
                        const double loss_markout_pnl =
                            rfq_markout_inventory_pnl(problem_, tier, side, v, q, markout);

                        if (!grid.admissible(q, dq)) {
                            op.reward[from] += rfq_rate * loss_markout_pnl;
                            continue;
                        }

                        const double delta = deltas[j];
                        const double win_rate = tier.flow().arrival_rate(delta, z);
                        const double loss_rate = std::max(0.0, rfq_rate - win_rate);
                        const double q_next = q + dq;
                        const double win_markout_pnl =
                            rfq_markout_inventory_pnl(problem_, tier, side, v, q_next, markout);
                        op.reward[from] += win_rate *
                            (z * (problem_.spread() * (0.5 - delta) - tier.fee())
                             + win_markout_pnl)
                            + loss_rate * loss_markout_pnl;
                        add_inventory_transition(op.generator, states, v, i, q_next, win_rate);
                    }
                }
            }

            if (problem_.dark_pool() && p_slice.dark_pool) {
                const auto& venue = *problem_.dark_pool();
                const auto& p = *p_slice.dark_pool;
                for (Side side : {Side::Bid, Side::Ask}) {
                    const bool active = side == Side::Bid ? p.bid_active[i] : p.ask_active[i];
                    const double posted = side == Side::Bid ? p.bid_size[i] : p.ask_size[i];
                    if (!active || posted <= 0.0 || !venue.risk_reducing(q, side, posted)) continue;
                    const int u = static_cast<int>(std::llround(posted));
                    const double dir = direction(side);
                    for (const auto& [fill, rate] : venue.arrivals(side).fill_rates(u)) {
                        op.reward[from] -= rate * venue.fee(side) * fill;
                        add_inventory_transition(op.generator, states, v, i, q + dir * fill, rate);
                    }
                }
            }

            if (problem_.passive_ecn() && p_slice.passive_ecn) {
                const auto& venue = *problem_.passive_ecn();
                const auto& p = *p_slice.passive_ecn;
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
                            op.reward[from] += rate * fill *
                                (problem_.spread() * (0.5 - delta) - venue.maker_fee());
                            add_inventory_transition(op.generator, states, v, i, q + dir * fill, rate);
                        }
                    }
                }
            }

            // Exogenous volatility-state transitions keep inventory unchanged.
            for (std::size_t u = 0; u < nv; ++u) {
                if (u == v) continue;
                const double rate = q_sigma[v][u];
                if (rate <= 0.0) continue;
                op.generator(from, joint_index(u, i, nq)) += rate;
                op.generator(from, from) -= rate;
            }
        }
    }
    return op;
}

MarkoutResolvent HowardSolver::markout_resolvent(const JointPolicy& policy) const {
    const auto generator_op = fixed_policy_operator(policy, nullptr);
    const auto& L = generator_op.generator;
    const auto& q_states = problem_.grid().states();
    const std::size_t nq = q_states.size();
    const std::size_t nv = problem_.volatility().state_count();
    const std::size_t n = nq * nv;

    std::vector<double> q_rhs(n, 0.0);
    for (std::size_t v = 0; v < nv; ++v) {
        for (std::size_t i = 0; i < nq; ++i) q_rhs[joint_index(v, i, nq)] = q_states[i];
    }

    MarkoutResolvent out;
    std::vector<double> unique_taus;
    for (const auto& tier : problem_.tiers()) {
        if (!tier.use_markout()) continue;
        const double tau = tier.markout().tau_minutes();
        const bool seen = std::any_of(unique_taus.begin(), unique_taus.end(), [&](double x) {
            return std::abs(x - tau) <= 1e-12 * std::max({1.0, std::abs(x), std::abs(tau)});
        });
        if (!seen) unique_taus.push_back(tau);
    }

    for (double tau : unique_taus) {
        const double beta = 1.0 / tau;
        DenseMatrix A(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) A(i, j) = -L(i, j);
            A(i, i) += beta;
        }
        const auto x = A.solve(q_rhs);
        MarkoutExposureCurve curve;
        curve.tau_minutes = tau;
        curve.effective_inventory.resize(n);
        for (std::size_t i = 0; i < n; ++i) curve.effective_inventory[i] = beta * x[i];
        out.curves.push_back(std::move(curve));
    }
    return out;
}

std::pair<std::vector<double>, double> HowardSolver::evaluate_policy(
        const JointPolicy& policy, const MarkoutResolvent& markout) const {
    const auto op = fixed_policy_operator(policy, &markout);
    const std::size_t n = op.reward.size();
    DenseMatrix A(n + 1, n + 1);
    std::vector<double> b(n + 1, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) A(i, j) = op.generator(i, j);
        A(i, n) = -1.0;
        b[i] = -op.reward[i];
    }
    const std::size_t ref = joint_index(problem_.volatility().initial_state(),
                                        problem_.grid().zero_index(),
                                        problem_.grid().states().size());
    A(n, ref) = 1.0;

    const auto x = A.solve(b);
    std::vector<double> h(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n));
    return {std::move(h), x[n]};
}

HowardSolver::JointPolicy HowardSolver::improve_policy(
        const std::vector<double>& value, const JointPolicy& previous,
        const MarkoutResolvent& markout) const {
    const std::size_t nq = problem_.grid().states().size();
    const std::size_t nv = problem_.volatility().state_count();
    JointPolicy out;
    out.reserve(nv);
    for (std::size_t v = 0; v < nv; ++v) {
        const auto h_slice = slice_values(value, v, nq);
        const auto m_slice = slice_markout(markout, v, nq);
        PolicyBuilder builder(problem_, &m_slice);
        out.push_back(builder.improve(h_slice, previous[v]));
    }
    return out;
}

double HowardSolver::policy_change(const JointPolicy& lhs, const JointPolicy& rhs) {
    if (lhs.size() != rhs.size()) return std::numeric_limits<double>::infinity();
    double out = 0.0;
    for (std::size_t v = 0; v < lhs.size(); ++v) out = std::max(out, policy_slice_change(lhs[v], rhs[v]));
    return out;
}

double HowardSolver::markout_change(const MarkoutResolvent& lhs, const MarkoutResolvent& rhs) {
    double out = 0.0;
    for (const auto& curve : lhs.curves) {
        const auto* other = rhs.effective_inventory(curve.tau_minutes);
        if (other == nullptr || other->size() != curve.effective_inventory.size()) return std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < other->size(); ++i) {
            out = std::max(out, std::abs(curve.effective_inventory[i] - (*other)[i]));
        }
    }
    if (lhs.curves.size() != rhs.curves.size()) return std::numeric_limits<double>::infinity();
    return out;
}

double HowardSolver::markout_scale(const MarkoutResolvent& markout) {
    double out = 1.0;
    for (const auto& curve : markout.curves) out = std::max(out, max_abs(curve.effective_inventory));
    return out;
}

double HowardSolver::policy_scale(const JointPolicy& policy) {
    double out = 1.0;
    for (const auto& p : policy) out = std::max(out, max_abs_policy_slice(p));
    return out;
}

Solution HowardSolver::solve() const {
    const auto& grid = problem_.grid();
    const auto& states = grid.states();
    const auto& op_indices = grid.operational_indices();
    const auto& volatility = problem_.volatility();
    const std::size_t nq = states.size();
    const std::size_t nv = volatility.state_count();
    const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());

    PolicyBuilder initial_builder(problem_);
    const Policy seed = initial_builder.initial_policy();
    JointPolicy policy(nv, seed);
    MarkoutResolvent markout = markout_resolvent(policy);
    std::vector<double> h(nq * nv, 0.0);
    double rho = 0.0;

    SolverDiagnostics diagnostics;
    constexpr int kMaximumIterations = 256;
    for (int iteration = 1; iteration <= kMaximumIterations; ++iteration) {
        const auto h_previous = h;
        std::tie(h, rho) = evaluate_policy(policy, markout);

        JointPolicy improved = improve_policy(h, policy, markout);
        const double p_change = policy_change(improved, policy);
        MarkoutResolvent improved_markout = markout_resolvent(improved);
        const double m_change = markout_change(improved_markout, markout);

        const auto op = fixed_policy_operator(improved, &improved_markout);
        double residual = 0.0;
        double rhs_scale = 1.0;
        for (std::size_t i = 0; i < h.size(); ++i) {
            double rhs = op.reward[i];
            for (std::size_t j = 0; j < h.size(); ++j) rhs += op.generator(i, j) * h[j];
            residual = std::max(residual, std::abs(rhs - rho));
            rhs_scale = std::max(rhs_scale, std::abs(rhs));
        }
        double h_change = 0.0;
        for (std::size_t i = 0; i < h.size(); ++i) {
            h_change = std::max(h_change, std::abs(h[i] - h_previous[i]));
        }
        const double scale = std::max({1.0, std::abs(rho), rhs_scale});
        const double residual_tol = sqrt_eps * scale;
        const double policy_tol = sqrt_eps * policy_scale(improved);
        const double markout_tol = sqrt_eps * markout_scale(improved_markout);

        diagnostics.iterations = iteration;
        diagnostics.average_reward = rho;
        diagnostics.reward_lower_bound = rho - residual;
        diagnostics.reward_upper_bound = rho + residual;
        diagnostics.value_change.push_back(h_change);
        diagnostics.bellman_residual.push_back(residual);
        diagnostics.markout_resolvent_change.push_back(m_change);

        policy = std::move(improved);
        markout = std::move(improved_markout);

        if (residual <= residual_tol && p_change <= policy_tol && m_change <= markout_tol) {
            diagnostics.converged = true;
            break;
        }
    }

    // Apply operational constraints independently in every volatility state.
    for (std::size_t v = 0; v < nv; ++v) {
        const auto m_slice = slice_markout(markout, v, nq);
        PolicyBuilder final_builder(problem_, &m_slice);
        policy[v] = final_builder.snap_operational_constraints(policy[v]);
    }
    markout = markout_resolvent(policy);
    std::tie(h, rho) = evaluate_policy(policy, markout);
    const auto final_op = fixed_policy_operator(policy, &markout);
    double residual = 0.0;
    for (std::size_t i = 0; i < h.size(); ++i) {
        double rhs = final_op.reward[i];
        for (std::size_t j = 0; j < h.size(); ++j) rhs += final_op.generator(i, j) * h[j];
        residual = std::max(residual, std::abs(rhs - rho));
    }
    diagnostics.average_reward = rho;
    diagnostics.reward_lower_bound = rho - residual;
    diagnostics.reward_upper_bound = rho + residual;

    Solution solution;
    solution.q_grid = grid.operational_states();
    solution.solve_q_grid = states;
    solution.average_reward = rho;
    solution.hard_inventory_limit = grid.hard_limit();
    solution.diagnostics = diagnostics;
    solution.volatility_states = volatility.sigma_states();

    solution.value_by_volatility.reserve(nv);
    solution.tier_policies_by_volatility.reserve(nv);
    solution.solve_tier_policies_by_volatility.reserve(nv);
    solution.dark_pool_policies_by_volatility.reserve(nv);
    solution.passive_ecn_policies_by_volatility.reserve(nv);
    solution.solve_dark_pool_policies_by_volatility.reserve(nv);
    solution.solve_passive_ecn_policies_by_volatility.reserve(nv);

    for (std::size_t v = 0; v < nv; ++v) {
        solution.value_by_volatility.push_back(slice_operational(h, v, nq, op_indices));
        solution.solve_tier_policies_by_volatility.push_back(policy[v].tiers);
        std::vector<LadderPolicy> op_tiers;
        op_tiers.reserve(policy[v].tiers.size());
        for (const auto& p : policy[v].tiers) {
            op_tiers.push_back(slice_policy(p, op_indices, grid.operational_states()));
        }
        solution.tier_policies_by_volatility.push_back(std::move(op_tiers));

        if (policy[v].dark_pool) {
            solution.solve_dark_pool_policies_by_volatility.push_back(*policy[v].dark_pool);
            solution.dark_pool_policies_by_volatility.push_back(
                slice_policy(*policy[v].dark_pool, op_indices, grid.operational_states()));
        } else {
            solution.solve_dark_pool_policies_by_volatility.push_back(std::nullopt);
            solution.dark_pool_policies_by_volatility.push_back(std::nullopt);
        }
        if (policy[v].passive_ecn) {
            solution.solve_passive_ecn_policies_by_volatility.push_back(*policy[v].passive_ecn);
            solution.passive_ecn_policies_by_volatility.push_back(
                slice_policy(*policy[v].passive_ecn, op_indices, grid.operational_states()));
        } else {
            solution.solve_passive_ecn_policies_by_volatility.push_back(std::nullopt);
            solution.passive_ecn_policies_by_volatility.push_back(std::nullopt);
        }
    }

    const std::size_t v0 = volatility.initial_state();
    solution.value = solution.value_by_volatility[v0];
    solution.value_solve = slice_values(h, v0, nq);
    solution.tier_policies = solution.tier_policies_by_volatility[v0];
    solution.solve_tier_policies = solution.solve_tier_policies_by_volatility[v0];
    solution.dark_pool_policy = solution.dark_pool_policies_by_volatility[v0];
    solution.passive_ecn_policy = solution.passive_ecn_policies_by_volatility[v0];
    solution.solve_dark_pool_policy = solution.solve_dark_pool_policies_by_volatility[v0];
    solution.solve_passive_ecn_policy = solution.solve_passive_ecn_policies_by_volatility[v0];

    // Keep a full operational [vol][q] markout surface and a legacy initial-vol slice.
    for (const auto& curve : markout.curves) {
        MarkoutExposureCurve joint_curve;
        joint_curve.tau_minutes = curve.tau_minutes;
        for (std::size_t v = 0; v < nv; ++v) {
            for (auto qi : op_indices) {
                joint_curve.effective_inventory.push_back(
                    curve.effective_inventory[joint_index(v, qi, nq)]);
            }
        }
        solution.markout_exposure_joint.push_back(joint_curve);

        MarkoutExposureCurve initial_curve;
        initial_curve.tau_minutes = curve.tau_minutes;
        initial_curve.effective_inventory = slice_operational(curve.effective_inventory, v0, nq, op_indices);
        solution.markout_exposure.push_back(std::move(initial_curve));
    }
    return solution;
}

}  // namespace ladder_pricer
