#include "ladder_pricer/policy_operator.hpp"
#include <algorithm>
#include <cmath>
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

} // namespace

AffineOperator fixed_policy_operator(const PricingProblem& problem, const Policy& policy) {
    const auto& grid = problem.grid();
    const auto& states = grid.states();
    const std::size_t n = states.size();
    AffineOperator op{std::vector<double>(n, 0.0), DenseMatrix(n, n)};

    for (std::size_t i = 0; i < n; ++i) {
        const double q = states[i];
        op.reward[i] = -problem.penalty().value(q);

        for (std::size_t k = 0; k < problem.tiers().size(); ++k) {
            const auto& tier = problem.tiers()[k];
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
                    op.reward[i] += rate * z * (problem.spread() * (0.5 - delta) - tier.fee());
                    add_transition(op.generator, states, i, q_next, rate);
                }
            }
        }

        if (problem.dark_pool() && policy.dark_pool) {
            const auto& venue = *problem.dark_pool();
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

        if (problem.passive_ecn() && policy.passive_ecn) {
            const auto& venue = *problem.passive_ecn();
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
                            (problem.spread() * (0.5 - delta) - venue.maker_fee());
                        add_transition(op.generator, states, i, q + dir * fill, rate);
                    }
                }
            }
        }
    }
    return op;
}

} // namespace ladder_pricer
