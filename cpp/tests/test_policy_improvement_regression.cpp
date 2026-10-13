// Regression for the inventory-ladder policy update near high adverse markout.
// The old update used the previous Howard iteration's inner-neighbour quote.
// This test reproduces the resulting q=0 value-function spike at -50 EUR/M.
#include "ladder_pricer/howard.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

using namespace ladder_pricer;

int main() {
    std::vector<double> q_grid;
    for (int q = -20; q <= 20; ++q) q_grid.push_back(static_cast<double>(q));

    constexpr double spot = 11.5;
    constexpr double markout_eur_per_m = 50.0;
    constexpr double tau_minutes = 36.7 / 60.0;
    const auto markout = SaturatingMarkout(markout_eur_per_m * spot / 1e6, tau_minutes);
    const std::vector<double> sizes{1.0, 2.0, 3.0, 5.0, 10.0, 20.0};
    std::vector<Tier> tiers;
    tiers.emplace_back("Tier 1", sizes, LogisticFlow(.0155, .144, .0857, .52, 8.42, .026),
                       markout, true, -10.0, 100.0);
    tiers.emplace_back("Tier 2", sizes, LogisticFlow(.0232, .303, .122, .48, 2.86, .020),
                       markout, true, -10.0, 100.0);

    // ECN off and gamma=1 make the regression easy to observe.
    PricingProblem problem(q_grid, .002, 0.0, QuadraticPenalty(1.0, .002),
                           InternalizationTime(4.0, .070, .0084), tiers);
    const auto sol = HowardSolver(std::move(problem)).solve();
    assert(sol.diagnostics.converged);
    assert(std::isfinite(sol.average_reward));

    const auto zero_it = std::find(sol.q_grid.begin(), sol.q_grid.end(), 0.0);
    assert(zero_it != sol.q_grid.end());
    const auto zero = static_cast<std::size_t>(std::distance(sol.q_grid.begin(), zero_it));
    assert(zero > 0 && zero + 1 < sol.value.size());

    // The old lagged-neighbour update creates a large isolated spike:
    // |h(-1)-2h(0)+h(1)| ~= 0.01246 instead of ~= 0.00070.
    const double local_curvature = std::abs(sol.value[zero - 1] -
                                            2.0 * sol.value[zero] +
                                            sol.value[zero + 1]);
    assert(local_curvature < 0.002);

    // The old solution also locks both 1M customer quotes at q=0 to -1.221.
    // This broad bound protects the regression without pinning exact optima.
    const auto& tier1 = sol.tier_policies.at(0);
    assert(tier1.bid.at(zero).at(0) > -0.5);
    assert(tier1.ask.at(zero).at(0) > -0.5);

    // Check that the operational constraints remain satisfied at every q,
    // for every tier and both sides (not just at q=0).
    const auto gaps = [](const std::vector<double>& ladder) {
        std::vector<double> output;
        for (std::size_t j = 1; j < ladder.size(); ++j)
            output.push_back(ladder[j - 1] - ladder[j]);
        return output;
    };
    for (const auto& tier : sol.tier_policies) {
        for (int side = 0; side < 2; ++side) {
            const auto& rows = side == 0 ? tier.bid : tier.ask;
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const double q = sol.q_grid[i];
                if (q == 0.0 || std::abs(q) > 20.0) continue;
                const std::size_t inward = q > 0.0 ? i - 1 : i + 1;
                const int sign = ((q > 0.0 && side == 1) ||
                                  (q < 0.0 && side == 0)) ? 1 : -1;
                for (std::size_t j = 0; j < rows[i].size(); ++j)
                    assert(sign * (rows[i][j] - rows[inward][j]) >= -1e-7);
                const auto own_gaps = gaps(rows[i]);
                const auto inner_gaps = gaps(rows[inward]);
                for (std::size_t j = 0; j < own_gaps.size(); ++j)
                    assert(-sign * (own_gaps[j] - inner_gaps[j]) >= -1e-7);
            }
        }
    }
    std::cout << "policy regression OK, center curvature=" << local_curvature
              << " bid_1m=" << tier1.bid[zero][0]
              << " ask_1m=" << tier1.ask[zero][0] << "\n";
}
