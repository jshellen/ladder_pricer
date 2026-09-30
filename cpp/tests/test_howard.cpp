#include "ladder_pricer/howard.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ladder_pricer;

std::vector<double> grid() {
    std::vector<double> pos;
    for (double q = 0.0; q <= 3.0 + 1e-12; q += 0.25) pos.push_back(q);
    for (double q = 4.0; q <= 20.0 + 1e-12; q += 1.0) pos.push_back(q);
    std::vector<double> out;
    for (auto it = pos.rbegin(); it != pos.rend() - 1; ++it) out.push_back(-*it);
    out.insert(out.end(), pos.begin(), pos.end());
    return out;
}

int main() {
    const std::vector<double> sizes{1,2,3,5,10,20};
    std::vector<Tier> tiers;
    tiers.emplace_back("Tier 1", sizes,
        LogisticFlow(0.0155,0.144,0.0857,0.52,8.42,0.026),
        SaturatingMarkout(1e-4,0.5,0.5), true, -10, 100);
    tiers.emplace_back("Tier 2", sizes,
        LogisticFlow(0.0232,0.303,0.122,0.48,2.86,0.02),
        SaturatingMarkout(1e-4,0.5,0.5), true, -10, 100);
    tiers.emplace_back("Tier 3", std::vector<double>{1},
        LogisticFlow(1.0,0.144,0.0857,0.52,20.0,0.026),
        SaturatingMarkout(4e-4,0.5,0.10), true, -10, 100);

    PricingProblem problem(grid(), 0.002, 0.0,
        QuadraticPenalty(0.1, 0.002),
        InternalizationTime(4.0,0.070,0.0084), tiers);
    const auto solution = HowardSolver(std::move(problem)).solve();
    std::cout << "converged=" << solution.diagnostics.converged
              << " iterations=" << solution.diagnostics.iterations
              << " rho=" << solution.average_reward << "\n";
    assert(solution.diagnostics.converged);
    assert(std::isfinite(solution.average_reward));
    assert(solution.value.size() == grid().size());
    assert(std::abs(solution.value[solution.value.size()/2]) < 1e-8);

    // Regression: constrained Howard can have multiple feasible fixed-point
    // branches when cold-started independently at nearby risk aversions.
    // A neighboring solved policy must be accepted as a valid warm start and
    // should not produce a worse ergodic objective than the known cold branch.
    {
        std::vector<Tier> local_tiers;
        local_tiers.emplace_back("Tier 1", sizes,
            LogisticFlow(0.0155,0.144,0.0857,0.52,8.42,0.026),
            SaturatingMarkout(1e-4,0.5,0.5), true, -10, 100);
        local_tiers.emplace_back("Tier 2", sizes,
            LogisticFlow(0.0232,0.303,0.122,0.48,2.86,0.02),
            SaturatingMarkout(1e-4,0.5,0.5), true, -10, 100);

        PricingProblem p0(grid(), 0.002, 0.0,
            QuadraticPenalty(1.196969696969697, 0.002),
            InternalizationTime(4.0,0.070,0.0084), local_tiers);
        const auto s0 = HowardSolver(std::move(p0)).solve();
        Policy warm;
        warm.tiers = s0.solve_tier_policies;
        if (s0.solve_dark_pool_policy) warm.dark_pool = *s0.solve_dark_pool_policy;

        PricingProblem p1_cold(grid(), 0.002, 0.0,
            QuadraticPenalty(1.2070707070707072, 0.002),
            InternalizationTime(4.0,0.070,0.0084), local_tiers);
        const auto cold = HowardSolver(std::move(p1_cold)).solve();

        PricingProblem p1_warm(grid(), 0.002, 0.0,
            QuadraticPenalty(1.2070707070707072, 0.002),
            InternalizationTime(4.0,0.070,0.0084), local_tiers);
        const auto continued = HowardSolver(std::move(p1_warm)).solve(warm);
        assert(continued.diagnostics.converged);
        assert(continued.average_reward >= cold.average_reward - 1e-12);
        // This calibration used to jump to a clearly dominated cold branch.
        assert(continued.average_reward > cold.average_reward + 1e-5);
    }
    return 0;
}
