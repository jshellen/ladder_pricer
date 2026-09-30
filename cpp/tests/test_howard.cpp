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
    return 0;
}
