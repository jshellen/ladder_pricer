#include "ladder_pricer/solver.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using namespace ladder_pricer;

int main() {
    std::vector<double> grid;
    for (int q=-8;q<=8;++q) grid.push_back(static_cast<double>(q));
    std::vector<Tier> tiers;
    tiers.emplace_back("T1",std::vector<double>{1.,2.,4.},
        LogisticFlow(.0155,.144,.0857,.52,8.42,.026),
        SaturatingMarkout(.00050,.5),true,-10.,100.);
    tiers.emplace_back("T2",std::vector<double>{1.,2.,4.},
        LogisticFlow(.0232,.303,.122,.48,2.86,.02),
        SaturatingMarkout(.00090,2.0),true,-10.,100.);
    PricingProblem problem(grid,.002,QuadraticPenalty(.1,.002),tiers);
    Solver solver(problem, {12.,20.,35.}, {{0.,.1,.02},{.04,0.,.08},{.02,.07,0.}});
    const auto result=solver.solve();
    assert(result.q_grid.size()==grid.size());
    assert(result.policies.size()==3);
    assert(result.values.size()==3);
    assert(result.sigma_pips==std::vector<double>({12.,20.,35.}));
    assert(std::isfinite(result.average_reward));
    assert(result.converged);
    assert(result.evaluation_residual<1.e-8);
    for (const auto& policy:result.policies) {
        assert(policy.tiers.size()==2);
        assert(policy.tiers.front().bid.size()==problem.grid().states().size());
        for (const auto& row:policy.tiers.front().bid)
            assert(row.size()==3);
    }
    // Joint generator and nonzero markout resolvents should be present.
    assert(result.weighted_inventory.size()==2);
    assert(result.weighted_inventory[0].size()==3*problem.grid().states().size());
    const std::size_t n=problem.grid().states().size();
    for (auto v:result.weighted_inventory[0]) assert(std::isfinite(v));
    assert(std::abs(result.weighted_inventory[0][n]-result.weighted_inventory[0][2*n])>1e-10);
    std::cout<<"Solver q x sigma result OK: "<<result.average_reward<<"\n";
}
