#include "ladder_pricer/expected_mtm.hpp"
#include "ladder_pricer/solver.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ladder_pricer;

static PricingProblem make_problem() {
    std::vector<double> states;
    for (int q=-20; q<=20; ++q) states.push_back(static_cast<double>(q));
    std::vector<Tier> tiers;
    tiers.emplace_back("T1", std::vector<double>{1.,5.},
                       LogisticFlow(.0155,.144,.0857,.52,8.42,.026),
                       SaturatingMarkout(.00050, .50), true, -10, 100);
    tiers.emplace_back("T2", std::vector<double>{1.,5.},
                       LogisticFlow(.0232,.303,.122,.48,2.86,.02),
                       SaturatingMarkout(.00090, 2.00), true, -10, 100);
    return PricingProblem(std::move(states), .002, QuadraticPenalty(.1,.002),
                          std::move(tiers));
}

static bool almost(double a, double b, double eps=1e-11) {
    return std::abs(a-b) < eps;
}

int main() {
    const auto problem = make_problem();
    const auto& grid = problem.grid().states();
    const std::size_t n = grid.size();
    const auto zero = problem.grid().zero_index();

    // 1) Without future inventory transitions, the full post-trade position
    // is hit by the complete asymptotic markout. The won-vs-lost increment
    // is only the added 5M (not the entire +15M inventory).
    DenseMatrix none(n,n);
    ExpectedMtm static_mtm(problem,none);
    assert(almost(static_mtm.weighted_inventory(0,15.),15.));
    assert(almost(static_mtm.trade_mtm(0,+1.,15.),-15.*.00050));
    assert(almost(static_mtm.win_increment(0,+1.,10.,15.),-5.*.00050));
    // Customer purchase -> dealer inventory decreases -> market goes up.
    // On an existing +10M long, the remaining +5M position gains MTM.
    assert(almost(static_mtm.trade_mtm(0,-1.,5.),+5.*.00050));
    // Short inventory after customer buys EUR loses MTM.
    assert(almost(static_mtm.trade_mtm(0,-1.,-15.),-15.*.00050));

    // 2) Test the resolvent against an analytically soluble CTMC. Every
    // nonzero inventory jumps to zero at r=0.5/min. E[Q_t]=q exp(-rt),
    // so u_tau(q)=q/(1+r*tau), with a DIFFERENT tau in each tier.
    constexpr double r=.5;
    DenseMatrix generator(n,n);
    for(std::size_t i=0;i<n;++i){
        if(i==zero) continue;
        generator(i,zero)=r;
        generator(i,i)=-r;
    }
    ExpectedMtm mtm(problem,generator);
    assert(almost(mtm.weighted_inventory(0,15.),15./(1.+r*.5)));
    assert(almost(mtm.weighted_inventory(1,15.),15./(1.+r*2.)));
    assert(almost(mtm.trade_mtm(0,+1.,15.),-15.*.00050/(1.+r*.5)));
    assert(almost(mtm.trade_mtm(1,+1.,15.),-15.*.00090/(1.+r*2.)));
    assert(almost(mtm.win_increment(0,+1.,10.,15.),-5.*.00050/(1.+r*.5)));
    assert(almost(mtm.weighted_inventory(0,0.),0.));

    // 3) The markout resolvent continues to supply finite values to the
    // production solver, without an exogenous internalization-time proxy.
    const auto solution = Solver(make_problem(),{12.,20.,35.},
        {{0.,.1,.02},{.04,0.,.08},{.02,.07,0.}}).solve();
    assert(solution.converged);
    assert(solution.evaluation_residual<1e-8);
    assert(solution.weighted_inventory.size()==2);
    for (auto x:solution.weighted_inventory[0]) assert(std::isfinite(x));
    std::cout << "Expected MTM resolvent and side signs OK\n";
}
