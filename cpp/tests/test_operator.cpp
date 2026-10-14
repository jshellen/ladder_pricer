#include "ladder_pricer/policy_operator.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using namespace ladder_pricer;
int main() {
    std::vector<double> grid;
    for (int q=-5;q<=5;++q) grid.push_back(static_cast<double>(q));
    std::vector<Tier> tiers;
    tiers.emplace_back("T1",std::vector<double>{1.,2.},
        LogisticFlow(.0155,.144,.0857,.52,8.42,.026),
        SaturatingMarkout(.00050,.5),true,-10.,100.);
    PricingProblem problem(grid,.002,QuadraticPenalty(.1,.002),tiers);
    const auto policy=PolicyBuilder(problem).initial_policy();
    const auto op=fixed_policy_operator(problem,policy);
    const std::size_t n=problem.grid().states().size();
    assert(op.reward.size()==n);
    assert(op.generator.rows()==n && op.generator.cols()==n);
    for (std::size_t i=0;i<n;++i) {
        double sum=0;
        for (std::size_t j=0;j<n;++j) {
            const double x=op.generator(i,j);
            if (j!=i) assert(x>=-1e-13);
            sum+=x;
        }
        assert(std::abs(sum)<1e-12);
        assert(std::isfinite(op.reward[i]));
    }
    const auto zero=problem.grid().zero_index();
    assert(op.generator(zero,zero)<0.0);
    std::cout<<"Fixed-policy operator rows and rewards OK\n";
}
