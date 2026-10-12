#include "historical_test_flow.hpp"
#include "ladder_pricer/core.hpp"
#include "ladder_pricer/howard.hpp"
#include "ladder_pricer/detail/brent.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ladder_pricer {
namespace {
void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
}

// Exercise the actual coordinate optimizer and independently re-evaluate its
// objective via the full RFQ Hamiltonian (including interpolated RFQ sizes).
void validate_single_pricing_knot(const PolicyBuilder& builder, const Tier& tier,
        const std::vector<double>& value, double q, Side side,
        std::size_t j, double initial_delta) {
    std::vector<double> seed(tier.sizes().size(), initial_delta);
    // Project before optimization, same as production, then freeze all knots
    // except j at the values of the finished coordinate sweep.
    const auto optimized = builder.coordinate_optimize_ladder(tier, value, q, side, seed);
    auto row = optimized;
    auto [lo, hi] = builder.coordinate_bounds(tier, j, row, {});
    const double dir = side == Side::Bid ? 1.0 : -1.0;
    const auto& grid = builder.problem_.grid();
    const double hq = grid.interpolate(value, q);
    const auto* exposure = builder.markout_ ?
        builder.markout_->effective_inventory(tier.markout().tau_minutes()) : nullptr;
    require(!tier.use_markout() || exposure != nullptr, "missing markout resolvent");
    const double loss = exposure ? -dir * tier.markout().asymptotic() *
        grid.interpolate(*exposure, q) : 0.0;
    // Production's exact upper-bound break-even clipping: a zero contribution
    // from each admissible RFQ atom defines its break-even interpolated quote.
    double max_break_even = -std::numeric_limits<double>::infinity();
    for (const auto& source : tier.flow().sources()) {
        for (const auto& atom : source.rfq_atoms()) {
            const double z = atom.target_size;
            if (atom.one_side_arrival_rate <= 0 || !grid.admissible(q, dir * z)) continue;
            const auto& sizes = tier.sizes();
            double weight = 0.0;
            if (z <= sizes.front() + 1e-12) weight = (j == 0 ? 1. : 0.);
            else if (z >= sizes.back() - 1e-12) weight = (j+1 == sizes.size() ? 1. : 0.);
            else {
                auto it = std::upper_bound(sizes.begin(), sizes.end(), z);
                const auto right = static_cast<std::size_t>(it-sizes.begin());
                const double wr = (z-sizes[right-1])/(sizes[right]-sizes[right-1]);
                if (j == right-1) weight = 1.-wr;
                if (j == right) weight = wr;
            }
            if (weight <= 1e-14) continue;
            auto interpolated = [&](const std::vector<double>& xs) {
                if (z <= sizes.front()) return xs.front();
                if (z >= sizes.back()) return xs.back();
                auto it = std::upper_bound(sizes.begin(), sizes.end(), z);
                const auto k = static_cast<std::size_t>(it-sizes.begin());
                const double w = (z-sizes[k-1])/(sizes[k]-sizes[k-1]);
                return xs[k-1]*(1.-w)+xs[k]*w;
            };
            const double constant_delta = interpolated(row)-weight*row[j];
            const double next_q = q + dir*z;
            const double win_markout = exposure ? -dir * tier.markout().asymptotic() *
                grid.interpolate(*exposure, next_q) : 0.0;
            const double additive = -z*tier.fee() +
                win_markout - loss
                + grid.interpolate(value,next_q)-hq;
            const double break_even = .5 + additive/(z*builder.problem_.spread());
            max_break_even = std::max(max_break_even,(break_even-constant_delta)/weight);
        }
    }
    if (std::isfinite(max_break_even)) hi=std::min(hi,std::max(lo,max_break_even));
    if (hi <= lo + 1e-12) return;
    auto hamiltonian = [&](double x) {
        row[j]=x;
        return builder.ladder_hamiltonian(tier,value,q,side,row);
    };
    // Independent multi-bracket search: discover all sampled local maxima,
    // refine each on its own bracket and always include both endpoints.
    struct Result {double x, h; int peaks; int evaluations;};
    auto global_search = [&](int steps) -> Result {
        std::vector<double> ys(static_cast<std::size_t>(steps)+1);
        for (int i=0;i<=steps;++i) {
            ys[i]=hamiltonian(lo+(hi-lo)*static_cast<double>(i)/steps);
            require(std::isfinite(ys[i]),"non-finite Hamiltonian");
        }
        Result out{lo,ys.front(),0,steps+1};
        auto keep=[&](double x,double h) {if(h>out.h){out.x=x;out.h=h;}};
        keep(hi,ys.back());
        for (int i=1;i<steps;++i) {
            keep(lo+(hi-lo)*static_cast<double>(i)/steps,ys[i]);
            if(ys[i]>=ys[i-1] && ys[i]>=ys[i+1] &&
                    (ys[i]>ys[i-1] || ys[i]>ys[i+1])) {
                ++out.peaks;
                double a=lo+(hi-lo)*static_cast<double>(i-1)/steps;
                double b=lo+(hi-lo)*static_cast<double>(i+1)/steps;
                auto optimum=detail::bounded_brent_minimize(
                    [&](double x) {++out.evaluations;return -hamiltonian(x);},a,b,30);
                keep(optimum.first,-optimum.second);
            }
        }
        return out;
    };
    auto coarse=global_search(400);
    auto medium=global_search(2000);
    auto fine=global_search(10000);
    const double actual_h=hamiltonian(optimized[j]);
    const double tolerance=1e-8+1e-8*std::max(std::abs(fine.h),std::abs(actual_h));
    const double shortfall=fine.h-actual_h;
    const double stability=std::abs(fine.h-medium.h);
    std::cout << std::setprecision(11) << "q=" << q
              << " side=" << (side==Side::Bid?"bid":"ask")
              << " knot=" << tier.sizes()[j] << " lo=" << lo << " hi=" << hi
              << " Brent_delta=" << optimized[j] << " global_delta=" << fine.x
              << " H_Brent=" << actual_h << " H_global=" << fine.h
              << " shortfall=" << shortfall
              << " peaks=" << coarse.peaks << "/" << medium.peaks << "/" << fine.peaks
              << " stability=" << stability
              << " evals=" << fine.evaluations << '\n';
    require(shortfall<=tolerance,"production Brent below multi-bracket global search");
    require(stability<=tolerance,"global search has not stabilized under grid refinement");
}
} // namespace ladder_pricer

int main() {
    try {
        using namespace ladder_pricer;
        std::vector<double> qgrid;
        for (int i=-20;i<=20;++i) qgrid.push_back(static_cast<double>(i));
        const HistoricalLogisticFixture win(0.0155,0.144,0.0857,0.52,8.42,0.026);
        std::vector<FlowSource> sources;
        sources.emplace_back("EURSEK", DiscreteRfqDistribution(12.0,
            {1.,2.,3.,4.,5.,7.,10.,15.,20.},
            {.13,.14,.12,.11,.13,.09,.12,.09,.07}), historical_win(win));
        sources.emplace_back("USDSEK", DiscreteRfqDistribution(7.0,
            {1.25,2.5,6.25,12.5,25.}, {.10,.20,.30,.25,.15}), historical_win(win), FXMapping(.9, 1.25));
        std::vector<Tier> tiers;
        tiers.emplace_back("Tier 1",std::vector<double>{1,2,3,5,10,20},
            AggregatedFlow(std::move(sources)),SaturatingMarkout(.0001,.5),
            true,-1.,1.,.0003,.5);
        PricingProblem problem(qgrid,.002,0.,QuadraticPenalty(.1,.002),
            InternalizationTime(4.,.070,.0084),std::move(tiers));
        const auto& tier=problem.tiers().front();
        // Solve the actual HJB; validate coordinates against its converged V(q).
        const auto solution=HowardSolver(problem).solve();
        require(solution.diagnostics.converged,"HJB did not converge");
        require(!solution.markout_resolvent_solve.curves.empty(),
                "markout resolvent was not active");
        PolicyBuilder builder(problem, &solution.markout_resolvent_solve);
        require(solution.value_solve.size()==problem.grid().states().size(),
                "HJB value and inventory grids differ");
        int cases=0;
        for (double q : {-15.,-5.,0.,5.,15.}) {
            for (auto side : {Side::Bid,Side::Ask}) {
                for (std::size_t j : {std::size_t(0),std::size_t(2),std::size_t(4)}) {
                    validate_single_pricing_knot(builder,tier,solution.value_solve,q,side,j,.25);
                    ++cases;
                }
            }
        }
        std::cout << "PASS: " << cases
                  << " markout-enabled real-HJB single-knot multi-bracket global comparisons\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAIL: " << ex.what() << '\n';
        return 1;
    }
}
