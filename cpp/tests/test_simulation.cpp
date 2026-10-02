#include "ladder_pricer/howard.hpp"
#include "ladder_pricer/analytics.hpp"
#include "ladder_pricer/simulation.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace ladder_pricer;

std::vector<double> grid() {
    std::vector<double> p;
    for (double q = 0; q <= 3.0 + 1e-12; q += .25) p.push_back(q);
    for (double q = 4; q <= 20 + 1e-12; q += 1) p.push_back(q);
    std::vector<double> o;
    for (auto it = p.rbegin(); it != p.rend() - 1; ++it) o.push_back(-*it);
    o.insert(o.end(), p.begin(), p.end());
    return o;
}

int main() {

    // Dark-pool ZIP transition rates must represent an exogenous arrival clock
    // followed by X~ZIP and F=min(X,u).  Positive fill rates therefore sum to
    // lambda*(1-p0), and the full-fill bucket absorbs the entire upper tail.
    {
        const double lambda = 2.0;
        const double mu = 2.0;
        const double p0 = 0.25;
        ZeroInflatedPoissonArrival dist(lambda, mu, p0);
        const auto rates = dist.fill_rates(2);
        assert(rates.size() == 2);
        const double sum_rates = rates[0].second + rates[1].second;
        assert(std::abs(sum_rates - lambda * (1.0 - p0)) < 1e-12);

        const double pzero = std::exp(-mu);
        const double positive_norm = 1.0 - pzero;
        const double p1_cond = (pzero * mu) / positive_norm;
        const double expected_full = lambda * (1.0 - p0) * (1.0 - p1_cond);
        assert(std::abs(rates[1].second - expected_full) < 1e-12);

        std::mt19937_64 rng(123456);
        int zeros = 0;
        long long positive_sum = 0;
        int positives = 0;
        constexpr int draws = 200000;
        for (int n=0; n<draws; ++n) {
            const int x = dist.sample_incoming_size(rng);
            if (x == 0) ++zeros; else { ++positives; positive_sum += x; }
        }
        const double zero_freq = static_cast<double>(zeros) / draws;
        assert(std::abs(zero_freq - p0) < 0.01);
        assert(positives > 0);
        assert(static_cast<double>(positive_sum) / positives > 2.0);
    }
    std::vector<double> sizes{1,2,3,5,10,20};
    std::vector<Tier> tiers;
    tiers.emplace_back("Tier 1", sizes,
        LogisticFlow(.0155,.144,.0857,.52,8.42,.026),
        SaturatingMarkout(1e-4,.5,.5), true, -10, 100);
    tiers.emplace_back("Tier 2", sizes,
        LogisticFlow(.0232,.303,.122,.48,2.86,.02),
        SaturatingMarkout(1e-4,.5,.5), true, -10, 100);

    PricingProblem p(grid(), .002, 0,
        QuadraticPenalty(.1,.002), InternalizationTime(4,.07,.0084), tiers);
    auto s = HowardSolver(p).solve();

    // Existing analytical-vs-Monte-Carlo mean regression (sigma=0).
    auto stats = PnlAnalytics(p,s,11.5).statistics(570,0,0);
    auto mc = MonteCarloSimulator(p,s,11.5).run(570,3000,0,0,12345,2,51);
    double m = std::accumulate(mc.pnl_base_ccy.begin(),mc.pnl_base_ccy.end(),0.0)
             / mc.pnl_base_ccy.size();
    std::cout << "cf=" << stats.expected_base_ccy << " mc=" << m << "\n";
    if (std::abs(m-stats.expected_base_ccy)>600) return 1;

    // The market path must evolve independently of fills, and output sampling must
    // not alter the underlying simulation. Both runs use identical event/spot RNG
    // streams but different inventory-output grids.
    auto coarse = MonteCarloSimulator(p,s,11.5).run(10.0, 4, .002, 0, 777, 1, 11);
    auto fine   = MonteCarloSimulator(p,s,11.5).run(10.0, 4, .002, 0, 777, 1, 41);

    assert(coarse.trade_count == fine.trade_count);
    assert(coarse.final_inventory == fine.final_inventory);
    assert(coarse.final_spot.size() == fine.final_spot.size());
    for (std::size_t i=0; i<coarse.final_spot.size(); ++i) {
        assert(std::abs(coarse.final_spot[i] - fine.final_spot[i]) < 1e-12);
        assert(std::abs(coarse.pnl_base_ccy[i] - fine.pnl_base_ccy[i]) < 1e-8);
    }

    assert(coarse.sample_paths.size() == 1);
    assert(fine.sample_paths.size() == 1);
    const auto& a = coarse.sample_paths.front();
    const auto& b = fine.sample_paths.front();
    assert(a.times == b.times);
    assert(a.inventories == b.inventories);
    assert(a.spots.size() == b.spots.size());
    assert(a.spots.size() > 500);  // One-second market clock over 10 minutes.
    bool moved = false;
    for (std::size_t i=1; i<a.spots.size(); ++i) {
        assert(std::abs(a.spots[i] - b.spots[i]) < 1e-12);
        if (std::abs(a.spots[i] - a.spots[i-1]) > 1e-12) moved = true;
    }
    assert(moved);

    std::cout << "market_points=" << a.spots.size()
              << " trades=" << coarse.trade_count.front() << "\n";
    return 0;
}
