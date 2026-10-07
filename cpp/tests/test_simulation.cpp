#include "ladder_pricer/howard.hpp"
#include "ladder_pricer/analytics.hpp"
#include "ladder_pricer/simulation.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
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
    if (std::abs(m-stats.expected_base_ccy)>1500) return 1;

    // The market path must evolve independently of fills, and output sampling must
    // not alter the underlying simulation. Both runs use identical event/spot RNG
    // streams but different inventory-output grids.
    auto coarse = MonteCarloSimulator(p,s,11.5).run(10.0, 4, .002, 0, 777, 1, 11);
    auto fine   = MonteCarloSimulator(p,s,11.5).run(10.0, 4, .002, 0, 777, 1, 41);

    // Population diagnostics must aggregate every simulated path rather than
    // only the retained sample used for interactive charting.
    std::uint64_t aggregate_trade_count = 0;
    double aggregate_volume = 0.0;
    for (const auto& aggregate : coarse.fill_aggregates) {
        aggregate_trade_count += aggregate.trade_count;
        aggregate_volume += aggregate.volume;
    }
    const long long path_trade_count = std::accumulate(
        coarse.trade_count.begin(), coarse.trade_count.end(), 0LL);
    assert(aggregate_trade_count == static_cast<std::uint64_t>(path_trade_count));
    assert(aggregate_volume >= 0.0);

    std::uint64_t aggregate_rfq_wins = 0;
    std::uint64_t aggregate_rfq_requests = 0;
    for (const auto& aggregate : coarse.rfq_aggregates) {
        aggregate_rfq_wins += aggregate.wins;
        aggregate_rfq_requests += aggregate.requests;
        assert(aggregate.wins <= aggregate.requests);
    }
    // This fixture has customer tiers only, so every trade is an RFQ win.
    assert(aggregate_rfq_wins == aggregate_trade_count);
    assert(aggregate_rfq_requests >= aggregate_rfq_wins);

    // RFQs are generated from one pooled two-sided source clock.  The requested
    // size is sampled from the *same exogenous intensity curve* used by the HJB.
    // Pricing knots are {1,2,4}, but customer RFQs use the full 1M grid
    // {1,2,3,4}.  For theta=1,beta=0 the curve weights are
    // {1,1/2,1/3,1/4}; 3M therefore tests a genuinely interpolated quote.
    {
        const std::vector<double> sampled_sizes{1.0, 2.0, 4.0};
        const LogisticFlow sampled_flow(1.0, 1.0, 0.0, 0.50, 5.0, 0.0);
        std::vector<Tier> size_tiers;
        size_tiers.emplace_back(
            "RFQ size sampling", sampled_sizes,
            AggregatedFlow(std::vector<FlowSource>{
                FlowSource("EURSEK", sampled_flow, 1.0, 1.0),
            }),
            SaturatingMarkout(0.0, 0.5, 0.5), false, -10.0, 100.0);
        PricingProblem size_problem(
            grid(), .002, 0.0, QuadraticPenalty(.1, .002),
            InternalizationTime(4,.07,.0084), size_tiers);
        auto size_solution = HowardSolver(size_problem).solve();
        auto size_mc = MonteCarloSimulator(size_problem, size_solution, 11.5)
                           .run(2.0, 1000, 0.0, 0.0, 20261003, 10, 5);

        std::map<double, std::uint64_t> requests_by_size;
        std::uint64_t total_requests = 0;
        for (const auto& aggregate : size_mc.rfq_aggregates) {
            requests_by_size[aggregate.size] = aggregate.requests;
            total_requests += aggregate.requests;
        }
        assert((size_problem.tiers()[0].rfq_sizes() == std::vector<double>{1.0, 2.0, 3.0, 4.0}));
        // One-sided curve mass is 1 + 1/2 + 1/3 + 1/4 = 25/12, so
        // the pooled bid+ask clock is 25/6 RFQs/min.
        const double expected_requests = (25.0 / 6.0) * 2.0 * 1000.0;
        assert(std::abs(static_cast<double>(total_requests) - expected_requests)
               / expected_requests < 0.05);

        const std::vector<double> rfq_sizes{1.0, 2.0, 3.0, 4.0};
        const std::vector<double> expected_probabilities{12.0/25.0, 6.0/25.0, 4.0/25.0, 3.0/25.0};
        for (std::size_t j = 0; j < rfq_sizes.size(); ++j) {
            const double realized = static_cast<double>(requests_by_size[rfq_sizes[j]])
                                  / static_cast<double>(total_requests);
            assert(std::abs(realized - expected_probabilities[j]) < 0.025);
        }
        assert(requests_by_size[3.0] > 0);

        std::uint64_t tier_size_requests = 0;
        for (const auto& aggregate : size_mc.rfq_tier_size_aggregates) {
            assert(aggregate.tier == "RFQ size sampling");
            assert(aggregate.wins <= aggregate.requests);
            tier_size_requests += aggregate.requests;
        }
        assert(tier_size_requests == total_requests);

        std::uint64_t delta_requests = 0;
        for (const auto& aggregate : size_mc.rfq_delta_aggregates) {
            assert(aggregate.tier == "RFQ size sampling");
            assert(aggregate.wins <= aggregate.admissible_requests);
            assert(aggregate.admissible_requests <= aggregate.requests);
            assert(aggregate.expected_wins >= -1e-12);
            assert(aggregate.expected_wins <= static_cast<double>(aggregate.admissible_requests) + 1e-12);
            delta_requests += aggregate.requests;
        }
        assert(delta_requests == total_requests);

        std::uint64_t inventory_requests = 0;
        for (const auto& aggregate : size_mc.rfq_inventory_aggregates) {
            assert(aggregate.tier == "RFQ size sampling");
            assert(aggregate.side == "bid" || aggregate.side == "ask");
            assert(aggregate.wins <= aggregate.requests);
            assert(aggregate.expected_wins >= -1e-12);
            assert(aggregate.expected_wins <= static_cast<double>(aggregate.requests) + 1e-12);
            inventory_requests += aggregate.requests;
        }
        assert(inventory_requests == total_requests);

        bool saw_bid = false;
        bool saw_ask = false;
        for (const auto& path : size_mc.sample_paths) {
            for (const auto& event : path.rfq_events) {
                saw_bid = saw_bid || event.side == "bid";
                saw_ask = saw_ask || event.side == "ask";
            }
        }
        assert(saw_bid && saw_ask);
    }

    // Native progress reporting is based on genuinely completed paths and must
    // always finish at exactly paths / paths.
    int progress_reports = 0;
    int progress_completed = 0;
    int progress_total = 0;
    auto progress_mc = MonteCarloSimulator(p,s,11.5).run(0.2, 37, 0.0, 0, 991, 0, 3,
        [&](int completed, int total) {
            ++progress_reports;
            assert(completed >= progress_completed);
            progress_completed = completed;
            progress_total = total;
        });
    assert(progress_mc.pnl_base_ccy.size() == 37);
    assert(progress_reports > 0);
    assert(progress_completed == 37);
    assert(progress_total == 37);

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

    // Retained paths must record every customer RFQ, not just wins, so realized
    // hit ratios can use the true RFQ denominator. Won RFQ events correspond
    // one-for-one with tier fills in this tiers-only problem.
    assert(!a.rfq_events.empty());
    std::size_t won_rfq_events = 0;
    for (const auto& event : a.rfq_events) {
        assert(event.size > 0.0);
        assert(event.win_probability >= 0.0 && event.win_probability <= 1.0);
        if (event.won) ++won_rfq_events;
    }
    assert(a.rfq_events.size() >= a.fills.size());
    assert(won_rfq_events == a.fills.size());

    // Passive ECN parent trades use the fixed discrete source-size pillars. A
    // hit partially fills our posted quote when the incoming parent trade is
    // smaller, and can never execute above the posted target-equivalent size.
    PricingProblem ecn_problem(
        grid(), .002, 0,
        QuadraticPenalty(.1,.002), InternalizationTime(4,.07,.0084), tiers,
        std::nullopt,
        PassiveECN(
            {0.5},
            AggregatedECNFlow(std::vector<ECNFlowSource>{
                ECNFlowSource("EURSEK", ExponentialFlow(100.0, 1.0), 1.0, 1.0,
                              std::vector<double>{0.20, 0.20, 0.20, 0.20, 0.20}),
            }),
            1.0, 0.0));
    auto ecn_solution = HowardSolver(ecn_problem).solve();
    assert(ecn_solution.diagnostics.converged);
    auto ecn_mc = MonteCarloSimulator(ecn_problem, ecn_solution, 11.5)
                      .run(0.5, 1, 0.0, 5.0, 424242, 1, 11);
    assert(ecn_mc.sample_paths.size() == 1);
    int ecn_fills = 0;
    bool saw_partial = false;
    for (const auto& fill : ecn_mc.sample_paths.front().fills) {
        if (fill.tier.rfind("Passive ECN", 0) != 0) continue;
        ++ecn_fills;
        assert(fill.size > 0.0);
        assert(fill.size <= 1.0 + 1e-12);
        if (fill.size < 0.999) saw_partial = true;
    }
    assert(ecn_fills > 0);
    assert(saw_partial);
    for (const auto& event : ecn_mc.sample_paths.front().ecn_arrivals) {
        const std::vector<double> pillars{1.0, 0.75, 0.50, 0.25, 0.10};
        assert(std::any_of(pillars.begin(), pillars.end(), [&](double x) {
            return std::abs(event.source_trade_size - x) < 1e-12;
        }));
        assert(event.target_fill_size >= 0.0);
        assert(event.target_fill_size <= 1.0 + 1e-12);
    }

    std::cout << "market_points=" << a.spots.size()
              << " trades=" << coarse.trade_count.front() << "\n";
    return 0;
}
