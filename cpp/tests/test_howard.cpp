#include "ladder_pricer/howard.hpp"

#include <algorithm>
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
    {
        PassiveECN ecn({-0.10, 0.0, 0.25, 0.50},
            ExponentialFlow(0.15, 2.4), 1.0, 0.0);
        assert(ecn.side_allowed(2.0, Side::Ask));
        assert(!ecn.side_allowed(2.0, Side::Bid));
        assert(ecn.side_allowed(-2.0, Side::Bid));
        assert(!ecn.side_allowed(-2.0, Side::Ask));
        assert(!ecn.side_allowed(0.0, Side::Bid));
        assert(!ecn.side_allowed(0.0, Side::Ask));
        assert(ecn.risk_reducing(2.0, Side::Ask, 1.0));
        assert(!ecn.risk_reducing(0.5, Side::Ask, 1.0));  // cannot cross through flat
        assert(ecn.risk_reducing(-2.0, Side::Bid, 1.0));
        assert(!ecn.risk_reducing(-0.5, Side::Bid, 1.0));
        // ECN uses the tier delta convention: d=0 touch, d=0.5 mid, with
        // lambda(d)=A exp(-k(0.5-d)).  A is therefore the intensity at mid.
        assert(std::abs(ecn.flow().arrival_rate(0.5) - 0.15) < 1e-15);
        assert(ecn.flow().arrival_rate(0.25) < ecn.flow().arrival_rate(0.5));
        assert(ecn.flow().arrival_rate(0.0) < ecn.flow().arrival_rate(0.25));
        assert(std::abs(ecn.flow().arrival_rate(0.0) - 0.15 * std::exp(-1.2)) < 1e-15);
    }
    {
        const double cross_mid = 1.20;
        const double target_spread = 20.0;
        const double source_spread = 25.0;
        const double alpha = target_spread / (cross_mid * source_spread);
        ECNFlowSource direct("EURSEK", ExponentialFlow(0.15, 2.4), 1.0, 1.0);
        ECNFlowSource crossed("USDSEK", ExponentialFlow(0.20, 3.0), alpha, cross_mid);
        assert(std::abs(crossed.implied_delta(0.30) -
                        (0.5 + alpha * (0.30 - 0.5))) < 1e-15);
        assert(std::abs(crossed.target_reach(alpha * 0.25) - 0.25) < 1e-15);
        AggregatedECNFlow combined(std::vector<ECNFlowSource>{direct, crossed});
        const double expected = direct.arrival_rate(0.30) + crossed.arrival_rate(0.30);
        assert(std::abs(combined.arrival_rate(0.30) - expected) < 1e-15);
        assert(crossed.arrival_rate(0.30) > 0.0);
    }
    {
        std::vector<double> deltas;
        for (int i = 0; i <= 50; ++i) deltas.push_back(0.01 * i);
        const std::vector<double> ecn_sizes{1,2,3,5,10,20};
        std::vector<Tier> ecn_tiers;
        ecn_tiers.emplace_back("Tier 1", ecn_sizes,
            LogisticFlow(0.0155,0.144,0.0857,0.52,8.42,0.026),
            SaturatingMarkout(1e-4,0.5,0.5), true, -10, 100);
        ecn_tiers.emplace_back("Tier 2", ecn_sizes,
            LogisticFlow(0.0232,0.303,0.122,0.48,2.86,0.02),
            SaturatingMarkout(1e-4,0.5,0.5), true, -10, 100);
        PricingProblem ecn_problem(
            grid(), 0.002, 0.0,
            QuadraticPenalty(0.1, 0.002),
            InternalizationTime(4.0, 0.070, 0.0084),
            std::move(ecn_tiers), std::nullopt,
            PassiveECN(
                deltas,
                AggregatedECNFlow(std::vector<ECNFlowSource>{
                    ECNFlowSource("EURSEK", ExponentialFlow(2.5, 8.4), 1.0, 1.0),
                    ECNFlowSource("USDSEK", ExponentialFlow(1.5, 7.0), 0.75, 1.18),
                }),
                1.0, 3e-4));
        const auto ecn_solution = HowardSolver(std::move(ecn_problem)).solve();
        assert(ecn_solution.diagnostics.converged);
        assert(ecn_solution.passive_ecn_policy.has_value());
        const auto& policy = *ecn_solution.passive_ecn_policy;
        auto at = [&](double q) {
            const auto it = std::find_if(policy.q_grid.begin(), policy.q_grid.end(),
                [&](double x) { return std::abs(x - q) < 1e-12; });
            assert(it != policy.q_grid.end());
            return static_cast<std::size_t>(std::distance(policy.q_grid.begin(), it));
        };
        const std::size_t pos = at(2.0);
        const std::size_t neg = at(-2.0);
        const std::size_t zero = at(0.0);
        assert(policy.ask_active[pos]);
        assert(!policy.bid_active[pos]);
        assert(policy.bid_active[neg]);
        assert(!policy.ask_active[neg]);
        assert(!policy.bid_active[zero] && !policy.ask_active[zero]);
        assert(policy.ask_delta[pos] >= 0.0 && policy.ask_delta[pos] <= 0.5);
        assert(policy.bid_delta[neg] >= 0.0 && policy.bid_delta[neg] <= 0.5);
    }
    {
        auto arrivals = std::make_shared<ZeroInflatedPoissonArrival>(0.0012, 2.9, 0.7172);
        DarkPool dark(arrivals, 3e-4, {1.0, 2.0, 3.0, 4.0, 5.0});
        assert(dark.side_allowed(2.0, Side::Ask));
        assert(!dark.side_allowed(2.0, Side::Bid));
        assert(dark.side_allowed(-2.0, Side::Bid));
        assert(!dark.side_allowed(-2.0, Side::Ask));
        assert(!dark.side_allowed(0.0, Side::Bid));
        assert(!dark.side_allowed(0.0, Side::Ask));
        assert(dark.risk_reducing(2.0, Side::Ask, 2.0));
        assert(!dark.risk_reducing(2.0, Side::Ask, 3.0));
        assert(dark.risk_reducing(-2.0, Side::Bid, 2.0));
        assert(!dark.risk_reducing(-2.0, Side::Bid, 3.0));
        assert(std::abs(dark.fee(Side::Bid) - dark.fee(Side::Ask)) < 1e-15);
        assert(std::abs(dark.arrivals(Side::Bid).arrival_intensity() -
                        dark.arrivals(Side::Ask).arrival_intensity()) < 1e-15);
    }
    {
        Tier fee_tier("Fee tier", {1.0},
            LogisticFlow(0.1,0.1,0.1,0.5,5.0,0.0),
            SaturatingMarkout(0.0,0.5,0.5), false, -10.0, 100.0, 2e-4);
        assert(std::abs(fee_tier.fee() - 2e-4) < 1e-15);
    }
    {
        const LogisticFlow eur_flow(0.10, 0.2, 0.01, 0.50, 8.0, 0.0);
        const LogisticFlow usd_flow(0.20, 0.2, 0.01, 0.50, 6.0, 0.0);
        const double cross_mid = 1.20;
        const double target_spread = 20.0;
        const double source_spread = 25.0;
        const double delta_scale = target_spread / (cross_mid * source_spread);
        FlowSource eur("EURSEK", eur_flow, 1.0, 1.0);
        FlowSource usd("USDSEK", usd_flow, delta_scale, cross_mid);
        assert(std::abs(usd.implied_delta(0.30) -
                        (0.5 + delta_scale * (0.30 - 0.5))) < 1e-15);
        assert(std::abs(usd.source_size(2.0) - 2.4) < 1e-15);
        AggregatedFlow combined(std::vector<FlowSource>{eur, usd});
        const double expected = eur.arrival_rate(0.30, 2.0) + usd.arrival_rate(0.30, 2.0);
        assert(std::abs(combined.arrival_rate(0.30, 2.0) - expected) < 1e-15);
        const double rfq = eur.rfq_arrival_rate(2.0) + usd.rfq_arrival_rate(2.0);
        assert(std::abs(combined.rfq_arrival_rate(2.0) - rfq) < 1e-15);
        assert(std::abs(combined.win_probability(0.30, 2.0) - expected / rfq) < 1e-15);
        const double d = combined.optimal_delta(2.0, 0.002, 0.0, -10.0, 100.0);
        assert(std::isfinite(d));
        assert(d >= -10.0 && d <= 100.0);
        auto objective = [&](double x) {
            return combined.arrival_rate(x, 2.0) * (2.0 * 0.002 * (0.5 - x));
        };
        double brute_best = -10.0;
        for (int i = 0; i <= 20000; ++i) {
            const double x = -10.0 + 10.5 * static_cast<double>(i) / 20000.0;
            if (objective(x) > objective(brute_best)) brute_best = x;
        }
        assert(objective(d) + 1e-10 >= objective(brute_best));
    }
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
