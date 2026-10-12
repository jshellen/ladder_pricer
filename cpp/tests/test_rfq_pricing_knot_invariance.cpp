#include "historical_test_flow.hpp"
#include "ladder_pricer/core.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace ladder_pricer;

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double lhs, double rhs, const char* message) {
    check(std::abs(lhs - rhs) < 1e-12, message);
}

// Test the actual Tier construction path without rebuilding the RFQ distribution.
Tier make_tier(const std::vector<double>& pricing_knots, double rfq_size_step) {
    const HistoricalLogisticFixture win_curve(0.0155, 0.144, 0.0857, 0.52, 8.42, 0.026);
    std::vector<FlowSource> sources;
    sources.emplace_back("EURSEK", DiscreteRfqDistribution(12.0, {1., 2., 3., 5., 10., 20.},
                               {0.20, 0.25, 0.15, 0.20, 0.12, 0.08}), historical_win(win_curve));
    // A crossed source, with distinct native sizes and non-unit FX size scale.
    sources.emplace_back("USDSEK", DiscreteRfqDistribution(7.0, {1.25, 2.5, 6.25, 12.5, 25.},
                               {0.10, 0.20, 0.30, 0.25, 0.15}), historical_win(win_curve), FXMapping(0.9, 1.25));
    return Tier("Tier 1", pricing_knots, AggregatedFlow(std::move(sources)),
                SaturatingMarkout(0.0001, 0.5), true, -1., 1., 0., rfq_size_step);
}

void compare_source(const FlowSource& a, const FlowSource& b) {
    check(a.name() == b.name(), "source identity changed");
    close(a.total_rfq_rate(), b.total_rfq_rate(), "total RFQ arrival rate changed");
    const auto& da = a.rfq_distribution();
    const auto& db = b.rfq_distribution();
    check(da.source_sizes() == db.source_sizes(), "source-native RFQ sizes changed");
    check(da.probabilities() == db.probabilities(), "RFQ size PMF changed");

    const auto aa = a.rfq_atoms();
    const auto bb = b.rfq_atoms();
    check(aa.size() == bb.size(), "HJB RFQ atom count changed");
    double side_rate_sum = 0.;
    double expected_size_a = 0., expected_size_b = 0.;
    for (std::size_t i = 0; i < aa.size(); ++i) {
        close(aa[i].target_size, bb[i].target_size, "HJB RFQ atom size changed");
        close(aa[i].one_side_arrival_rate, bb[i].one_side_arrival_rate,
              "HJB RFQ atom arrival intensity changed");
        close(a.rfq_arrival_rate(aa[i].target_size),
              b.rfq_arrival_rate(bb[i].target_size), "size-specific RFQ rate changed");
        side_rate_sum += aa[i].one_side_arrival_rate;
        expected_size_a += aa[i].target_size * da.probabilities()[i];
        expected_size_b += bb[i].target_size * db.probabilities()[i];
    }
    close(2. * side_rate_sum, a.total_rfq_rate(), "HJB atom rates do not sum to Lambda");
    close(expected_size_a, expected_size_b, "expected RFQ size changed");

    // Monte Carlo calls sample_rfq_target_size(event_rng) before win/loss.
    // With identical RNGs, every pre-win RFQ draw must agree, including sizes
    // lying BETWEEN pricing knots, not merely average histogram frequencies.
    std::mt19937_64 rng_a(20261009), rng_b(20261009);
    std::vector<std::uint64_t> counts(aa.size(), 0);
    constexpr int draws = 20000;
    for (int n = 0; n < draws; ++n) {
        const double za = a.sample_rfq_target_size(rng_a);
        const double zb = b.sample_rfq_target_size(rng_b);
        close(za, zb, "Monte Carlo pre-win sampled RFQ size changed");
        bool found = false;
        for (std::size_t i = 0; i < aa.size(); ++i) {
            if (std::abs(za - aa[i].target_size) < 1e-12) {
                ++counts[i]; found = true; break;
            }
        }
        check(found, "sampled size not in configured exogenous RFQ support");
    }
    for (std::size_t i = 0; i < aa.size(); ++i) {
        const double observed = static_cast<double>(counts[i]) / draws;
        const double p = da.probabilities()[i];
        check(std::abs(observed - p) < 0.025, "sample frequencies disagree with RFQ PMF");
    }
}
} // namespace

int main() {
    try {
        const auto sparse = make_tier({1., 5., 10., 20.}, 1.);
        const auto dense = make_tier({1., 2., 3., 5., 7., 10., 15., 20.}, .25);
        check(sparse.sizes() != dense.sizes(), "test did not vary pricing knots");
        check(sparse.rfq_sizes() != dense.rfq_sizes(), "test did not vary legacy pricing support");
        const auto& sa = sparse.flow().sources();
        const auto& sb = dense.flow().sources();
        check(sa.size() == 2 && sb.size() == 2, "expected two independent flow sources");
        for (std::size_t i = 0; i < sa.size(); ++i) compare_source(sa[i], sb[i]);
        std::cout << "RFQ pricing-knot invariance passed: 2 flow sources, "
                     "sparse/dense grids, exact HJB atoms and 20,000 paired MC size draws per source\n";
    } catch (const std::exception& e) {
        std::cerr << "RFQ pricing-knot invariance FAILED: " << e.what() << '\n';
        return 1;
    }
}
