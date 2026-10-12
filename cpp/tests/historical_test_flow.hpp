#pragma once
// Test-only adapter for historical regression fixtures. The public C++ API
// accepts explicit RFQ distributions; no legacy Tier constructor exists.
#include "ladder_pricer/core.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

struct HistoricalLogisticFixture {
    double a0_, theta_, beta_, shift_, steepness_, volume_shift_;
    HistoricalLogisticFixture(double a0, double theta, double beta, double shift, double steepness, double volume_shift)
        : a0_(a0), theta_(theta), beta_(beta), shift_(shift), steepness_(steepness), volume_shift_(volume_shift) {}
    double a0() const { return a0_; }
    double theta() const { return theta_; }
    double beta() const { return beta_; }
    double shift() const { return shift_; }
    double steepness() const { return steepness_; }
    double volume_shift() const { return volume_shift_; }
    double win_probability(double delta, double size) const {
        return ladder_pricer::LogisticWinProbability(shift_, steepness_, volume_shift_).win_probability(delta, size);
    }
    double optimal_delta(double size, double spread, double additive_value) const {
        return ladder_pricer::LogisticWinProbability(shift_, steepness_, volume_shift_).optimal_delta(size, spread, additive_value);
    }
    double rfq_arrival_rate(double size) const {
        return a0_ * std::pow(size, -theta_ - beta_ * size);
    }
};
inline ladder_pricer::LogisticWinProbability historical_win(const HistoricalLogisticFixture& f) {
    return {f.shift(), f.steepness(), f.volume_shift()};
}
inline ladder_pricer::AggregatedFlow historical_test_flow(
    const HistoricalLogisticFixture& flow,
    const std::vector<double>& knots, double step = 1.0) {
    if (knots.empty() || !std::isfinite(step) || step <= 0)
        throw std::invalid_argument("invalid historical test RFQ grid");
    const double lo = knots.front(), hi = knots.back();
    std::vector<double> support;
    for (std::size_t i = 0;; ++i) {
        const double z = lo + static_cast<double>(i) * step;
        if (z > hi + 1e-10) break;
        support.push_back(std::min(z, hi));
    }
    if (support.empty() || std::abs(support.back() - hi) > 1e-10) support.push_back(hi);
    support.insert(support.end(), knots.begin(), knots.end());
    std::sort(support.begin(), support.end());
    support.erase(std::unique(support.begin(), support.end(), [](double a, double b){
        return std::abs(a-b) <= 1e-10;
    }), support.end());
    std::vector<double> weights;
    double total = 0.0;
    for (const double z : support) {
        const double w = flow.rfq_arrival_rate(z);
        if (!std::isfinite(w) || w < 0.0) throw std::runtime_error("invalid historical test RFQ rate");
        weights.push_back(w);
        total += w;
    }
    if (!(total > 0.0)) throw std::runtime_error("zero historical test RFQ rate");
    for (auto& w : weights) w /= total;
    return ladder_pricer::AggregatedFlow(std::vector<ladder_pricer::FlowSource>{
        ladder_pricer::FlowSource("direct", ladder_pricer::DiscreteRfqDistribution(
            2.0 * total, std::move(support), std::move(weights)),
            historical_win(flow), ladder_pricer::FXMapping{})});
}
