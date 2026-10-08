#pragma once

#include "ladder_pricer/core.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ladder_pricer {

struct FillEvent {
    double time_minutes = 0.0;
    std::string tier;
    std::string side;
    double size = 0.0;
    double execution_price = 0.0;
    double inventory_before = 0.0;
    double inventory_after = 0.0;
};

struct RfqEvent {
    double time_minutes = 0.0;
    std::string tier;
    std::string side;
    double size = 0.0;
    double delta = 0.0;
    double win_probability = 0.0;
    bool won = false;
};

struct EcnArrivalEvent {
    double time_minutes = 0.0;
    std::string source;
    std::string side;
    double reference_price = 0.0;
    double trade_distance_pips = 0.0;
    double trade_price = 0.0;
    double quote_depth_pips = 0.0;
    double source_trade_size = 0.0;
    double target_fill_size = 0.0;
    bool quote_active = false;
    bool won = false;
};

// Compact Monte Carlo population summaries. These are accumulated over every
// simulated path so long-run diagnostics do not depend on the small retained
// path sample used for interactive path inspection.
struct FillAggregate {
    std::string tier;
    std::string side;
    std::uint64_t trade_count = 0;
    double volume = 0.0;
};

struct RfqAggregate {
    double size = 0.0;
    std::uint64_t wins = 0;
    std::uint64_t requests = 0;
};

// Tier-specific RFQ summaries used to validate the Monte Carlo sampling
// against the HJB/customer-flow assumptions.  These are population aggregates
// over every simulated path (not only the retained paths shown in the UI).
struct RfqTierSizeAggregate {
    std::string tier;
    double size = 0.0;
    std::uint64_t wins = 0;
    std::uint64_t requests = 0;
};

struct RfqDeltaAggregate {
    std::string tier;
    double size = 0.0;
    double delta = 0.0;  // mean quoted delta in this compact 1%-wide bucket
    std::uint64_t wins = 0;
    std::uint64_t requests = 0;
    std::uint64_t admissible_requests = 0;
    double expected_wins = 0.0;  // sum of Bernoulli probabilities used by MC
};

struct RfqInventoryAggregate {
    std::string tier;
    std::string side;
    double size = 0.0;
    double inventory = 0.0;  // inventory-grid bucket used for the diagnostic
    std::uint64_t wins = 0;
    std::uint64_t requests = 0;
    double expected_wins = 0.0;  // sum of Bernoulli probabilities used by MC
};

struct SamplePath {
    std::vector<double> times;
    std::vector<double> spots;
    std::vector<double> inventories;
    std::vector<double> cashes;
    std::vector<int> volatility_states;
    std::vector<FillEvent> fills;
    std::vector<RfqEvent> rfq_events;
    std::vector<EcnArrivalEvent> ecn_arrivals;
};

struct MonteCarloResult {
    std::vector<double> pnl_quote_ccy;
    std::vector<double> pnl_base_ccy;
    std::vector<double> final_inventory;
    std::vector<double> final_spot;
    std::vector<int> trade_count;

    std::vector<double> sample_times;
    std::vector<double> inventory_lower;
    std::vector<double> inventory_median;
    std::vector<double> inventory_upper;
    std::vector<FillAggregate> fill_aggregates;
    std::vector<RfqAggregate> rfq_aggregates;
    std::vector<RfqTierSizeAggregate> rfq_tier_size_aggregates;
    std::vector<RfqDeltaAggregate> rfq_delta_aggregates;
    std::vector<RfqInventoryAggregate> rfq_inventory_aggregates;
    std::vector<SamplePath> sample_paths;
};

class MonteCarloSimulator {
public:
    MonteCarloSimulator(const PricingProblem& problem, const Solution& solution,
                        double reference_spot)
        : problem_(problem), solution_(solution), reference_spot_(reference_spot) {}

    MonteCarloResult run(double horizon_minutes, int paths, double sigma,
                         double initial_inventory, std::uint64_t seed,
                         int retained_paths = 6, int sample_points = 191,
                         const std::function<void(int, int)>& progress = {}) const;

private:
    const PricingProblem& problem_;
    const Solution& solution_;
    double reference_spot_;
};

}  // namespace ladder_pricer
