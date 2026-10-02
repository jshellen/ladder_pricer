#pragma once

#include "ladder_pricer/core.hpp"

#include <cstdint>
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

struct EcnArrivalEvent {
    double time_minutes = 0.0;
    std::string side;
    double reference_price = 0.0;
    double trade_distance_pips = 0.0;
    double trade_price = 0.0;
    double quote_depth_pips = 0.0;
    bool quote_active = false;
    bool won = false;
};

struct SamplePath {
    std::vector<double> times;
    std::vector<double> spots;
    std::vector<double> inventories;
    std::vector<double> cashes;
    std::vector<FillEvent> fills;
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
    std::vector<SamplePath> sample_paths;
};

class MonteCarloSimulator {
public:
    MonteCarloSimulator(const PricingProblem& problem, const Solution& solution,
                        double reference_spot)
        : problem_(problem), solution_(solution), reference_spot_(reference_spot) {}

    MonteCarloResult run(double horizon_minutes, int paths, double sigma,
                         double initial_inventory, std::uint64_t seed,
                         int retained_paths = 6, int sample_points = 191) const;

private:
    const PricingProblem& problem_;
    const Solution& solution_;
    double reference_spot_;
};

}  // namespace ladder_pricer
