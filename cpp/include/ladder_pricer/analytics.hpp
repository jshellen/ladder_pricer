#pragma once

#include "ladder_pricer/solver.hpp"
#include "ladder_pricer/matrix.hpp"

namespace ladder_pricer {

struct PnlStatistics {
    double horizon_minutes = 0.0;
    double initial_inventory = 0.0;
    double reference_spot = 0.0;
    double expected_quote_ccy = 0.0;
    double std_quote_ccy = 0.0;
    double expected_base_ccy = 0.0;
    double std_base_ccy = 0.0;
};

class PnlAnalytics {
public:
    PnlAnalytics(const PricingProblem& problem, const SolverResult& solution,
                 const std::vector<std::vector<double>>& switches, double reference_spot)
        : problem_(problem), solution_(solution), switches_(switches), reference_spot_(reference_spot) {}

    PnlStatistics statistics(double horizon_minutes, double initial_inventory = 0.0,
                             int initial_regime = -1) const;
    // First passage to zero inventory, switching regimes while waiting.
    // Returns a value per operational inventory node for the selected initial regime.
    std::vector<double> internalization_times(int initial_regime = -1) const;

private:
    const PricingProblem& problem_;
    const SolverResult& solution_;
    const std::vector<std::vector<double>>& switches_;
    double reference_spot_;
};

}  // namespace ladder_pricer
