#pragma once

#include "ladder_pricer/core.hpp"
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
    PnlAnalytics(const PricingProblem& problem, const Solution& solution, double reference_spot)
        : problem_(problem), solution_(solution), reference_spot_(reference_spot) {}

    PnlStatistics statistics(double horizon_minutes, double sigma,
                             double initial_inventory = 0.0) const;

private:
    const PricingProblem& problem_;
    const Solution& solution_;
    double reference_spot_;
};

}  // namespace ladder_pricer
