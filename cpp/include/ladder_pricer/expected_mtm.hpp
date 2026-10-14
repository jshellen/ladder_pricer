#pragma once

#include "ladder_pricer/core.hpp"
#include "ladder_pricer/matrix.hpp"

#include <vector>

namespace ladder_pricer {

// Under a FIXED inventory policy, u_tau(q)=E_q[Q_E], E~Exp(1/tau).
// Each tier's adverse RFQ price shock decays with its own time constant.
// The price impact is created for ALL customer RFQs, including lost requests.
// With symmetric exogenous bid/ask arrivals, the price impact of an unfilled
// baseline cancels in expected instantaneous reward. The incremental payoff
// of winning (vs losing) one request is therefore
//     -direction * markout_asymptote * (u(q_after) - u(q_before)).
// This is distinct from the GROSS MTM of a winning RFQ, which is
//     -direction * markout_asymptote * u(q_after).
class ExpectedMtm {
public:
    ExpectedMtm(const PricingProblem& problem, const DenseMatrix& generator);
    // Build a regime-specific MTM view from the JOINT regime/inventory resolvents.
    ExpectedMtm(const PricingProblem& problem, std::vector<std::vector<double>> weighted_inventories);

    double weighted_inventory(std::size_t tier, double inventory) const;
    double trade_mtm(std::size_t tier, double direction, double inventory_after) const;
    double win_increment(std::size_t tier, double direction,
                         double inventory_before, double inventory_after) const;

private:
    const PricingProblem& problem_;
    std::vector<std::vector<double>> weighted_inventories_;
};

}  // namespace ladder_pricer
