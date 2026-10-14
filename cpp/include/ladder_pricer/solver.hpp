#pragma once
#include "ladder_pricer/policy_operator.hpp"
#include "ladder_pricer/expected_mtm.hpp"
#include <vector>

namespace ladder_pricer {

// Joint finite-state Markov chain (inventory q, volatility regime v).
// Volatility switching has exogenous row-generator K (1/minute) and does NOT
// execute an inventory trade. Each regime can choose its own RFQ/ECN policy.
struct SolverResult {
    std::vector<double> q_grid;
    std::vector<double> sigma_pips;
    std::vector<std::vector<double>> values;
    std::vector<Policy> policies; // full hidden inventory grid, one per regime
    std::vector<std::vector<double>> weighted_inventory; // RFQ-tier resolvents over all (regime,q) states
    double average_reward = 0.0;
    int iterations = 0;
    bool converged = false;
    double evaluation_residual = 0.0;
};

class Solver {
public:
    Solver(const PricingProblem& base,
                     std::vector<double> sigma_pips,
                     std::vector<std::vector<double>> switch_rates);
    SolverResult solve() const;

private:
    struct Evaluation {
        std::vector<double> value;
        double rho = 0.0;
        std::vector<std::vector<double>> weighted_inventory; // tier-major, joint states
        double residual = 0.0;
    };
    Evaluation evaluate(const std::vector<Policy>& policies) const;
    std::vector<ExpectedMtm> mtm_views(const Evaluation& e) const;

    std::vector<PricingProblem> problems_;
    std::vector<double> sigmas_;
    std::vector<std::vector<double>> switch_rates_;
};
}  // namespace ladder_pricer
