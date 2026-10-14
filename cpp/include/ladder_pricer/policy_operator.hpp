#pragma once
#include "ladder_pricer/core.hpp"
#include "ladder_pricer/matrix.hpp"

namespace ladder_pricer {

// Operator for a fixed inventory policy in a single volatility regime.
// The joint Solver assembles these regime blocks and their transition rates.
struct AffineOperator {
    std::vector<double> reward;
    DenseMatrix generator;
};
AffineOperator fixed_policy_operator(const PricingProblem& problem, const Policy& policy);

} // namespace ladder_pricer
