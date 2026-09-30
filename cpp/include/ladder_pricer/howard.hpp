#pragma once

#include "ladder_pricer/core.hpp"
#include "ladder_pricer/matrix.hpp"

namespace ladder_pricer {

class HowardSolver {
public:
    explicit HowardSolver(PricingProblem problem) : problem_(std::move(problem)) {}

    const PricingProblem& problem() const noexcept { return problem_; }
    Solution solve() const;
    Solution solve(const Policy& initial_policy) const;

private:
    struct AffineOperator {
        std::vector<double> reward;
        DenseMatrix generator;
    };

    AffineOperator fixed_policy_operator(const Policy& policy) const;
    std::pair<std::vector<double>, double> evaluate_policy(const Policy& policy) const;

    static double policy_change(const Policy& lhs, const Policy& rhs);
    static double policy_scale(const Policy& policy);
    Solution solve_impl(const Policy* initial_policy) const;

    PricingProblem problem_;
};

}  // namespace ladder_pricer
