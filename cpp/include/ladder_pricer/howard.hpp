#pragma once

#include "ladder_pricer/core.hpp"
#include "ladder_pricer/matrix.hpp"

namespace ladder_pricer {

class HowardSolver {
public:
    explicit HowardSolver(PricingProblem problem) : problem_(std::move(problem)) {}

    const PricingProblem& problem() const noexcept { return problem_; }
    Solution solve() const;

private:
    using JointPolicy = std::vector<Policy>;  // one ordinary inventory policy per volatility state

    struct AffineOperator {
        std::vector<double> reward;
        DenseMatrix generator;
    };

    AffineOperator fixed_policy_operator(const JointPolicy& policy,
                                         const MarkoutResolvent* markout = nullptr) const;
    MarkoutResolvent markout_resolvent(const JointPolicy& policy) const;
    std::pair<std::vector<double>, double> evaluate_policy(
        const JointPolicy& policy, const MarkoutResolvent& markout) const;
    JointPolicy improve_policy(const std::vector<double>& value,
                               const JointPolicy& previous,
                               const MarkoutResolvent& markout) const;

    static double policy_change(const JointPolicy& lhs, const JointPolicy& rhs);
    static double markout_change(const MarkoutResolvent& lhs, const MarkoutResolvent& rhs);
    static double markout_scale(const MarkoutResolvent& markout);
    static double policy_scale(const JointPolicy& policy);

    PricingProblem problem_;
};

}  // namespace ladder_pricer
