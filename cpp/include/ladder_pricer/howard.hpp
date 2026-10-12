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

    struct IterationResult {
        JointPolicy policy;
        MarkoutResolvent markout;
        SolverDiagnostics diagnostics;
    };

    IterationResult solve_policy_iteration() const;
    Solution build_solution(IterationResult result) const;

    struct AffineOperator {
        std::vector<double> reward;
        DenseMatrix generator;
        bool with_reward = true;
        bool with_generator = true;
    };

    AffineOperator assemble_operator(const JointPolicy& policy,
                                     const MarkoutResolvent* markout,
                                     bool with_reward, bool with_generator) const;
    DenseMatrix build_generator(const JointPolicy& policy) const;
    std::vector<double> build_reward(const JointPolicy& policy,
                                     const MarkoutResolvent& markout) const;
    void add_rfq_contributions(AffineOperator& op, const Policy& policy,
                               std::size_t vol, std::size_t inventory,
                               const MarkoutResolvent* markout) const;
    void add_darkpool_contributions(AffineOperator& op, const Policy& policy,
                                    std::size_t vol, std::size_t inventory) const;
    void add_ecn_contributions(AffineOperator& op, const Policy& policy,
                               std::size_t vol, std::size_t inventory) const;
    void add_volatility_transitions(AffineOperator& op,
                                    std::size_t vol, std::size_t inventory) const;
    MarkoutResolvent markout_resolvent(const DenseMatrix& generator) const;
    std::pair<std::vector<double>, double> evaluate_policy(
        const DenseMatrix& generator, const std::vector<double>& reward) const;
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
