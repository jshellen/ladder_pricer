#pragma once
// Data-only compatibility structs for unlinked pre-regime analytic/MC source.
// The sole production HJB optimizer is Solver (solver.hpp).
#include "ladder_pricer/core.hpp"

namespace ladder_pricer {
struct SolverDiagnostics {
    bool converged = false;
    int iterations = 0;
    double average_reward = 0.0;
    double reward_lower_bound = 0.0;
    double reward_upper_bound = 0.0;
    std::vector<double> value_change;
    std::vector<double> bellman_residual;
};

struct Solution {
    std::vector<double> q_grid;
    std::vector<double> value;
    std::vector<double> solve_q_grid;
    std::vector<double> value_solve;
    // Operational policies are convenient for pricing/UI use.
    std::vector<LadderPolicy> tier_policies;
    std::optional<DarkPoolPolicy> dark_pool_policy;
    std::optional<PassiveECNPolicy> passive_ecn_policy;

    // Full hidden-grid policies are retained for simulation and diagnostics.
    std::vector<LadderPolicy> solve_tier_policies;
    // Expected gross MTM of won RFQs and incremental won-vs-lost MTM.
    // Values are in millions of quote currency; pybind converts to EUR.
    std::vector<LadderPolicy> gross_mtm_tiers;
    std::vector<LadderPolicy> incremental_mtm_tiers;
    std::optional<DarkPoolPolicy> solve_dark_pool_policy;
    std::optional<PassiveECNPolicy> solve_passive_ecn_policy;
    double average_reward = 0.0;
    double hard_inventory_limit = 0.0;
    SolverDiagnostics diagnostics;
};

} // namespace ladder_pricer
