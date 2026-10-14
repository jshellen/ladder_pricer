#include "ladder_pricer/expected_mtm.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ladder_pricer {

ExpectedMtm::ExpectedMtm(const PricingProblem& problem, const DenseMatrix& generator)
    : problem_(problem), weighted_inventories_(problem.tiers().size()) {
    const auto& states = problem.grid().states();
    const auto n = states.size();
    if (generator.rows() != n || generator.cols() != n) {
        throw std::invalid_argument("expected MTM generator/grid dimension mismatch");
    }
    // Reuse the resolvent when two tiers share a time constant.
    std::vector<std::pair<double, std::vector<double>>> solved;
    for (std::size_t k = 0; k < problem.tiers().size(); ++k) {
        const auto& tier = problem.tiers()[k];
        if (!tier.use_markout() || tier.markout().asymptotic() == 0.0) continue;
        const double tau = tier.markout().tau_minutes();
        bool cached = false;
        for (const auto& entry : solved) {
            if (entry.first == tau) {
                weighted_inventories_[k] = entry.second;
                cached = true;
                break;
            }
        }
        if (cached) continue;

        DenseMatrix A(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                A(i, j) = (i == j ? 1.0 : 0.0) - tau * generator(i, j);
            }
        }
        auto u = A.solve(states);
        for (double x : u) {
            if (!std::isfinite(x)) throw std::runtime_error("nonfinite expected MTM inventory");
        }
        solved.emplace_back(tau, u);
        weighted_inventories_[k] = std::move(u);
    }
}

ExpectedMtm::ExpectedMtm(const PricingProblem& problem,
                         std::vector<std::vector<double>> weighted_inventories)
    : problem_(problem), weighted_inventories_(std::move(weighted_inventories)) {
    if (weighted_inventories_.size() != problem_.tiers().size())
        throw std::invalid_argument("expected MTM tier count mismatch");
    for (const auto& values : weighted_inventories_) {
        if (!values.empty() && values.size() != problem_.grid().states().size())
            throw std::invalid_argument("expected MTM regime inventory grid mismatch");
    }
}

double ExpectedMtm::weighted_inventory(std::size_t tier, double inventory) const {
    if (weighted_inventories_.at(tier).empty()) return 0.0;
    return problem_.grid().interpolate(weighted_inventories_[tier], inventory);
}

double ExpectedMtm::trade_mtm(std::size_t tier, double direction, double inventory_after) const {
    const auto& t = problem_.tiers().at(tier);
    if (!t.use_markout()) return 0.0;
    return -direction * t.markout().asymptotic() * weighted_inventory(tier, inventory_after);
}

double ExpectedMtm::win_increment(std::size_t tier, double direction,
                                   double inventory_before, double inventory_after) const {
    return trade_mtm(tier, direction, inventory_after) -
           trade_mtm(tier, direction, inventory_before);
}

}  // namespace ladder_pricer
