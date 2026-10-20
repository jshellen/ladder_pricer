#pragma once
// Interpolate a continuously evolving Heston/CIR variance on the HJB variance grid.
// Weights are computed in VARIANCE (sigma^2), never in sigma.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ladder_pricer {
struct VarianceBracket {
    std::size_t lower = 0;
    std::size_t upper = 0;
    double upper_weight = 0.0;

    double blend(double left, double right) const noexcept {
        return left + upper_weight * (right - left);
    }
    // Randomized selection for discrete actions (e.g. dark-pool posted size).
    // The expectation of the chosen action is linearly interpolated.
    std::size_t choose(double uniform_draw) const noexcept {
        return upper_weight > uniform_draw ? upper : lower;
    }
};

inline VarianceBracket bracket_variance(const std::vector<double>& sigma_pips, double variance) {
    if (sigma_pips.empty() || !std::isfinite(variance))
        throw std::invalid_argument("invalid variance or empty Heston grid");
    if (variance <= sigma_pips.front()*sigma_pips.front()) return {0, 0, 0.0};
    const std::size_t last = sigma_pips.size() - 1;
    if (variance >= sigma_pips[last]*sigma_pips[last]) return {last, last, 0.0};
    const auto it = std::lower_bound(sigma_pips.begin(), sigma_pips.end(), std::sqrt(variance));
    const std::size_t upper = static_cast<std::size_t>(it - sigma_pips.begin());
    const std::size_t lower = upper - 1;
    const double vl = sigma_pips[lower]*sigma_pips[lower];
    const double vu = sigma_pips[upper]*sigma_pips[upper];
    return {lower, upper, std::clamp((variance - vl)/(vu - vl), 0.0, 1.0)};
}
} // namespace ladder_pricer
