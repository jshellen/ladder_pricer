#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hjb {

using Matrix = std::vector<std::vector<double>>;
using OptionalDeltaRow = std::optional<std::reference_wrapper<const std::vector<double>>>;

inline double clamp(double x, double lo, double hi) {
    return std::min(std::max(x, lo), hi);
}

inline void validate_strictly_increasing(
    const std::vector<double>& x,
    const std::string& name
) {
    if (x.empty()) {
        throw std::invalid_argument(name + " cannot be empty.");
    }
    for (std::size_t i = 1; i < x.size(); ++i) {
        if (x[i] <= x[i - 1]) {
            throw std::invalid_argument(name + " must be strictly increasing.");
        }
    }
}

inline void validate_positive_strictly_increasing(
    const std::vector<double>& x,
    const std::string& name
) {
    validate_strictly_increasing(x, name);
    for (double v : x) {
        if (v <= 0.0) {
            throw std::invalid_argument(name + " must contain positive values only.");
        }
    }
}

inline void validate_centered_symmetric_grid(
    const std::vector<double>& x,
    const std::string& name,
    double tol = 1e-10
) {
    validate_strictly_increasing(x, name);

    if (x.size() < 3) {
        throw std::invalid_argument(name + " must contain at least 3 points.");
    }
    if (x.size() % 2 == 0) {
        throw std::invalid_argument(
            name + " must have odd length so that 0 is exactly in the middle."
        );
    }

    const std::size_t mid = x.size() / 2;
    if (std::abs(x[mid]) > tol) {
        throw std::invalid_argument(name + " must contain 0 exactly at the middle index.");
    }

    for (std::size_t i = 0; i < mid; ++i) {
        if (std::abs(x[i] + x[x.size() - 1 - i]) > tol) {
            throw std::invalid_argument(name + " must be symmetric around 0.");
        }
    }
}

inline std::pair<std::size_t, double> locate_segment_with_weight(
    const std::vector<double>& grid,
    double x
) {
    if (grid.empty()) {
        throw std::invalid_argument("locate_segment_with_weight: grid cannot be empty.");
    }
    if (grid.size() == 1) {
        return {0, 0.0};
    }

    if (x <= grid.front()) {
        const double x0 = grid[0];
        const double x1 = grid[1];
        return {0, (x - x0) / (x1 - x0)};
    }

    if (x >= grid.back()) {
        const std::size_t i = grid.size() - 2;
        const double x0 = grid[i];
        const double x1 = grid[i + 1];
        return {i, (x - x0) / (x1 - x0)};
    }

    const auto it = std::upper_bound(grid.begin(), grid.end(), x);
    const std::size_t i = static_cast<std::size_t>(it - grid.begin() - 1);
    const double x0 = grid[i];
    const double x1 = grid[i + 1];
    return {i, (x - x0) / (x1 - x0)};
}

inline double interp_linear(
    const std::vector<double>& grid,
    const std::vector<double>& vals,
    double x
) {
    if (grid.size() != vals.size()) {
        throw std::invalid_argument("interp_linear: grid and vals size mismatch.");
    }
    if (grid.empty()) {
        throw std::invalid_argument("interp_linear: empty input.");
    }
    if (grid.size() == 1) {
        return vals.front();
    }

    const auto [i, t] = locate_segment_with_weight(grid, x);
    return vals[i] + t * (vals[i + 1] - vals[i]);
}

enum class Side : unsigned char { Bid, Ask };

} // namespace hjb
