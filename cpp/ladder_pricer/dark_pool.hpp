#pragma once

#include "common.hpp"
#include "policy.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace hjb {

struct DarkPoolVenue {
    double lambda_bid{0.0};
    double lambda_ask{0.0};
    double p_bid{0.5};
    double p_ask{0.5};
    double fee_per_unit_bid{0.0};
    double fee_per_unit_ask{0.0};
    std::vector<double> posted_sizes{1.0, 2.0, 3.0};
    bool allow_both_sides{false};
    double min_fill_value{0.0};
    DarkPoolPolicy policy;

    DarkPoolVenue() = default;

    DarkPoolVenue(
        double lambda_bid_,
        double lambda_ask_,
        double p_bid_,
        double p_ask_,
        double fee_per_unit_bid_,
        double fee_per_unit_ask_,
        std::vector<double> posted_sizes_,
        bool allow_both_sides_ = false,
        double min_fill_value_ = 0.0
    )
        : lambda_bid(lambda_bid_),
          lambda_ask(lambda_ask_),
          p_bid(p_bid_),
          p_ask(p_ask_),
          fee_per_unit_bid(fee_per_unit_bid_),
          fee_per_unit_ask(fee_per_unit_ask_),
          posted_sizes(std::move(posted_sizes_)),
          allow_both_sides(allow_both_sides_),
          min_fill_value(min_fill_value_) {
        validate();
    }

    static bool is_integer_like(double x, double tol = 1e-10) {
        return std::abs(x - std::round(x)) <= tol;
    }

    void validate() const {
        if (lambda_bid < 0.0 || lambda_ask < 0.0) {
            throw std::invalid_argument("DarkPoolVenue: arrival intensities must be nonnegative.");
        }
        if (!(p_bid > 0.0 && p_bid <= 1.0) || !(p_ask > 0.0 && p_ask <= 1.0)) {
            throw std::invalid_argument("DarkPoolVenue: geometric probabilities must lie in (0, 1].");
        }
        validate_positive_strictly_increasing(posted_sizes, "DarkPoolVenue.posted_sizes");
        for (double u : posted_sizes) {
            if (!is_integer_like(u)) {
                throw std::invalid_argument(
                    "DarkPoolVenue: posted_sizes must be integer-valued because arrival sizes are geometric in unit chunks."
                );
            }
        }
    }

    void reset_policy_shape(const std::vector<double>& q_grid) {
        policy.reset_shape(q_grid);
    }

    bool is_admissible(double q, Side side, double tol = 1e-12) const {
        if (allow_both_sides) {
            return true;
        }
        return (q > tol && side == Side::Ask) || (q < -tol && side == Side::Bid);
    }
};

} // namespace hjb
