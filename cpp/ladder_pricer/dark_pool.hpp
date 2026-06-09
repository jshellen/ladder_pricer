#pragma once

#include "arrival_distributions.hpp"
#include "common.hpp"
#include "policy.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hjb {

struct DarkPoolVenue {
    std::shared_ptr<ArrivalDistribution> dist_bid;
    std::shared_ptr<ArrivalDistribution> dist_ask;
    double fee_per_unit_bid{0.0};
    double fee_per_unit_ask{0.0};
    std::vector<double> posted_sizes{1.0, 2.0, 3.0};
    bool allow_both_sides{false};
    double min_fill_value{0.0};
    DarkPoolPolicy policy;

    DarkPoolVenue() = default;

    DarkPoolVenue(DarkPoolVenue&&) = default;
    DarkPoolVenue& operator=(DarkPoolVenue&&) = default;
    DarkPoolVenue(const DarkPoolVenue&) = default;
    DarkPoolVenue& operator=(const DarkPoolVenue&) = default;

    /// Constructor with geometric arrival distributions (legacy compatibility).
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
        : dist_bid(std::make_shared<GeometricArrivalDist>(lambda_bid_, p_bid_)),
          dist_ask(std::make_shared<GeometricArrivalDist>(lambda_ask_, p_ask_)),
          fee_per_unit_bid(fee_per_unit_bid_),
          fee_per_unit_ask(fee_per_unit_ask_),
          posted_sizes(std::move(posted_sizes_)),
          allow_both_sides(allow_both_sides_),
          min_fill_value(min_fill_value_) {
        validate();
    }

    /// Constructor with explicit arrival distributions.
    DarkPoolVenue(
        std::shared_ptr<ArrivalDistribution> dist_bid_,
        std::shared_ptr<ArrivalDistribution> dist_ask_,
        double fee_per_unit_bid_,
        double fee_per_unit_ask_,
        std::vector<double> posted_sizes_,
        bool allow_both_sides_ = false,
        double min_fill_value_ = 0.0
    )
        : dist_bid(std::move(dist_bid_)),
          dist_ask(std::move(dist_ask_)),
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
        if (!dist_bid || !dist_ask) {
            throw std::invalid_argument("DarkPoolVenue: dist_bid and dist_ask must be set.");
        }
        dist_bid->validate();
        dist_ask->validate();
        validate_positive_strictly_increasing(posted_sizes, "DarkPoolVenue.posted_sizes");
        for (double u : posted_sizes) {
            if (!is_integer_like(u)) {
                throw std::invalid_argument(
                    "DarkPoolVenue: posted_sizes must be integer-valued."
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
