#pragma once

#include "common.hpp"
#include "models.hpp"
#include "policy.hpp"
#include "quote_analytics.hpp"

#include <string>
#include <vector>

namespace hjb {

struct Tier {
    std::string name;
    LogisticFlowCurve flow_curve;
    SqrtMarkoutModel markout_model;
    QuotePolicy policy;

    Tier() = default;

    Tier(
        std::string name_,
        LogisticFlowCurve flow_curve_,
        SqrtMarkoutModel markout_model_
    )
        : name(std::move(name_)),
          flow_curve(std::move(flow_curve_)),
          markout_model(std::move(markout_model_)) {}

    virtual ~Tier() = default;

    virtual const char* tier_type_name() const = 0;
    virtual const std::vector<double>& sizes() const = 0;
    virtual bool is_admissible(double q, double z, Side side) const = 0;

    virtual void validate() const = 0;

    void reset_policy_shape(const std::vector<double>& q_grid) {
        policy.reset_shape(q_grid, sizes());
    }

    double quote(double q, double z, Side side) const {
        return policy.delta(q, z, side);
    }

    QuoteSummary quote_summary(
        double q,
        double z,
        Side side,
        double mid,
        double spread
    ) const {
        return policy.quote_summary(q, z, side, mid, spread);
    }
};

struct MDPTier final : public Tier {
    std::vector<double> sizes_;
    double delta_min{-5.0};
    double delta_max{5.0};

    MDPTier() = default;

    MDPTier(
        std::string name_,
        std::vector<double> sizes__,
        LogisticFlowCurve flow_curve_,
        SqrtMarkoutModel markout_model_,
        double delta_min_ = -5.0,
        double delta_max_ = 5.0
    )
        : Tier(std::move(name_), std::move(flow_curve_), std::move(markout_model_)),
          sizes_(std::move(sizes__)),
          delta_min(delta_min_),
          delta_max(delta_max_) {
        validate();
    }

    const char* tier_type_name() const override {
        return "mdp";
    }

    const std::vector<double>& sizes() const override {
        return sizes_;
    }

    void validate() const override {
        validate_positive_strictly_increasing(sizes_, "MDPTier.sizes");
        if (delta_max <= delta_min) {
            throw std::invalid_argument("MDPTier: delta_max must be > delta_min.");
        }
    }

    bool is_admissible(double, double, Side) const override {
        return true;
    }
};

} // namespace hjb
