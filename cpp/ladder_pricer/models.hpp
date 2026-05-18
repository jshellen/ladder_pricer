#pragma once

#include "common.hpp"

#include <cmath>

namespace hjb {

struct LogisticFlowCurve {
    double A0{1.0};
    double theta{0.0};
    double shift{0.20};
    double steepness{10.0};
    double volume_shift{0.05};
    double z_floor{1e-8};

    LogisticFlowCurve() = default;

    LogisticFlowCurve(
        double A0_,
        double theta_,
        double shift_,
        double steepness_,
        double volume_shift_,
        double z_floor_ = 1e-8
    )
        : A0(A0_),
          theta(theta_),
          shift(shift_),
          steepness(steepness_),
          volume_shift(volume_shift_),
          z_floor(z_floor_) {}

    double A(double z) const {
        z = std::max(z, z_floor);
        return A0 * std::pow(z, -theta);
    }

    double hit_ratio(double delta, double z) const {
        const double y = (delta - shift + volume_shift * (z - 1.0)) * steepness;

        if (y >= 0.0) {
            const double e = std::exp(-y);
            return 1.0 / (1.0 + e);
        }

        const double e = std::exp(y);
        return e / (1.0 + e);
    }

    double arrival_rate(double delta, double z) const {
        return A(z) * hit_ratio(delta, z);
    }
};

struct SqrtMarkoutModel {
    double base{0.005};
    double coeff{0.003};

    SqrtMarkoutModel() = default;
    SqrtMarkoutModel(double base_, double coeff_) : base(base_), coeff(coeff_) {}

    double expected_markout(double z) const {
        return base + coeff * std::sqrt(z);
    }
};

struct PolynomialInventoryPenalty {
    double risk_aversion{2.0};
    double sigma{0.25};
    double tau0{2.0};
    double tau1{0.1};
    double tau2{0.0015};

    PolynomialInventoryPenalty() = default;

    PolynomialInventoryPenalty(
        double risk_aversion_,
        double sigma_,
        double tau0_,
        double tau1_,
        double tau2_
    )
        : risk_aversion(risk_aversion_),
          sigma(sigma_),
          tau0(tau0_),
          tau1(tau1_),
          tau2(tau2_) {}

    double value(double q) const {
        const double x = std::abs(q);
        const double x2 = x * x;
        return risk_aversion * sigma * sigma *
               (tau0 * x2 + tau1 * x2 * x + tau2 * x2 * x2);
    }
};

} // namespace hjb
