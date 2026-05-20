#pragma once

#include "common.hpp"

#include <cmath>

namespace hjb {

struct LogisticFlowCurve {
    double A0{1.0};
    double theta{0.0};
    double beta{0.0};
    double shift{0.20};
    double steepness{10.0};
    double volume_shift{0.05};
    double z_floor{1e-8};

    LogisticFlowCurve() = default;

    LogisticFlowCurve(
        double A0_,
        double theta_,
        double beta_,
        double shift_,
        double steepness_,
        double volume_shift_,
        double z_floor_ = 1e-8
    )
        : A0(A0_),
          theta(theta_),
          beta(beta_),
          shift(shift_),
          steepness(steepness_),
          volume_shift(volume_shift_),
          z_floor(z_floor_) {}

    double A(double z) const {
        z = std::max(z, z_floor);
        return A0 * std::pow(z, -theta - beta * z);
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

} // namespace hjb
