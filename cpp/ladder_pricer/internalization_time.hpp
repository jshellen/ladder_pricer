#pragma once

#include "common.hpp"

#include <cmath>

namespace hjb {

struct PolynomialInternalizationTime {
    double tau0{2.0};
    double tau1{0.1};
    double tau2{0.0015};

    PolynomialInternalizationTime() = default;

    PolynomialInternalizationTime(double tau0_, double tau1_, double tau2_)
        : tau0(tau0_), tau1(tau1_), tau2(tau2_) {}

    double value(double q) const {
        const double x = std::abs(q);
        return tau0 + tau1 * x + tau2 * x * x;
    }
};

} // namespace hjb
