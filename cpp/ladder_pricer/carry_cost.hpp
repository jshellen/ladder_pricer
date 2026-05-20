#pragma once

#include "common.hpp"

#include <cmath>

namespace hjb {

struct CarryCost {
    double risk_aversion{2.0};
    double sigma{0.25};

    CarryCost() = default;

    CarryCost(double risk_aversion_, double sigma_)
        : risk_aversion(risk_aversion_), sigma(sigma_) {}

    double value() const {
        return risk_aversion * sigma * sigma;
    }
};


} // namespace hjb
