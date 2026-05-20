#pragma once

#include "carry_cost.hpp"
#include "internalization_time.hpp"

namespace hjb {

struct PolynomialInventoryPenalty {
    CarryCost carry_cost;
    PolynomialInternalizationTime internalization_time;

    PolynomialInventoryPenalty() = default;

    PolynomialInventoryPenalty(
        CarryCost carry_cost_,
        PolynomialInternalizationTime internalization_time_
    )
        : carry_cost(carry_cost_),
          internalization_time(internalization_time_) {}

    double value(double q) const {
        return carry_cost.value() * q * q * internalization_time.value(q);
    }
};

} // namespace hjb
