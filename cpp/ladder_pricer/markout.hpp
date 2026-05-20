#pragma once

#include "common.hpp"

#include <cmath>
#include <variant>

namespace hjb {

struct SqrtMarkoutModel {
    double base{0.005};
    double coeff{0.003};

    SqrtMarkoutModel() = default;
    SqrtMarkoutModel(double base_, double coeff_) : base(base_), coeff(coeff_) {}

    double expected_markout(double z, double /*t*/) const {
        return base + coeff * std::sqrt(z);
    }
};

struct SqrtTimeMarkoutModel {
    double a0{0.0};
    double a1{0.0};
    double a2{0.0};
    double b0{0.0};
    double b1{0.0};
    double b2{0.0};

    SqrtTimeMarkoutModel() = default;

    SqrtTimeMarkoutModel(double a0_, double a1_, double a2_,
                         double b0_, double b1_, double b2_)
        : a0(a0_), a1(a1_), a2(a2_),
          b0(b0_), b1(b1_), b2(b2_) {}

    double alpha(double z) const { return a0 + a1 * z + a2 * z * z; }
    double beta(double z) const { return b0 + b1 * z + b2 * z * z; }

    double expected_markout(double z, double t) const {
        return alpha(z) * std::sqrt(t) + beta(z) * t;
    }
};

using MarkoutModel = std::variant<SqrtMarkoutModel, SqrtTimeMarkoutModel>;

inline double expected_markout(const MarkoutModel& m, double z, double t) {
    return std::visit([z, t](const auto& model) { return model.expected_markout(z, t); }, m);
}

} // namespace hjb
