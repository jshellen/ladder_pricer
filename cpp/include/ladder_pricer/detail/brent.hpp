#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ladder_pricer::detail {

// Bounded Brent minimization: parabolic interpolation with golden-section fallback.
// Assumes the objective is approximately unimodal on [lower, upper].
// Returns the best evaluated (argument, objective value) pair.
template <class Function>
std::pair<double, double> bounded_brent_minimize(Function&& objective,
                                                  double lower,
                                                  double upper,
                                                  int precision_bits = 20) {
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper)
        throw std::invalid_argument("bounded_brent_minimize requires finite lower <= upper");
    if (precision_bits < 1 || precision_bits > 52)
        throw std::invalid_argument("bounded_brent_minimize invalid precision_bits");
    auto evaluate = [&](double x) {
        const double f = objective(x);
        if (!std::isfinite(f))
            throw std::domain_error("bounded_brent_minimize nonfinite objective");
        return f;
    };
    if (lower == upper) return {lower, evaluate(lower)};

    constexpr double golden = 0.3819660112501051518;
    const double eps = std::numeric_limits<double>::epsilon();
    double a = lower, b = upper;
    double x = a + golden * (b - a);
    double w = x, v = x;
    double fx = evaluate(x), fw = fx, fv = fx;
    double d = 0.0, e = 0.0;
    for (int iter = 0; iter < 256; ++iter) {
        const double scale = std::max({1.0, std::abs(a), std::abs(b)});
        const double tol = std::max(std::ldexp(scale, -precision_bits), 4.0 * eps * scale);
        const double midpoint = a + 0.5 * (b - a);
        const double tol2 = 2.0 * tol;
        if (std::abs(x - midpoint) <= tol2 - 0.5 * (b - a)) break;

        bool parabolic = false;
        if (std::abs(e) > tol) {
            const double r = (x - w) * (fx - fv);
            const double q0 = (x - v) * (fx - fw);
            double p = (x - v) * q0 - (x - w) * r;
            double q = 2.0 * (q0 - r);
            if (q > 0.0) p = -p;
            q = std::abs(q);
            const double prev_e = e;
            e = d;
            if (q > 0.0 && std::abs(p) < std::abs(0.5 * q * prev_e) &&
                p > q * (a - x) && p < q * (b - x)) {
                d = p / q;
                const double candidate = x + d;
                if (candidate - a < tol2 || b - candidate < tol2)
                    d = (x < midpoint ? tol : -tol);
                parabolic = true;
            }
        }
        if (!parabolic) {
            e = x < midpoint ? b - x : a - x;
            d = golden * e;
        }
        const double u = std::clamp(x + (std::abs(d) >= tol ? d : (d > 0 ? tol : -tol)), a, b);
        if (u == x) break;
        const double fu = evaluate(u);
        if (fu <= fx) {
            if (u < x) b = x; else a = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }
    return {x, fx};
}
} // namespace ladder_pricer::detail
