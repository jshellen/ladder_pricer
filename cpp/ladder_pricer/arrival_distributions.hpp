#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace hjb {

/// Abstract interface for discrete arrival size distributions.
/// Used by the dark pool to model fill size probabilities.
class ArrivalDistribution {
public:
    virtual ~ArrivalDistribution() = default;

    /// Validate the distribution parameters.
    virtual void validate() const = 0;

    /// Compute expected fill value for posting size u.
    /// - direction: +1 for bid (inventory increases), -1 for ask (inventory decreases)
    /// - h_interpolator: function to evaluate value function h at any inventory
    /// - q: current inventory
    /// - hq: current value h(q)
    /// - u: posted size (cap)
    /// - fee: per-unit execution cost
    /// Returns lambda * E[h(q + direction*k) - h(q) - fee*k] where k is arrival size.
    virtual double expected_fill_value(
        const std::function<double(double)>& h_interpolator,
        double q,
        double hq,
        int u,
        double fee,
        double direction
    ) const = 0;

    /// Human-readable name of this distribution.
    virtual const char* name() const = 0;
};

/// Geometric arrival distribution (current behavior).
/// P(fill = k) = p*(1-p)^{k-1} for k < u,
/// P(fill >= u) = (1-p)^{u-1} (absorbed into the last term).
struct GeometricArrivalDist : public ArrivalDistribution {
    double lambda{1.0};
    double p{0.5};

    GeometricArrivalDist() = default;

    GeometricArrivalDist(double lambda_, double p_)
        : lambda(lambda_), p(p_) {}

    void validate() const override {
        if (lambda < 0.0) {
            throw std::invalid_argument("GeometricArrivalDist: lambda must be nonnegative.");
        }
        if (!(p > 0.0 && p <= 1.0)) {
            throw std::invalid_argument("GeometricArrivalDist: p must lie in (0, 1].");
        }
    }

    double expected_fill_value(
        const std::function<double(double)>& h_interp,
        double q,
        double hq,
        int u,
        double fee,
        double direction
    ) const override {
        const double r = 1.0 - p;
        double expected = 0.0;
        for (int k = 1; k < u; ++k) {
            const double kd = static_cast<double>(k);
            expected += p * std::pow(r, kd - 1.0) * (h_interp(q + direction * kd) - hq - fee * kd);
        }
        const double ud = static_cast<double>(u);
        expected += std::pow(r, ud - 1.0) * (h_interp(q + direction * ud) - hq - fee * ud);
        return lambda * expected;
    }

    const char* name() const override { return "geometric"; }
};

/// Zero-inflated Poisson arrival distribution.
/// P(fill = 0) = p0 (zero-inflation)
/// P(fill = k | k > 0) ∝ λ^k / k! for k ∈ {1, ..., u}
/// where λ is the Poisson rate parameter.
struct ZeroInflatedPoissonArrivalDist : public ArrivalDistribution {
    double lambda_arr{1.0};   ///< Arrival intensity (per-unit-time fill rate)
    double mu{2.0};           ///< Poisson rate parameter (average fill size when positive)
    double p0{0.1};           ///< Zero-inflation probability

    ZeroInflatedPoissonArrivalDist() = default;

    ZeroInflatedPoissonArrivalDist(double lambda_arr_, double mu_, double p0_)
        : lambda_arr(lambda_arr_), mu(mu_), p0(p0_) {}

    void validate() const override {
        if (lambda_arr < 0.0) {
            throw std::invalid_argument("ZeroInflatedPoissonArrivalDist: lambda_arr must be nonnegative.");
        }
        if (mu <= 0.0) {
            throw std::invalid_argument("ZeroInflatedPoissonArrivalDist: mu must be positive.");
        }
        if (!(p0 >= 0.0 && p0 < 1.0)) {
            throw std::invalid_argument("ZeroInflatedPoissonArrivalDist: p0 must lie in [0, 1).");
        }
    }

    double expected_fill_value(
        const std::function<double(double)>& h_interp,
        double q,
        double hq,
        int u,
        double fee,
        double direction
    ) const override {
        // Precompute Poisson PMF values and their normalization
        std::vector<double> pmf(u + 1, 0.0);
        double Z = 0.0;  // Normalization constant for the non-zero part

        // Compute Poisson probabilities: exp(-mu) * mu^k / k!
        double log_poisson_coeff = -mu;
        double log_mu_power = 0.0;
        double log_factorial = 0.0;
        
        for (int k = 0; k <= u; ++k) {
            double log_pmf = log_poisson_coeff + log_mu_power - log_factorial;
            pmf[k] = std::exp(log_pmf);
            if (k > 0) {
                Z += pmf[k];
            }
            // Update for next iteration
            if (k < u) {
                log_mu_power += std::log(mu);
                log_factorial += std::log(static_cast<double>(k + 1));
            }
        }

        // Normalize the non-zero part
        if (Z > 1e-15) {
            for (int k = 1; k <= u; ++k) {
                pmf[k] /= Z;
            }
        }

        // Expected fill value: zero contributes 0, k > 0 contributes weighted by (1 - p0)
        double expected = 0.0;
        for (int k = 1; k <= u; ++k) {
            const double kd = static_cast<double>(k);
            expected += (1.0 - p0) * pmf[k] * (h_interp(q + direction * kd) - hq - fee * kd);
        }
        return lambda_arr * expected;
    }

    const char* name() const override { return "zero_inflated_poisson"; }
};

}  // namespace hjb
