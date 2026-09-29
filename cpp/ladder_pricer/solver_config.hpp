#pragma once

#include "common.hpp"

#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hjb {

struct LinearInterpolator1D {
    std::reference_wrapper<const std::vector<double>> grid;
    std::reference_wrapper<const std::vector<double>> vals;

    LinearInterpolator1D(
        const std::vector<double>& grid_,
        const std::vector<double>& vals_
    )
        : grid(grid_), vals(vals_) {}

    double operator()(double x) const {
        return interp_linear(grid.get(), vals.get(), x);
    }
};

struct SolverConfig {
    std::vector<double> q_grid;
    double dt{0.002};
    int n_iter{140};

    double spot{1.0};
    double spot_drift{0.0};
    double spread{20.0 / 10000.0};

    double golden_tol{1e-4};
    int golden_max_iter{32};

    bool early_stop{true};
    double tol_h{1e-5};
    double tol_rhs{1e-4};
    int min_iter{5};
    int consecutive_passes_required{3};

    void validate() const {
        validate_centered_symmetric_grid(q_grid, "SolverConfig.q_grid");

        if (dt <= 0.0) {
            throw std::invalid_argument("SolverConfig: dt must be positive.");
        }
        if (n_iter < 1) {
            throw std::invalid_argument("SolverConfig: n_iter must be at least 1.");
        }
        if (spot <= 0.0) {
            throw std::invalid_argument("SolverConfig: spot must be positive.");
        }
        if (spread <= 0.0) {
            throw std::invalid_argument("SolverConfig: spread must be positive.");
        }
        if (golden_tol <= 0.0) {
            throw std::invalid_argument("SolverConfig: golden_tol must be positive.");
        }
        if (golden_max_iter < 1) {
            throw std::invalid_argument("SolverConfig: golden_max_iter must be at least 1.");
        }
        if (tol_h < 0.0 || tol_rhs < 0.0) {
            throw std::invalid_argument("SolverConfig: tolerances must be nonnegative.");
        }
        if (min_iter < 0 || consecutive_passes_required < 1) {
            throw std::invalid_argument("SolverConfig: invalid stopping parameters.");
        }
    }
};

struct SolverGridMeta {
    std::size_t nq{0};
    std::size_t q0_idx{0};
    double q_min{0.0};
    double q_max{0.0};
};

inline SolverGridMeta build_solver_grid_meta(const std::vector<double>& q_grid) {
    if (q_grid.empty()) {
        throw std::invalid_argument("build_solver_grid_meta: q_grid cannot be empty.");
    }

    SolverGridMeta meta;
    meta.nq = q_grid.size();
    meta.q0_idx = q_grid.size() / 2;
    meta.q_min = q_grid.front();
    meta.q_max = q_grid.back();
    return meta;
}

struct GoldenSectionSearch {
    double tol{1e-4};
    int max_iter{32};

    GoldenSectionSearch() = default;

    GoldenSectionSearch(double tol_, int max_iter_)
        : tol(tol_), max_iter(max_iter_) {
        validate();
    }

    void validate() const {
        if (tol <= 0.0) {
            throw std::invalid_argument("GoldenSectionSearch: tol must be positive.");
        }
        if (max_iter < 1) {
            throw std::invalid_argument("GoldenSectionSearch: max_iter must be at least 1.");
        }
    }

    template <class ObjectiveFn>
    double maximize(ObjectiveFn&& objective_fn, double lower, double upper) const {
        if (upper < lower) {
            throw std::invalid_argument("GoldenSectionSearch::maximize: invalid interval.");
        }
        if (std::abs(upper - lower) < tol) {
            return 0.5 * (lower + upper);
        }

        constexpr double gr = 1.6180339887498948482;
        double a = lower;
        double b = upper;
        double c = b - (b - a) / gr;
        double d = a + (b - a) / gr;
        double fc = objective_fn(c);
        double fd = objective_fn(d);

        for (int it = 0; it < max_iter; ++it) {
            if (std::abs(b - a) < tol) {
                break;
            }

            if (fc > fd) {
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) / gr;
                fc = objective_fn(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) / gr;
                fd = objective_fn(d);
            }
        }

        return 0.5 * (a + b);
    }
};

struct SolverDiagnostics {
    bool converged{false};
    int iterations_used{0};
    double final_max_h_change{std::numeric_limits<double>::infinity()};
    double final_max_rhs{std::numeric_limits<double>::infinity()};
    int consecutive_passes{0};

    std::vector<double> history_max_h_change;
    std::vector<double> history_max_rhs;

    SolverDiagnostics() = default;

    explicit SolverDiagnostics(int n_iter) {
        reserve(n_iter);
    }

    void reserve(int n_iter) {
        const std::size_t n = n_iter > 0 ? static_cast<std::size_t>(n_iter) : 0u;
        history_max_h_change.reserve(n);
        history_max_rhs.reserve(n);
    }

    void record_iteration(int iteration, double max_h_change, double max_rhs) {
        iterations_used = iteration;
        final_max_h_change = max_h_change;
        final_max_rhs = max_rhs;
        history_max_h_change.push_back(max_h_change);
        history_max_rhs.push_back(max_rhs);
    }

    bool update_stopping_state(
        bool passes_now,
        bool early_stop,
        int consecutive_passes_required
    ) {
        consecutive_passes = passes_now ? consecutive_passes + 1 : 0;

        if (early_stop && consecutive_passes >= consecutive_passes_required) {
            converged = true;
            return true;
        }
        return false;
    }
};

} // namespace hjb
