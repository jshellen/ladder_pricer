#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hjb {

// ============================================================
// Helpers
// ============================================================

inline double clamp(double x, double lo, double hi) {
    return std::min(std::max(x, lo), hi);
}

inline std::pair<std::size_t, double> locate_segment_with_weight(
    const std::vector<double>& grid,
    double x
) {
    if (grid.empty()) {
        throw std::invalid_argument("Grid cannot be empty.");
    }
    if (grid.size() == 1) {
        return {0, 0.0};
    }

    if (x <= grid.front()) {
        const double x0 = grid[0];
        const double x1 = grid[1];
        const double t = (x - x0) / (x1 - x0);
        return {0, t};
    }

    if (x >= grid.back()) {
        const std::size_t i = grid.size() - 2;
        const double x0 = grid[i];
        const double x1 = grid[i + 1];
        const double t = (x - x0) / (x1 - x0);
        return {i, t};
    }

    auto it = std::upper_bound(grid.begin(), grid.end(), x);
    const std::size_t i = static_cast<std::size_t>(it - grid.begin() - 1);
    const double x0 = grid[i];
    const double x1 = grid[i + 1];
    const double t = (x - x0) / (x1 - x0);
    return {i, t};
}

inline double interp_linear(
    const std::vector<double>& grid,
    const std::vector<double>& vals,
    double x
) {
    if (grid.size() != vals.size()) {
        throw std::invalid_argument("interp_linear: size mismatch.");
    }
    if (grid.empty()) {
        throw std::invalid_argument("interp_linear: empty input.");
    }
    if (grid.size() == 1) {
        return vals.front();
    }

    const auto [i, t] = locate_segment_with_weight(grid, x);
    return vals[i] + t * (vals[i + 1] - vals[i]);
}

inline double interp_trilinear(
    const std::vector<double>& q_grid,
    const std::vector<double>& y_grid,
    const std::vector<double>& nu_grid,
    const std::vector<std::vector<std::vector<double>>>& vals,
    double q,
    double y,
    double nu
) {
    if (vals.size() != q_grid.size()) {
        throw std::invalid_argument("interp_trilinear: q dimension mismatch.");
    }
    for (const auto& slab : vals) {
        if (slab.size() != y_grid.size()) {
            throw std::invalid_argument("interp_trilinear: y dimension mismatch.");
        }
        for (const auto& row : slab) {
            if (row.size() != nu_grid.size()) {
                throw std::invalid_argument("interp_trilinear: nu dimension mismatch.");
            }
        }
    }

    const auto [iq, tq] = locate_segment_with_weight(q_grid, q);
    const auto [iy, ty] = locate_segment_with_weight(y_grid, y);
    const auto [in, tn] = locate_segment_with_weight(nu_grid, nu);

    const double c000 = vals[iq][iy][in];
    const double c100 = vals[iq + 1][iy][in];
    const double c010 = vals[iq][iy + 1][in];
    const double c110 = vals[iq + 1][iy + 1][in];

    const double c001 = vals[iq][iy][in + 1];
    const double c101 = vals[iq + 1][iy][in + 1];
    const double c011 = vals[iq][iy + 1][in + 1];
    const double c111 = vals[iq + 1][iy + 1][in + 1];

    const double c00 = c000 + tq * (c100 - c000);
    const double c10 = c010 + tq * (c110 - c010);
    const double c01 = c001 + tq * (c101 - c001);
    const double c11 = c011 + tq * (c111 - c011);

    const double c0 = c00 + ty * (c10 - c00);
    const double c1 = c01 + ty * (c11 - c01);

    return c0 + tn * (c1 - c0);
}

inline double max_abs_vec(const std::vector<double>& x) {
    double out = 0.0;
    for (double v : x) {
        out = std::max(out, std::abs(v));
    }
    return out;
}

inline double max_abs_3d(const std::vector<std::vector<std::vector<double>>>& x) {
    double out = 0.0;
    for (const auto& slab : x) {
        for (const auto& row : slab) {
            out = std::max(out, max_abs_vec(row));
        }
    }
    return out;
}

inline std::vector<double> diff(const std::vector<double>& x) {
    if (x.size() < 2) return {};
    std::vector<double> out(x.size() - 1);
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        out[i] = x[i + 1] - x[i];
    }
    return out;
}

// ============================================================
// No-copy interpolator
// ============================================================

struct HInterp3D {
    const std::vector<double>& q_grid;
    const std::vector<double>& y_grid;
    const std::vector<double>& nu_grid;
    const std::vector<std::vector<std::vector<double>>>& h;

    double operator()(double q, double y, double nu) const {
        return interp_trilinear(q_grid, y_grid, nu_grid, h, q, y, nu);
    }
};

// ============================================================
// Abstract model components
// ============================================================

struct FlowCurve {
    virtual ~FlowCurve() = default;
    virtual double arrival_rate(double delta, double z) const = 0;
};

struct DriftJumpModel {
    virtual ~DriftJumpModel() = default;
    virtual double jump_size(double z) const = 0;
};

struct InventoryPenalty {
    virtual ~InventoryPenalty() = default;
    // Base penalty ψ(q). Solver multiplies it by exp(2*nu).
    virtual double value(double q) const = 0;
};

// ============================================================
// Concrete model components
// ============================================================

struct LogisticFlowCurve final : public FlowCurve {
    double A0{1.0};
    double theta{0.0};
    double k{2.0};
    double m0{0.2};
    double m_alpha{0.0};
    double z_floor{1e-8};

    LogisticFlowCurve() = default;

    LogisticFlowCurve(
        double A0_,
        double theta_,
        double k_,
        double m0_,
        double m_alpha_,
        double z_floor_ = 1e-8
    )
        : A0(A0_),
          theta(theta_),
          k(k_),
          m0(m0_),
          m_alpha(m_alpha_),
          z_floor(z_floor_) {}

    double A(double z) const {
        z = std::max(z, z_floor);
        return A0 * std::pow(z, -theta);
    }

    double m(double z) const {
        z = std::max(z, z_floor);
        return m0 + m_alpha * z;
    }

    double arrival_rate(double delta, double z) const override {
        const double Az = A(z);
        const double mz = m(z);

        if (Az < 0.0) {
            throw std::invalid_argument("A(z) must be nonnegative.");
        }
        if (k <= 0.0) {
            throw std::invalid_argument("k must be positive.");
        }

        const double x = k * (delta - mz);

        if (x >= 0.0) {
            const double ex = std::exp(-x);
            return Az * ex / (1.0 + ex);
        }
        const double ex = std::exp(x);
        return Az / (1.0 + ex);
    }
};

struct SqrtDriftJumpModel final : public DriftJumpModel {
    double base{0.0};
    double coeff{0.01};

    SqrtDriftJumpModel() = default;

    SqrtDriftJumpModel(double base_, double coeff_)
        : base(base_), coeff(coeff_) {}

    double jump_size(double z) const override {
        return base + coeff * std::sqrt(z);
    }
};

struct PolynomialInventoryPenalty final : public InventoryPenalty {
    double risk_aversion{2.0};
    double sigma_ref{1.0};
    double tau0{2.0};
    double cubic_coeff{0.1};
    double quartic_coeff{0.0015};

    PolynomialInventoryPenalty() = default;

    PolynomialInventoryPenalty(
        double risk_aversion_,
        double sigma_ref_,
        double tau0_,
        double cubic_coeff_,
        double quartic_coeff_
    )
        : risk_aversion(risk_aversion_),
          sigma_ref(sigma_ref_),
          tau0(tau0_),
          cubic_coeff(cubic_coeff_),
          quartic_coeff(quartic_coeff_) {}

    double value(double q) const override {
        const double x = std::abs(q);
        return risk_aversion * sigma_ref * sigma_ref *
               (tau0 * x * x + cubic_coeff * x * x * x + quartic_coeff * x * x * x * x);
    }
};

// ============================================================
// Quote policy: indexed by q, y, nu, size
// ============================================================

struct QuotePolicy {
    std::vector<double> q_grid;
    std::vector<double> y_grid;
    std::vector<double> nu_grid;
    std::vector<double> sizes;

    // shapes: [nq][ny][nnu][nz]
    std::vector<std::vector<std::vector<std::vector<double>>>> bid;
    std::vector<std::vector<std::vector<std::vector<double>>>> ask;

    QuotePolicy() = default;

    QuotePolicy(
        std::vector<double> q_grid_,
        std::vector<double> y_grid_,
        std::vector<double> nu_grid_,
        std::vector<double> sizes_,
        std::vector<std::vector<std::vector<std::vector<double>>>> bid_,
        std::vector<std::vector<std::vector<std::vector<double>>>> ask_
    )
        : q_grid(std::move(q_grid_)),
          y_grid(std::move(y_grid_)),
          nu_grid(std::move(nu_grid_)),
          sizes(std::move(sizes_)),
          bid(std::move(bid_)),
          ask(std::move(ask_)) {
        validate();
    }

    void validate() const {
        const std::size_t nq = q_grid.size();
        const std::size_t ny = y_grid.size();
        const std::size_t nnu = nu_grid.size();
        const std::size_t nz = sizes.size();

        if (bid.size() != nq || ask.size() != nq) {
            throw std::invalid_argument("QuotePolicy: q dimension mismatch.");
        }

        auto check_cube = [&](const auto& cube, const std::string& name) {
            for (const auto& slab_y : cube) {
                if (slab_y.size() != ny) {
                    throw std::invalid_argument("QuotePolicy: " + name + " y dimension mismatch.");
                }
                for (const auto& slab_nu : slab_y) {
                    if (slab_nu.size() != nnu) {
                        throw std::invalid_argument("QuotePolicy: " + name + " nu dimension mismatch.");
                    }
                    for (const auto& row : slab_nu) {
                        if (row.size() != nz) {
                            throw std::invalid_argument("QuotePolicy: " + name + " size dimension mismatch.");
                        }
                    }
                }
            }
        };

        check_cube(bid, "bid");
        check_cube(ask, "ask");
    }

    double delta_at_index(
        std::size_t iq,
        std::size_t iy,
        std::size_t inu,
        std::size_t iz,
        const std::string& side
    ) const {
        if (side == "bid") return bid.at(iq).at(iy).at(inu).at(iz);
        if (side == "ask") return ask.at(iq).at(iy).at(inu).at(iz);
        throw std::invalid_argument("Unknown side: " + side);
    }

    double quote(double q, double y, double nu, double z, const std::string& side) const {
        const auto& cube = (side == "bid") ? bid : ask;
        if (side != "bid" && side != "ask") {
            throw std::invalid_argument("Unknown side: " + side);
        }

        std::vector<double> vals_by_z(sizes.size(), 0.0);

        for (std::size_t iz = 0; iz < sizes.size(); ++iz) {
            std::vector<std::vector<std::vector<double>>> slice(
                q_grid.size(),
                std::vector<std::vector<double>>(y_grid.size(), std::vector<double>(nu_grid.size(), 0.0))
            );

            for (std::size_t iq = 0; iq < q_grid.size(); ++iq) {
                for (std::size_t iy = 0; iy < y_grid.size(); ++iy) {
                    for (std::size_t inu = 0; inu < nu_grid.size(); ++inu) {
                        slice[iq][iy][inu] = cube[iq][iy][inu][iz];
                    }
                }
            }

            vals_by_z[iz] = interp_trilinear(q_grid, y_grid, nu_grid, slice, q, y, nu);
        }

        return interp_linear(sizes, vals_by_z, z);
    }
};

// ============================================================
// Price tier
// ============================================================

struct PriceTier {
    std::string name;
    std::vector<double> sizes;
    std::shared_ptr<FlowCurve> flow_curve;
    std::shared_ptr<DriftJumpModel> jump_model;

    double delta_min{-0.5};
    double delta_max{4.0};
    double golden_tol{1e-4};
    int golden_max_iter{32};

    QuotePolicy policy;

    PriceTier() = default;

    PriceTier(
        std::string name_,
        std::vector<double> sizes_,
        std::shared_ptr<FlowCurve> flow_curve_,
        std::shared_ptr<DriftJumpModel> jump_model_,
        double delta_min_ = -0.5,
        double delta_max_ = 4.0,
        double golden_tol_ = 1e-4,
        int golden_max_iter_ = 32
    )
        : name(std::move(name_)),
          sizes(std::move(sizes_)),
          flow_curve(std::move(flow_curve_)),
          jump_model(std::move(jump_model_)),
          delta_min(delta_min_),
          delta_max(delta_max_),
          golden_tol(golden_tol_),
          golden_max_iter(golden_max_iter_) {
        validate();
    }

    void validate() const {
        if (sizes.empty()) {
            throw std::invalid_argument("PriceTier: sizes cannot be empty.");
        }
        if (!flow_curve) {
            throw std::invalid_argument("PriceTier: flow_curve cannot be null.");
        }
        if (!jump_model) {
            throw std::invalid_argument("PriceTier: jump_model cannot be null.");
        }
        if (delta_max <= delta_min) {
            throw std::invalid_argument("PriceTier: delta_max must be > delta_min.");
        }
        for (double z : sizes) {
            if (z <= 0.0) {
                throw std::invalid_argument("PriceTier: sizes must be positive.");
            }
        }
        for (std::size_t i = 1; i < sizes.size(); ++i) {
            if (sizes[i] <= sizes[i - 1]) {
                throw std::invalid_argument("PriceTier: sizes must be strictly increasing.");
            }
        }
    }

    static double next_inventory(double q, double z, const std::string& side) {
        if (side == "bid") return q + z;
        if (side == "ask") return q - z;
        throw std::invalid_argument("Unknown side: " + side);
    }

    static double next_drift(double y, double jump, const std::string& side) {
        if (side == "bid") return y - jump;
        if (side == "ask") return y + jump;
        throw std::invalid_argument("Unknown side: " + side);
    }

    double arrival_rate(double delta, double z) const {
        return flow_curve->arrival_rate(delta, z);
    }

    double jump_size(double z) const {
        return jump_model->jump_size(z);
    }

    double single_size_objective(
        double delta,
        const HInterp3D& h_fn,
        double q,
        double y,
        double nu,
        double z,
        const std::string& side
    ) const {
        const double q_next = next_inventory(q, z, side);
        const double y_next = next_drift(y, jump_size(z), side);
        const double dh = h_fn(q_next, y_next, nu) - h_fn(q, y, nu);
        const double lam = arrival_rate(delta, z);
        return lam * (z * delta + dh);
    }

    double golden_search_max(
        const std::function<double(double)>& objective_fn,
        double lower,
        double upper
    ) const {
        if (upper < lower) {
            throw std::invalid_argument("golden_search_max: invalid interval.");
        }
        if (std::abs(upper - lower) < golden_tol) {
            return 0.5 * (lower + upper);
        }

        const double gr = (std::sqrt(5.0) + 1.0) / 2.0;
        double a = lower;
        double b = upper;
        double c = b - (b - a) / gr;
        double d = a + (b - a) / gr;
        double fc = objective_fn(c);
        double fd = objective_fn(d);

        for (int it = 0; it < golden_max_iter; ++it) {
            if (std::abs(b - a) < golden_tol) break;

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

    static bool is_shrink_mode(double q, const std::string& side) {
        return (q > 0.0 && side == "ask") || (q < 0.0 && side == "bid");
    }

    static bool is_expand_mode(double q, const std::string& side) {
        return (q > 0.0 && side == "bid") || (q < 0.0 && side == "ask");
    }

    std::pair<double, double> rung_bounds(
        std::size_t rung_idx,
        const std::vector<double>& current_delta,
        double q,
        const std::string& side,
        const std::vector<double>* prev_delta
    ) const {
        const std::size_t n = sizes.size();
        double lower = delta_min;
        double upper = delta_max;

        std::vector<double> prev_gaps;
        if (prev_delta != nullptr) {
            prev_gaps = diff(*prev_delta);
        }

        if (rung_idx == 0) {
            if (!prev_gaps.empty() && is_expand_mode(q, side)) {
                double required_total_gap = 0.0;
                for (double g : prev_gaps) required_total_gap += g;
                upper = std::min(upper, delta_max - required_total_gap);
            }
            return {lower, upper};
        }

        const double prev_curr = current_delta[rung_idx - 1];
        lower = std::max(lower, prev_curr);

        if (prev_gaps.empty()) {
            return {lower, upper};
        }

        const double prev_gap = prev_gaps[rung_idx - 1];

        if (is_shrink_mode(q, side)) {
            upper = std::min(upper, prev_curr + prev_gap);
        } else if (is_expand_mode(q, side)) {
            lower = std::max(lower, prev_curr + prev_gap);

            double remaining_future_gap = 0.0;
            for (std::size_t k = rung_idx; k + 1 < n; ++k) {
                remaining_future_gap += prev_gaps[k];
            }
            upper = std::min(upper, delta_max - remaining_future_gap);
        }

        return {lower, upper};
    }

    std::vector<double> solve_side_ladder(
        const HInterp3D& h_fn,
        double q,
        double y,
        double nu,
        const std::string& side,
        const std::vector<double>* prev_delta = nullptr
    ) const {
        const std::size_t n = sizes.size();
        std::vector<double> delta(n, 0.0);

        for (std::size_t j = 0; j < n; ++j) {
            auto [lower0, upper0] = rung_bounds(j, delta, q, side, prev_delta);
            double lower = std::max(delta_min, lower0);
            double upper = std::min(delta_max, upper0);

            if (lower > upper + 1e-12) {
                throw std::runtime_error(
                    "Infeasible ladder bounds for tier '" + name + "', side '" + side + "'."
                );
            }
            if (upper < lower) {
                upper = lower;
            }

            const double z = sizes[j];
            auto obj = [&](double d) {
                return single_size_objective(d, h_fn, q, y, nu, z, side);
            };

            delta[j] = golden_search_max(obj, lower, upper);
            delta[j] = clamp(delta[j], lower, upper);
        }

        return delta;
    }

    QuotePolicy build_policy(
        const HInterp3D& h_fn,
        const std::vector<double>& q_grid,
        const std::vector<double>& y_grid,
        const std::vector<double>& nu_grid
    ) {
        const std::size_t nq = q_grid.size();
        const std::size_t ny = y_grid.size();
        const std::size_t nnu = nu_grid.size();
        const std::size_t nz = sizes.size();

        std::vector<std::vector<std::vector<std::vector<double>>>> bid(
            nq,
            std::vector<std::vector<std::vector<double>>>(
                ny,
                std::vector<std::vector<double>>(nnu, std::vector<double>(nz, 0.0))
            )
        );

        std::vector<std::vector<std::vector<std::vector<double>>>> ask = bid;

        std::size_t q0_idx = 0;
        double best_abs = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < nq; ++i) {
            const double a = std::abs(q_grid[i]);
            if (a < best_abs) {
                best_abs = a;
                q0_idx = i;
            }
        }

        for (std::size_t iy = 0; iy < ny; ++iy) {
            const double y = y_grid[iy];

            for (std::size_t inu = 0; inu < nnu; ++inu) {
                const double nu = nu_grid[inu];
                const double q0 = q_grid[q0_idx];

                ask[q0_idx][iy][inu] = solve_side_ladder(h_fn, q0, y, nu, "ask", nullptr);
                bid[q0_idx][iy][inu] = solve_side_ladder(h_fn, q0, y, nu, "bid", nullptr);

                for (std::size_t iq = q0_idx + 1; iq < nq; ++iq) {
                    const double q = q_grid[iq];
                    ask[iq][iy][inu] = solve_side_ladder(h_fn, q, y, nu, "ask", &ask[iq - 1][iy][inu]);
                    bid[iq][iy][inu] = solve_side_ladder(h_fn, q, y, nu, "bid", &bid[iq - 1][iy][inu]);
                }

                for (std::size_t ii = q0_idx; ii-- > 0;) {
                    const double q = q_grid[ii];
                    ask[ii][iy][inu] = solve_side_ladder(h_fn, q, y, nu, "ask", &ask[ii + 1][iy][inu]);
                    bid[ii][iy][inu] = solve_side_ladder(h_fn, q, y, nu, "bid", &bid[ii + 1][iy][inu]);
                }
            }
        }

        policy = QuotePolicy(q_grid, y_grid, nu_grid, sizes, bid, ask);
        return policy;
    }

    double quote(double q, double y, double nu, double z, const std::string& side) const {
        return policy.quote(q, y, nu, z, side);
    }
};

// ============================================================
// Solver
// ============================================================

struct SolverConfig {
    std::vector<double> q_grid;
    std::vector<double> y_grid;
    std::vector<double> nu_grid;

    double dt{0.001};
    int n_iter{200};

    double kappa_y{2.0};
    double kappa_nu{1.0};
    double nu_bar{0.0};
    double eta_nu{0.2};

    bool early_stop{true};
    double tol_h{1e-5};
    double tol_rhs{1e-4};
    int min_iter{5};
    int consecutive_passes_required{3};

    void validate() const {
        if (q_grid.size() < 2 || y_grid.size() < 2 || nu_grid.size() < 2) {
            throw std::invalid_argument("SolverConfig: all grids must have at least 2 points.");
        }
        if (dt <= 0.0) {
            throw std::invalid_argument("SolverConfig: dt must be positive.");
        }
        if (n_iter < 1) {
            throw std::invalid_argument("SolverConfig: n_iter must be at least 1.");
        }
        if (kappa_y < 0.0 || kappa_nu < 0.0 || eta_nu < 0.0) {
            throw std::invalid_argument("SolverConfig: mean reversion and vol-of-vol must be nonnegative.");
        }
        if (tol_h < 0.0 || tol_rhs < 0.0) {
            throw std::invalid_argument("SolverConfig: tolerances must be nonnegative.");
        }
        if (min_iter < 0 || consecutive_passes_required < 1) {
            throw std::invalid_argument("SolverConfig: invalid stopping parameters.");
        }

        auto check_increasing = [](const std::vector<double>& g, const std::string& name) {
            for (std::size_t i = 1; i < g.size(); ++i) {
                if (g[i] <= g[i - 1]) {
                    throw std::invalid_argument("SolverConfig: " + name + " must be strictly increasing.");
                }
            }
        };

        check_increasing(q_grid, "q_grid");
        check_increasing(y_grid, "y_grid");
        check_increasing(nu_grid, "nu_grid");
    }
};

struct HJBSolution {
    // [nq][ny][nnu]
    std::vector<std::vector<std::vector<double>>> h;
    std::vector<double> q_grid;
    std::vector<double> y_grid;
    std::vector<double> nu_grid;
    std::vector<std::shared_ptr<PriceTier>> tiers;

    bool converged{false};
    int iterations_used{0};
    double final_max_h_change{std::numeric_limits<double>::infinity()};
    double final_max_rhs{std::numeric_limits<double>::infinity()};
    std::vector<double> history_max_h_change;
    std::vector<double> history_max_rhs;
};

struct HJBLadderSolver {
    SolverConfig config;
    std::shared_ptr<InventoryPenalty> penalty;
    std::vector<std::shared_ptr<PriceTier>> tiers;

    HJBLadderSolver(
        SolverConfig config_,
        std::shared_ptr<InventoryPenalty> penalty_,
        std::vector<std::shared_ptr<PriceTier>> tiers_
    )
        : config(std::move(config_)),
          penalty(std::move(penalty_)),
          tiers(std::move(tiers_)) {
        config.validate();
        if (!penalty) {
            throw std::invalid_argument("HJBLadderSolver: penalty cannot be null.");
        }
    }

    HInterp3D make_h_interp(
        const std::vector<std::vector<std::vector<double>>>& h_mat
    ) const {
        return HInterp3D{config.q_grid, config.y_grid, config.nu_grid, h_mat};
    }

    double y_drift_term(
        const std::vector<std::vector<std::vector<double>>>& h_mat,
        std::size_t iq,
        std::size_t iy,
        std::size_t inu
    ) const {
        const double y = config.y_grid[iy];
        const double b = -config.kappa_y * y;

        if (std::abs(b) < 1e-14 || config.y_grid.size() == 1) {
            return 0.0;
        }

        double dh_dy = 0.0;
        const std::size_t ny = config.y_grid.size();

        if (b > 0.0) {
            if (iy == 0) {
                const double dy = config.y_grid[1] - config.y_grid[0];
                dh_dy = (h_mat[iq][1][inu] - h_mat[iq][0][inu]) / dy;
            } else {
                const double dy = config.y_grid[iy] - config.y_grid[iy - 1];
                dh_dy = (h_mat[iq][iy][inu] - h_mat[iq][iy - 1][inu]) / dy;
            }
        } else {
            if (iy + 1 >= ny) {
                const double dy = config.y_grid[ny - 1] - config.y_grid[ny - 2];
                dh_dy = (h_mat[iq][ny - 1][inu] - h_mat[iq][ny - 2][inu]) / dy;
            } else {
                const double dy = config.y_grid[iy + 1] - config.y_grid[iy];
                dh_dy = (h_mat[iq][iy + 1][inu] - h_mat[iq][iy][inu]) / dy;
            }
        }

        return b * dh_dy;
    }

    double nu_drift_term(
        const std::vector<std::vector<std::vector<double>>>& h_mat,
        std::size_t iq,
        std::size_t iy,
        std::size_t inu
    ) const {
        const double nu = config.nu_grid[inu];
        const double b = config.kappa_nu * (config.nu_bar - nu);

        if (std::abs(b) < 1e-14 || config.nu_grid.size() == 1) {
            return 0.0;
        }

        double dh_dnu = 0.0;
        const std::size_t nnu = config.nu_grid.size();

        if (b > 0.0) {
            if (inu == 0) {
                const double dnu = config.nu_grid[1] - config.nu_grid[0];
                dh_dnu = (h_mat[iq][iy][1] - h_mat[iq][iy][0]) / dnu;
            } else {
                const double dnu = config.nu_grid[inu] - config.nu_grid[inu - 1];
                dh_dnu = (h_mat[iq][iy][inu] - h_mat[iq][iy][inu - 1]) / dnu;
            }
        } else {
            if (inu + 1 >= nnu) {
                const double dnu = config.nu_grid[nnu - 1] - config.nu_grid[nnu - 2];
                dh_dnu = (h_mat[iq][iy][nnu - 1] - h_mat[iq][iy][nnu - 2]) / dnu;
            } else {
                const double dnu = config.nu_grid[inu + 1] - config.nu_grid[inu];
                dh_dnu = (h_mat[iq][iy][inu + 1] - h_mat[iq][iy][inu]) / dnu;
            }
        }

        return b * dh_dnu;
    }

    double nu_diffusion_term(
        const std::vector<std::vector<std::vector<double>>>& h_mat,
        std::size_t iq,
        std::size_t iy,
        std::size_t inu
    ) const {
        if (config.eta_nu == 0.0 || config.nu_grid.size() < 3) {
            return 0.0;
        }

        const std::size_t nnu = config.nu_grid.size();
        double second = 0.0;

        if (inu == 0) {
            const double d = config.nu_grid[1] - config.nu_grid[0];
            second = (h_mat[iq][iy][0] - 2.0 * h_mat[iq][iy][1] + h_mat[iq][iy][2]) / (d * d);
        } else if (inu + 1 == nnu) {
            const double d = config.nu_grid[nnu - 1] - config.nu_grid[nnu - 2];
            second = (h_mat[iq][iy][nnu - 3] - 2.0 * h_mat[iq][iy][nnu - 2] + h_mat[iq][iy][nnu - 1]) / (d * d);
        } else {
            const double d = config.nu_grid[inu + 1] - config.nu_grid[inu];
            second = (h_mat[iq][iy][inu + 1] - 2.0 * h_mat[iq][iy][inu] + h_mat[iq][iy][inu - 1]) / (d * d);
        }

        return 0.5 * config.eta_nu * config.eta_nu * second;
    }

    void update_policies(const std::vector<std::vector<std::vector<double>>>& h_mat) {
        const auto h_fn = make_h_interp(h_mat);
        for (auto& tier : tiers) {
            tier->build_policy(h_fn, config.q_grid, config.y_grid, config.nu_grid);
        }
    }

    std::vector<std::vector<std::vector<double>>> bellman_rhs_from_policies(
        const std::vector<std::vector<std::vector<double>>>& h_mat
    ) const {
        const auto h_fn = make_h_interp(h_mat);

        const std::size_t nq = config.q_grid.size();
        const std::size_t ny = config.y_grid.size();
        const std::size_t nnu = config.nu_grid.size();

        std::vector<std::vector<std::vector<double>>> rhs(
            nq,
            std::vector<std::vector<double>>(ny, std::vector<double>(nnu, 0.0))
        );

        for (std::size_t iq = 0; iq < nq; ++iq) {
            const double q = config.q_grid[iq];

            for (std::size_t iy = 0; iy < ny; ++iy) {
                const double y = config.y_grid[iy];

                for (std::size_t inu = 0; inu < nnu; ++inu) {
                    const double nu = config.nu_grid[inu];
                    double val = q * y;

                    val += y_drift_term(h_mat, iq, iy, inu);
                    val += nu_drift_term(h_mat, iq, iy, inu);
                    val += nu_diffusion_term(h_mat, iq, iy, inu);

                    val -= std::exp(2.0 * nu) * penalty->value(q);

                    for (const auto& tier : tiers) {
                        for (std::size_t iz = 0; iz < tier->sizes.size(); ++iz) {
                            const double z = tier->sizes[iz];
                            const double jump = tier->jump_size(z);

                            const double d_b = tier->policy.delta_at_index(iq, iy, inu, iz, "bid");
                            const double lam_b = tier->arrival_rate(d_b, z);
                            const double cont_b = h_fn(q + z, y - jump, nu) - h_fn(q, y, nu);
                            val += lam_b * (z * d_b + cont_b);

                            const double d_a = tier->policy.delta_at_index(iq, iy, inu, iz, "ask");
                            const double lam_a = tier->arrival_rate(d_a, z);
                            const double cont_a = h_fn(q - z, y + jump, nu) - h_fn(q, y, nu);
                            val += lam_a * (z * d_a + cont_a);
                        }
                    }

                    rhs[iq][iy][inu] = val;
                }
            }
        }

        return rhs;
    }

    HJBSolution solve() {
        const std::size_t nq = config.q_grid.size();
        const std::size_t ny = config.y_grid.size();
        const std::size_t nnu = config.nu_grid.size();

        std::vector<std::vector<std::vector<double>>> h(
            nq,
            std::vector<std::vector<double>>(ny, std::vector<double>(nnu, 0.0))
        );

        std::size_t q0_idx = 0;
        std::size_t y0_idx = 0;
        std::size_t nu0_idx = 0;

        double best_abs_q = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < nq; ++i) {
            const double a = std::abs(config.q_grid[i]);
            if (a < best_abs_q) {
                best_abs_q = a;
                q0_idx = i;
            }
        }

        double best_abs_y = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < ny; ++i) {
            const double a = std::abs(config.y_grid[i]);
            if (a < best_abs_y) {
                best_abs_y = a;
                y0_idx = i;
            }
        }

        double best_abs_nu = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < nnu; ++i) {
            const double a = std::abs(config.nu_grid[i]);
            if (a < best_abs_nu) {
                best_abs_nu = a;
                nu0_idx = i;
            }
        }

        bool converged = false;
        int iterations_used = 0;
        double final_max_h_change = std::numeric_limits<double>::infinity();
        double final_max_rhs = std::numeric_limits<double>::infinity();
        int consecutive_passes = 0;

        std::vector<double> hist_h;
        std::vector<double> hist_rhs;

        for (int it = 1; it <= config.n_iter; ++it) {
            const auto h_old = h;

            update_policies(h_old);
            const auto rhs = bellman_rhs_from_policies(h_old);

            std::vector<std::vector<std::vector<double>>> dh = h;

            for (std::size_t iq = 0; iq < nq; ++iq) {
                for (std::size_t iy = 0; iy < ny; ++iy) {
                    for (std::size_t inu = 0; inu < nnu; ++inu) {
                        h[iq][iy][inu] = h_old[iq][iy][inu] + config.dt * rhs[iq][iy][inu];
                        dh[iq][iy][inu] = h[iq][iy][inu] - h_old[iq][iy][inu];
                    }
                }
            }

            const double anchor = h[q0_idx][y0_idx][nu0_idx];
            for (std::size_t iq = 0; iq < nq; ++iq) {
                for (std::size_t iy = 0; iy < ny; ++iy) {
                    for (std::size_t inu = 0; inu < nnu; ++inu) {
                        h[iq][iy][inu] -= anchor;
                    }
                }
            }

            const double max_h_change = max_abs_3d(dh);
            const double max_rhs_now = max_abs_3d(rhs);

            hist_h.push_back(max_h_change);
            hist_rhs.push_back(max_rhs_now);

            final_max_h_change = max_h_change;
            final_max_rhs = max_rhs_now;
            iterations_used = it;

            const bool passes_now =
                it >= config.min_iter &&
                max_h_change <= config.tol_h &&
                max_rhs_now <= config.tol_rhs;

            if (passes_now) {
                consecutive_passes += 1;
            } else {
                consecutive_passes = 0;
            }

            if (config.early_stop &&
                consecutive_passes >= config.consecutive_passes_required) {
                converged = true;
                break;
            }
        }

        update_policies(h);

        HJBSolution out;
        out.h = std::move(h);
        out.q_grid = config.q_grid;
        out.y_grid = config.y_grid;
        out.nu_grid = config.nu_grid;
        out.tiers = tiers;
        out.converged = converged;
        out.iterations_used = iterations_used;
        out.final_max_h_change = final_max_h_change;
        out.final_max_rhs = final_max_rhs;
        out.history_max_h_change = std::move(hist_h);
        out.history_max_rhs = std::move(hist_rhs);
        return out;
    }
};

} // namespace hjb