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

inline double max_abs(const std::vector<double>& x) {
    double m = 0.0;
    for (double v : x) {
        m = std::max(m, std::abs(v));
    }
    return m;
}

inline std::vector<double> diff(const std::vector<double>& x) {
    if (x.size() < 2) return {};
    std::vector<double> out(x.size() - 1);
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        out[i] = x[i + 1] - x[i];
    }
    return out;
}

inline double interp_linear(
    const std::vector<double>& grid,
    const std::vector<double>& vals,
    double x
) {
    if (grid.size() != vals.size()) {
        throw std::invalid_argument("interp_linear: grid and vals size mismatch.");
    }
    if (grid.empty()) {
        throw std::invalid_argument("interp_linear: empty input.");
    }
    if (grid.size() == 1) {
        return vals.front();
    }

    if (x <= grid.front()) {
        double x0 = grid[0], x1 = grid[1];
        double y0 = vals[0], y1 = vals[1];
        double t = (x - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }
    if (x >= grid.back()) {
        std::size_t n = grid.size();
        double x0 = grid[n - 2], x1 = grid[n - 1];
        double y0 = vals[n - 2], y1 = vals[n - 1];
        double t = (x - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

    auto it = std::upper_bound(grid.begin(), grid.end(), x);
    std::size_t idx = static_cast<std::size_t>(it - grid.begin() - 1);
    double x0 = grid[idx], x1 = grid[idx + 1];
    double y0 = vals[idx], y1 = vals[idx + 1];
    double t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

// ============================================================
// Abstract model components
// ============================================================

struct FlowCurve {
    virtual ~FlowCurve() = default;
    virtual double arrival_rate(double delta, double z) const = 0;
};

struct MarkoutModel {
    virtual ~MarkoutModel() = default;
    virtual double expected_markout(double z) const = 0;
};

struct InventoryPenalty {
    virtual ~InventoryPenalty() = default;
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
        double Az = A(z);
        double mz = m(z);

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

struct SqrtMarkoutModel final : public MarkoutModel {
    double base{0.005};
    double coeff{0.003};

    SqrtMarkoutModel() = default;
    SqrtMarkoutModel(double base_, double coeff_) : base(base_), coeff(coeff_) {}

    double expected_markout(double z) const override {
        return base + coeff * std::sqrt(z);
    }
};

struct PolynomialInventoryPenalty final : public InventoryPenalty {
    double risk_aversion{2.0};
    double sigma{0.25};
    double tau0{2.0};
    double cubic_coeff{0.1};
    double quartic_coeff{0.0015};

    PolynomialInventoryPenalty() = default;

    PolynomialInventoryPenalty(
        double risk_aversion_,
        double sigma_,
        double tau0_,
        double cubic_coeff_,
        double quartic_coeff_
    )
        : risk_aversion(risk_aversion_),
          sigma(sigma_),
          tau0(tau0_),
          cubic_coeff(cubic_coeff_),
          quartic_coeff(quartic_coeff_) {}

    double value(double q) const override {
        const double x = std::abs(q);
        return risk_aversion * sigma * sigma *
               (tau0 * x * x + cubic_coeff * x * x * x + quartic_coeff * x * x * x * x);
    }
};

// ============================================================
// Quote policy
// ============================================================

struct QuotePolicy {
    std::vector<double> q_grid;
    std::vector<double> sizes;
    std::vector<std::vector<double>> bid;  // shape: nq x nz
    std::vector<std::vector<double>> ask;  // shape: nq x nz

    QuotePolicy() = default;

    QuotePolicy(
        std::vector<double> q_grid_,
        std::vector<double> sizes_,
        std::vector<std::vector<double>> bid_,
        std::vector<std::vector<double>> ask_
    )
        : q_grid(std::move(q_grid_)),
          sizes(std::move(sizes_)),
          bid(std::move(bid_)),
          ask(std::move(ask_)) {
        validate();
    }

    void validate() const {
        const std::size_t nq = q_grid.size();
        const std::size_t nz = sizes.size();

        if (bid.size() != nq || ask.size() != nq) {
            throw std::invalid_argument("QuotePolicy: row count mismatch.");
        }
        for (const auto& row : bid) {
            if (row.size() != nz) {
                throw std::invalid_argument("QuotePolicy: bid column count mismatch.");
            }
        }
        for (const auto& row : ask) {
            if (row.size() != nz) {
                throw std::invalid_argument("QuotePolicy: ask column count mismatch.");
            }
        }
    }

    double delta_at_index(std::size_t i, std::size_t j, const std::string& side) const {
        if (side == "bid") return bid.at(i).at(j);
        if (side == "ask") return ask.at(i).at(j);
        throw std::invalid_argument("Unknown side: " + side);
    }

    double quote(double q, double z, const std::string& side) const {
        const auto& mat = (side == "bid") ? bid : ask;
        if (side != "bid" && side != "ask") {
            throw std::invalid_argument("Unknown side: " + side);
        }

        std::vector<double> vals_by_z(sizes.size(), 0.0);
        for (std::size_t j = 0; j < sizes.size(); ++j) {
            std::vector<double> col(q_grid.size(), 0.0);
            for (std::size_t i = 0; i < q_grid.size(); ++i) {
                col[i] = mat[i][j];
            }
            vals_by_z[j] = interp_linear(q_grid, col, q);
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
    std::shared_ptr<MarkoutModel> markout_model;

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
        std::shared_ptr<MarkoutModel> markout_model_,
        double delta_min_ = -0.5,
        double delta_max_ = 4.0,
        double golden_tol_ = 1e-4,
        int golden_max_iter_ = 32
    )
        : name(std::move(name_)),
          sizes(std::move(sizes_)),
          flow_curve(std::move(flow_curve_)),
          markout_model(std::move(markout_model_)),
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
        if (!markout_model) {
            throw std::invalid_argument("PriceTier: markout_model cannot be null.");
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

    double arrival_rate(double delta, double z) const {
        return flow_curve->arrival_rate(delta, z);
    }

    double expected_markout(double z) const {
        return markout_model->expected_markout(z);
    }

    double single_size_objective(
        double delta,
        const std::function<double(double)>& h_fn,
        double q,
        double z,
        const std::string& side
    ) const {
        const double q_next = next_inventory(q, z, side);
        const double dh = h_fn(q_next) - h_fn(q);
        const double lam = arrival_rate(delta, z);
        const double mu = expected_markout(z);
        return lam * (z * (delta - mu) + dh);
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

        // Smallest rung
        if (rung_idx == 0) {
            if (!prev_gaps.empty() && is_expand_mode(q, side)) {
                double required_total_gap = 0.0;
                for (double g : prev_gaps) required_total_gap += g;
                upper = std::min(upper, delta_max - required_total_gap);
            }
            return {lower, upper};
        }

        // Weak monotonicity in size: d_j >= d_{j-1}
        const double prev_curr = current_delta[rung_idx - 1];
        lower = std::max(lower, prev_curr);

        if (prev_gaps.empty()) {
            return {lower, upper};
        }

        const double prev_gap = prev_gaps[rung_idx - 1];

        if (is_shrink_mode(q, side)) {
            // d_j - d_{j-1} <= prev_gap
            upper = std::min(upper, prev_curr + prev_gap);
        } else if (is_expand_mode(q, side)) {
            // d_j - d_{j-1} >= prev_gap
            lower = std::max(lower, prev_curr + prev_gap);

            // Reserve room for future mandatory gaps
            double remaining_future_gap = 0.0;
            for (std::size_t k = rung_idx; k + 1 < n; ++k) {
                remaining_future_gap += prev_gaps[k];
            }
            upper = std::min(upper, delta_max - remaining_future_gap);
        }

        return {lower, upper};
    }

    std::vector<double> solve_side_ladder(
        const std::function<double(double)>& h_fn,
        double q,
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
                return single_size_objective(d, h_fn, q, z, side);
            };

            delta[j] = golden_search_max(obj, lower, upper);
            delta[j] = clamp(delta[j], lower, upper);
        }

        return delta;
    }

    QuotePolicy build_policy(
        const std::function<double(double)>& h_fn,
        const std::vector<double>& q_grid
    ) {
        const std::size_t nq = q_grid.size();
        const std::size_t nz = sizes.size();

        std::vector<std::vector<double>> bid(nq, std::vector<double>(nz, 0.0));
        std::vector<std::vector<double>> ask(nq, std::vector<double>(nz, 0.0));

        std::size_t q0_idx = 0;
        double best_abs = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < nq; ++i) {
            const double a = std::abs(q_grid[i]);
            if (a < best_abs) {
                best_abs = a;
                q0_idx = i;
            }
        }

        const double q0 = q_grid[q0_idx];
        ask[q0_idx] = solve_side_ladder(h_fn, q0, "ask", nullptr);
        bid[q0_idx] = solve_side_ladder(h_fn, q0, "bid", nullptr);

        // q > 0 outward
        for (std::size_t i = q0_idx + 1; i < nq; ++i) {
            const double q = q_grid[i];
            ask[i] = solve_side_ladder(h_fn, q, "ask", &ask[i - 1]);
            bid[i] = solve_side_ladder(h_fn, q, "bid", &bid[i - 1]);
        }

        // q < 0 outward
        for (std::size_t ii = q0_idx; ii-- > 0;) {
            const double q = q_grid[ii];
            ask[ii] = solve_side_ladder(h_fn, q, "ask", &ask[ii + 1]);
            bid[ii] = solve_side_ladder(h_fn, q, "bid", &bid[ii + 1]);
        }

        policy = QuotePolicy(q_grid, sizes, bid, ask);
        return policy;
    }

    double quote(double q, double z, const std::string& side) const {
        return policy.quote(q, z, side);
    }
};

// ============================================================
// Solver
// ============================================================

struct SolverConfig {
    std::vector<double> q_grid;
    double dt{0.002};
    int n_iter{140};

    bool early_stop{true};
    double tol_h{1e-5};
    double tol_rhs{1e-4};
    int min_iter{5};
    int consecutive_passes_required{3};

    void validate() const {
        if (q_grid.empty()) {
            throw std::invalid_argument("SolverConfig: q_grid cannot be empty.");
        }
        if (dt <= 0.0) {
            throw std::invalid_argument("SolverConfig: dt must be positive.");
        }
        if (n_iter < 1) {
            throw std::invalid_argument("SolverConfig: n_iter must be at least 1.");
        }
        if (tol_h < 0.0 || tol_rhs < 0.0) {
            throw std::invalid_argument("SolverConfig: tolerances must be nonnegative.");
        }
        if (min_iter < 0 || consecutive_passes_required < 1) {
            throw std::invalid_argument("SolverConfig: invalid stopping parameters.");
        }
    }
};

struct HJBSolution {
    std::vector<double> h;
    std::vector<double> q_grid;
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

    std::function<double(double)> make_h_interp(const std::vector<double>& h_vec) const {
        std::vector<double> grid = config.q_grid;
        std::vector<double> vals = h_vec;
        return [grid = std::move(grid), vals = std::move(vals)](double q) {
            return interp_linear(grid, vals, q);
        };
    }

    void update_policies(const std::vector<double>& h_vec) {
        auto h_fn = make_h_interp(h_vec);
        for (auto& tier : tiers) {
            tier->build_policy(h_fn, config.q_grid);
        }
    }

    std::vector<double> bellman_rhs_from_policies(const std::vector<double>& h_vec) const {
        auto h_fn = make_h_interp(h_vec);
        std::vector<double> rhs(h_vec.size(), 0.0);

        for (std::size_t i = 0; i < config.q_grid.size(); ++i) {
            const double q = config.q_grid[i];
            double val = -penalty->value(q);

            for (const auto& tier : tiers) {
                for (std::size_t j = 0; j < tier->sizes.size(); ++j) {
                    const double z = tier->sizes[j];
                    const double mu = tier->expected_markout(z);

                    const double d_b = tier->policy.delta_at_index(i, j, "bid");
                    const double qpb = h_fn(q + z) - h_fn(q);
                    const double lam_b = tier->arrival_rate(d_b, z);
                    val += lam_b * (z * (d_b - mu) + qpb);

                    const double d_a = tier->policy.delta_at_index(i, j, "ask");
                    const double qpa = h_fn(q - z) - h_fn(q);
                    const double lam_a = tier->arrival_rate(d_a, z);
                    val += lam_a * (z * (d_a - mu) + qpa);
                }
            }

            rhs[i] = val;
        }

        return rhs;
    }

    HJBSolution solve() {
        std::vector<double> h(config.q_grid.size(), 0.0);

        std::size_t q0_idx = 0;
        double best_abs = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < config.q_grid.size(); ++i) {
            const double a = std::abs(config.q_grid[i]);
            if (a < best_abs) {
                best_abs = a;
                q0_idx = i;
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
            const std::vector<double> h_old = h;

            update_policies(h_old);
            const std::vector<double> rhs = bellman_rhs_from_policies(h_old);

            for (std::size_t i = 0; i < h.size(); ++i) {
                h[i] = h_old[i] + config.dt * rhs[i];
            }

            const double h0 = h[q0_idx];
            for (double& v : h) {
                v -= h0;
            }

            std::vector<double> dh(h.size(), 0.0);
            for (std::size_t i = 0; i < h.size(); ++i) {
                dh[i] = h[i] - h_old[i];
            }

            const double max_h_change = max_abs(dh);
            const double max_rhs_now = max_abs(rhs);

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