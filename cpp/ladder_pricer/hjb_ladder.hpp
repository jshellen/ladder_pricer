#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hjb {

// ============================================================
// Common aliases and helpers
// ============================================================

using Matrix = std::vector<std::vector<double>>;

inline double clamp(double x, double lo, double hi) {
    return std::min(std::max(x, lo), hi);
}

inline void validate_strictly_increasing(
    const std::vector<double>& x,
    const std::string& name
) {
    if (x.empty()) {
        throw std::invalid_argument(name + " cannot be empty.");
    }
    for (std::size_t i = 1; i < x.size(); ++i) {
        if (x[i] <= x[i - 1]) {
            throw std::invalid_argument(name + " must be strictly increasing.");
        }
    }
}

inline void validate_positive_strictly_increasing(
    const std::vector<double>& x,
    const std::string& name
) {
    validate_strictly_increasing(x, name);
    for (double v : x) {
        if (v <= 0.0) {
            throw std::invalid_argument(name + " must contain positive values only.");
        }
    }
}

inline void validate_centered_symmetric_grid(
    const std::vector<double>& x,
    const std::string& name,
    double tol = 1e-10
) {
    validate_strictly_increasing(x, name);

    if (x.size() < 3) {
        throw std::invalid_argument(name + " must contain at least 3 points.");
    }
    if (x.size() % 2 == 0) {
        throw std::invalid_argument(name + " must have odd length so that 0 is exactly in the middle.");
    }

    const std::size_t mid = x.size() / 2;
    if (std::abs(x[mid]) > tol) {
        throw std::invalid_argument(name + " must contain 0 exactly at the middle index.");
    }

    for (std::size_t i = 0; i < mid; ++i) {
        if (std::abs(x[i] + x[x.size() - 1 - i]) > tol) {
            throw std::invalid_argument(name + " must be symmetric around 0.");
        }
    }
}

inline std::pair<std::size_t, double> locate_segment_with_weight(
    const std::vector<double>& grid,
    double x
) {
    if (grid.empty()) {
        throw std::invalid_argument("locate_segment_with_weight: grid cannot be empty.");
    }
    if (grid.size() == 1) {
        return {0, 0.0};
    }

    if (x <= grid.front()) {
        const double x0 = grid[0];
        const double x1 = grid[1];
        return {0, (x - x0) / (x1 - x0)};
    }

    if (x >= grid.back()) {
        const std::size_t i = grid.size() - 2;
        const double x0 = grid[i];
        const double x1 = grid[i + 1];
        return {i, (x - x0) / (x1 - x0)};
    }

    const auto it = std::upper_bound(grid.begin(), grid.end(), x);
    const std::size_t i = static_cast<std::size_t>(it - grid.begin() - 1);
    const double x0 = grid[i];
    const double x1 = grid[i + 1];
    return {i, (x - x0) / (x1 - x0)};
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

    const auto [i, t] = locate_segment_with_weight(grid, x);
    return vals[i] + t * (vals[i + 1] - vals[i]);
}

enum class Side : unsigned char { Bid, Ask };

inline const char* side_name(Side side) {
    return side == Side::Bid ? "bid" : "ask";
}

inline constexpr Side kAllSides[2] = {Side::Bid, Side::Ask};

inline double next_inventory(double q, double z, Side side) {
    return side == Side::Bid ? q + z : q - z;
}

// ============================================================
// Abstract model components
// ============================================================

struct FlowCurve {
    virtual ~FlowCurve() = default;
    virtual double hit_ratio(double delta, double z) const = 0;
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
    double shift{0.20};
    double steepness{10.0};
    double volume_shift{0.05};
    double z_floor{1e-8};

    LogisticFlowCurve() = default;

    LogisticFlowCurve(
        double A0_,
        double theta_,
        double shift_,
        double steepness_,
        double volume_shift_,
        double z_floor_ = 1e-8
    )
        : A0(A0_),
          theta(theta_),
          shift(shift_),
          steepness(steepness_),
          volume_shift(volume_shift_),
          z_floor(z_floor_) {}

    double A(double z) const {
        z = std::max(z, z_floor);
        return A0 * std::pow(z, -theta);
    }

    double hit_ratio(double delta, double z) const override {
        const double y = (delta - shift + volume_shift * (z - 1.0)) * steepness;

        if (y >= 0.0) {
            const double e = std::exp(-y);
            return 1.0 / (1.0 + e);
        } else {
            const double e = std::exp(y);
            return e / (1.0 + e);
        }
    }

    double arrival_rate(double delta, double z) const override {
        return A(z) * hit_ratio(delta, z);
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
// Lightweight interpolation view for h(q)
// ============================================================

struct LinearInterpolator1D {
    const std::vector<double>* grid{nullptr};
    const std::vector<double>* vals{nullptr};

    double operator()(double x) const {
        if (grid == nullptr || vals == nullptr) {
            throw std::runtime_error("LinearInterpolator1D is not initialized.");
        }
        return interp_linear(*grid, *vals, x);
    }
};

// ============================================================
// Common trade-value helpers
// ============================================================

inline double continuation_change(
    const LinearInterpolator1D& h,
    double q,
    double z,
    Side side
) {
    return h(next_inventory(q, z, side)) - h(q);
}

inline double immediate_edge(
    double spread,
    double z,
    double delta,
    double mu
) {
    return z * spread * (0.5 - delta) - z * mu;
}

inline double trade_contribution(
    const FlowCurve& flow_curve,
    const MarkoutModel& markout_model,
    const LinearInterpolator1D& h,
    double spread,
    double q,
    double z,
    Side side,
    double delta
) {
    const double lam = flow_curve.arrival_rate(delta, z);
    const double mu = markout_model.expected_markout(z);
    const double dq = continuation_change(h, q, z, side);
    return lam * (immediate_edge(spread, z, delta, mu) + dq);
}

// ============================================================
// ECN parametric policy
// ============================================================

struct ExponentialECNPolicy {
    double delta_start{0.10};
    double delta_target{0.50};
    double decay{2.0};
    double decay_min{0.25};
    double decay_max{10.0};

    ExponentialECNPolicy() = default;

    ExponentialECNPolicy(
        double delta_start_,
        double delta_target_,
        double decay_,
        double decay_min_,
        double decay_max_
    )
        : delta_start(delta_start_),
          delta_target(delta_target_),
          decay(decay_),
          decay_min(decay_min_),
          decay_max(decay_max_) {
        validate();
    }

    void validate() const {
        if (decay_min <= 0.0) {
            throw std::invalid_argument("ExponentialECNPolicy: decay_min must be positive.");
        }
        if (decay_max <= decay_min) {
            throw std::invalid_argument("ExponentialECNPolicy: decay_max must be > decay_min.");
        }
        if (decay <= 0.0) {
            throw std::invalid_argument("ExponentialECNPolicy: decay must be positive.");
        }
        if (delta_target < delta_start) {
            throw std::invalid_argument(
                "ExponentialECNPolicy: delta_target must be >= delta_start."
            );
        }
    }

    double delta_at_abs_inventory(double q_abs) const {
        return delta_at_abs_inventory_with_decay(q_abs, decay);
    }

    double delta_at_abs_inventory_with_decay(double q_abs, double decay_) const {
        const double x = std::max(q_abs - 1.0, 0.0);
        return delta_target - (delta_target - delta_start) * std::exp(-x / decay_);
    }
};

// ============================================================
// Quote analytics
// ============================================================

struct QuoteSummary {
    double delta{0.0};

    double price_improvement_frac{0.0};
    double price_improvement_pct_of_spread{0.0};
    double price_improvement_pips{0.0};

    double quote_relative_to_mid{0.0};
    double quote_relative_to_mid_pips{0.0};

    double distance_to_mid{0.0};
    double distance_to_mid_pips{0.0};

    double reference_delta{0.0};
    double volume_premium_pips{0.0};

    double quote_price{0.0};
};

struct QuoteMetrics {
    static constexpr double pips_per_unit = 10000.0;

    static double price_improvement(double delta, double spread) {
        return spread * delta;
    }

    static double price_improvement_pct_of_spread(double delta) {
        return 100.0 * delta;
    }

    static double price_improvement_pips(double delta, double spread) {
        return pips_per_unit * price_improvement(delta, spread);
    }

    static double quote_relative_to_mid(double delta, Side side, double spread) {
        return side == Side::Bid
            ? spread * (delta - 0.5)
            : spread * (0.5 - delta);
    }

    static double quote_relative_to_mid_pips(double delta, Side side, double spread) {
        return pips_per_unit * quote_relative_to_mid(delta, side, spread);
    }

    static double distance_to_mid(double delta, double spread) {
        return std::abs(spread * (0.5 - delta));
    }

    static double distance_to_mid_pips(double delta, double spread) {
        return pips_per_unit * distance_to_mid(delta, spread);
    }

    static double volume_premium_pips(double delta_ref, double delta_cur, double spread) {
        return pips_per_unit * spread * (delta_ref - delta_cur);
    }

    static double quote_price(double mid, double delta, Side side, double spread) {
        return mid + quote_relative_to_mid(delta, side, spread);
    }

    static QuoteSummary make_summary(
        double delta,
        double delta_ref,
        Side side,
        double mid,
        double spread
    ) {
        QuoteSummary out;
        out.delta = delta;
        out.price_improvement_frac = price_improvement(delta, spread);
        out.price_improvement_pct_of_spread = price_improvement_pct_of_spread(delta);
        out.price_improvement_pips = price_improvement_pips(delta, spread);
        out.quote_relative_to_mid = quote_relative_to_mid(delta, side, spread);
        out.quote_relative_to_mid_pips = quote_relative_to_mid_pips(delta, side, spread);
        out.distance_to_mid = distance_to_mid(delta, spread);
        out.distance_to_mid_pips = distance_to_mid_pips(delta, spread);
        out.reference_delta = delta_ref;
        out.volume_premium_pips = volume_premium_pips(delta_ref, delta, spread);
        out.quote_price = quote_price(mid, delta, side, spread);
        return out;
    }
};

// ============================================================
// Quote policy
// ============================================================

struct QuotePolicy {
    std::vector<double> q_grid;
    std::vector<double> sizes;
    Matrix bid;
    Matrix ask;

    QuotePolicy() = default;

    QuotePolicy(
        std::vector<double> q_grid_,
        std::vector<double> sizes_,
        Matrix bid_,
        Matrix ask_
    )
        : q_grid(std::move(q_grid_)),
          sizes(std::move(sizes_)),
          bid(std::move(bid_)),
          ask(std::move(ask_)) {
        validate();
    }

    static void resize_matrix(Matrix& mat, std::size_t rows, std::size_t cols) {
        mat.resize(rows);
        for (auto& row : mat) {
            row.resize(cols);
            std::fill(row.begin(), row.end(), 0.0);
        }
    }

    void reset_shape(
        const std::vector<double>& q_grid_,
        const std::vector<double>& sizes_
    ) {
        q_grid = q_grid_;
        sizes = sizes_;

        const std::size_t nq = q_grid.size();
        const std::size_t nz = sizes.size();

        resize_matrix(bid, nq, nz);
        resize_matrix(ask, nq, nz);
    }

    void validate() const {
        validate_strictly_increasing(q_grid, "QuotePolicy.q_grid");
        validate_positive_strictly_increasing(sizes, "QuotePolicy.sizes");

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

    const Matrix& matrix(Side side) const {
        return side == Side::Bid ? bid : ask;
    }

    Matrix& matrix_mut(Side side) {
        return side == Side::Bid ? bid : ask;
    }

    double delta_at_index(std::size_t i, std::size_t j, Side side) const {
        return matrix(side)[i][j];
    }

    double& delta_ref(std::size_t i, std::size_t j, Side side) {
        return matrix_mut(side)[i][j];
    }

    double interp_q_column(const Matrix& mat, std::size_t j, double q) const {
        if (q_grid.size() == 1) {
            return mat[0][j];
        }
        const auto [i, tq] = locate_segment_with_weight(q_grid, q);
        return mat[i][j] + tq * (mat[i + 1][j] - mat[i][j]);
    }

    double delta(double q, double z, Side side) const {
        const Matrix& mat = matrix(side);

        if (sizes.size() == 1) {
            return interp_q_column(mat, 0, q);
        }

        const auto [j, tz] = locate_segment_with_weight(sizes, z);
        const double v0 = interp_q_column(mat, j, q);
        const double v1 = interp_q_column(mat, j + 1, q);
        return v0 + tz * (v1 - v0);
    }

    double reference_delta(double q, Side side) const {
        if (sizes.empty()) {
            throw std::runtime_error("QuotePolicy::reference_delta: no sizes available.");
        }
        return delta(q, sizes.front(), side);
    }

    QuoteSummary quote_summary(
        double q,
        double z,
        Side side,
        double mid,
        double spread
    ) const {
        const double d = delta(q, z, side);
        const double d_ref = reference_delta(q, side);
        return QuoteMetrics::make_summary(d, d_ref, side, mid, spread);
    }
};

// ============================================================
// Solver config and cached grid metadata
// ============================================================

struct SolverConfig {
    std::vector<double> q_grid;
    double dt{0.002};
    int n_iter{140};

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

// ============================================================
// Optimization utilities
// ============================================================

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

struct LadderBoundsPolicy {
    double delta_min{-5.0};
    double delta_max{5.0};

    LadderBoundsPolicy() = default;

    LadderBoundsPolicy(double delta_min_, double delta_max_)
        : delta_min(delta_min_), delta_max(delta_max_) {
        validate();
    }

    void validate() const {
        if (delta_max <= delta_min) {
            throw std::invalid_argument("LadderBoundsPolicy: delta_max must be > delta_min.");
        }
    }

    static bool is_shrink_mode(double q, Side side) {
        return (q > 0.0 && side == Side::Ask) || (q < 0.0 && side == Side::Bid);
    }

    static bool is_expand_mode(double q, Side side) {
        return (q > 0.0 && side == Side::Bid) || (q < 0.0 && side == Side::Ask);
    }

    static double total_gap(const std::vector<double>& delta) {
        if (delta.size() < 2) {
            return 0.0;
        }
        return delta.front() - delta.back();
    }

    static double gap_at(const std::vector<double>& delta, std::size_t j_minus_one) {
        return delta[j_minus_one] - delta[j_minus_one + 1];
    }

    static double remaining_future_gap(
        const std::vector<double>& delta,
        std::size_t rung_idx
    ) {
        if (delta.size() < 2 || rung_idx >= delta.size() - 1) {
            return 0.0;
        }
        return delta[rung_idx] - delta.back();
    }

    std::pair<double, double> bounds_for_rung(
        std::size_t rung_idx,
        const std::vector<double>& current_delta,
        double q,
        Side side,
        const std::vector<double>* prev_delta
    ) const {
        double lower = delta_min;
        double upper = delta_max;

        if (rung_idx == 0) {
            if (prev_delta != nullptr && is_expand_mode(q, side)) {
                lower = std::max(lower, delta_min + total_gap(*prev_delta));
            }
            return {lower, upper};
        }

        const double prev_curr = current_delta[rung_idx - 1];
        upper = std::min(upper, prev_curr);

        if (prev_delta == nullptr || prev_delta->size() < 2) {
            return {lower, upper};
        }

        const double prev_gap = gap_at(*prev_delta, rung_idx - 1);

        if (is_shrink_mode(q, side)) {
            lower = std::max(lower, prev_curr - prev_gap);
        } else if (is_expand_mode(q, side)) {
            upper = std::min(upper, prev_curr - prev_gap);
            lower = std::max(
                lower,
                delta_min + remaining_future_gap(*prev_delta, rung_idx)
            );
        }

        return {lower, upper};
    }
};

// ============================================================
// Tier hierarchy (data + validation only)
// ============================================================

struct Tier {
    std::string name;
    std::shared_ptr<FlowCurve> flow_curve;
    std::shared_ptr<MarkoutModel> markout_model;
    QuotePolicy policy;

    Tier() = default;

    Tier(
        std::string name_,
        std::shared_ptr<FlowCurve> flow_curve_,
        std::shared_ptr<MarkoutModel> markout_model_
    )
        : name(std::move(name_)),
          flow_curve(std::move(flow_curve_)),
          markout_model(std::move(markout_model_)) {}

    virtual ~Tier() = default;

    virtual const char* tier_type_name() const = 0;
    virtual const std::vector<double>& sizes() const = 0;
    virtual bool is_admissible(double q, double z, Side side) const = 0;

    virtual void validate() const {
        if (!flow_curve) {
            throw std::invalid_argument(
                std::string(tier_type_name()) + " tier: flow_curve cannot be null."
            );
        }
        if (!markout_model) {
            throw std::invalid_argument(
                std::string(tier_type_name()) + " tier: markout_model cannot be null."
            );
        }
    }

    void reset_policy_shape(const std::vector<double>& q_grid) {
        policy.reset_shape(q_grid, sizes());
    }

    double arrival_rate(double delta, double z) const {
        return flow_curve->arrival_rate(delta, z);
    }

    double expected_markout(double z) const {
        return markout_model->expected_markout(z);
    }

    double quote(double q, double z, Side side) const {
        return policy.delta(q, z, side);
    }

    QuoteSummary quote_summary(
        double q,
        double z,
        Side side,
        double mid,
        double spread
    ) const {
        return policy.quote_summary(q, z, side, mid, spread);
    }
};

struct MDPTier final : public Tier {
    std::vector<double> sizes_;
    double delta_min{-5.0};
    double delta_max{5.0};

    MDPTier() = default;

    MDPTier(
        std::string name_,
        std::vector<double> sizes__,
        std::shared_ptr<FlowCurve> flow_curve_,
        std::shared_ptr<MarkoutModel> markout_model_,
        double delta_min_ = -5.0,
        double delta_max_ = 5.0
    )
        : Tier(
              std::move(name_),
              std::move(flow_curve_),
              std::move(markout_model_)
          ),
          sizes_(std::move(sizes__)),
          delta_min(delta_min_),
          delta_max(delta_max_) {
        validate();
    }

    const char* tier_type_name() const override {
        return "mdp";
    }

    const std::vector<double>& sizes() const override {
        return sizes_;
    }

    void validate() const override {
        Tier::validate();
        validate_positive_strictly_increasing(sizes_, "MDPTier.sizes");
        if (delta_max <= delta_min) {
            throw std::invalid_argument("MDPTier: delta_max must be > delta_min.");
        }
    }

    bool is_admissible(double, double, Side) const override {
        return true;
    }
};

struct ECNTier final : public Tier {
    double delta_min{-5.0};
    double delta_max{5.0};
    ExponentialECNPolicy ecn_policy{};

    ECNTier() = default;

    ECNTier(
        std::string name_,
        std::shared_ptr<FlowCurve> flow_curve_,
        std::shared_ptr<MarkoutModel> markout_model_,
        double delta_min_ = -5.0,
        double delta_max_ = 5.0,
        ExponentialECNPolicy ecn_policy_ = ExponentialECNPolicy{}
    )
        : Tier(
              std::move(name_),
              std::move(flow_curve_),
              std::move(markout_model_)
          ),
          delta_min(delta_min_),
          delta_max(delta_max_),
          ecn_policy(std::move(ecn_policy_)) {
        validate();
    }

    const char* tier_type_name() const override {
        return "ecn";
    }

    const std::vector<double>& sizes() const override {
        static const std::vector<double> fixed_size{1.0};
        return fixed_size;
    }

    void validate() const override {
        Tier::validate();

        if (delta_max <= delta_min) {
            throw std::invalid_argument("ECNTier: delta_max must be > delta_min.");
        }

        ecn_policy.validate();

        if (ecn_policy.delta_start < delta_min || ecn_policy.delta_start > delta_max) {
            throw std::invalid_argument(
                "ECNTier: delta_start must lie inside [delta_min, delta_max]."
            );
        }
        if (ecn_policy.delta_target < delta_min || ecn_policy.delta_target > delta_max) {
            throw std::invalid_argument(
                "ECNTier: delta_target must lie inside [delta_min, delta_max]."
            );
        }
    }

    static bool is_active_side(double q, Side side, double tol = 1e-12) {
        return (q > tol && side == Side::Ask) || (q < -tol && side == Side::Bid);
    }

    bool is_admissible(double q, double z, Side side) const override {
        constexpr double tol = 1e-12;
        return std::abs(z - 1.0) <= tol && is_active_side(q, side, tol);
    }

    double active_delta(double q) const {
        return ecn_policy.delta_at_abs_inventory(std::abs(q));
    }

    double active_delta_with_decay(double q, double decay) const {
        return ecn_policy.delta_at_abs_inventory_with_decay(std::abs(q), decay);
    }
};

// ============================================================
// Diagnostics and solution
// ============================================================

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
        if (passes_now) {
            ++consecutive_passes;
        } else {
            consecutive_passes = 0;
        }

        if (early_stop && consecutive_passes >= consecutive_passes_required) {
            converged = true;
            return true;
        }
        return false;
    }
};

struct HJBSolution {
    std::vector<double> h;
    std::vector<double> q_grid;
    std::vector<std::shared_ptr<Tier>> tiers;
    SolverDiagnostics diagnostics;
};

// ============================================================
// Solver
// ============================================================

struct HJBLadderSolver {
    SolverConfig config;
    std::shared_ptr<InventoryPenalty> penalty;
    std::vector<std::shared_ptr<Tier>> tiers;

    GoldenSectionSearch optimizer;
    SolverGridMeta grid_meta_;

    HJBLadderSolver(
        SolverConfig config_,
        std::shared_ptr<InventoryPenalty> penalty_,
        std::vector<std::shared_ptr<Tier>> tiers_
    )
        : config(std::move(config_)),
          penalty(std::move(penalty_)),
          tiers(std::move(tiers_)),
          optimizer(config.golden_tol, config.golden_max_iter) {
        config.validate();

        if (!penalty) {
            throw std::invalid_argument("HJBLadderSolver: penalty cannot be null.");
        }
        for (const auto& tier : tiers) {
            if (!tier) {
                throw std::invalid_argument("HJBLadderSolver: tier cannot be null.");
            }
        }
    }

    void validate_problem_definition() const {
        config.validate();

        if (!penalty) {
            throw std::invalid_argument("HJBLadderSolver: penalty cannot be null.");
        }
        for (const auto& tier : tiers) {
            if (!tier) {
                throw std::invalid_argument("HJBLadderSolver: tier cannot be null.");
            }
            tier->validate();
        }
    }

    void initialize_policy_shapes() {
        for (auto& tier : tiers) {
            tier->reset_policy_shape(config.q_grid);
        }
    }

    void prepare_solve_context() {
        validate_problem_definition();
        grid_meta_ = build_solver_grid_meta(config.q_grid);
        initialize_policy_shapes();
    }

    LinearInterpolator1D make_h_view(const std::vector<double>& h_vec) const {
        return LinearInterpolator1D{&config.q_grid, &h_vec};
    }

    // --------------------------------------------------------
    // MDP policy construction
    // --------------------------------------------------------

    double mdp_objective(
        const MDPTier& tier,
        const LinearInterpolator1D& h,
        double q,
        double z,
        Side side,
        double delta
    ) const {
        return trade_contribution(
            *tier.flow_curve,
            *tier.markout_model,
            h,
            config.spread,
            q,
            z,
            side,
            delta
        );
    }

    void build_mdp_side_ladder(
        const MDPTier& tier,
        const LinearInterpolator1D& h,
        std::vector<double>& delta_row,
        double q,
        Side side,
        const std::vector<double>* prev_delta
    ) const {
        const std::size_t nz = tier.sizes_.size();
        const LadderBoundsPolicy bounds(tier.delta_min, tier.delta_max);

        delta_row.assign(nz, tier.delta_min);

        for (std::size_t j = 0; j < nz; ++j) {
            const double z = tier.sizes_[j];

            auto [lower, upper] =
                bounds.bounds_for_rung(j, delta_row, q, side, prev_delta);

            lower = std::max(lower, tier.delta_min);
            upper = std::min(upper, tier.delta_max);

            if (lower > upper + 1e-12) {
                throw std::runtime_error(
                    "Infeasible ladder bounds for MDP tier '" + tier.name +
                    "', side '" + std::string(side_name(side)) + "'."
                );
            }
            if (upper < lower) {
                upper = lower;
            }

            const auto obj = [&](double d) {
                return mdp_objective(tier, h, q, z, side, d);
            };

            delta_row[j] = clamp(optimizer.maximize(obj, lower, upper), lower, upper);
        }
    }

    void build_mdp_side_policy(
        const MDPTier& tier,
        const LinearInterpolator1D& h,
        Matrix& side_matrix,
        Side side
    ) const {
        const auto& q_grid = config.q_grid;
        const std::size_t nq = grid_meta_.nq;
        const std::size_t q0_idx = grid_meta_.q0_idx;

        build_mdp_side_ladder(
            tier,
            h,
            side_matrix[q0_idx],
            q_grid[q0_idx],
            side,
            nullptr
        );

        for (std::size_t i = q0_idx + 1; i < nq; ++i) {
            build_mdp_side_ladder(
                tier,
                h,
                side_matrix[i],
                q_grid[i],
                side,
                &side_matrix[i - 1]
            );
        }

        for (std::size_t i = q0_idx; i-- > 0;) {
            build_mdp_side_ladder(
                tier,
                h,
                side_matrix[i],
                q_grid[i],
                side,
                &side_matrix[i + 1]
            );
        }
    }

    void update_mdp_policy(
        MDPTier& tier,
        const LinearInterpolator1D& h
    ) const {
        build_mdp_side_policy(tier, h, tier.policy.bid, Side::Bid);
        build_mdp_side_policy(tier, h, tier.policy.ask, Side::Ask);
    }

    // --------------------------------------------------------
    // ECN policy construction
    // --------------------------------------------------------

    double ecn_local_objective(
        const ECNTier& tier,
        const LinearInterpolator1D& h,
        double q,
        double decay
    ) const {
        if (q > 0.0) {
            return trade_contribution(
                *tier.flow_curve,
                *tier.markout_model,
                h,
                config.spread,
                q,
                1.0,
                Side::Ask,
                tier.active_delta_with_decay(q, decay)
            );
        }
        if (q < 0.0) {
            return trade_contribution(
                *tier.flow_curve,
                *tier.markout_model,
                h,
                config.spread,
                q,
                1.0,
                Side::Bid,
                tier.active_delta_with_decay(q, decay)
            );
        }
        return 0.0;
    }

    double optimize_ecn_decay(
        const ECNTier& tier,
        const LinearInterpolator1D& h
    ) const {
        const auto objective = [&](double decay) {
            double total = 0.0;
            for (double q : config.q_grid) {
                total += ecn_local_objective(tier, h, q, decay);
            }
            return total;
        };

        return clamp(
            optimizer.maximize(
                objective,
                tier.ecn_policy.decay_min,
                tier.ecn_policy.decay_max
            ),
            tier.ecn_policy.decay_min,
            tier.ecn_policy.decay_max
        );
    }

    void build_ecn_policy_from_decay(
        ECNTier& tier,
        double decay
    ) const {
        const double parked_delta = tier.ecn_policy.delta_target;

        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            const double q = config.q_grid[i];

            tier.policy.delta_ref(i, 0, Side::Bid) = parked_delta;
            tier.policy.delta_ref(i, 0, Side::Ask) = parked_delta;

            if (q > 0.0) {
                tier.policy.delta_ref(i, 0, Side::Ask) =
                    tier.active_delta_with_decay(q, decay);
            } else if (q < 0.0) {
                tier.policy.delta_ref(i, 0, Side::Bid) =
                    tier.active_delta_with_decay(q, decay);
            }
        }
    }

    void update_ecn_policy(
        ECNTier& tier,
        const LinearInterpolator1D& h
    ) const {
        const double decay_star = optimize_ecn_decay(tier, h);
        tier.ecn_policy.decay = decay_star;
        build_ecn_policy_from_decay(tier, decay_star);
    }

    // --------------------------------------------------------
    // Tier dispatch
    // --------------------------------------------------------

    void update_tier_policy(
        Tier& tier,
        const LinearInterpolator1D& h
    ) const {
        if (auto* mdp = dynamic_cast<MDPTier*>(&tier)) {
            update_mdp_policy(*mdp, h);
            return;
        }
        if (auto* ecn = dynamic_cast<ECNTier*>(&tier)) {
            update_ecn_policy(*ecn, h);
            return;
        }
        throw std::runtime_error("Unknown tier type in update_tier_policy.");
    }

    double mdp_bellman_contribution(
        const MDPTier& tier,
        double q,
        std::size_t q_index,
        const LinearInterpolator1D& h
    ) const {
        double total = 0.0;

        for (Side side : kAllSides) {
            for (std::size_t j = 0; j < tier.sizes_.size(); ++j) {
                const double z = tier.sizes_[j];
                const double delta = tier.policy.delta_at_index(q_index, j, side);

                total += trade_contribution(
                    *tier.flow_curve,
                    *tier.markout_model,
                    h,
                    config.spread,
                    q,
                    z,
                    side,
                    delta
                );
            }
        }

        return total;
    }

    double ecn_bellman_contribution(
        const ECNTier& tier,
        double q,
        std::size_t q_index,
        const LinearInterpolator1D& h
    ) const {
        if (q > 0.0) {
            return trade_contribution(
                *tier.flow_curve,
                *tier.markout_model,
                h,
                config.spread,
                q,
                1.0,
                Side::Ask,
                tier.policy.delta_at_index(q_index, 0, Side::Ask)
            );
        }
        if (q < 0.0) {
            return trade_contribution(
                *tier.flow_curve,
                *tier.markout_model,
                h,
                config.spread,
                q,
                1.0,
                Side::Bid,
                tier.policy.delta_at_index(q_index, 0, Side::Bid)
            );
        }
        return 0.0;
    }

    double tier_bellman_contribution(
        const Tier& tier,
        double q,
        std::size_t q_index,
        const LinearInterpolator1D& h
    ) const {
        if (const auto* mdp = dynamic_cast<const MDPTier*>(&tier)) {
            return mdp_bellman_contribution(*mdp, q, q_index, h);
        }
        if (const auto* ecn = dynamic_cast<const ECNTier*>(&tier)) {
            return ecn_bellman_contribution(*ecn, q, q_index, h);
        }
        throw std::runtime_error("Unknown tier type in tier_bellman_contribution.");
    }

    // --------------------------------------------------------
    // Public-ish internal solver steps
    // --------------------------------------------------------

    void update_policies_unchecked(const std::vector<double>& h_vec) {
        const LinearInterpolator1D h_view = make_h_view(h_vec);
        for (auto& tier : tiers) {
            update_tier_policy(*tier, h_view);
        }
    }

    double bellman_rhs_from_policies_unchecked(
        const std::vector<double>& h_vec,
        std::vector<double>& rhs
    ) const {
        const LinearInterpolator1D h_view = make_h_view(h_vec);
        double max_rhs_abs = 0.0;

        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            const double q = config.q_grid[i];
            double val = -penalty->value(q) + config.spot_drift * q;

            for (const auto& tier : tiers) {
                val += tier_bellman_contribution(*tier, q, i, h_view);
            }

            rhs[i] = val;
            max_rhs_abs = std::max(max_rhs_abs, std::abs(val));
        }

        return max_rhs_abs;
    }

    HJBSolution solve() {
        prepare_solve_context();

        std::vector<double> h(grid_meta_.nq, 0.0);
        std::vector<double> h_next(grid_meta_.nq, 0.0);
        std::vector<double> rhs(grid_meta_.nq, 0.0);

        SolverDiagnostics diagnostics(config.n_iter);

        for (int it = 1; it <= config.n_iter; ++it) {
            update_policies_unchecked(h);
            const double max_rhs_now = bellman_rhs_from_policies_unchecked(h, rhs);

            for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
                h_next[i] = h[i] + config.dt * rhs[i];
            }

            const double h0 = h_next[grid_meta_.q0_idx];
            double max_h_change = 0.0;
            for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
                h_next[i] -= h0;
                max_h_change = std::max(max_h_change, std::abs(h_next[i] - h[i]));
            }

            diagnostics.record_iteration(it, max_h_change, max_rhs_now);

            const bool passes_now =
                it >= config.min_iter &&
                max_h_change <= config.tol_h &&
                max_rhs_now <= config.tol_rhs;

            h.swap(h_next);

            if (diagnostics.update_stopping_state(
                    passes_now,
                    config.early_stop,
                    config.consecutive_passes_required)) {
                break;
            }
        }

        update_policies_unchecked(h);

        HJBSolution out;
        out.h = std::move(h);
        out.q_grid = config.q_grid;
        out.tiers = tiers;
        out.diagnostics = std::move(diagnostics);
        return out;
    }
};

} // namespace hjb