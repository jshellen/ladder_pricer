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

inline std::size_t nearest_to_zero_index(const std::vector<double>& x) {
    if (x.empty()) {
        throw std::invalid_argument("nearest_to_zero_index: input cannot be empty.");
    }

    std::size_t best_idx = 0;
    double best_abs = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double a = std::abs(x[i]);
        if (a < best_abs) {
            best_abs = a;
            best_idx = i;
        }
    }
    return best_idx;
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
        const double x = -(delta - shift + volume_shift * (z - 1.0)) * steepness;
        const double ex = std::exp(x);
        return 1.0 / (1.0 + ex);
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
        validate_strictly_increasing(sizes, "QuotePolicy.sizes");

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

    double delta_at_index(std::size_t i, std::size_t j, Side side) const {
        return matrix(side)[i][j];
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
// Price tier
// ============================================================

struct PriceTier {
    std::string name;
    std::vector<double> sizes;
    std::shared_ptr<FlowCurve> flow_curve;
    std::shared_ptr<MarkoutModel> markout_model;

    double delta_min{-5.0};
    double delta_max{5.0};

    QuotePolicy policy;

    PriceTier() = default;

    PriceTier(
        std::string name_,
        std::vector<double> sizes_,
        std::shared_ptr<FlowCurve> flow_curve_,
        std::shared_ptr<MarkoutModel> markout_model_,
        double delta_min_ = -5.0,
        double delta_max_ = 5.0
    )
        : name(std::move(name_)),
          sizes(std::move(sizes_)),
          flow_curve(std::move(flow_curve_)),
          markout_model(std::move(markout_model_)),
          delta_min(delta_min_),
          delta_max(delta_max_) {
        validate();
    }

    void validate() const {
        validate_strictly_increasing(sizes, "PriceTier.sizes");

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

// ============================================================
// Solver config
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
        validate_strictly_increasing(q_grid, "SolverConfig.q_grid");

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
// Policy builder
// ============================================================

struct TierPolicyBuilder {
    const PriceTier* tier{nullptr};
    double spread{0.0};
    const GoldenSectionSearch* optimizer{nullptr};
    const LadderBoundsPolicy* bounds_policy{nullptr};
    LinearInterpolator1D h;

    TierPolicyBuilder() = default;

    TierPolicyBuilder(
        const PriceTier& tier_,
        double spread_,
        const GoldenSectionSearch& optimizer_,
        const LadderBoundsPolicy& bounds_policy_
    )
        : tier(&tier_),
          spread(spread_),
          optimizer(&optimizer_),
          bounds_policy(&bounds_policy_) {}

    void set_h_view(const std::vector<double>& q_grid, const std::vector<double>& h_vec) {
        h = LinearInterpolator1D{&q_grid, &h_vec};
    }

    const PriceTier& price_tier() const {
        if (tier == nullptr) {
            throw std::runtime_error("TierPolicyBuilder is not initialized.");
        }
        return *tier;
    }

    const GoldenSectionSearch& searcher() const {
        if (optimizer == nullptr) {
            throw std::runtime_error("TierPolicyBuilder optimizer is not initialized.");
        }
        return *optimizer;
    }

    const LadderBoundsPolicy& bounds() const {
        if (bounds_policy == nullptr) {
            throw std::runtime_error("TierPolicyBuilder bounds are not initialized.");
        }
        return *bounds_policy;
    }

    static double next_inventory(double q, double z, Side side) {
        return side == Side::Bid ? q + z : q - z;
    }

    double continuation_change(double q, double z, Side side) const {
        return h(next_inventory(q, z, side)) - h(q);
    }

    double immediate_edge(double z, double delta, double mu) const {
        return z * spread * (0.5 - delta) - z * mu;
    }

    double objective(double q, double z, Side side, double delta) const {
        const double lam = price_tier().arrival_rate(delta, z);
        const double mu = price_tier().expected_markout(z);
        return lam * (immediate_edge(z, delta, mu) + continuation_change(q, z, side));
    }

    void build_side_ladder(
        std::vector<double>& delta,
        double q,
        Side side,
        const std::vector<double>* prev_delta = nullptr
    ) const {
        const std::size_t n = price_tier().sizes.size();
        delta.assign(n, 0.0);

        for (std::size_t j = 0; j < n; ++j) {
            auto [lower, upper] =
                bounds().bounds_for_rung(j, delta, q, side, prev_delta);

            lower = std::max(lower, price_tier().delta_min);
            upper = std::min(upper, price_tier().delta_max);

            if (lower > upper + 1e-12) {
                throw std::runtime_error(
                    "Infeasible ladder bounds for tier '" + price_tier().name +
                    "', side '" + std::string(side_name(side)) + "'."
                );
            }
            if (upper < lower) {
                upper = lower;
            }

            const double z = price_tier().sizes[j];
            const auto obj = [&](double d) {
                return objective(q, z, side, d);
            };

            delta[j] = clamp(searcher().maximize(obj, lower, upper), lower, upper);
        }
    }

    void build_policy_inplace(QuotePolicy& policy) const {
        const auto& q_grid = policy.q_grid;
        const std::size_t nq = q_grid.size();
        const std::size_t q0_idx = nearest_to_zero_index(q_grid);

        build_side_ladder(policy.ask[q0_idx], q_grid[q0_idx], Side::Ask, nullptr);
        build_side_ladder(policy.bid[q0_idx], q_grid[q0_idx], Side::Bid, nullptr);

        for (std::size_t i = q0_idx + 1; i < nq; ++i) {
            build_side_ladder(policy.ask[i], q_grid[i], Side::Ask, &policy.ask[i - 1]);
            build_side_ladder(policy.bid[i], q_grid[i], Side::Bid, &policy.bid[i - 1]);
        }

        for (std::size_t i = q0_idx; i-- > 0;) {
            build_side_ladder(policy.ask[i], q_grid[i], Side::Ask, &policy.ask[i + 1]);
            build_side_ladder(policy.bid[i], q_grid[i], Side::Bid, &policy.bid[i + 1]);
        }
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
    std::vector<std::shared_ptr<PriceTier>> tiers;
    SolverDiagnostics diagnostics;
};

// ============================================================
// Solver
// ============================================================

struct HJBLadderSolver {
    SolverConfig config;
    std::shared_ptr<InventoryPenalty> penalty;
    std::vector<std::shared_ptr<PriceTier>> tiers;

    GoldenSectionSearch optimizer;
    std::vector<LadderBoundsPolicy> tier_bounds_;
    std::vector<TierPolicyBuilder> tier_policy_builders_;

    HJBLadderSolver(
        SolverConfig config_,
        std::shared_ptr<InventoryPenalty> penalty_,
        std::vector<std::shared_ptr<PriceTier>> tiers_
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
            tier->policy.reset_shape(config.q_grid, tier->sizes);
        }
    }

    void prepare_solve_context() {
        validate_problem_definition();
        initialize_policy_shapes();

        tier_bounds_.clear();
        tier_bounds_.reserve(tiers.size());
        for (const auto& tier : tiers) {
            tier_bounds_.emplace_back(tier->delta_min, tier->delta_max);
        }

        tier_policy_builders_.clear();
        tier_policy_builders_.reserve(tiers.size());
        for (std::size_t k = 0; k < tiers.size(); ++k) {
            tier_policy_builders_.emplace_back(
                *tiers[k],
                config.spread,
                optimizer,
                tier_bounds_[k]
            );
        }
    }

    LinearInterpolator1D make_h_view(const std::vector<double>& h_vec) const {
        return LinearInterpolator1D{&config.q_grid, &h_vec};
    }

    void update_policies_unchecked(const std::vector<double>& h_vec) {
        for (std::size_t k = 0; k < tiers.size(); ++k) {
            tier_policy_builders_[k].set_h_view(config.q_grid, h_vec);
            tier_policy_builders_[k].build_policy_inplace(tiers[k]->policy);
        }
    }

    double bellman_rhs_from_policies_unchecked(
        const std::vector<double>& h_vec,
        std::vector<double>& rhs
    ) const {
        const LinearInterpolator1D h_view = make_h_view(h_vec);
        double max_rhs_abs = 0.0;

        for (std::size_t i = 0; i < config.q_grid.size(); ++i) {
            const double q = config.q_grid[i];
            double val = -penalty->value(q) + config.spot_drift * q;

            for (const auto& tier : tiers) {
                for (std::size_t j = 0; j < tier->sizes.size(); ++j) {
                    const double z = tier->sizes[j];
                    const double mu = tier->expected_markout(z);

                    const double d_b = tier->policy.delta_at_index(i, j, Side::Bid);
                    const double dq_b = h_view(q + z) - h_view(q);
                    const double lam_b = tier->arrival_rate(d_b, z);
                    val += lam_b * (config.spread * z * (0.5 - d_b) - z * mu + dq_b);

                    const double d_a = tier->policy.delta_at_index(i, j, Side::Ask);
                    const double dq_a = h_view(q - z) - h_view(q);
                    const double lam_a = tier->arrival_rate(d_a, z);
                    val += lam_a * (config.spread * z * (0.5 - d_a) - z * mu + dq_a);
                }
            }

            rhs[i] = val;
            max_rhs_abs = std::max(max_rhs_abs, std::abs(val));
        }

        return max_rhs_abs;
    }

    HJBSolution solve() {
        prepare_solve_context();

        const std::size_t nq = config.q_grid.size();
        const std::size_t q0_idx = nearest_to_zero_index(config.q_grid);

        std::vector<double> h(nq, 0.0);
        std::vector<double> h_next(nq, 0.0);
        std::vector<double> rhs(nq, 0.0);

        SolverDiagnostics diagnostics(config.n_iter);

        for (int it = 1; it <= config.n_iter; ++it) {
            update_policies_unchecked(h);
            const double max_rhs_now = bellman_rhs_from_policies_unchecked(h, rhs);

            for (std::size_t i = 0; i < nq; ++i) {
                h_next[i] = h[i] + config.dt * rhs[i];
            }

            const double h0 = h_next[q0_idx];
            double max_h_change = 0.0;
            for (std::size_t i = 0; i < nq; ++i) {
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