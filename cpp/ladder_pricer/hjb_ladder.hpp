#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
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
        throw std::invalid_argument(
            name + " must have odd length so that 0 is exactly in the middle."
        );
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

// ============================================================
// Concrete model components
// ============================================================

struct LogisticFlowCurve {
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

    double hit_ratio(double delta, double z) const {
        const double y = (delta - shift + volume_shift * (z - 1.0)) * steepness;

        if (y >= 0.0) {
            const double e = std::exp(-y);
            return 1.0 / (1.0 + e);
        }

        const double e = std::exp(y);
        return e / (1.0 + e);
    }

    double arrival_rate(double delta, double z) const {
        return A(z) * hit_ratio(delta, z);
    }
};

struct SqrtMarkoutModel {
    double base{0.005};
    double coeff{0.003};

    SqrtMarkoutModel() = default;
    SqrtMarkoutModel(double base_, double coeff_) : base(base_), coeff(coeff_) {}

    double expected_markout(double z) const {
        return base + coeff * std::sqrt(z);
    }
};

struct ECNAdverseSelectionModel {
    double ecn_toxicity{0.003};
    double ecn_fee{0.0};
    double ecn_convergence_inventory{4.0};

    ECNAdverseSelectionModel() = default;

    ECNAdverseSelectionModel(
        double ecn_toxicity_,
        double ecn_fee_,
        double ecn_convergence_inventory_
    )
        : ecn_toxicity(ecn_toxicity_),
          ecn_fee(ecn_fee_),
          ecn_convergence_inventory(ecn_convergence_inventory_) {
        validate();
    }

    void validate() const {
        if (ecn_toxicity < 0.0) {
            throw std::invalid_argument(
                "ECNAdverseSelectionModel: ecn_toxicity must be nonnegative."
            );
        }
        if (ecn_fee < 0.0) {
            throw std::invalid_argument(
                "ECNAdverseSelectionModel: ecn_fee must be nonnegative."
            );
        }
        if (ecn_convergence_inventory <= 0.0) {
            throw std::invalid_argument(
                "ECNAdverseSelectionModel: ecn_convergence_inventory must be positive."
            );
        }
    }

    double toxicity_cost(double delta, double z) const {
        const double delta_pos = std::max(delta, 0.0);
        return z * ecn_toxicity * delta_pos;
    }

    double expected_cost(
        const SqrtMarkoutModel& base_markout_model,
        double delta,
        double z
    ) const {
        return z * base_markout_model.expected_markout(z) +
               toxicity_cost(delta, z) +
               ecn_fee;
    }
};

struct PolynomialInventoryPenalty {
    double risk_aversion{2.0};
    double sigma{0.25};
    double tau0{2.0};
    double tau1{0.1};
    double tau2{0.0015};

    PolynomialInventoryPenalty() = default;

    PolynomialInventoryPenalty(
        double risk_aversion_,
        double sigma_,
        double tau0_,
        double tau1_,
        double tau2_
    )
        : risk_aversion(risk_aversion_),
          sigma(sigma_),
          tau0(tau0_),
          tau1(tau1_),
          tau2(tau2_) {}

    double value(double q) const {
        const double x = std::abs(q);
        return risk_aversion * sigma * sigma *
               (tau0 * x * x + tau1 * x * x * x + tau2 * x * x * x * x);
    }
};

// ============================================================
// Lightweight interpolation view for h(q)
// ============================================================

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

// ============================================================
// Trade-value helpers
// ============================================================

inline double bid_trade_value(
    const LogisticFlowCurve& flow_curve,
    const SqrtMarkoutModel& markout_model,
    const LinearInterpolator1D& h,
    double spread,
    double q,
    double z,
    double delta
) {
    const double lam = flow_curve.arrival_rate(delta, z);
    const double mu = markout_model.expected_markout(z);
    const double dq = h(q + z) - h(q);
    return lam * (z * spread * (0.5 - delta) - z * mu + dq);
}

inline double ask_trade_value(
    const LogisticFlowCurve& flow_curve,
    const SqrtMarkoutModel& markout_model,
    const LinearInterpolator1D& h,
    double spread,
    double q,
    double z,
    double delta
) {
    const double lam = flow_curve.arrival_rate(delta, z);
    const double mu = markout_model.expected_markout(z);
    const double dq = h(q - z) - h(q);
    return lam * (z * spread * (0.5 - delta) - z * mu + dq);
}

inline double ecn_bid_trade_value(
    const LogisticFlowCurve& flow_curve,
    const SqrtMarkoutModel& base_markout_model,
    const ECNAdverseSelectionModel& adverse_selection_model,
    const LinearInterpolator1D& h,
    double spread,
    double q,
    double z,
    double delta
) {
    const double lam = flow_curve.arrival_rate(delta, z);
    const double cost = adverse_selection_model.expected_cost(base_markout_model, delta, z);
    const double dq = h(q + z) - h(q);
    return lam * (z * spread * (0.5 - delta) - cost + dq);
}

inline double ecn_ask_trade_value(
    const LogisticFlowCurve& flow_curve,
    const SqrtMarkoutModel& base_markout_model,
    const ECNAdverseSelectionModel& adverse_selection_model,
    const LinearInterpolator1D& h,
    double spread,
    double q,
    double z,
    double delta
) {
    const double lam = flow_curve.arrival_rate(delta, z);
    const double cost = adverse_selection_model.expected_cost(base_markout_model, delta, z);
    const double dq = h(q - z) - h(q);
    return lam * (z * spread * (0.5 - delta) - cost + dq);
}

inline double dark_pool_poisson_trade_value(
    double arrival_lambda,
    double size_lambda,
    double fee_per_unit,
    const LinearInterpolator1D& h,
    double q,
    double posted_size,
    Side side
) {
    if (arrival_lambda <= 0.0 || size_lambda <= 0.0 || posted_size <= 0.0) {
        return 0.0;
    }

    const int u = static_cast<int>(std::llround(posted_size));
    if (u <= 0) {
        return 0.0;
    }

    // Opportunity arrivals are controlled by arrival_lambda.
    // Conditional opportunity size is modeled as:
    //
    //     K ~ Poisson(size_lambda) | K >= 1
    //
    // The executed size is min(K, posted_size).
    const double p0 = std::exp(-size_lambda);
    const double positive_mass = 1.0 - p0;

    if (positive_mass <= 0.0) {
        return 0.0;
    }

    const double hq = h(q);

    auto continuation_value = [&](int k) {
        const double size = static_cast<double>(k);

        if (side == Side::Ask) {
            return h(q - size) - hq - fee_per_unit * size;
        }

        return h(q + size) - hq - fee_per_unit * size;
    };

    double expected = 0.0;

    double pk = p0;
    double cumulative_positive_prob = 0.0;

    for (int k = 1; k < u; ++k) {
        pk *= size_lambda / static_cast<double>(k);

        const double prob = pk / positive_mass;
        cumulative_positive_prob += prob;

        expected += prob * continuation_value(k);
    }

    const double tail_prob = std::max(0.0, 1.0 - cumulative_positive_prob);
    expected += tail_prob * continuation_value(u);

    return arrival_lambda * expected;
}

inline double dark_pool_ask_trade_value(
    double arrival_lambda,
    double size_lambda,
    double fee_per_unit,
    const LinearInterpolator1D& h,
    double q,
    double posted_size
) {
    return dark_pool_poisson_trade_value(
        arrival_lambda,
        size_lambda,
        fee_per_unit,
        h,
        q,
        posted_size,
        Side::Ask
    );
}

inline double dark_pool_bid_trade_value(
    double arrival_lambda,
    double size_lambda,
    double fee_per_unit,
    const LinearInterpolator1D& h,
    double q,
    double posted_size
) {
    return dark_pool_poisson_trade_value(
        arrival_lambda,
        size_lambda,
        fee_per_unit,
        h,
        q,
        posted_size,
        Side::Bid
    );
}

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

struct DarkPoolPolicy {
    std::vector<double> q_grid;
    std::vector<double> bid_size;
    std::vector<double> ask_size;
    std::vector<unsigned char> bid_active;
    std::vector<unsigned char> ask_active;

    DarkPoolPolicy() = default;

    explicit DarkPoolPolicy(std::vector<double> q_grid_)
        : q_grid(std::move(q_grid_)) {
        reset_shape(q_grid);
    }

    void reset_shape(const std::vector<double>& q_grid_) {
        q_grid = q_grid_;
        bid_size.assign(q_grid.size(), 0.0);
        ask_size.assign(q_grid.size(), 0.0);
        bid_active.assign(q_grid.size(), 0u);
        ask_active.assign(q_grid.size(), 0u);
    }

    void validate() const {
        validate_strictly_increasing(q_grid, "DarkPoolPolicy.q_grid");
        if (bid_size.size() != q_grid.size() || ask_size.size() != q_grid.size()) {
            throw std::invalid_argument("DarkPoolPolicy: size vector length mismatch.");
        }
        if (bid_active.size() != q_grid.size() || ask_active.size() != q_grid.size()) {
            throw std::invalid_argument("DarkPoolPolicy: active flag length mismatch.");
        }
    }

    const std::vector<double>& size_vector(Side side) const {
        return side == Side::Bid ? bid_size : ask_size;
    }

    std::vector<double>& size_vector_mut(Side side) {
        return side == Side::Bid ? bid_size : ask_size;
    }

    const std::vector<unsigned char>& active_vector(Side side) const {
        return side == Side::Bid ? bid_active : ask_active;
    }

    std::vector<unsigned char>& active_vector_mut(Side side) {
        return side == Side::Bid ? bid_active : ask_active;
    }

    double posted_size(double q, Side side) const {
        const auto& vals = size_vector(side);
        if (q_grid.empty()) {
            throw std::runtime_error("DarkPoolPolicy::posted_size: q_grid is empty.");
        }
        if (q_grid.size() == 1) {
            return vals.front();
        }

        const auto [i, tq] = locate_segment_with_weight(q_grid, q);
        return tq <= 0.5 ? vals[i] : vals[i + 1];
    }

    bool is_active(std::size_t q_index, Side side) const {
        const auto& flags = active_vector(side);
        return q_index < flags.size() && flags[q_index] != 0u;
    }

    void set_posted_size(std::size_t q_index, Side side, double size, bool active) {
        auto& vals = size_vector_mut(side);
        auto& flags = active_vector_mut(side);
        if (q_index >= vals.size() || q_index >= flags.size()) {
            throw std::out_of_range("DarkPoolPolicy::set_posted_size: q_index out of range.");
        }
        vals[q_index] = size;
        flags[q_index] = active ? 1u : 0u;
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

struct ECNQuoteDecision {
    double delta{0.0};
    double contribution{0.0};
    bool active{false};
};

struct DarkPoolSizeDecision {
    double posted_size{0.0};
    double contribution{0.0};
    bool active{false};
};

// ============================================================
// Tier hierarchy
// ============================================================

struct Tier {
    std::string name;
    LogisticFlowCurve flow_curve;
    SqrtMarkoutModel markout_model;
    QuotePolicy policy;

    Tier() = default;

    Tier(
        std::string name_,
        LogisticFlowCurve flow_curve_,
        SqrtMarkoutModel markout_model_
    )
        : name(std::move(name_)),
          flow_curve(std::move(flow_curve_)),
          markout_model(std::move(markout_model_)) {}

    virtual ~Tier() = default;

    virtual const char* tier_type_name() const = 0;
    virtual const std::vector<double>& sizes() const = 0;
    virtual bool is_admissible(double q, double z, Side side) const = 0;

    virtual void validate() const = 0;

    void reset_policy_shape(const std::vector<double>& q_grid) {
        policy.reset_shape(q_grid, sizes());
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
        LogisticFlowCurve flow_curve_,
        SqrtMarkoutModel markout_model_,
        double delta_min_ = -5.0,
        double delta_max_ = 5.0
    )
        : Tier(std::move(name_), std::move(flow_curve_), std::move(markout_model_)),
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
    ECNAdverseSelectionModel adverse_selection_model{};
    std::vector<unsigned char> bid_active;
    std::vector<unsigned char> ask_active;

    ECNTier() = default;

    ECNTier(
        std::string name_,
        LogisticFlowCurve flow_curve_,
        SqrtMarkoutModel markout_model_,
        double delta_min_ = -5.0,
        double delta_max_ = 5.0,
        ECNAdverseSelectionModel adverse_selection_model_ = ECNAdverseSelectionModel{}
    )
        : Tier(std::move(name_), std::move(flow_curve_), std::move(markout_model_)),
          delta_min(delta_min_),
          delta_max(delta_max_),
          adverse_selection_model(std::move(adverse_selection_model_)) {
        validate();
    }

    const char* tier_type_name() const override {
        return "ecn";
    }

    const std::vector<double>& sizes() const override {
        static const std::vector<double> fixed_size{1.0};
        return fixed_size;
    }

    static double smoothstep01(double x) {
        const double u = clamp(x, 0.0, 1.0);
        return u * u * (3.0 - 2.0 * u);
    }

    double delta_mid_cap() const {
        return std::min(delta_max, 0.5);
    }

    double delta_profile(double q_abs) const {
        const double q = std::max(q_abs, 0.0);
        const double q_star = adverse_selection_model.ecn_convergence_inventory;
        const double u = q_star > 0.0 ? std::min(q / q_star, 1.0) : 1.0;
        const double s = smoothstep01(u);
        return delta_min + (delta_mid_cap() - delta_min) * s;
    }

    double delta_cap(double q_abs) const {
        return delta_profile(q_abs);
    }

    void validate() const override {
        if (delta_max <= delta_min) {
            throw std::invalid_argument("ECNTier: delta_max must be > delta_min.");
        }
        adverse_selection_model.validate();
        if (delta_min >= delta_mid_cap()) {
            throw std::invalid_argument(
                "ECNTier: delta_min must be < min(delta_max, 0.5) for passive ECN quoting."
            );
        }
    }

    void reset_activity_shape(std::size_t nq) {
        bid_active.assign(nq, 0u);
        ask_active.assign(nq, 0u);
    }

    static bool is_active_side(double q, Side side, double tol = 1e-12) {
        return (q > tol && side == Side::Ask) || (q < -tol && side == Side::Bid);
    }

    bool is_admissible(double q, double z, Side side) const override {
        constexpr double tol = 1e-12;
        return std::abs(z - 1.0) <= tol && is_active_side(q, side, tol);
    }

    bool is_policy_active(std::size_t q_index, Side side) const {
        const auto& flags = side == Side::Bid ? bid_active : ask_active;
        return q_index < flags.size() && flags[q_index] != 0u;
    }

    void set_policy_active(std::size_t q_index, Side side, bool active) {
        auto& flags = side == Side::Bid ? bid_active : ask_active;
        if (q_index >= flags.size()) {
            throw std::out_of_range("ECNTier::set_policy_active: q_index out of range.");
        }
        flags[q_index] = active ? 1u : 0u;
    }
};

struct DarkPoolVenue {
    double arrival_lambda_bid{0.0};
    double arrival_lambda_ask{0.0};

    double size_lambda_bid{1.0};
    double size_lambda_ask{1.0};

    double fee_per_unit_bid{0.0};
    double fee_per_unit_ask{0.0};

    std::vector<double> posted_sizes{1.0, 2.0, 3.0};
    bool allow_both_sides{false};

    DarkPoolPolicy policy;

    DarkPoolVenue() = default;

    DarkPoolVenue(
        double arrival_lambda_bid_,
        double arrival_lambda_ask_,
        double size_lambda_bid_,
        double size_lambda_ask_,
        double fee_per_unit_bid_,
        double fee_per_unit_ask_,
        std::vector<double> posted_sizes_,
        bool allow_both_sides_ = false
    )
        : arrival_lambda_bid(arrival_lambda_bid_),
          arrival_lambda_ask(arrival_lambda_ask_),
          size_lambda_bid(size_lambda_bid_),
          size_lambda_ask(size_lambda_ask_),
          fee_per_unit_bid(fee_per_unit_bid_),
          fee_per_unit_ask(fee_per_unit_ask_),
          posted_sizes(std::move(posted_sizes_)),
          allow_both_sides(allow_both_sides_) {
        validate();
    }

    static bool is_integer_like(double x, double tol = 1e-10) {
        return std::abs(x - std::round(x)) <= tol;
    }

    void validate() const {
        if (arrival_lambda_bid < 0.0 || arrival_lambda_ask < 0.0) {
            throw std::invalid_argument("DarkPoolVenue: arrival lambdas must be nonnegative.");
        }
        if (size_lambda_bid <= 0.0 || size_lambda_ask <= 0.0) {
            throw std::invalid_argument(
                "DarkPoolVenue: size-distribution lambdas must be positive."
            );
        }
        if (fee_per_unit_bid < 0.0 || fee_per_unit_ask < 0.0) {
            throw std::invalid_argument("DarkPoolVenue: fees must be nonnegative.");
        }

        validate_positive_strictly_increasing(posted_sizes, "DarkPoolVenue.posted_sizes");

        for (double u : posted_sizes) {
            if (!is_integer_like(u)) {
                throw std::invalid_argument(
                    "DarkPoolVenue: posted_sizes must be integer-valued because dark-pool fills are modeled in unit chunks."
                );
            }
        }
    }

    void reset_policy_shape(const std::vector<double>& q_grid) {
        policy.reset_shape(q_grid);
    }

    bool is_admissible(double q, Side side, double tol = 1e-12) const {
        if (allow_both_sides) {
            return true;
        }

        return (q > tol && side == Side::Ask) ||
               (q < -tol && side == Side::Bid);
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
    std::vector<MDPTier> mdp_tiers;
    std::vector<ECNTier> ecn_tiers;
    std::optional<DarkPoolVenue> dark_pool;
    SolverDiagnostics diagnostics;
};

// ============================================================
// Solver
// ============================================================

struct HJBLadderSolver {
    SolverConfig config;
    PolynomialInventoryPenalty penalty;

    std::vector<MDPTier> mdp_tiers;
    std::vector<ECNTier> ecn_tiers;
    std::optional<DarkPoolVenue> dark_pool;

    GoldenSectionSearch optimizer;
    SolverGridMeta grid_meta_;

    HJBLadderSolver(
        SolverConfig config_,
        PolynomialInventoryPenalty penalty_,
        std::vector<MDPTier> mdp_tiers_ = {},
        std::vector<ECNTier> ecn_tiers_ = {},
        std::optional<DarkPoolVenue> dark_pool_ = std::nullopt
    )
        : config(std::move(config_)),
          penalty(std::move(penalty_)),
          mdp_tiers(std::move(mdp_tiers_)),
          ecn_tiers(std::move(ecn_tiers_)),
          dark_pool(std::move(dark_pool_)),
          optimizer(config.golden_tol, config.golden_max_iter) {
        config.validate();
    }

    static const char* side_name(Side side) {
        return side == Side::Bid ? "bid" : "ask";
    }

    static void repair_or_throw_bounds(
        double& lower,
        double& upper,
        const std::string& tier_name,
        const char* side_name_
    ) {
        constexpr double tol = 1e-12;

        if (upper < lower) {
            if (lower - upper > tol) {
                throw std::runtime_error(
                    "Infeasible ladder bounds for MDP tier '" + tier_name +
                    "', side '" + std::string(side_name_) + "'."
                );
            }
            upper = lower;
        }
    }

    void validate_problem_definition() const {
        config.validate();

        for (const auto& tier : mdp_tiers) {
            tier.validate();
        }
        for (const auto& tier : ecn_tiers) {
            tier.validate();
        }
        if (dark_pool.has_value()) {
            dark_pool->validate();
        }
    }

    void initialize_policy_shapes() {
        for (auto& tier : mdp_tiers) {
            tier.reset_policy_shape(config.q_grid);
        }
        for (auto& tier : ecn_tiers) {
            tier.reset_policy_shape(config.q_grid);
            tier.reset_activity_shape(config.q_grid.size());
        }
        if (dark_pool.has_value()) {
            dark_pool->reset_policy_shape(config.q_grid);
        }
    }

    void prepare_solve_context() {
        validate_problem_definition();
        grid_meta_ = build_solver_grid_meta(config.q_grid);
        initialize_policy_shapes();
    }

    // --------------------------------------------------------
    // MDP policy construction
    // --------------------------------------------------------

    std::pair<double, double> mdp_bounds_for_rung(
        const MDPTier& tier,
        std::size_t rung_idx,
        const std::vector<double>& current_delta,
        const std::vector<double>* prev_delta_row,
        double q,
        Side side
    ) const {
        if (rung_idx >= current_delta.size()) {
            throw std::out_of_range("mdp_bounds_for_rung: rung_idx out of range.");
        }

        if (prev_delta_row != nullptr && prev_delta_row->size() != current_delta.size()) {
            throw std::invalid_argument(
                "mdp_bounds_for_rung: prev_delta_row must have same size as current_delta."
            );
        }

        double lower = tier.delta_min;
        double upper = tier.delta_max;

        // 1. Size-ladder monotonicity.
        // Larger size rungs should not be more aggressive than smaller size rungs.
        if (rung_idx > 0) {
            upper = std::min(upper, current_delta[rung_idx - 1]);
        }

        // 2. Zero-inventory row.
        constexpr double q_tol = 1e-12;
        if (prev_delta_row == nullptr || std::abs(q) <= q_tol) {
            return {lower, upper};
        }

        const auto& prev = *prev_delta_row;

        const bool risk_decreasing =
            (side == Side::Bid && q < -q_tol) ||
            (side == Side::Ask && q > q_tol);

        const bool risk_increasing =
            (side == Side::Bid && q > q_tol) ||
            (side == Side::Ask && q < -q_tol);

        // 3. First rung inventory monotonicity.
        if (rung_idx == 0) {
            if (risk_decreasing) {
                lower = std::max(lower, prev[0]);
            } else if (risk_increasing) {
                upper = std::min(upper, prev[0]);
            }

            return {lower, upper};
        }

        // 4. Larger rung inventory monotonicity.
        const double prev_curr = current_delta[rung_idx - 1];
        const double prev_row_gap = prev[rung_idx - 1] - prev[rung_idx];

        if (risk_decreasing) {
            lower = std::max(lower, prev_curr - prev_row_gap);
        } else if (risk_increasing) {
            const double required_tail_span = prev[rung_idx] - prev.back();
            lower = std::max(lower, tier.delta_min + required_tail_span);
            upper = std::min(upper, prev_curr - prev_row_gap);
        }

        return {lower, upper};
    }

    double optimize_mdp_rung(
        const MDPTier& tier,
        const LinearInterpolator1D& h,
        double q,
        double z,
        Side side,
        double lower,
        double upper
    ) const {
        repair_or_throw_bounds(lower, upper, tier.name, side_name(side));

        if (side == Side::Bid) {
            const auto obj = [&](double d) {
                return bid_trade_value(
                    tier.flow_curve,
                    tier.markout_model,
                    h,
                    config.spread,
                    q,
                    z,
                    d
                );
            };

            return clamp(optimizer.maximize(obj, lower, upper), lower, upper);
        }

        const auto obj = [&](double d) {
            return ask_trade_value(
                tier.flow_curve,
                tier.markout_model,
                h,
                config.spread,
                q,
                z,
                d
            );
        };

        return clamp(optimizer.maximize(obj, lower, upper), lower, upper);
    }

    void build_mdp_ladder_row(
        const MDPTier& tier,
        const LinearInterpolator1D& h,
        std::vector<double>& delta_row,
        const std::vector<double>* prev_delta_row,
        double q,
        Side side
    ) const {
        const std::size_t nz = tier.sizes_.size();
        delta_row.assign(nz, tier.delta_min);

        for (std::size_t j = 0; j < nz; ++j) {
            const double z = tier.sizes_[j];

            auto [lower, upper] = mdp_bounds_for_rung(
                tier,
                j,
                delta_row,
                prev_delta_row,
                q,
                side
            );

            delta_row[j] = optimize_mdp_rung(
                tier,
                h,
                q,
                z,
                side,
                lower,
                upper
            );
        }
    }

    void build_mdp_bid_policy(
        MDPTier& tier,
        const LinearInterpolator1D& h
    ) const {
        const auto& q_grid = config.q_grid;

        const std::size_t nq = grid_meta_.nq;
        const std::size_t q0_idx = grid_meta_.q0_idx;

        build_mdp_ladder_row(
            tier,
            h,
            tier.policy.bid[q0_idx],
            nullptr,
            q_grid[q0_idx],
            Side::Bid
        );

        for (std::size_t i = q0_idx + 1; i < nq; ++i) {
            build_mdp_ladder_row(
                tier,
                h,
                tier.policy.bid[i],
                &tier.policy.bid[i - 1],
                q_grid[i],
                Side::Bid
            );
        }

        for (std::size_t i = q0_idx; i-- > 0;) {
            build_mdp_ladder_row(
                tier,
                h,
                tier.policy.bid[i],
                &tier.policy.bid[i + 1],
                q_grid[i],
                Side::Bid
            );
        }
    }

    void build_mdp_ask_policy(
        MDPTier& tier,
        const LinearInterpolator1D& h
    ) const {
        const auto& q_grid = config.q_grid;

        const std::size_t nq = grid_meta_.nq;
        const std::size_t q0_idx = grid_meta_.q0_idx;

        build_mdp_ladder_row(
            tier,
            h,
            tier.policy.ask[q0_idx],
            nullptr,
            q_grid[q0_idx],
            Side::Ask
        );

        for (std::size_t i = q0_idx + 1; i < nq; ++i) {
            build_mdp_ladder_row(
                tier,
                h,
                tier.policy.ask[i],
                &tier.policy.ask[i - 1],
                q_grid[i],
                Side::Ask
            );
        }

        for (std::size_t i = q0_idx; i-- > 0;) {
            build_mdp_ladder_row(
                tier,
                h,
                tier.policy.ask[i],
                &tier.policy.ask[i + 1],
                q_grid[i],
                Side::Ask
            );
        }
    }

    // --------------------------------------------------------
    // ECN policy construction
    // --------------------------------------------------------

    ECNQuoteDecision optimize_ecn_quote_for_state(
        const ECNTier& tier,
        const LinearInterpolator1D& h,
        double q,
        Side side,
        std::optional<double> lower_bound = std::nullopt
    ) const {
        (void) lower_bound;

        ECNQuoteDecision out;
        out.delta = tier.delta_min;
        out.contribution = 0.0;
        out.active = false;

        if (!tier.is_admissible(q, 1.0, side)) {
            return out;
        }

        const double d = clamp(tier.delta_profile(std::abs(q)), tier.delta_min, tier.delta_mid_cap());
        const double v = side == Side::Ask
            ? ecn_ask_trade_value(
                tier.flow_curve,
                tier.markout_model,
                tier.adverse_selection_model,
                h,
                config.spread,
                q,
                1.0,
                d
            )
            : ecn_bid_trade_value(
                tier.flow_curve,
                tier.markout_model,
                tier.adverse_selection_model,
                h,
                config.spread,
                q,
                1.0,
                d
            );

        out.delta = d;
        out.contribution = v;
        out.active = true;
        return out;
    }

    void clear_ecn_policy(ECNTier& tier) const {
        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            tier.policy.delta_ref(i, 0, Side::Bid) = tier.delta_min;
            tier.policy.delta_ref(i, 0, Side::Ask) = tier.delta_min;
            tier.set_policy_active(i, Side::Bid, false);
            tier.set_policy_active(i, Side::Ask, false);
        }
    }

    void build_ecn_policy(ECNTier& tier, const LinearInterpolator1D&) const {
        clear_ecn_policy(tier);

        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            const double q = config.q_grid[i];
            if (q > 0.0) {
                tier.policy.delta_ref(i, 0, Side::Ask) = tier.delta_profile(std::abs(q));
                tier.set_policy_active(i, Side::Ask, true);
            } else if (q < 0.0) {
                tier.policy.delta_ref(i, 0, Side::Bid) = tier.delta_profile(std::abs(q));
                tier.set_policy_active(i, Side::Bid, true);
            }
        }
    }

    // --------------------------------------------------------
    // Dark-pool policy construction
    // --------------------------------------------------------

    DarkPoolSizeDecision optimize_dark_pool_size_for_state(
        const DarkPoolVenue& venue,
        const LinearInterpolator1D& h,
        double q,
        Side side
    ) const {
        DarkPoolSizeDecision out;

        if (!venue.is_admissible(q, side)) {
            return out;
        }

        for (double u : venue.posted_sizes) {
            const double value = side == Side::Ask
                ? dark_pool_ask_trade_value(
                    venue.arrival_lambda_ask,
                    venue.size_lambda_ask,
                    venue.fee_per_unit_ask,
                    h,
                    q,
                    u
                )
                : dark_pool_bid_trade_value(
                    venue.arrival_lambda_bid,
                    venue.size_lambda_bid,
                    venue.fee_per_unit_bid,
                    h,
                    q,
                    u
                );

            if (value > out.contribution) {
                out.posted_size = u;
                out.contribution = value;
                out.active = value > 0.0;
            }
        }

        if (!out.active) {
            out.posted_size = 0.0;
            out.contribution = 0.0;
        }

        return out;
    }

    void clear_dark_pool_policy(DarkPoolVenue& venue) const {
        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            venue.policy.set_posted_size(i, Side::Bid, 0.0, false);
            venue.policy.set_posted_size(i, Side::Ask, 0.0, false);
        }
    }

    void build_dark_pool_policy(DarkPoolVenue& venue, const LinearInterpolator1D& h) const {
        clear_dark_pool_policy(venue);

        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            const double q = config.q_grid[i];

            if (venue.allow_both_sides || q < 0.0) {
                const DarkPoolSizeDecision bid_decision =
                    optimize_dark_pool_size_for_state(venue, h, q, Side::Bid);
                venue.policy.set_posted_size(
                    i, Side::Bid, bid_decision.posted_size, bid_decision.active
                );
            }

            if (venue.allow_both_sides || q > 0.0) {
                const DarkPoolSizeDecision ask_decision =
                    optimize_dark_pool_size_for_state(venue, h, q, Side::Ask);
                venue.policy.set_posted_size(
                    i, Side::Ask, ask_decision.posted_size, ask_decision.active
                );
            }
        }
    }

    // --------------------------------------------------------
    // Bellman contributions
    // --------------------------------------------------------

    double mdp_bid_bellman_contribution(
        const MDPTier& tier,
        double q,
        std::size_t q_index,
        const LinearInterpolator1D& h
    ) const {
        double total = 0.0;

        const auto& bid_row = tier.policy.bid[q_index];
        const std::size_t nz = tier.sizes_.size();

        for (std::size_t j = 0; j < nz; ++j) {
            const double z = tier.sizes_[j];
            total += bid_trade_value(
                tier.flow_curve,
                tier.markout_model,
                h,
                config.spread,
                q,
                z,
                bid_row[j]
            );
        }

        return total;
    }

    double mdp_ask_bellman_contribution(
        const MDPTier& tier,
        double q,
        std::size_t q_index,
        const LinearInterpolator1D& h
    ) const {
        double total = 0.0;

        const auto& ask_row = tier.policy.ask[q_index];
        const std::size_t nz = tier.sizes_.size();

        for (std::size_t j = 0; j < nz; ++j) {
            const double z = tier.sizes_[j];
            total += ask_trade_value(
                tier.flow_curve,
                tier.markout_model,
                h,
                config.spread,
                q,
                z,
                ask_row[j]
            );
        }

        return total;
    }

    double ecn_bellman_contribution(
        const ECNTier& tier,
        double q,
        std::size_t q_index,
        const LinearInterpolator1D& h
    ) const {
        if (q > 0.0 && tier.is_policy_active(q_index, Side::Ask)) {
            return ecn_ask_trade_value(
                tier.flow_curve,
                tier.markout_model,
                tier.adverse_selection_model,
                h,
                config.spread,
                q,
                1.0,
                tier.policy.ask[q_index][0]
            );
        }
        if (q < 0.0 && tier.is_policy_active(q_index, Side::Bid)) {
            return ecn_bid_trade_value(
                tier.flow_curve,
                tier.markout_model,
                tier.adverse_selection_model,
                h,
                config.spread,
                q,
                1.0,
                tier.policy.bid[q_index][0]
            );
        }
        return 0.0;
    }

    double dark_pool_bellman_contribution(
        const DarkPoolVenue& venue,
        double q,
        std::size_t q_index,
        const LinearInterpolator1D& h
    ) const {
        double total = 0.0;

        if (venue.policy.is_active(q_index, Side::Bid)) {
            total += dark_pool_bid_trade_value(
                venue.arrival_lambda_bid,
                venue.size_lambda_bid,
                venue.fee_per_unit_bid,
                h,
                q,
                venue.policy.bid_size[q_index]
            );
        }

        if (venue.policy.is_active(q_index, Side::Ask)) {
            total += dark_pool_ask_trade_value(
                venue.arrival_lambda_ask,
                venue.size_lambda_ask,
                venue.fee_per_unit_ask,
                h,
                q,
                venue.policy.ask_size[q_index]
            );
        }

        return total;
    }

    // --------------------------------------------------------
    // Public-ish internal solver steps
    // --------------------------------------------------------

    void update_policies(const LinearInterpolator1D& h) {
        for (auto& tier : mdp_tiers) {
            build_mdp_bid_policy(tier, h);
            build_mdp_ask_policy(tier, h);
        }

        for (auto& tier : ecn_tiers) {
            build_ecn_policy(tier, h);
        }

        if (dark_pool.has_value()) {
            build_dark_pool_policy(*dark_pool, h);
        }
    }

    double bellman_rhs(const LinearInterpolator1D& h, std::vector<double>& rhs) const {
        double max_rhs_abs = 0.0;

        for (std::size_t i = 0; i < grid_meta_.nq; ++i) {
            const double q = config.q_grid[i];
            double val = -penalty.value(q) + config.spot_drift * q;

            for (const auto& tier : mdp_tiers) {
                val += mdp_bid_bellman_contribution(tier, q, i, h);
                val += mdp_ask_bellman_contribution(tier, q, i, h);
            }
            for (const auto& tier : ecn_tiers) {
                val += ecn_bellman_contribution(tier, q, i, h);
            }
            if (dark_pool.has_value()) {
                val += dark_pool_bellman_contribution(*dark_pool, q, i, h);
            }

            rhs[i] = val;
            max_rhs_abs = std::max(max_rhs_abs, std::abs(val));
        }

        return max_rhs_abs;
    }

    std::vector<double> bellman_rhs_from_policies(const std::vector<double>& h_vec) {
        if (h_vec.size() != config.q_grid.size()) {
            throw std::invalid_argument(
                "bellman_rhs_from_policies: h_vec size must match config.q_grid size."
            );
        }

        prepare_solve_context();

        const LinearInterpolator1D h{config.q_grid, h_vec};
        update_policies(h);

        std::vector<double> rhs(config.q_grid.size(), 0.0);
        bellman_rhs(h, rhs);
        return rhs;
    }

    HJBSolution solve() {
        prepare_solve_context();

        std::vector<double> h(grid_meta_.nq, 0.0);
        std::vector<double> h_next(grid_meta_.nq, 0.0);
        std::vector<double> rhs(grid_meta_.nq, 0.0);

        const LinearInterpolator1D h_view{config.q_grid, h};
        SolverDiagnostics diagnostics(config.n_iter);

        for (int it = 1; it <= config.n_iter; ++it) {
            update_policies(h_view);
            const double max_rhs_now = bellman_rhs(h_view, rhs);

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
                    config.consecutive_passes_required
                )) {
                break;
            }
        }

        update_policies(h_view);

        HJBSolution out;
        out.h = std::move(h);
        out.q_grid = config.q_grid;
        out.mdp_tiers = mdp_tiers;
        out.ecn_tiers = ecn_tiers;
        out.dark_pool = dark_pool;
        out.diagnostics = std::move(diagnostics);
        return out;
    }
};

} // namespace hjb