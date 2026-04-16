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
// Flat array containers
// ============================================================

struct FlatCube3D {
    std::vector<double> data;
    std::size_t nq{0}, ny{0}, nnu{0};

    FlatCube3D() = default;

    FlatCube3D(std::size_t nq_, std::size_t ny_, std::size_t nnu_, double init = 0.0)
        : data(nq_ * ny_ * nnu_, init), nq(nq_), ny(ny_), nnu(nnu_) {}

    void reset(std::size_t nq_, std::size_t ny_, std::size_t nnu_, double init = 0.0) {
        nq = nq_;
        ny = ny_;
        nnu = nnu_;
        data.assign(nq_ * ny_ * nnu_, init);
    }

    std::size_t index(std::size_t iq, std::size_t iy, std::size_t inu) const noexcept {
        return (iq * ny + iy) * nnu + inu;
    }

    double& at(std::size_t iq, std::size_t iy, std::size_t inu) noexcept {
        return data[index(iq, iy, inu)];
    }

    double at(std::size_t iq, std::size_t iy, std::size_t inu) const noexcept {
        return data[index(iq, iy, inu)];
    }

    bool empty() const noexcept { return data.empty(); }
};

struct FlatCube4D {
    std::vector<double> data;
    std::size_t nq{0}, ny{0}, nnu{0}, nz{0};

    FlatCube4D() = default;

    FlatCube4D(std::size_t nq_, std::size_t ny_, std::size_t nnu_, std::size_t nz_,
               double init = 0.0)
        : data(nq_ * ny_ * nnu_ * nz_, init),
          nq(nq_), ny(ny_), nnu(nnu_), nz(nz_) {}

    void ensure_shape(std::size_t nq_, std::size_t ny_, std::size_t nnu_, std::size_t nz_) {
        if (nq_ == nq && ny_ == ny && nnu_ == nnu && nz_ == nz) return;
        nq = nq_;
        ny = ny_;
        nnu = nnu_;
        nz = nz_;
        data.assign(nq_ * ny_ * nnu_ * nz_, 0.0);
    }

    std::size_t index(std::size_t iq, std::size_t iy,
                      std::size_t inu, std::size_t iz) const noexcept {
        return ((iq * ny + iy) * nnu + inu) * nz + iz;
    }

    double& at(std::size_t iq, std::size_t iy,
               std::size_t inu, std::size_t iz) noexcept {
        return data[index(iq, iy, inu, iz)];
    }

    double at(std::size_t iq, std::size_t iy,
              std::size_t inu, std::size_t iz) const noexcept {
        return data[index(iq, iy, inu, iz)];
    }

    double* row_ptr(std::size_t iq, std::size_t iy, std::size_t inu) noexcept {
        return data.data() + ((iq * ny + iy) * nnu + inu) * nz;
    }

    const double* row_ptr(std::size_t iq, std::size_t iy,
                          std::size_t inu) const noexcept {
        return data.data() + ((iq * ny + iy) * nnu + inu) * nz;
    }

    bool empty() const noexcept { return data.empty(); }
};

// ============================================================
// Helpers
// ============================================================

inline double clamp(double x, double lo, double hi) {
    return std::min(std::max(x, lo), hi);
}

inline bool approx_equal(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol;
}

enum class Side : unsigned char { Bid, Ask };

inline Side parse_side(const std::string& side) {
    if (side == "bid") return Side::Bid;
    if (side == "ask") return Side::Ask;
    throw std::invalid_argument("Unknown side: " + side);
}

inline const char* side_name(Side side) {
    return (side == Side::Bid) ? "bid" : "ask";
}

inline std::pair<std::size_t, double> locate_segment_with_weight(
    const std::vector<double>& grid,
    double x
) {
    if (grid.empty()) throw std::invalid_argument("Grid cannot be empty.");
    if (grid.size() == 1) return {0, 0.0};

    if (x <= grid.front()) {
        const double t = (x - grid[0]) / (grid[1] - grid[0]);
        return {0, t};
    }
    if (x >= grid.back()) {
        const std::size_t i = grid.size() - 2;
        const double t = (x - grid[i]) / (grid[i + 1] - grid[i]);
        return {i, t};
    }

    auto it = std::upper_bound(grid.begin(), grid.end(), x);
    const std::size_t i = static_cast<std::size_t>(it - grid.begin() - 1);
    const double t = (x - grid[i]) / (grid[i + 1] - grid[i]);
    return {i, t};
}

inline double interp_linear(
    const std::vector<double>& grid,
    const std::vector<double>& vals,
    double x
) {
    if (grid.size() != vals.size()) throw std::invalid_argument("interp_linear: size mismatch.");
    if (grid.empty()) throw std::invalid_argument("interp_linear: empty input.");
    if (grid.size() == 1) return vals.front();

    const auto [i, t] = locate_segment_with_weight(grid, x);
    return vals[i] + t * (vals[i + 1] - vals[i]);
}

inline double interp_trilinear(
    const std::vector<double>& q_grid,
    const std::vector<double>& y_grid,
    const std::vector<double>& nu_grid,
    const FlatCube3D& vals,
    double q, double y, double nu
) {
    if (vals.nq != q_grid.size())
        throw std::invalid_argument("interp_trilinear: q dimension mismatch.");
    if (vals.ny != y_grid.size())
        throw std::invalid_argument("interp_trilinear: y dimension mismatch.");
    if (vals.nnu != nu_grid.size())
        throw std::invalid_argument("interp_trilinear: nu dimension mismatch.");

    const auto [iq, tq] = locate_segment_with_weight(q_grid, q);
    const auto [iy, ty] = locate_segment_with_weight(y_grid, y);
    const auto [in, tn] = locate_segment_with_weight(nu_grid, nu);

    const double c000 = vals.at(iq,     iy,     in    );
    const double c100 = vals.at(iq + 1, iy,     in    );
    const double c010 = vals.at(iq,     iy + 1, in    );
    const double c110 = vals.at(iq + 1, iy + 1, in    );
    const double c001 = vals.at(iq,     iy,     in + 1);
    const double c101 = vals.at(iq + 1, iy,     in + 1);
    const double c011 = vals.at(iq,     iy + 1, in + 1);
    const double c111 = vals.at(iq + 1, iy + 1, in + 1);

    const double c00 = c000 + tq * (c100 - c000);
    const double c10 = c010 + tq * (c110 - c010);
    const double c01 = c001 + tq * (c101 - c001);
    const double c11 = c011 + tq * (c111 - c011);
    const double c0  = c00  + ty * (c10  - c00 );
    const double c1  = c01  + ty * (c11  - c01 );
    return c0 + tn * (c1 - c0);
}

inline double interp_trilinear_rung(
    const std::vector<double>& q_grid,
    const std::vector<double>& y_grid,
    const std::vector<double>& nu_grid,
    const FlatCube4D& vals,
    double q, double y, double nu,
    std::size_t iz
) {
    const auto [iq, tq] = locate_segment_with_weight(q_grid, q);
    const auto [iy, ty] = locate_segment_with_weight(y_grid, y);
    const auto [in, tn] = locate_segment_with_weight(nu_grid, nu);

    const double c000 = vals.at(iq,     iy,     in,     iz);
    const double c100 = vals.at(iq + 1, iy,     in,     iz);
    const double c010 = vals.at(iq,     iy + 1, in,     iz);
    const double c110 = vals.at(iq + 1, iy + 1, in,     iz);
    const double c001 = vals.at(iq,     iy,     in + 1, iz);
    const double c101 = vals.at(iq + 1, iy,     in + 1, iz);
    const double c011 = vals.at(iq,     iy + 1, in + 1, iz);
    const double c111 = vals.at(iq + 1, iy + 1, in + 1, iz);

    const double c00 = c000 + tq * (c100 - c000);
    const double c10 = c010 + tq * (c110 - c010);
    const double c01 = c001 + tq * (c101 - c001);
    const double c11 = c011 + tq * (c111 - c011);
    const double c0  = c00  + ty * (c10  - c00 );
    const double c1  = c01  + ty * (c11  - c01 );
    return c0 + tn * (c1 - c0);
}

inline double max_abs_vec(const std::vector<double>& x) {
    double out = 0.0;
    for (double v : x) out = std::max(out, std::abs(v));
    return out;
}

inline double max_abs_3d(const FlatCube3D& x) {
    return max_abs_vec(x.data);
}

inline void diff_inplace(const double* x, std::size_t n, std::vector<double>& out) {
    if (n < 2) {
        out.clear();
        return;
    }
    out.resize(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) out[i] = x[i + 1] - x[i];
}

// ============================================================
// Grid state
// ============================================================

struct GridState {
    std::size_t iq{0};
    std::size_t iy{0};
    std::size_t inu{0};
    double q{0.0};
    double y{0.0};
    double nu{0.0};
};

// ============================================================
// No-copy interpolator over FlatCube3D
// ============================================================

struct HInterp3D {
    const std::vector<double>& q_grid;
    const std::vector<double>& y_grid;
    const std::vector<double>& nu_grid;
    const FlatCube3D& h;

    HInterp3D(
        const std::vector<double>& q_grid_,
        const std::vector<double>& y_grid_,
        const std::vector<double>& nu_grid_,
        const FlatCube3D& h_
    )
        : q_grid(q_grid_), y_grid(y_grid_), nu_grid(nu_grid_), h(h_) {
        if (h.nq != q_grid.size())
            throw std::invalid_argument("HInterp3D: q dimension mismatch.");
        if (h.ny != y_grid.size())
            throw std::invalid_argument("HInterp3D: y dimension mismatch.");
        if (h.nnu != nu_grid.size())
            throw std::invalid_argument("HInterp3D: nu dimension mismatch.");
    }

    double operator()(double q, double y, double nu) const {
        const auto [iq, tq] = locate_segment_with_weight(q_grid, q);
        const auto [iy, ty] = locate_segment_with_weight(y_grid, y);
        const auto [in, tn] = locate_segment_with_weight(nu_grid, nu);

        const double c000 = h.at(iq,     iy,     in    );
        const double c100 = h.at(iq + 1, iy,     in    );
        const double c010 = h.at(iq,     iy + 1, in    );
        const double c110 = h.at(iq + 1, iy + 1, in    );
        const double c001 = h.at(iq,     iy,     in + 1);
        const double c101 = h.at(iq + 1, iy,     in + 1);
        const double c011 = h.at(iq,     iy + 1, in + 1);
        const double c111 = h.at(iq + 1, iy + 1, in + 1);

        const double c00 = c000 + tq * (c100 - c000);
        const double c10 = c010 + tq * (c110 - c010);
        const double c01 = c001 + tq * (c101 - c001);
        const double c11 = c011 + tq * (c111 - c011);
        const double c0  = c00  + ty * (c10  - c00 );
        const double c1  = c01  + ty * (c11  - c01 );
        return c0 + tn * (c1 - c0);
    }
};

// ============================================================
// Optimization utility
// ============================================================

namespace opt {

struct GoldenSectionOptions {
    double tol{1e-4};
    int max_iter{32};
};

struct GoldenSectionResult {
    double x_star{0.0};
    double f_star{0.0};
    int iterations{0};
    bool converged{false};
};

struct GoldenSectionOptimizer {
    GoldenSectionOptions opts{};

    GoldenSectionOptimizer() = default;

    explicit GoldenSectionOptimizer(const GoldenSectionOptions& opts_) : opts(opts_) {
        validate_options(opts);
    }

    void set_options(const GoldenSectionOptions& opts_) {
        validate_options(opts_);
        opts = opts_;
    }

    static void validate_options(const GoldenSectionOptions& opts_) {
        if (opts_.tol <= 0.0)
            throw std::invalid_argument("GoldenSectionOptimizer: tol must be positive.");
        if (opts_.max_iter < 1)
            throw std::invalid_argument("GoldenSectionOptimizer: max_iter must be >= 1.");
    }

    template <class ObjectiveFn>
    GoldenSectionResult maximize(
        const ObjectiveFn& f,
        double lower,
        double upper
    ) const {
        if (upper < lower)
            throw std::invalid_argument("GoldenSectionOptimizer::maximize: invalid bounds.");

        if (std::abs(upper - lower) < opts.tol) {
            const double x = 0.5 * (lower + upper);
            return {x, f(x), 0, true};
        }

        constexpr double gr = 1.6180339887498948482;

        double a = lower;
        double b = upper;
        double c = b - (b - a) / gr;
        double d = a + (b - a) / gr;

        double fc = f(c);
        double fd = f(d);

        int it = 0;
        for (; it < opts.max_iter; ++it) {
            if (std::abs(b - a) < opts.tol) break;

            if (fc > fd) {
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) / gr;
                fc = f(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) / gr;
                fd = f(d);
            }
        }

        const double x_star = 0.5 * (a + b);
        const double f_star = (fc > fd) ? fc : fd;
        return {x_star, f_star, it, std::abs(b - a) < opts.tol};
    }

    template <class ObjectiveFn>
    GoldenSectionResult minimize(
        const ObjectiveFn& f,
        double lower,
        double upper
    ) const {
        auto neg_f = [&](double x) { return -f(x); };
        auto res = maximize(neg_f, lower, upper);
        res.f_star = f(res.x_star);
        return res;
    }
};

} // namespace opt

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

    LogisticFlowCurve(double A0_, double theta_, double k_,
                      double m0_, double m_alpha_, double z_floor_ = 1e-8)
        : A0(A0_), theta(theta_), k(k_),
          m0(m0_), m_alpha(m_alpha_), z_floor(z_floor_) {}

    double A(double z) const {
        return A0 * std::pow(std::max(z, z_floor), -theta);
    }

    double m(double z) const {
        return m0 + m_alpha * std::max(z, z_floor);
    }

    double arrival_rate(double delta, double z) const override {
        if (A0 < 0.0) throw std::invalid_argument("A(z) must be nonnegative.");
        if (k <= 0.0) throw std::invalid_argument("k must be positive.");

        const double x = k * (delta - m(z));
        if (x >= 0.0) {
            const double ex = std::exp(-x);
            return A(z) * ex / (1.0 + ex);
        }
        const double ex = std::exp(x);
        return A(z) / (1.0 + ex);
    }
};

struct SqrtDriftJumpModel final : public DriftJumpModel {
    double base{0.0};
    double coeff{0.01};

    SqrtDriftJumpModel() = default;
    SqrtDriftJumpModel(double base_, double coeff_) : base(base_), coeff(coeff_) {}

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

    PolynomialInventoryPenalty(double risk_aversion_, double sigma_ref_, double tau0_,
                               double cubic_coeff_, double quartic_coeff_)
        : risk_aversion(risk_aversion_), sigma_ref(sigma_ref_), tau0(tau0_),
          cubic_coeff(cubic_coeff_), quartic_coeff(quartic_coeff_) {}

    double value(double q) const override {
        const double x = std::abs(q);
        const double x2 = x * x;
        return risk_aversion * sigma_ref * sigma_ref *
               (tau0 * x2 + cubic_coeff * x2 * x + quartic_coeff * x2 * x2);
    }
};

// ============================================================
// Quote policy
// ============================================================

struct QuotePolicy {
    std::vector<double> q_grid;
    std::vector<double> y_grid;
    std::vector<double> nu_grid;
    std::vector<double> sizes;

    FlatCube4D bid;
    FlatCube4D ask;

    QuotePolicy() = default;

    QuotePolicy(
        std::vector<double> q_grid_,
        std::vector<double> y_grid_,
        std::vector<double> nu_grid_,
        std::vector<double> sizes_,
        FlatCube4D bid_,
        FlatCube4D ask_
    )
        : q_grid(std::move(q_grid_)),
          y_grid(std::move(y_grid_)),
          nu_grid(std::move(nu_grid_)),
          sizes(std::move(sizes_)),
          bid(std::move(bid_)),
          ask(std::move(ask_)) {
        validate();
    }

    void set_axes(
        const std::vector<double>& q_grid_,
        const std::vector<double>& y_grid_,
        const std::vector<double>& nu_grid_,
        const std::vector<double>& sizes_
    ) {
        q_grid = q_grid_;
        y_grid = y_grid_;
        nu_grid = nu_grid_;
        sizes = sizes_;
    }

    void ensure_storage_shape() {
        bid.ensure_shape(q_grid.size(), y_grid.size(), nu_grid.size(), sizes.size());
        ask.ensure_shape(q_grid.size(), y_grid.size(), nu_grid.size(), sizes.size());
    }

    void validate() const {
        const std::size_t nq  = q_grid.size();
        const std::size_t ny  = y_grid.size();
        const std::size_t nnu = nu_grid.size();
        const std::size_t nz  = sizes.size();

        auto check = [&](const FlatCube4D& c, const char* name) {
            if (c.nq != nq || c.ny != ny || c.nnu != nnu || c.nz != nz)
                throw std::invalid_argument(
                    std::string("QuotePolicy: ") + name + " shape mismatch.");
        };
        check(bid, "bid");
        check(ask, "ask");
    }

    double delta_at_index(
        std::size_t iq, std::size_t iy, std::size_t inu, std::size_t iz, Side side
    ) const {
        return (side == Side::Bid) ? bid.at(iq, iy, inu, iz)
                                   : ask.at(iq, iy, inu, iz);
    }

    double quote(double q, double y, double nu, double z, Side side) const {
        const FlatCube4D& cube = (side == Side::Bid) ? bid : ask;

        if (sizes.empty())
            throw std::invalid_argument("QuotePolicy::quote: sizes cannot be empty.");

        if (sizes.size() == 1)
            return interp_trilinear_rung(q_grid, y_grid, nu_grid, cube, q, y, nu, 0);

        const auto [iz, tz] = locate_segment_with_weight(sizes, z);
        const double v0 = interp_trilinear_rung(q_grid, y_grid, nu_grid, cube, q, y, nu, iz);
        const double v1 = interp_trilinear_rung(q_grid, y_grid, nu_grid, cube, q, y, nu, iz + 1);
        return v0 + tz * (v1 - v0);
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

    QuotePolicy policy;

    PriceTier() = default;

    PriceTier(
        std::string name_,
        std::vector<double> sizes_,
        std::shared_ptr<FlowCurve> flow_curve_,
        std::shared_ptr<DriftJumpModel> jump_model_,
        double delta_min_ = -0.5,
        double delta_max_ = 4.0
    )
        : name(std::move(name_)),
          sizes(std::move(sizes_)),
          flow_curve(std::move(flow_curve_)),
          jump_model(std::move(jump_model_)),
          delta_min(delta_min_),
          delta_max(delta_max_) {
        validate();
    }

    void validate() const {
        if (sizes.empty()) throw std::invalid_argument("PriceTier: sizes cannot be empty.");
        if (!flow_curve) throw std::invalid_argument("PriceTier: flow_curve cannot be null.");
        if (!jump_model) throw std::invalid_argument("PriceTier: jump_model cannot be null.");
        if (delta_max <= delta_min)
            throw std::invalid_argument("PriceTier: delta_max must be > delta_min.");
        for (double z : sizes)
            if (z <= 0.0) throw std::invalid_argument("PriceTier: sizes must be positive.");
        for (std::size_t i = 1; i < sizes.size(); ++i)
            if (sizes[i] <= sizes[i - 1])
                throw std::invalid_argument("PriceTier: sizes must be strictly increasing.");
    }

    double arrival_rate(double delta, double z) const { return flow_curve->arrival_rate(delta, z); }
    double jump_size(double z) const { return jump_model->jump_size(z); }

    double quote(double q, double y, double nu, double z, Side side) const {
        return policy.quote(q, y, nu, z, side);
    }
};

// ============================================================
// Stateful ladder policy builder
// ============================================================

struct LadderPolicyBuilder {
    const std::vector<double>& q_grid;
    const std::vector<double>& y_grid;
    const std::vector<double>& nu_grid;
    const opt::GoldenSectionOptimizer& optimizer;
    std::size_t q0_idx{0};

    const FlatCube3D* h_mat{nullptr};
    const HInterp3D* h_fn{nullptr};

    std::vector<double> prev_gaps_storage;

    LadderPolicyBuilder(
        const std::vector<double>& q_grid_,
        const std::vector<double>& y_grid_,
        const std::vector<double>& nu_grid_,
        const opt::GoldenSectionOptimizer& optimizer_
    )
        : q_grid(q_grid_),
          y_grid(y_grid_),
          nu_grid(nu_grid_),
          optimizer(optimizer_) {
        if (q_grid.empty())
            throw std::invalid_argument("LadderPolicyBuilder: q_grid cannot be empty.");
        if (q_grid.size() % 2 == 0)
            throw std::invalid_argument("LadderPolicyBuilder: q_grid must have odd length.");

        q0_idx = q_grid.size() / 2;
        if (!approx_equal(q_grid[q0_idx], 0.0)) {
            throw std::invalid_argument(
                "LadderPolicyBuilder: q_grid middle point must be 0.");
        }
    }

    LadderPolicyBuilder(const LadderPolicyBuilder&) = delete;
    LadderPolicyBuilder& operator=(const LadderPolicyBuilder&) = delete;
    LadderPolicyBuilder(LadderPolicyBuilder&&) = delete;
    LadderPolicyBuilder& operator=(LadderPolicyBuilder&&) = delete;

    GridState make_state(std::size_t iq, std::size_t iy, std::size_t inu) const {
        return GridState{iq, iy, inu, q_grid[iq], y_grid[iy], nu_grid[inu]};
    }

    void set_value_function(const FlatCube3D& h_mat_, const HInterp3D& h_fn_) {
        h_mat = &h_mat_;
        h_fn = &h_fn_;
    }

    static double next_inventory(double q, double z, Side side) {
        return (side == Side::Bid) ? (q + z) : (q - z);
    }

    static double next_drift(double y, double jump, Side side) {
        return (side == Side::Bid) ? (y - jump) : (y + jump);
    }

    static bool is_shrink_mode(double q, Side side) {
        return (q > 0.0 && side == Side::Ask) || (q < 0.0 && side == Side::Bid);
    }

    static bool is_expand_mode(double q, Side side) {
        return (q > 0.0 && side == Side::Bid) || (q < 0.0 && side == Side::Ask);
    }

    static std::pair<double, double> rung_bounds(
        const PriceTier& tier,
        std::size_t rung_idx,
        const double* current_delta,
        double q,
        Side side,
        const std::vector<double>* prev_gaps
    ) {
        const std::size_t n = tier.sizes.size();
        double lower = tier.delta_min;
        double upper = tier.delta_max;

        if (rung_idx == 0) {
            if (prev_gaps != nullptr && is_expand_mode(q, side)) {
                double total = 0.0;
                for (double g : *prev_gaps) total += g;
                upper = std::min(upper, tier.delta_max - total);
            }
            return {lower, upper};
        }

        const double prev_curr = current_delta[rung_idx - 1];
        lower = std::max(lower, prev_curr);

        if (prev_gaps == nullptr || prev_gaps->empty())
            return {lower, upper};

        const double prev_gap = (*prev_gaps)[rung_idx - 1];

        if (is_shrink_mode(q, side)) {
            upper = std::min(upper, prev_curr + prev_gap);
        } else if (is_expand_mode(q, side)) {
            lower = std::max(lower, prev_curr + prev_gap);

            double remaining = 0.0;
            for (std::size_t k = rung_idx; k + 1 < n; ++k)
                remaining += (*prev_gaps)[k];
            upper = std::min(upper, tier.delta_max - remaining);
        }

        return {lower, upper};
    }

    void solve_side_ladder_inplace(
        const PriceTier& tier,
        const GridState& s,
        Side side,
        double* delta,
        const double* prev_delta = nullptr
    ) {
        if (!h_mat || !h_fn) {
            throw std::logic_error(
                "LadderPolicyBuilder::solve_side_ladder_inplace: value function not set.");
        }

        const std::size_t n = tier.sizes.size();
        const double h_here = h_mat->at(s.iq, s.iy, s.inu);

        const std::vector<double>* prev_gaps = nullptr;
        if (prev_delta != nullptr) {
            diff_inplace(prev_delta, n, prev_gaps_storage);
            prev_gaps = &prev_gaps_storage;
        } else {
            prev_gaps_storage.clear();
        }

        for (std::size_t j = 0; j < n; ++j) {
            auto [lower0, upper0] = rung_bounds(tier, j, delta, s.q, side, prev_gaps);
            double lower = std::max(tier.delta_min, lower0);
            double upper = std::min(tier.delta_max, upper0);

            if (lower > upper + 1e-12) {
                throw std::runtime_error(
                    "Infeasible ladder bounds for tier '" + tier.name +
                    "', side '" + std::string(side_name(side)) + "'.");
            }
            if (upper < lower) upper = lower;

            const double z    = tier.sizes[j];
            const double jump = tier.jump_size(z);
            const double dh   = (*h_fn)(next_inventory(s.q, z, side),
                                        next_drift(s.y, jump, side), s.nu) - h_here;

            auto objective = [&](double d) {
                return tier.arrival_rate(d, z) * (z * d + dh);
            };

            const auto res = optimizer.maximize(objective, lower, upper);
            delta[j] = clamp(res.x_star, lower, upper);
        }
    }

    void build_policy_inplace(const PriceTier& tier, QuotePolicy& policy) {
        if (!h_mat || !h_fn) {
            throw std::logic_error(
                "LadderPolicyBuilder::build_policy_inplace: value function not set.");
        }

        const std::size_t nq  = q_grid.size();
        const std::size_t ny  = y_grid.size();
        const std::size_t nnu = nu_grid.size();

        for (std::size_t iy = 0; iy < ny; ++iy) {
            for (std::size_t inu = 0; inu < nnu; ++inu) {
                const GridState s0 = make_state(q0_idx, iy, inu);

                solve_side_ladder_inplace(
                    tier, s0, Side::Ask,
                    policy.ask.row_ptr(q0_idx, iy, inu), nullptr);
                solve_side_ladder_inplace(
                    tier, s0, Side::Bid,
                    policy.bid.row_ptr(q0_idx, iy, inu), nullptr);

                for (std::size_t iq = q0_idx + 1; iq < nq; ++iq) {
                    const GridState s = make_state(iq, iy, inu);

                    solve_side_ladder_inplace(
                        tier, s, Side::Ask,
                        policy.ask.row_ptr(iq, iy, inu),
                        policy.ask.row_ptr(iq - 1, iy, inu));
                    solve_side_ladder_inplace(
                        tier, s, Side::Bid,
                        policy.bid.row_ptr(iq, iy, inu),
                        policy.bid.row_ptr(iq - 1, iy, inu));
                }

                for (std::size_t ii = q0_idx; ii-- > 0;) {
                    const GridState s = make_state(ii, iy, inu);

                    solve_side_ladder_inplace(
                        tier, s, Side::Ask,
                        policy.ask.row_ptr(ii, iy, inu),
                        policy.ask.row_ptr(ii + 1, iy, inu));
                    solve_side_ladder_inplace(
                        tier, s, Side::Bid,
                        policy.bid.row_ptr(ii, iy, inu),
                        policy.bid.row_ptr(ii + 1, iy, inu));
                }
            }
        }
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

    double golden_tol{1e-4};
    int golden_max_iter{32};

    bool early_stop{true};
    double tol_h{1e-5};
    double tol_rhs{1e-4};
    int min_iter{5};
    int consecutive_passes_required{3};

    void validate() const {
        if (q_grid.size() < 3 || y_grid.size() < 3 || nu_grid.size() < 3)
            throw std::invalid_argument(
                "SolverConfig: q_grid, y_grid and nu_grid must each have at least 3 points.");

        if (q_grid.size() % 2 == 0)
            throw std::invalid_argument("SolverConfig: q_grid must have odd length.");
        if (y_grid.size() % 2 == 0)
            throw std::invalid_argument("SolverConfig: y_grid must have odd length.");
        if (nu_grid.size() % 2 == 0)
            throw std::invalid_argument("SolverConfig: nu_grid must have odd length.");

        if (dt <= 0.0)
            throw std::invalid_argument("SolverConfig: dt must be positive.");
        if (n_iter < 1)
            throw std::invalid_argument("SolverConfig: n_iter must be at least 1.");
        if (kappa_y < 0.0 || kappa_nu < 0.0 || eta_nu < 0.0)
            throw std::invalid_argument(
                "SolverConfig: mean reversion and vol-of-vol must be nonnegative.");
        if (golden_tol <= 0.0)
            throw std::invalid_argument("SolverConfig: golden_tol must be positive.");
        if (golden_max_iter < 1)
            throw std::invalid_argument("SolverConfig: golden_max_iter must be at least 1.");
        if (tol_h < 0.0 || tol_rhs < 0.0)
            throw std::invalid_argument("SolverConfig: tolerances must be nonnegative.");
        if (min_iter < 0 || consecutive_passes_required < 1)
            throw std::invalid_argument("SolverConfig: invalid stopping parameters.");

        auto check_inc = [](const std::vector<double>& g, const std::string& name) {
            for (std::size_t i = 1; i < g.size(); ++i) {
                if (g[i] <= g[i - 1]) {
                    throw std::invalid_argument(
                        "SolverConfig: " + name + " must be strictly increasing.");
                }
            }
        };

        check_inc(q_grid,  "q_grid");
        check_inc(y_grid,  "y_grid");
        check_inc(nu_grid, "nu_grid");

        const std::size_t q_mid  = q_grid.size() / 2;
        const std::size_t y_mid  = y_grid.size() / 2;
        const std::size_t nu_mid = nu_grid.size() / 2;

        if (!approx_equal(q_grid[q_mid], 0.0))
            throw std::invalid_argument("SolverConfig: q_grid middle point must be 0.");
        if (!approx_equal(y_grid[y_mid], 0.0))
            throw std::invalid_argument("SolverConfig: y_grid middle point must be 0.");
        if (!approx_equal(nu_grid[nu_mid], nu_bar))
            throw std::invalid_argument(
                "SolverConfig: nu_grid middle point must equal nu_bar.");
    }
};

struct HJBSolution {
    FlatCube3D h;
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
    opt::GoldenSectionOptimizer optimizer;
    LadderPolicyBuilder policy_builder;

    HJBLadderSolver(
        SolverConfig config_,
        std::shared_ptr<InventoryPenalty> penalty_,
        std::vector<std::shared_ptr<PriceTier>> tiers_
    )
        : config(std::move(config_)),
          penalty(std::move(penalty_)),
          tiers(std::move(tiers_)),
          optimizer(opt::GoldenSectionOptions{config.golden_tol, config.golden_max_iter}),
          policy_builder(config.q_grid, config.y_grid, config.nu_grid, optimizer) {
        config.validate();
        if (!penalty) throw std::invalid_argument("HJBLadderSolver: penalty cannot be null.");
        initialize_policies();
    }

    void sync_optimizer_from_config() {
        optimizer.set_options(opt::GoldenSectionOptions{config.golden_tol, config.golden_max_iter});
    }

    GridState make_state(std::size_t iq, std::size_t iy, std::size_t inu) const {
        return GridState{iq, iy, inu, config.q_grid[iq], config.y_grid[iy], config.nu_grid[inu]};
    }

    void initialize_policies() {
        for (auto& tier : tiers) {
            if (!tier) throw std::invalid_argument("HJBLadderSolver: tier cannot be null.");
            tier->validate();
            tier->policy.set_axes(config.q_grid, config.y_grid, config.nu_grid, tier->sizes);
            tier->policy.ensure_storage_shape();
            tier->policy.validate();
        }
    }

    double y_drift_term(const FlatCube3D& h_mat, const GridState& s) const {
        const double b = -config.kappa_y * s.y;
        if (std::abs(b) < 1e-14 || config.y_grid.size() == 1) return 0.0;

        const std::size_t ny = config.y_grid.size();
        double dh_dy = 0.0;

        if (b > 0.0) {
            const std::size_t lo = (s.iy == 0) ? 0 : s.iy - 1;
            const std::size_t hi = lo + 1;
            const double dy = config.y_grid[hi] - config.y_grid[lo];
            dh_dy = (h_mat.at(s.iq, hi, s.inu) - h_mat.at(s.iq, lo, s.inu)) / dy;
        } else {
            const std::size_t hi = (s.iy + 1 >= ny) ? ny - 1 : s.iy + 1;
            const std::size_t lo = hi - 1;
            const double dy = config.y_grid[hi] - config.y_grid[lo];
            dh_dy = (h_mat.at(s.iq, hi, s.inu) - h_mat.at(s.iq, lo, s.inu)) / dy;
        }
        return b * dh_dy;
    }

    double nu_drift_term(const FlatCube3D& h_mat, const GridState& s) const {
        const double b = config.kappa_nu * (config.nu_bar - s.nu);
        if (std::abs(b) < 1e-14 || config.nu_grid.size() == 1) return 0.0;

        const std::size_t nnu = config.nu_grid.size();
        double dh_dnu = 0.0;

        if (b > 0.0) {
            const std::size_t lo = (s.inu == 0) ? 0 : s.inu - 1;
            const std::size_t hi = lo + 1;
            const double dnu = config.nu_grid[hi] - config.nu_grid[lo];
            dh_dnu = (h_mat.at(s.iq, s.iy, hi) - h_mat.at(s.iq, s.iy, lo)) / dnu;
        } else {
            const std::size_t hi = (s.inu + 1 >= nnu) ? nnu - 1 : s.inu + 1;
            const std::size_t lo = hi - 1;
            const double dnu = config.nu_grid[hi] - config.nu_grid[lo];
            dh_dnu = (h_mat.at(s.iq, s.iy, hi) - h_mat.at(s.iq, s.iy, lo)) / dnu;
        }
        return b * dh_dnu;
    }

    double nu_diffusion_term(const FlatCube3D& h_mat, const GridState& s) const {
        if (config.eta_nu == 0.0 || config.nu_grid.size() < 3) return 0.0;

        const std::size_t nnu = config.nu_grid.size();
        double second = 0.0;

        if (s.inu == 0) {
            const double d = config.nu_grid[1] - config.nu_grid[0];
            second = (h_mat.at(s.iq, s.iy, 0) - 2.0 * h_mat.at(s.iq, s.iy, 1)
                      + h_mat.at(s.iq, s.iy, 2)) / (d * d);
        } else if (s.inu + 1 == nnu) {
            const double d = config.nu_grid[nnu - 1] - config.nu_grid[nnu - 2];
            second = (h_mat.at(s.iq, s.iy, nnu - 3) - 2.0 * h_mat.at(s.iq, s.iy, nnu - 2)
                      + h_mat.at(s.iq, s.iy, nnu - 1)) / (d * d);
        } else {
            const double d = config.nu_grid[s.inu + 1] - config.nu_grid[s.inu];
            second = (h_mat.at(s.iq, s.iy, s.inu + 1) - 2.0 * h_mat.at(s.iq, s.iy, s.inu)
                      + h_mat.at(s.iq, s.iy, s.inu - 1)) / (d * d);
        }
        return 0.5 * config.eta_nu * config.eta_nu * second;
    }

    void update_policies(const FlatCube3D& h_mat, const HInterp3D& h_fn) {
        sync_optimizer_from_config();
        policy_builder.set_value_function(h_mat, h_fn);
        for (auto& tier : tiers) {
            policy_builder.build_policy_inplace(*tier, tier->policy);
        }
    }

    void update_policies(const FlatCube3D& h_mat) {
        HInterp3D h_fn{config.q_grid, config.y_grid, config.nu_grid, h_mat};
        update_policies(h_mat, h_fn);
    }

    HJBSolution solve() {
        sync_optimizer_from_config();

        const std::size_t nq  = config.q_grid.size();
        const std::size_t ny  = config.y_grid.size();
        const std::size_t nnu = config.nu_grid.size();

        FlatCube3D h_curr(nq, ny, nnu, 0.0);
        FlatCube3D h_next(nq, ny, nnu, 0.0);

        HInterp3D h_fn{config.q_grid, config.y_grid, config.nu_grid, h_curr};

        const std::size_t q0_idx  = nq / 2;
        const std::size_t y0_idx  = ny / 2;
        const std::size_t nu0_idx = nnu / 2;

        std::vector<double> penalty_q(nq);
        for (std::size_t iq = 0; iq < nq; ++iq)
            penalty_q[iq] = penalty->value(config.q_grid[iq]);

        std::vector<double> exp_2nu(nnu);
        for (std::size_t inu = 0; inu < nnu; ++inu)
            exp_2nu[inu] = std::exp(2.0 * config.nu_grid[inu]);

        bool converged = false;
        int iterations_used = 0;
        double final_max_h_change = std::numeric_limits<double>::infinity();
        double final_max_rhs = std::numeric_limits<double>::infinity();
        int consecutive_passes = 0;

        std::vector<double> hist_h, hist_rhs;
        hist_h.reserve(static_cast<std::size_t>(config.n_iter));
        hist_rhs.reserve(static_cast<std::size_t>(config.n_iter));

        for (int it = 1; it <= config.n_iter; ++it) {
            update_policies(h_curr, h_fn);

            double max_h_change = 0.0;
            double max_rhs_now  = 0.0;

            for (std::size_t iq = 0; iq < nq; ++iq) {
                for (std::size_t iy = 0; iy < ny; ++iy) {
                    for (std::size_t inu = 0; inu < nnu; ++inu) {
                        const GridState s = make_state(iq, iy, inu);
                        const double h_here = h_curr.at(s.iq, s.iy, s.inu);

                        double rhs = s.q * s.y
                                   + y_drift_term(h_curr, s)
                                   + nu_drift_term(h_curr, s)
                                   + nu_diffusion_term(h_curr, s)
                                   - exp_2nu[s.inu] * penalty_q[s.iq];

                        // ----------------------------------------------------------------- //
                        // Price tier optimization
                        // ----------------------------------------------------------------- //
                        for (const auto& tier : tiers) {
                            const double* bid_row = tier->policy.bid.row_ptr(s.iq, s.iy, s.inu);
                            const double* ask_row = tier->policy.ask.row_ptr(s.iq, s.iy, s.inu);

                            for (std::size_t iz = 0; iz < tier->sizes.size(); ++iz) {
                                const double z    = tier->sizes[iz];
                                const double jump = tier->jump_size(z);

                                const double d_b   = bid_row[iz];
                                const double lam_b = tier->arrival_rate(d_b, z);
                                rhs += lam_b * (z * d_b + h_fn(s.q + z, s.y - jump, s.nu) - h_here);

                                const double d_a   = ask_row[iz];
                                const double lam_a = tier->arrival_rate(d_a, z);
                                rhs += lam_a * (z * d_a + h_fn(s.q - z, s.y + jump, s.nu) - h_here);
                            }
                        }

                        const double new_h = h_here + config.dt * rhs;
                        h_next.at(s.iq, s.iy, s.inu) = new_h;

                        max_h_change = std::max(max_h_change, std::abs(new_h - h_here));
                        max_rhs_now  = std::max(max_rhs_now,  std::abs(rhs));
                    }
                }
            }

            const double anchor = h_next.at(q0_idx, y0_idx, nu0_idx);
            for (double& v : h_next.data) v -= anchor;

            hist_h.push_back(max_h_change);
            hist_rhs.push_back(max_rhs_now);
            final_max_h_change = max_h_change;
            final_max_rhs = max_rhs_now;
            iterations_used = it;

            const bool passes_now =
                it >= config.min_iter &&
                max_h_change <= config.tol_h &&
                max_rhs_now  <= config.tol_rhs;

            consecutive_passes = passes_now ? consecutive_passes + 1 : 0;

            h_curr.data.swap(h_next.data);

            if (config.early_stop &&
                consecutive_passes >= config.consecutive_passes_required) {
                converged = true;
                break;
            }
        }

        // ----------------------------------------------------------------- //
        // Final policy update after convergence or max iterations
        // ----------------------------------------------------------------- //
        update_policies(h_curr, h_fn);

        HJBSolution out;
        out.h                    = std::move(h_curr);
        out.q_grid               = config.q_grid;
        out.y_grid               = config.y_grid;
        out.nu_grid              = config.nu_grid;
        out.tiers                = tiers;
        out.converged            = converged;
        out.iterations_used      = iterations_used;
        out.final_max_h_change   = final_max_h_change;
        out.final_max_rhs        = final_max_rhs;
        out.history_max_h_change = std::move(hist_h);
        out.history_max_rhs      = std::move(hist_rhs);
        return out;
    }
};

} // namespace hjb