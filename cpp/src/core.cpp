#include "ladder_pricer/core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ladder_pricer {
namespace {

constexpr double kTolerance = 1e-10;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void require_positive_increasing(const std::vector<double>& values, const std::string& name) {
    require(!values.empty(), name + " must not be empty");
    for (std::size_t i = 0; i < values.size(); ++i) {
        require(values[i] > 0.0, name + " must contain positive values");
        if (i > 0) {
            require(values[i] > values[i - 1], name + " must be strictly increasing");
        }
    }
}

void require_centered_grid(const std::vector<double>& grid) {
    require(grid.size() >= 3 && grid.size() % 2 == 1,
            "inventory grid must have odd length and at least three states");
    for (std::size_t i = 1; i < grid.size(); ++i) {
        require(grid[i] > grid[i - 1], "inventory grid must be strictly increasing");
    }
    const std::size_t mid = grid.size() / 2;
    require(std::abs(grid[mid]) <= kTolerance, "inventory grid must contain zero at its center");
    for (std::size_t i = 0; i < mid; ++i) {
        require(std::abs(grid[i] + grid[grid.size() - 1 - i]) <= kTolerance,
                "inventory grid must be symmetric around zero");
    }
}

std::vector<double> fill_breakpoints(const std::vector<double>& grid, std::size_t i,
                                     double dir, double posted_size) {
    std::vector<double> out;
    const double q = grid[i];
    if (dir > 0.0) {
        for (std::size_t j = i + 1; j < grid.size(); ++j) {
            const double x = grid[j] - q;
            if (x >= posted_size - kTolerance) break;
            if (x > kTolerance) out.push_back(x);
        }
    } else {
        for (std::size_t j = i; j-- > 0;) {
            const double x = q - grid[j];
            if (x >= posted_size - kTolerance) break;
            if (x > kTolerance) out.push_back(x);
        }
    }
    return out;
}

std::size_t exact_index(const std::vector<double>& grid, double value) {
    auto it = std::lower_bound(grid.begin(), grid.end(), value - kTolerance);
    if (it != grid.end() && std::abs(*it - value) <= kTolerance) {
        return static_cast<std::size_t>(std::distance(grid.begin(), it));
    }
    throw std::runtime_error("inventory state missing from solve grid");
}

struct InterpolationWeights {
    std::size_t left;
    std::size_t right;
    double left_weight;
    double right_weight;
};

InterpolationWeights interpolation_weights(const std::vector<double>& grid, double q) {
    if (q <= grid.front() + kTolerance) {
        return {0, 0, 1.0, 0.0};
    }
    if (q >= grid.back() - kTolerance) {
        const auto i = grid.size() - 1;
        return {i, i, 1.0, 0.0};
    }

    auto upper = std::lower_bound(grid.begin(), grid.end(), q);
    const auto right = static_cast<std::size_t>(std::distance(grid.begin(), upper));
    if (std::abs(grid[right] - q) <= kTolerance) {
        return {right, right, 1.0, 0.0};
    }
    const auto left = right - 1;
    const double width = grid[right] - grid[left];
    const double w_right = (q - grid[left]) / width;
    return {left, right, 1.0 - w_right, w_right};
}

std::vector<double>& matrix(LadderPolicy& policy, Side side, std::size_t row) {
    return side == Side::Bid ? policy.bid[row] : policy.ask[row];
}

const std::vector<double>& matrix(const LadderPolicy& policy, Side side, std::size_t row) {
    return side == Side::Bid ? policy.bid[row] : policy.ask[row];
}

double direction(Side side) noexcept {
    return side == Side::Bid ? 1.0 : -1.0;
}


}  // namespace

// ---------------------------------------------------------------------------
// Inventory grid
// ---------------------------------------------------------------------------

InventoryGrid::InventoryGrid(std::vector<double> operational_states, double max_jump)
    : operational_(std::move(operational_states)) {
    require_centered_grid(operational_);
    require(max_jump >= 0.0, "max inventory jump must be nonnegative");

    operational_limit_ = operational_.back();
    hard_limit_ = operational_limit_ + max_jump;

    if (max_jump <= 0.0) {
        states_ = operational_;
    } else {
        const std::size_t mid = operational_.size() / 2;
        const double outer_step = operational_.back() - operational_[operational_.size() - 2];
        require(outer_step > 0.0, "inventory grid outer spacing must be positive");

        std::vector<double> positive(operational_.begin() + static_cast<std::ptrdiff_t>(mid),
                                     operational_.end());
        for (double q = operational_limit_ + outer_step;
             q < hard_limit_ - kTolerance;
             q += outer_step) {
            positive.push_back(q);
        }
        if (std::abs(positive.back() - hard_limit_) > kTolerance) {
            positive.push_back(hard_limit_);
        }

        states_.reserve(2 * positive.size() - 1);
        for (auto it = positive.rbegin(); it != positive.rend() - 1; ++it) {
            states_.push_back(-*it);
        }
        states_.insert(states_.end(), positive.begin(), positive.end());
    }

    zero_index_ = states_.size() / 2;
    operational_indices_.reserve(operational_.size());
    for (double q : operational_) {
        operational_indices_.push_back(exact_index(states_, q));
    }
}

bool InventoryGrid::admissible(double q, double inventory_change) const noexcept {
    const double target = q + inventory_change;
    return target >= -hard_limit_ - kTolerance && target <= hard_limit_ + kTolerance;
}

double InventoryGrid::interpolate(const std::vector<double>& values, double q) const {
    require(values.size() == states_.size(), "value vector does not match solve inventory grid");
    const auto w = interpolation_weights(states_, q);
    return w.left_weight * values[w.left] + w.right_weight * values[w.right];
}

// ---------------------------------------------------------------------------
// Economic primitives
// ---------------------------------------------------------------------------

LogisticFlow::LogisticFlow(double a0, double theta, double beta, double shift,
                           double steepness, double volume_shift)
    : a0_(a0), theta_(theta), beta_(beta), shift_(shift),
      steepness_(steepness), volume_shift_(volume_shift) {
    require(a0_ >= 0.0, "RFQ arrival-intensity scale must be nonnegative");
    require(steepness_ > 0.0, "logistic steepness must be positive");
}

double LogisticFlow::size_weight(double size) const {
    require(size > 0.0, "trade size must be positive");
    return std::pow(size, -theta_ - beta_ * size);
}

double LogisticFlow::rfq_arrival_rate(double size) const {
    return a0_ * size_weight(size);
}

double LogisticFlow::center(double size) const {
    return shift_ - volume_shift_ * (size - 1.0);
}

double LogisticFlow::hit_ratio(double delta, double size) const {
    const double y = steepness_ * (delta - center(size));
    if (y >= 0.0) {
        const double e = y < 745.0 ? std::exp(-y) : 0.0;
        return 1.0 / (1.0 + e);
    }
    const double e = y > -745.0 ? std::exp(y) : 0.0;
    return e / (1.0 + e);
}

double LogisticFlow::lambert_w_exp(double x) {
    if (std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == -std::numeric_limits<double>::infinity()) {
        return 0.0;
    }
    if (x == std::numeric_limits<double>::infinity()) {
        return std::numeric_limits<double>::infinity();
    }

    double y = x <= 0.0 ? x : std::log1p(x);
    const double eps = std::numeric_limits<double>::epsilon();
    for (int i = 0; i < 24; ++i) {
        const double ey = y > -745.0 ? std::exp(y) : 0.0;
        const double step = (y + ey - x) / (1.0 + ey);
        y -= step;
        if (std::abs(step) <= 8.0 * eps * std::max(1.0, std::abs(y))) {
            break;
        }
    }
    return y < 709.0 ? std::exp(y) : std::numeric_limits<double>::infinity();
}

double LogisticFlow::optimal_delta(double size, double spread, double additive_value) const {
    require(size > 0.0, "trade size must be positive");
    require(spread > 0.0, "spread must be positive");
    const double b = 0.5 + additive_value / (size * spread);
    const double x = steepness_ * (b - center(size)) - 1.0;
    const double w = lambert_w_exp(x);
    return b - (1.0 + w) / steepness_;
}


ExponentialFlow::ExponentialFlow(double a, double k) : a_(a), k_(k) {
    require(a_ >= 0.0, "exponential A must be nonnegative");
    require(k_ > 0.0, "exponential k must be positive");
}

double ExponentialFlow::arrival_rate(double delta) const {
    // delta uses the tier convention: d=0 is touch and d=0.5 is mid.
    // With X=0.5-d as distance from mid in spread fractions and X~Exp(k),
    // lambda_fill(d)=A*P(X_reach>=0.5-d)=A*exp(-k*(0.5-d)).
    const double x = -k_ * (0.5 - delta);
    if (x >= 709.0) return std::numeric_limits<double>::infinity();
    if (x <= -745.0) return 0.0;
    return a_ * std::exp(x);
}


ECNFlowSource::ECNFlowSource(std::string name, ExponentialFlow flow,
                             double delta_scale, double source_size_per_target,
                             double mean_trade_size)
    : name_(std::move(name)), flow_(std::move(flow)), delta_scale_(delta_scale),
      source_size_per_target_(source_size_per_target), mean_trade_size_(mean_trade_size) {
    require(!name_.empty(), "ECN flow source name must not be empty");
    require(delta_scale_ > 0.0, "ECN flow source delta scale must be positive");
    require(source_size_per_target_ > 0.0,
            "ECN flow source size scale must be positive");
    require(mean_trade_size_ > 0.0,
            "ECN flow source mean trade size must be positive");
}

double ECNFlowSource::implied_delta(double target_delta) const noexcept {
    return 0.5 + delta_scale_ * (target_delta - 0.5);
}

double ECNFlowSource::arrival_rate(double target_delta) const {
    return flow_.arrival_rate(implied_delta(target_delta));
}

double ECNFlowSource::target_reach(double source_reach) const noexcept {
    return source_reach / delta_scale_;
}

double ECNFlowSource::full_fill_probability(double target_posted_size) const {
    require(target_posted_size > 0.0, "ECN target posted size must be positive");
    const double mean = mean_target_trade_size();
    return std::exp(-target_posted_size / mean);
}

double ECNFlowSource::expected_fill_size(double target_posted_size) const {
    require(target_posted_size > 0.0, "ECN target posted size must be positive");
    const double mean = mean_target_trade_size();
    return mean * (-std::expm1(-target_posted_size / mean));
}

std::vector<std::pair<double, double>> ECNFlowSource::fill_components(
        double target_posted_size,
        const std::vector<double>& target_breakpoints) const {
    require(target_posted_size > 0.0, "ECN target posted size must be positive");
    const double mean = mean_target_trade_size();

    std::vector<double> cuts;
    cuts.reserve(target_breakpoints.size() + 1);
    for (double x : target_breakpoints) {
        if (x > kTolerance && x < target_posted_size - kTolerance) cuts.push_back(x);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end(), [](double a, double b) {
        return std::abs(a - b) <= kTolerance;
    }), cuts.end());
    cuts.push_back(target_posted_size);

    std::vector<std::pair<double, double>> out;
    out.reserve(cuts.size() + 1);
    double a = 0.0;
    for (double b : cuts) {
        const double length = b - a;
        if (length <= kTolerance) {
            a = b;
            continue;
        }
        const double survival_a = std::exp(-a / mean);
        const double interval_mass = survival_a * (-std::expm1(-length / mean));
        if (interval_mass > 0.0) {
            const double x = length / mean;
            double conditional_offset = 0.0;
            if (x < 1e-5) {
                // mu - L/(exp(L/mu)-1), evaluated with a cancellation-safe series.
                conditional_offset = length * (0.5 - x / 12.0 + x * x * x / 720.0);
            } else if (x > 700.0) {
                conditional_offset = mean;
            } else {
                conditional_offset = mean - length / std::expm1(x);
            }
            out.emplace_back(a + conditional_offset, interval_mass);
        }
        a = b;
    }

    // X >= posted size is an atom at the posted size after capping.
    const double full_mass = full_fill_probability(target_posted_size);
    if (full_mass > 0.0) out.emplace_back(target_posted_size, full_mass);
    return out;
}

AggregatedECNFlow::AggregatedECNFlow(ExponentialFlow direct_flow)
    : sources_{ECNFlowSource("direct", std::move(direct_flow), 1.0, 1.0, 1.0)} {}

AggregatedECNFlow::AggregatedECNFlow(std::vector<ECNFlowSource> sources)
    : sources_(std::move(sources)) {
    require(!sources_.empty(), "aggregated ECN flow requires at least one source");
}

double AggregatedECNFlow::arrival_rate(double target_delta) const {
    double total = 0.0;
    for (const auto& source : sources_) total += source.arrival_rate(target_delta);
    return total;
}


FlowSource::FlowSource(std::string name, LogisticFlow flow,
                       double delta_scale, double source_size_per_target)
    : name_(std::move(name)), flow_(std::move(flow)), delta_scale_(delta_scale),
      source_size_per_target_(source_size_per_target) {
    require(!name_.empty(), "flow source name must not be empty");
    require(delta_scale_ > 0.0, "flow source delta scale must be positive");
    require(source_size_per_target_ > 0.0,
            "flow source size scale must be positive");
}

double FlowSource::implied_delta(double target_delta) const noexcept {
    return 0.5 + delta_scale_ * (target_delta - 0.5);
}

double FlowSource::source_size(double target_size) const noexcept {
    return target_size * source_size_per_target_;
}

void FlowSource::configure_target_sizes(const std::vector<double>& target_sizes) {
    require(!target_sizes.empty(), "RFQ size distribution requires at least one target size");
    target_sizes_ = target_sizes;
    size_probabilities_.assign(target_sizes_.size(), 0.0);

    // The calibrated exogenous arrival-intensity curve itself defines the RFQ
    // size distribution.  For the current discrete rung model, normalize the
    // curve values across the configured target sizes.  The full two-sided
    // Monte Carlo clock is twice the sum of the one-sided rung intensities.
    double one_side_total = 0.0;
    for (std::size_t i = 0; i < target_sizes_.size(); ++i) {
        const double z = target_sizes_[i];
        require(z > 0.0, "RFQ target sizes must be positive");
        const double intensity = flow_.rfq_arrival_rate(source_size(z));
        require(std::isfinite(intensity) && intensity >= 0.0,
                "RFQ arrival intensities must be finite and nonnegative");
        size_probabilities_[i] = intensity;
        one_side_total += intensity;
    }
    require(one_side_total > 0.0, "RFQ arrival-intensity curve must have positive mass");
    for (double& probability : size_probabilities_) probability /= one_side_total;
    total_rfq_rate_ = 2.0 * one_side_total;
}

std::size_t FlowSource::sample_target_size_index(std::mt19937_64& rng) const {
    require(!size_probabilities_.empty(), "RFQ size distribution has not been configured");
    std::discrete_distribution<std::size_t> distribution(
        size_probabilities_.begin(), size_probabilities_.end());
    return distribution(rng);
}

double FlowSource::rfq_arrival_rate(double target_size) const {
    require(!target_sizes_.empty(), "RFQ size distribution has not been configured");
    for (double z : target_sizes_) {
        if (std::abs(z - target_size) <= kTolerance) {
            // This is the original calibrated one-sided exogenous intensity at
            // the requested source-currency size.  Equivalently it is
            // 0.5 * total_rfq_rate() * p(size).
            return flow_.rfq_arrival_rate(source_size(target_size));
        }
    }
    throw std::invalid_argument("target RFQ size is not one of the configured tier sizes");
}

double FlowSource::win_probability(double target_delta, double target_size) const {
    return flow_.win_probability(implied_delta(target_delta), source_size(target_size));
}

double FlowSource::arrival_rate(double target_delta, double target_size) const {
    return rfq_arrival_rate(target_size) * win_probability(target_delta, target_size);
}

double FlowSource::target_center(double target_size) const {
    return 0.5 + (flow_.center(source_size(target_size)) - 0.5) / delta_scale_;
}

double FlowSource::effective_steepness() const noexcept {
    return flow_.steepness() * delta_scale_;
}

AggregatedFlow::AggregatedFlow(LogisticFlow direct_flow)
    : sources_{FlowSource("direct", std::move(direct_flow), 1.0, 1.0)} {}

AggregatedFlow::AggregatedFlow(std::vector<FlowSource> sources)
    : sources_(std::move(sources)) {
    require(!sources_.empty(), "aggregated flow requires at least one source");
}

void AggregatedFlow::configure_target_sizes(const std::vector<double>& target_sizes) {
    for (auto& source : sources_) source.configure_target_sizes(target_sizes);
}

double AggregatedFlow::rfq_arrival_rate(double target_size) const {
    double total = 0.0;
    for (const auto& source : sources_) total += source.rfq_arrival_rate(target_size);
    return total;
}

double AggregatedFlow::arrival_rate(double target_delta, double target_size) const {
    double total = 0.0;
    for (const auto& source : sources_) total += source.arrival_rate(target_delta, target_size);
    return total;
}

double AggregatedFlow::win_probability(double target_delta, double target_size) const {
    const double rfq = rfq_arrival_rate(target_size);
    return rfq > 0.0 ? arrival_rate(target_delta, target_size) / rfq : 0.0;
}

double AggregatedFlow::optimal_delta(double target_size, double spread, double additive_value,
                                     double lower, double upper) const {
    require(target_size > 0.0, "trade size must be positive");
    require(spread > 0.0, "spread must be positive");
    require(lower <= upper, "lower delta bound must not exceed upper bound");

    if (sources_.size() == 1 &&
        std::abs(sources_.front().delta_scale() - 1.0) <= 1e-14 &&
        std::abs(sources_.front().source_size_per_target() - 1.0) <= 1e-14) {
        return std::clamp(
            sources_.front().flow().optimal_delta(target_size, spread, additive_value),
            lower, upper);
    }

    const auto objective = [&](double delta) {
        const double bracket = target_size * spread * (0.5 - delta) + additive_value;
        return arrival_rate(delta, target_size) * bracket;
    };

    const double break_even = 0.5 + additive_value / (target_size * spread);
    if (break_even <= lower) return lower;
    const double search_upper = std::min(upper, break_even);
    if (search_upper <= lower + 1e-14) return lower;

    std::vector<double> knots{lower, search_upper};
    for (const auto& source : sources_) {
        const double center = source.target_center(target_size);
        const double k = source.effective_steepness();
        knots.push_back(std::clamp(center, lower, search_upper));
        for (double width : {1.0, 2.0, 4.0, 8.0}) {
            knots.push_back(std::clamp(center - width / k, lower, search_upper));
            knots.push_back(std::clamp(center + width / k, lower, search_upper));
        }
    }
    knots.push_back(std::clamp(break_even, lower, search_upper));
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end(), [](double a, double b) {
        return std::abs(a - b) <= 1e-12;
    }), knots.end());

    std::size_t best = 0;
    double best_value = objective(knots[0]);
    for (std::size_t i = 1; i < knots.size(); ++i) {
        const double value = objective(knots[i]);
        if (value > best_value) { best = i; best_value = value; }
    }

    double a = best == 0 ? knots[best] : knots[best - 1];
    double b = best + 1 >= knots.size() ? knots[best] : knots[best + 1];
    if (b <= a + 1e-13) return knots[best];

    constexpr double phi = 0.6180339887498948482;
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = objective(c), fd = objective(d);
    for (int iter = 0; iter < 48; ++iter) {
        if (fc < fd) {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a); fd = objective(d);
        } else {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a); fc = objective(c);
        }
    }
    const double candidate = 0.5 * (a + b);
    return objective(candidate) >= best_value ? candidate : knots[best];
}

SaturatingMarkout::SaturatingMarkout(double impact_scale, double size_exponent, double tau_minutes)
    : impact_scale_(impact_scale), size_exponent_(size_exponent), tau_minutes_(tau_minutes) {
    require(impact_scale_ >= 0.0, "markout impact scale must be nonnegative");
    require(size_exponent_ >= 0.0, "markout size exponent must be nonnegative");
    require(tau_minutes_ > 0.0, "markout tau must be positive");
}

double SaturatingMarkout::asymptotic(double size) const {
    require(size > 0.0, "trade size must be positive");
    return impact_scale_ * std::pow(size, size_exponent_);
}

double SaturatingMarkout::expected(double size, double minutes) const {
    if (minutes <= 0.0) {
        return 0.0;
    }
    return asymptotic(size) * (-std::expm1(-minutes / tau_minutes_));
}

double InternalizationTime::value(double inventory) const noexcept {
    const double x = std::abs(inventory);
    return tau0_ + tau1_ * x + tau2_ * x * x;
}

QuadraticPenalty::QuadraticPenalty(double risk_aversion, double sigma)
    : risk_aversion_(risk_aversion), sigma_(sigma) {
    require(risk_aversion_ >= 0.0, "risk aversion must be nonnegative");
    require(sigma_ >= 0.0, "sigma must be nonnegative");
}

double QuadraticPenalty::value(double inventory) const noexcept {
    return risk_aversion_ * sigma_ * sigma_ * inventory * inventory;
}

namespace {

std::vector<double> build_rfq_size_support(const std::vector<double>& pricing_sizes,
                                           double step) {
    require(!pricing_sizes.empty(), "tier sizes must not be empty");
    require(step > 0.0 && std::isfinite(step), "RFQ size step must be positive");

    const double lo = pricing_sizes.front();
    const double hi = pricing_sizes.back();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(std::ceil((hi - lo) / step))
                + pricing_sizes.size() + 2);
    for (std::size_t n = 0;; ++n) {
        const double z = lo + static_cast<double>(n) * step;
        if (z > hi + kTolerance) break;
        out.push_back(std::min(z, hi));
    }
    if (out.empty() || std::abs(out.back() - hi) > kTolerance) out.push_back(hi);
    out.insert(out.end(), pricing_sizes.begin(), pricing_sizes.end());
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end(), [](double a, double b) {
        return std::abs(a - b) <= kTolerance;
    }), out.end());
    return out;
}

} // namespace

Tier::Tier(std::string name, std::vector<double> sizes, LogisticFlow flow,
           SaturatingMarkout markout, bool use_markout,
           double delta_min, double delta_max, double fee, double rfq_size_step)
    : Tier(std::move(name), std::move(sizes), AggregatedFlow(std::move(flow)),
           std::move(markout), use_markout, delta_min, delta_max, fee, rfq_size_step) {}

Tier::Tier(std::string name, std::vector<double> sizes, AggregatedFlow flow,
           SaturatingMarkout markout, bool use_markout,
           double delta_min, double delta_max, double fee, double rfq_size_step)
    : name_(std::move(name)), sizes_(std::move(sizes)), rfq_size_step_(rfq_size_step),
      flow_(std::move(flow)), markout_(std::move(markout)), use_markout_(use_markout),
      delta_min_(delta_min), delta_max_(delta_max), fee_(fee) {
    require(!name_.empty(), "tier name must not be empty");
    require_positive_increasing(sizes_, "tier sizes");
    rfq_sizes_ = build_rfq_size_support(sizes_, rfq_size_step_);
    flow_.configure_target_sizes(rfq_sizes_);
    require(delta_min_ <= delta_max_, "delta_min must not exceed delta_max");
    require(fee_ >= 0.0, "tier fee must be nonnegative");
}

// ---------------------------------------------------------------------------
// Dark pool
// ---------------------------------------------------------------------------

ZeroInflatedPoissonArrival::ZeroInflatedPoissonArrival(double intensity, double mean,
                                                       double zero_probability)
    : intensity_(intensity), mean_(mean), zero_probability_(zero_probability) {
    require(intensity_ >= 0.0, "arrival intensity must be nonnegative");
    require(mean_ > 0.0, "Poisson mean must be positive");
    require(zero_probability_ >= 0.0 && zero_probability_ < 1.0,
            "zero probability must lie in [0, 1)");
}

std::vector<std::pair<int, double>> ZeroInflatedPoissonArrival::fill_rates(int posted_size) const {
    require(posted_size > 0, "posted size must be positive");

    // Incoming dark-pool orders arrive at intensity_.  Their size is zero with
    // probability p0; otherwise it is a zero-truncated Poisson(mean_).  If our
    // posted size is u, the realized fill is min(X, u), so the u bucket absorbs
    // the entire upper tail X >= u.
    const double p_zero_pois = std::exp(-mean_);
    const double positive_norm = 1.0 - p_zero_pois;
    if (positive_norm <= 1e-15 || intensity_ <= 0.0 || zero_probability_ >= 1.0) {
        return {};
    }

    const double positive_mass = 1.0 - zero_probability_;
    std::vector<std::pair<int, double>> out;
    out.reserve(static_cast<std::size_t>(posted_size));

    double cumulative_positive = 0.0;
    double p = p_zero_pois;
    for (int k = 1; k < posted_size; ++k) {
        p *= mean_ / static_cast<double>(k);
        const double conditional = p / positive_norm;
        cumulative_positive += conditional;
        const double rate = intensity_ * positive_mass * conditional;
        if (rate > 0.0) out.emplace_back(k, rate);
    }

    const double tail_conditional = std::max(0.0, 1.0 - cumulative_positive);
    const double full_rate = intensity_ * positive_mass * tail_conditional;
    if (full_rate > 0.0) out.emplace_back(posted_size, full_rate);
    return out;
}

int ZeroInflatedPoissonArrival::sample_incoming_size(std::mt19937_64& rng) const {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    if (uniform(rng) < zero_probability_) return 0;

    std::poisson_distribution<int> poisson(mean_);
    int size = 0;
    do {
        size = poisson(rng);
    } while (size <= 0);
    return size;
}

DarkPool::DarkPool(std::shared_ptr<ArrivalDistribution> arrivals,
                   double fee,
                   std::vector<double> posted_sizes,
                   double min_fill_value)
    : arrivals_(std::move(arrivals)), fee_(fee),
      posted_sizes_(std::move(posted_sizes)), min_fill_value_(min_fill_value) {
    require(arrivals_ != nullptr, "dark-pool arrival distribution must not be null");
    require(fee_ >= 0.0, "dark-pool fee must be nonnegative");
    require_positive_increasing(posted_sizes_, "dark-pool posted sizes");
    for (double u : posted_sizes_) {
        require(std::abs(u - std::round(u)) <= kTolerance,
                "dark-pool posted sizes must be integer-valued");
    }
}

bool DarkPool::side_allowed(double inventory, Side side) const noexcept {
    return (inventory > kTolerance && side == Side::Ask) ||
           (inventory < -kTolerance && side == Side::Bid);
}

bool DarkPool::risk_reducing(double inventory, Side side, double size) const noexcept {
    if (size <= 0.0 || !side_allowed(inventory, side)) return false;
    const double change = direction(side) * size;
    const double after = inventory + change;
    return std::abs(after) <= std::abs(inventory) + kTolerance &&
           inventory * after >= -kTolerance;
}

PassiveECN::PassiveECN(std::vector<double> deltas, ExponentialFlow flow,
                       double quote_size, double maker_fee)
    : PassiveECN(std::move(deltas), AggregatedECNFlow(std::move(flow)),
                 quote_size, maker_fee) {}

PassiveECN::PassiveECN(std::vector<double> deltas, AggregatedECNFlow flow,
                       double quote_size, double maker_fee)
    : deltas_(std::move(deltas)), flow_(std::move(flow)),
      quote_size_(quote_size), maker_fee_(maker_fee) {
    require(!deltas_.empty(), "passive ECN deltas must not be empty");
    require(deltas_.back() <= 0.5 + kTolerance, "passive ECN delta must not cross through mid (d <= 0.5)");
    require(quote_size_ > 0.0, "passive ECN quote size must be positive");
    require(maker_fee_ >= 0.0, "passive ECN maker fee must be nonnegative");
    for (std::size_t i = 1; i < deltas_.size(); ++i) {
        require(deltas_[i] > deltas_[i - 1] + kTolerance,
                "passive ECN deltas must be strictly increasing");
    }
}

bool PassiveECN::side_allowed(double inventory, Side side) const noexcept {
    return (inventory > kTolerance && side == Side::Ask) ||
           (inventory < -kTolerance && side == Side::Bid);
}

bool PassiveECN::risk_reducing(double inventory, Side side, double size) const noexcept {
    if (size <= 0.0 || !side_allowed(inventory, side)) return false;
    const double change = direction(side) * size;
    const double after = inventory + change;
    return std::abs(after) <= std::abs(inventory) + kTolerance &&
           inventory * after >= -kTolerance;
}

// ---------------------------------------------------------------------------
// Problem
// ---------------------------------------------------------------------------

double PricingProblem::max_inventory_jump(const std::vector<Tier>& tiers,
                                          const std::optional<DarkPool>& dark_pool,
                                          const std::optional<PassiveECN>& passive_ecn) {
    double out = 0.0;
    for (const auto& tier : tiers) {
        out = std::max(out, tier.sizes().back());
    }
    if (dark_pool && !dark_pool->posted_sizes().empty()) {
        out = std::max(out, dark_pool->posted_sizes().back());
    }
    if (passive_ecn) out = std::max(out, passive_ecn->quote_size());
    return out;
}

PricingProblem::PricingProblem(std::vector<double> operational_inventory_grid,
                               double spread,
                               double spot_drift,
                               QuadraticPenalty penalty,
                               InternalizationTime internalization_time,
                               std::vector<Tier> tiers,
                               std::optional<DarkPool> dark_pool,
                               std::optional<PassiveECN> passive_ecn)
    : grid_(std::move(operational_inventory_grid), max_inventory_jump(tiers, dark_pool, passive_ecn)),
      spread_(spread), spot_drift_(spot_drift), penalty_(std::move(penalty)),
      internalization_time_(std::move(internalization_time)), tiers_(std::move(tiers)),
      dark_pool_(std::move(dark_pool)), passive_ecn_(std::move(passive_ecn)) {
    require(spread_ > 0.0, "spread must be positive");
    require(!tiers_.empty() || dark_pool_.has_value() || passive_ecn_.has_value(),
            "problem must contain at least one venue");
}

// ---------------------------------------------------------------------------
// Policy construction
// ---------------------------------------------------------------------------

std::vector<double> PolicyBuilder::gaps(const std::vector<double>& row) const {
    std::vector<double> out;
    if (row.size() < 2) {
        return out;
    }
    out.reserve(row.size() - 1);
    for (std::size_t j = 1; j < row.size(); ++j) {
        out.push_back(row[j - 1] - row[j]);
    }
    return out;
}

std::pair<double, double> PolicyBuilder::rung_bounds(const Tier& tier,
                                                     std::size_t rung,
                                                     const std::vector<double>& current_row,
                                                     const LadderBounds& bounds) const {
    const std::size_t nz = tier.sizes().size();
    auto check_gaps = [nz](const std::vector<double>& x, const char* name) {
        if (!x.empty() && x.size() != nz - 1) {
            throw std::invalid_argument(std::string(name) + " has wrong size");
        }
    };
    auto check_rungs = [nz](const std::vector<double>& x, const char* name) {
        if (!x.empty() && x.size() != nz) {
            throw std::invalid_argument(std::string(name) + " has wrong size");
        }
    };
    check_gaps(bounds.min_gaps, "min_gaps");
    check_gaps(bounds.max_gaps, "max_gaps");
    check_rungs(bounds.lower, "lower bounds");
    check_rungs(bounds.upper, "upper bounds");

    double lower = tier.delta_min();
    if (!bounds.min_gaps.empty()) {
        lower += std::accumulate(bounds.min_gaps.begin() + static_cast<std::ptrdiff_t>(rung),
                                 bounds.min_gaps.end(), 0.0);
    }
    double upper = tier.delta_max();
    if (!bounds.lower.empty()) {
        lower = std::max(lower, bounds.lower[rung]);
    }
    if (!bounds.upper.empty()) {
        upper = std::min(upper, bounds.upper[rung]);
    }

    if (rung > 0) {
        const double previous = current_row[rung - 1];
        const double min_gap = bounds.min_gaps.empty() ? 0.0 : bounds.min_gaps[rung - 1];
        upper = std::min(upper, previous - min_gap);
        if (!bounds.max_gaps.empty()) {
            lower = std::max(lower, previous - bounds.max_gaps[rung - 1]);
        }
    }

    if (upper < lower) {
        if (lower - upper > 1e-12) {
            throw std::runtime_error("infeasible trader ladder constraints for tier '" + tier.name() + "'");
        }
        upper = lower;
    }
    return {lower, upper};
}

std::vector<double> PolicyBuilder::build_ladder(const Tier& tier,
                                                const std::vector<double>& value,
                                                double inventory,
                                                Side side,
                                                const LadderBounds& bounds) const {
    const auto& grid = problem_.grid();
    const double hq = grid.interpolate(value, inventory);
    std::vector<double> row;
    row.reserve(tier.sizes().size());

    for (std::size_t j = 0; j < tier.sizes().size(); ++j) {
        const double z = tier.sizes()[j];
        const auto [lower, upper] = rung_bounds(tier, j, row, bounds);
        const double dq = direction(side) * z;
        if (!grid.admissible(inventory, dq)) {
            row.push_back(tier.delta_min());
            continue;
        }

        const double q_next = inventory + dq;
        const double markout = tier.use_markout()
            ? tier.markout().expected(z, problem_.internalization_time().value(q_next))
            : 0.0;
        const double additive = -z * (markout + tier.fee()) + grid.interpolate(value, q_next) - hq;
        const double unconstrained = tier.flow().optimal_delta(
            z, problem_.spread(), additive, lower, upper);
        row.push_back(std::clamp(unconstrained, lower, upper));
    }
    return row;
}

std::vector<double> PolicyBuilder::project_ladder(const Tier& tier,
                                                  const std::vector<double>& row,
                                                  const LadderBounds& bounds) const {
    std::vector<double> out;
    out.reserve(row.size());
    for (std::size_t j = 0; j < row.size(); ++j) {
        const auto [lower, upper] = rung_bounds(tier, j, out, bounds);
        out.push_back(std::clamp(row[j], lower, upper));
    }
    return out;
}

double PolicyBuilder::ladder_hamiltonian(const Tier& tier,
                                          const std::vector<double>& value,
                                          double inventory, Side side,
                                          const std::vector<double>& row_values) const {
    const auto& grid = problem_.grid();
    const double dir = direction(side);
    const double hq = grid.interpolate(value, inventory);
    double total = 0.0;
    for (std::size_t j = 0; j < tier.sizes().size(); ++j) {
        const double z = tier.sizes()[j];
        if (!grid.admissible(inventory, dir * z)) continue;
        const double q_next = inventory + dir * z;
        const double markout = tier.use_markout()
            ? tier.markout().expected(z, problem_.internalization_time().value(q_next))
            : 0.0;
        const double delta = row_values[j];
        const double rate = tier.flow().arrival_rate(delta, z);
        total += rate * (z * (problem_.spread() * (0.5 - delta) - tier.fee())
                         - z * markout
                         + grid.interpolate(value, q_next) - hq);
    }
    return total;
}

std::vector<double> PolicyBuilder::choose_howard_safe(
        const Tier& tier, const std::vector<double>& value,
        double inventory, Side side,
        const std::vector<double>& current,
        const std::vector<double>& candidate,
        const LadderBounds& bounds) const {
    const auto baseline = project_ladder(tier, current, bounds);
    return ladder_hamiltonian(tier, value, inventory, side, candidate)
            >= ladder_hamiltonian(tier, value, inventory, side, baseline)
        ? candidate : baseline;
}

Policy PolicyBuilder::initial_policy() const {
    const auto& q_grid = problem_.grid().states();
    const std::size_t q0 = problem_.grid().zero_index();
    const std::vector<double> zero(q_grid.size(), 0.0);

    Policy policy;
    policy.tiers.reserve(problem_.tiers().size());
    for (const auto& tier : problem_.tiers()) {
        LadderPolicy p;
        p.q_grid = q_grid;
        p.sizes = tier.sizes();
        p.bid.assign(q_grid.size(), std::vector<double>(tier.sizes().size(), 0.0));
        p.ask.assign(q_grid.size(), std::vector<double>(tier.sizes().size(), 0.0));

        const auto zero_bid = build_ladder(tier, zero, 0.0, Side::Bid);
        const auto zero_ask = build_ladder(tier, zero, 0.0, Side::Ask);

        auto defensive = [&tier](const std::vector<double>& row) {
            std::vector<double> out = row;
            const double shift = std::max(0.0, row.back() - tier.delta_min());
            for (double& x : out) {
                x = std::max(tier.delta_min(), x - shift);
            }
            return out;
        };
        const auto defensive_bid = defensive(zero_bid);
        const auto defensive_ask = defensive(zero_ask);

        for (std::size_t i = 0; i < q_grid.size(); ++i) {
            if (i > q0) {
                p.bid[i] = defensive_bid;
                p.ask[i] = zero_ask;
            } else if (i < q0) {
                p.bid[i] = zero_bid;
                p.ask[i] = defensive_ask;
            } else {
                p.bid[i] = zero_bid;
                p.ask[i] = zero_ask;
            }
        }
        policy.tiers.push_back(std::move(p));
    }

    if (problem_.dark_pool()) {
        DarkPoolPolicy p;
        p.q_grid = q_grid;
        p.bid_size.assign(q_grid.size(), 0.0);
        p.ask_size.assign(q_grid.size(), 0.0);
        p.bid_active.assign(q_grid.size(), false);
        p.ask_active.assign(q_grid.size(), false);
        const auto& venue = *problem_.dark_pool();
        if (!venue.posted_sizes().empty()) {
            const double u = venue.posted_sizes().back();
            for (std::size_t i = 0; i < q_grid.size(); ++i) {
                if (q_grid[i] < 0.0 && venue.risk_reducing(q_grid[i], Side::Bid, u) &&
                    problem_.grid().admissible(q_grid[i], u)) {
                    p.bid_size[i] = u;
                    p.bid_active[i] = true;
                }
                if (q_grid[i] > 0.0 && venue.risk_reducing(q_grid[i], Side::Ask, u) &&
                    problem_.grid().admissible(q_grid[i], -u)) {
                    p.ask_size[i] = u;
                    p.ask_active[i] = true;
                }
            }
        }
        policy.dark_pool = std::move(p);
    }
    if (problem_.passive_ecn()) {
        PassiveECNPolicy p;
        p.q_grid = q_grid;
        p.bid_delta.assign(q_grid.size(), 0.0);
        p.ask_delta.assign(q_grid.size(), 0.0);
        p.bid_active.assign(q_grid.size(), false);
        p.ask_active.assign(q_grid.size(), false);
        policy.passive_ecn = std::move(p);
    }
    return policy;
}

DarkPoolPolicy PolicyBuilder::improve_dark_pool(const std::vector<double>& value) const {
    const auto& venue = *problem_.dark_pool();
    const auto& q_grid = problem_.grid().states();
    DarkPoolPolicy policy;
    policy.q_grid = q_grid;
    policy.bid_size.assign(q_grid.size(), 0.0);
    policy.ask_size.assign(q_grid.size(), 0.0);
    policy.bid_active.assign(q_grid.size(), false);
    policy.ask_active.assign(q_grid.size(), false);

    for (Side side : {Side::Bid, Side::Ask}) {
        const double dir = direction(side);
        const auto& arrivals = venue.arrivals(side);
        const double fee = venue.fee(side);
        for (std::size_t i = 0; i < q_grid.size(); ++i) {
            const double q = q_grid[i];
            if (!venue.side_allowed(q, side)) {
                continue;
            }
            const double hq = problem_.grid().interpolate(value, q);
            double best_size = 0.0;
            double best_value = -std::numeric_limits<double>::infinity();
            for (double u_raw : venue.posted_sizes()) {
                const int u = static_cast<int>(std::llround(u_raw));
                if (!venue.risk_reducing(q, side, u_raw) ||
                    !problem_.grid().admissible(q, dir * u_raw)) {
                    continue;
                }
                double candidate = 0.0;
                for (const auto& [k, rate] : arrivals.fill_rates(u)) {
                    candidate += rate * (problem_.grid().interpolate(value, q + dir * k) - hq - fee * k);
                }
                if (candidate > best_value) {
                    best_value = candidate;
                    best_size = u_raw;
                }
            }
            const bool active = best_value > venue.min_fill_value();
            if (side == Side::Bid) {
                policy.bid_size[i] = active ? best_size : 0.0;
                policy.bid_active[i] = active;
            } else {
                policy.ask_size[i] = active ? best_size : 0.0;
                policy.ask_active[i] = active;
            }
        }
    }
    return policy;
}

PassiveECNPolicy PolicyBuilder::improve_passive_ecn(const std::vector<double>& value) const {
    const auto& venue = *problem_.passive_ecn();
    const auto& q_grid = problem_.grid().states();
    PassiveECNPolicy policy;
    policy.q_grid = q_grid;
    policy.bid_delta.assign(q_grid.size(), 0.0);
    policy.ask_delta.assign(q_grid.size(), 0.0);
    policy.bid_active.assign(q_grid.size(), false);
    policy.ask_active.assign(q_grid.size(), false);
    const double z = venue.quote_size();
    for (Side side : {Side::Bid, Side::Ask}) {
        const double dir = direction(side);
        for (std::size_t i = 0; i < q_grid.size(); ++i) {
            const double q = q_grid[i];
            if (!venue.risk_reducing(q, side, z) || !problem_.grid().admissible(q, dir * z)) continue;
            const double hq = problem_.grid().interpolate(value, q);
            const auto breaks = fill_breakpoints(q_grid, i, dir, z);
            double best_value = 0.0;  // OFF is always available.
            double best_delta = 0.0;
            bool active = false;
            for (std::size_t j = 0; j < venue.deltas().size(); ++j) {
                const double delta = venue.deltas()[j];
                double candidate = 0.0;
                for (const auto& source : venue.flow().sources()) {
                    const double hit_rate = source.arrival_rate(delta);
                    if (hit_rate <= 0.0) continue;
                    for (const auto& [fill, probability] : source.fill_components(z, breaks)) {
                        if (probability <= 0.0) continue;
                        const double edge = fill * (problem_.spread() * (0.5 - delta) - venue.maker_fee());
                        candidate += hit_rate * probability *
                            (edge + problem_.grid().interpolate(value, q + dir * fill) - hq);
                    }
                }
                if (candidate > best_value + kTolerance) {
                    best_value = candidate;
                    best_delta = delta;
                    active = true;
                }
            }
            if (side == Side::Bid) { policy.bid_active[i] = active; policy.bid_delta[i] = best_delta; }
            else { policy.ask_active[i] = active; policy.ask_delta[i] = best_delta; }
        }
    }
    return policy;
}

Policy PolicyBuilder::improve(const std::vector<double>& value, const Policy& previous) const {
    require(previous.tiers.size() == problem_.tiers().size(), "policy tier count mismatch");
    const auto& q_grid = problem_.grid().states();
    const std::size_t q0 = problem_.grid().zero_index();
    const double q_operational = problem_.grid().operational_limit();

    Policy improved = previous;
    for (std::size_t k = 0; k < problem_.tiers().size(); ++k) {
        const auto& tier = problem_.tiers()[k];
        const auto& old = previous.tiers[k];
        auto& out = improved.tiers[k];

        for (Side side : {Side::Bid, Side::Ask}) {
            {
                const auto candidate = build_ladder(tier, value, q_grid[q0], side);
                matrix(out, side, q0) = choose_howard_safe(
                    tier, value, q_grid[q0], side, matrix(old, side, q0), candidate);
            }

            for (std::size_t i = q0 + 1; i < q_grid.size(); ++i) {
                if (q_grid[i] > q_operational + kTolerance) {
                    const auto candidate = build_ladder(tier, value, q_grid[i], side);
                    matrix(out, side, i) = choose_howard_safe(
                        tier, value, q_grid[i], side, matrix(old, side, i), candidate);
                    continue;
                }
                const auto& inner = matrix(old, side, i - 1);
                LadderBounds bounds;
                if (side == Side::Bid) {
                    bounds.min_gaps = gaps(inner);
                    bounds.upper = inner;
                } else {
                    bounds.max_gaps = gaps(inner);
                    bounds.lower = inner;
                }
                {
                    const auto candidate = build_ladder(tier, value, q_grid[i], side, bounds);
                    matrix(out, side, i) = choose_howard_safe(
                        tier, value, q_grid[i], side, matrix(old, side, i), candidate, bounds);
                }
            }

            for (std::size_t offset = 1; offset <= q0; ++offset) {
                const std::size_t i = q0 - offset;
                if (q_grid[i] < -q_operational - kTolerance) {
                    const auto candidate = build_ladder(tier, value, q_grid[i], side);
                    matrix(out, side, i) = choose_howard_safe(
                        tier, value, q_grid[i], side, matrix(old, side, i), candidate);
                    continue;
                }
                const auto& inner = matrix(old, side, i + 1);
                LadderBounds bounds;
                if (side == Side::Bid) {
                    bounds.max_gaps = gaps(inner);
                    bounds.lower = inner;
                } else {
                    bounds.min_gaps = gaps(inner);
                    bounds.upper = inner;
                }
                {
                    const auto candidate = build_ladder(tier, value, q_grid[i], side, bounds);
                    matrix(out, side, i) = choose_howard_safe(
                        tier, value, q_grid[i], side, matrix(old, side, i), candidate, bounds);
                }
            }
        }
    }

    if (problem_.dark_pool()) {
        improved.dark_pool = improve_dark_pool(value);
    }
    if (problem_.passive_ecn()) {
        improved.passive_ecn = improve_passive_ecn(value);
    }
    return improved;
}

Policy PolicyBuilder::snap_operational_constraints(const Policy& policy) const {
    Policy out = policy;
    const auto& q_grid = problem_.grid().states();
    const std::size_t q0 = problem_.grid().zero_index();
    const double q_operational = problem_.grid().operational_limit();

    for (std::size_t k = 0; k < problem_.tiers().size(); ++k) {
        const auto& tier = problem_.tiers()[k];
        for (std::size_t i = q0 + 1; i < q_grid.size() && q_grid[i] <= q_operational + kTolerance; ++i) {
            const auto inner_bid = out.tiers[k].bid[i - 1];
            const auto inner_ask = out.tiers[k].ask[i - 1];
            LadderBounds bid_bounds;
            bid_bounds.min_gaps = gaps(inner_bid);
            bid_bounds.upper = inner_bid;
            out.tiers[k].bid[i] = project_ladder(tier, out.tiers[k].bid[i], bid_bounds);
            LadderBounds ask_bounds;
            ask_bounds.max_gaps = gaps(inner_ask);
            ask_bounds.lower = inner_ask;
            out.tiers[k].ask[i] = project_ladder(tier, out.tiers[k].ask[i], ask_bounds);
        }
        for (std::size_t offset = 1; offset <= q0; ++offset) {
            const std::size_t i = q0 - offset;
            if (q_grid[i] < -q_operational - kTolerance) {
                continue;
            }
            const auto inner_bid = out.tiers[k].bid[i + 1];
            const auto inner_ask = out.tiers[k].ask[i + 1];
            LadderBounds bid_bounds;
            bid_bounds.max_gaps = gaps(inner_bid);
            bid_bounds.lower = inner_bid;
            out.tiers[k].bid[i] = project_ladder(tier, out.tiers[k].bid[i], bid_bounds);
            LadderBounds ask_bounds;
            ask_bounds.min_gaps = gaps(inner_ask);
            ask_bounds.upper = inner_ask;
            out.tiers[k].ask[i] = project_ladder(tier, out.tiers[k].ask[i], ask_bounds);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Bellman operator
// ---------------------------------------------------------------------------

std::vector<double> BellmanModel::rhs(const std::vector<double>& value, const Policy& policy) const {
    const auto& grid = problem_.grid();
    const auto& q_grid = grid.states();
    require(value.size() == q_grid.size(), "value vector does not match solve inventory grid");
    require(policy.tiers.size() == problem_.tiers().size(), "policy tier count mismatch");

    std::vector<double> out(q_grid.size(), 0.0);
    for (std::size_t i = 0; i < q_grid.size(); ++i) {
        const double q = q_grid[i];
        const double hq = grid.interpolate(value, q);
        double total = -problem_.penalty().value(q) + problem_.spot_drift() * q;

        for (std::size_t k = 0; k < problem_.tiers().size(); ++k) {
            const auto& tier = problem_.tiers()[k];
            const auto& p = policy.tiers[k];
            for (Side side : {Side::Bid, Side::Ask}) {
                const auto& row = matrix(p, side, i);
                for (std::size_t j = 0; j < tier.sizes().size(); ++j) {
                    const double z = tier.sizes()[j];
                    const double dq = direction(side) * z;
                    if (!grid.admissible(q, dq)) {
                        continue;
                    }
                    const double delta = row[j];
                    const double rate = tier.flow().arrival_rate(delta, z);
                    const double q_next = q + dq;
                    const double markout = tier.use_markout()
                        ? tier.markout().expected(z, problem_.internalization_time().value(q_next))
                        : 0.0;
                    const double fill_value = z * (problem_.spread() * (0.5 - delta) - tier.fee())
                                            - z * markout
                                            + grid.interpolate(value, q_next) - hq;
                    total += rate * fill_value;
                }
            }
        }

        if (problem_.dark_pool() && policy.dark_pool) {
            const auto& venue = *problem_.dark_pool();
            const auto& p = *policy.dark_pool;
            for (Side side : {Side::Bid, Side::Ask}) {
                const bool active = side == Side::Bid ? p.bid_active[i] : p.ask_active[i];
                const double posted = side == Side::Bid ? p.bid_size[i] : p.ask_size[i];
                if (!active || posted <= 0.0) {
                    continue;
                }
                const int u = static_cast<int>(std::llround(posted));
                const double dir = direction(side);
                for (const auto& [fill, rate] : venue.arrivals(side).fill_rates(u)) {
                    total += rate * (grid.interpolate(value, q + dir * fill) - hq - venue.fee(side) * fill);
                }
            }
        }
        if (problem_.passive_ecn() && policy.passive_ecn) {
            const auto& venue = *problem_.passive_ecn();
            const auto& p = *policy.passive_ecn;
            const double z = venue.quote_size();
            for (Side side : {Side::Bid, Side::Ask}) {
                const bool active = side == Side::Bid ? p.bid_active[i] : p.ask_active[i];
                const double delta = side == Side::Bid ? p.bid_delta[i] : p.ask_delta[i];
                if (!active) continue;
                const double dir = direction(side);
                const auto breaks = fill_breakpoints(q_grid, i, dir, z);
                for (const auto& source : venue.flow().sources()) {
                    const double hit_rate = source.arrival_rate(delta);
                    if (hit_rate <= 0.0) continue;
                    for (const auto& [fill, probability] : source.fill_components(z, breaks)) {
                        const double rate = hit_rate * probability;
                        if (rate <= 0.0) continue;
                        const double edge = fill *
                            (problem_.spread() * (0.5 - delta) - venue.maker_fee());
                        total += rate *
                            (edge + grid.interpolate(value, q + dir * fill) - hq);
                    }
                }
            }
        }
        out[i] = total;
    }
    return out;
}

}  // namespace ladder_pricer
