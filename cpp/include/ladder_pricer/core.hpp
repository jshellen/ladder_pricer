#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ladder_pricer {

enum class Side { Bid, Ask };

class InventoryGrid {
public:
    InventoryGrid(std::vector<double> operational_states, double max_jump);

    const std::vector<double>& operational_states() const noexcept { return operational_; }
    const std::vector<double>& states() const noexcept { return states_; }
    const std::vector<std::size_t>& operational_indices() const noexcept { return operational_indices_; }
    std::size_t zero_index() const noexcept { return zero_index_; }
    double hard_limit() const noexcept { return hard_limit_; }
    double operational_limit() const noexcept { return operational_limit_; }

    bool admissible(double q, double inventory_change) const noexcept;
    double interpolate(const std::vector<double>& values, double q) const;

private:
    std::vector<double> operational_;
    std::vector<double> states_;
    std::vector<std::size_t> operational_indices_;
    std::size_t zero_index_ = 0;
    double hard_limit_ = 0.0;
    double operational_limit_ = 0.0;
};

class LogisticFlow {
public:
    LogisticFlow(double a0, double theta, double beta, double shift,
                 double steepness, double volume_shift);

    double scale(double size) const;
    double center(double size) const;
    double hit_ratio(double delta, double size) const;
    double arrival_rate(double delta, double size) const;
    double optimal_delta(double size, double spread, double additive_value) const;

    double a0() const noexcept { return a0_; }
    double theta() const noexcept { return theta_; }
    double beta() const noexcept { return beta_; }
    double shift() const noexcept { return shift_; }
    double steepness() const noexcept { return steepness_; }
    double volume_shift() const noexcept { return volume_shift_; }

private:
    static double lambert_w_exp(double log_argument);

    double a0_;
    double theta_;
    double beta_;
    double shift_;
    double steepness_;
    double volume_shift_;
};

class SaturatingMarkout {
public:
    SaturatingMarkout(double impact_scale, double size_exponent, double tau_minutes);

    double asymptotic(double size) const;
    double expected(double size, double minutes) const;

    double impact_scale() const noexcept { return impact_scale_; }
    double size_exponent() const noexcept { return size_exponent_; }
    double tau_minutes() const noexcept { return tau_minutes_; }

private:
    double impact_scale_;
    double size_exponent_;
    double tau_minutes_;
};

class InternalizationTime {
public:
    InternalizationTime(double tau0, double tau1, double tau2)
        : tau0_(tau0), tau1_(tau1), tau2_(tau2) {}

    double value(double inventory) const noexcept;
    double tau0() const noexcept { return tau0_; }
    double tau1() const noexcept { return tau1_; }
    double tau2() const noexcept { return tau2_; }

private:
    double tau0_;
    double tau1_;
    double tau2_;
};

class QuadraticPenalty {
public:
    QuadraticPenalty(double risk_aversion, double sigma);

    double value(double inventory) const noexcept;
    double risk_aversion() const noexcept { return risk_aversion_; }
    double sigma() const noexcept { return sigma_; }

private:
    double risk_aversion_;
    double sigma_;
};

class Tier {
public:
    Tier(std::string name, std::vector<double> sizes, LogisticFlow flow,
         SaturatingMarkout markout, bool use_markout,
         double delta_min, double delta_max);

    const std::string& name() const noexcept { return name_; }
    const std::vector<double>& sizes() const noexcept { return sizes_; }
    const LogisticFlow& flow() const noexcept { return flow_; }
    const SaturatingMarkout& markout() const noexcept { return markout_; }
    bool use_markout() const noexcept { return use_markout_; }
    double delta_min() const noexcept { return delta_min_; }
    double delta_max() const noexcept { return delta_max_; }

private:
    std::string name_;
    std::vector<double> sizes_;
    LogisticFlow flow_;
    SaturatingMarkout markout_;
    bool use_markout_;
    double delta_min_;
    double delta_max_;
};

class ArrivalDistribution {
public:
    virtual ~ArrivalDistribution() = default;
    virtual std::vector<std::pair<int, double>> fill_rates(int posted_size) const = 0;
    virtual double arrival_intensity() const noexcept = 0;
    virtual std::string name() const = 0;
};

class GeometricArrival final : public ArrivalDistribution {
public:
    GeometricArrival(double intensity, double probability);

    std::vector<std::pair<int, double>> fill_rates(int posted_size) const override;
    double arrival_intensity() const noexcept override { return intensity_; }
    std::string name() const override { return "geometric"; }

    double probability() const noexcept { return probability_; }

private:
    double intensity_;
    double probability_;
};

class ZeroInflatedPoissonArrival final : public ArrivalDistribution {
public:
    ZeroInflatedPoissonArrival(double intensity, double mean, double zero_probability);

    std::vector<std::pair<int, double>> fill_rates(int posted_size) const override;
    double arrival_intensity() const noexcept override { return intensity_; }
    std::string name() const override { return "zero_inflated_poisson"; }

    double mean() const noexcept { return mean_; }
    double zero_probability() const noexcept { return zero_probability_; }

private:
    double intensity_;
    double mean_;
    double zero_probability_;
};

class DarkPool {
public:
    DarkPool(std::shared_ptr<ArrivalDistribution> bid_arrivals,
             std::shared_ptr<ArrivalDistribution> ask_arrivals,
             double bid_fee, double ask_fee,
             std::vector<double> posted_sizes,
             bool allow_both_sides = false,
             double min_fill_value = 0.0);

    const ArrivalDistribution& arrivals(Side side) const;
    double fee(Side side) const noexcept;
    const std::vector<double>& posted_sizes() const noexcept { return posted_sizes_; }
    bool allow_both_sides() const noexcept { return allow_both_sides_; }
    double min_fill_value() const noexcept { return min_fill_value_; }
    bool side_allowed(double inventory, Side side) const noexcept;

private:
    std::shared_ptr<ArrivalDistribution> bid_arrivals_;
    std::shared_ptr<ArrivalDistribution> ask_arrivals_;
    double bid_fee_;
    double ask_fee_;
    std::vector<double> posted_sizes_;
    bool allow_both_sides_;
    double min_fill_value_;
};

struct LadderPolicy {
    std::vector<double> q_grid;
    std::vector<double> sizes;
    std::vector<std::vector<double>> bid;
    std::vector<std::vector<double>> ask;
};

struct DarkPoolPolicy {
    std::vector<double> q_grid;
    std::vector<double> bid_size;
    std::vector<double> ask_size;
    std::vector<bool> bid_active;
    std::vector<bool> ask_active;
};

struct Policy {
    std::vector<LadderPolicy> tiers;
    std::optional<DarkPoolPolicy> dark_pool;
};

class PricingProblem {
public:
    PricingProblem(std::vector<double> operational_inventory_grid,
                   double spread,
                   double spot_drift,
                   QuadraticPenalty penalty,
                   InternalizationTime internalization_time,
                   std::vector<Tier> tiers,
                   std::optional<DarkPool> dark_pool = std::nullopt);

    const InventoryGrid& grid() const noexcept { return grid_; }
    double spread() const noexcept { return spread_; }
    double spot_drift() const noexcept { return spot_drift_; }
    const QuadraticPenalty& penalty() const noexcept { return penalty_; }
    const InternalizationTime& internalization_time() const noexcept { return internalization_time_; }
    const std::vector<Tier>& tiers() const noexcept { return tiers_; }
    const std::optional<DarkPool>& dark_pool() const noexcept { return dark_pool_; }


private:
    static double max_inventory_jump(const std::vector<Tier>& tiers,
                                     const std::optional<DarkPool>& dark_pool);

    InventoryGrid grid_;
    double spread_;
    double spot_drift_;
    QuadraticPenalty penalty_;
    InternalizationTime internalization_time_;
    std::vector<Tier> tiers_;
    std::optional<DarkPool> dark_pool_;
};

struct SolverDiagnostics {
    bool converged = false;
    int iterations = 0;
    double average_reward = 0.0;
    double reward_lower_bound = 0.0;
    double reward_upper_bound = 0.0;
    std::vector<double> value_change;
    std::vector<double> bellman_residual;
};

struct Solution {
    std::vector<double> q_grid;
    std::vector<double> value;
    std::vector<double> solve_q_grid;
    std::vector<double> value_solve;
    // Operational policies are convenient for pricing/UI use.
    std::vector<LadderPolicy> tier_policies;
    std::optional<DarkPoolPolicy> dark_pool_policy;

    // Full hidden-grid policies are retained for simulation and diagnostics.
    std::vector<LadderPolicy> solve_tier_policies;
    std::optional<DarkPoolPolicy> solve_dark_pool_policy;
    double average_reward = 0.0;
    double hard_inventory_limit = 0.0;
    SolverDiagnostics diagnostics;
};

class BellmanModel {
public:
    explicit BellmanModel(const PricingProblem& problem) : problem_(problem) {}

    std::vector<double> rhs(const std::vector<double>& value, const Policy& policy) const;

private:
    const PricingProblem& problem_;
};

class PolicyBuilder {
public:
    explicit PolicyBuilder(const PricingProblem& problem) : problem_(problem) {}

    Policy initial_policy() const;
    Policy improve(const std::vector<double>& value, const Policy& previous) const;
    Policy snap_operational_constraints(const Policy& policy) const;

private:
    struct LadderBounds {
        std::vector<double> min_gaps;
        std::vector<double> max_gaps;
        std::vector<double> lower;
        std::vector<double> upper;
    };

    std::vector<double> build_ladder(const Tier& tier,
                                     const std::vector<double>& value,
                                     double inventory,
                                     Side side,
                                     const LadderBounds& bounds = {}) const;
    std::vector<double> project_ladder(const Tier& tier,
                                       const std::vector<double>& row,
                                       const LadderBounds& bounds) const;
    double ladder_hamiltonian(const Tier& tier, const std::vector<double>& value,
                              double inventory, Side side,
                              const std::vector<double>& row) const;
    std::vector<double> choose_howard_safe(const Tier& tier,
                                           const std::vector<double>& value,
                                           double inventory, Side side,
                                           const std::vector<double>& current,
                                           const std::vector<double>& candidate,
                                           const LadderBounds& bounds = {}) const;
    std::pair<double, double> rung_bounds(const Tier& tier,
                                         std::size_t rung,
                                         const std::vector<double>& current_row,
                                         const LadderBounds& bounds) const;
    std::vector<double> gaps(const std::vector<double>& row) const;
    DarkPoolPolicy improve_dark_pool(const std::vector<double>& value) const;

    const PricingProblem& problem_;
};


}  // namespace ladder_pricer
