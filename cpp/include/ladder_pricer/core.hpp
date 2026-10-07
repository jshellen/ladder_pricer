#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <random>
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

    // A0 is the scale of the calibrated exogenous RFQ-arrival intensity
    // curve for one customer side.  For source-currency size z,
    // lambda_RFQ(z) = A0 * z^(-theta-beta*z).  FlowSource uses this same curve
    // both as the HJB size-specific exogenous intensity and, after
    // normalization across configured RFQ rungs, as the Monte Carlo size PMF.
    double size_weight(double size) const;
    double rfq_arrival_rate(double size) const;
    double center(double size) const;
    // Probability that our quote wins a customer RFQ.
    double hit_ratio(double delta, double size) const;
    double win_probability(double delta, double size) const { return hit_ratio(delta, size); }
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


class FlowSource {
public:
    FlowSource(std::string name, LogisticFlow flow,
               double delta_scale = 1.0, double source_size_per_target = 1.0);

    const std::string& name() const noexcept { return name_; }
    const LogisticFlow& flow() const noexcept { return flow_; }
    double delta_scale() const noexcept { return delta_scale_; }
    double source_size_per_target() const noexcept { return source_size_per_target_; }

    double implied_delta(double target_delta) const noexcept;
    double source_size(double target_size) const noexcept;
    void configure_target_sizes(const std::vector<double>& target_sizes);
    const std::vector<double>& target_sizes() const noexcept { return target_sizes_; }
    const std::vector<double>& size_probabilities() const noexcept { return size_probabilities_; }
    double total_rfq_rate() const noexcept { return total_rfq_rate_; }
    std::size_t sample_target_size_index(std::mt19937_64& rng) const;
    double rfq_arrival_rate(double target_size) const;
    double win_probability(double target_delta, double target_size) const;
    double arrival_rate(double target_delta, double target_size) const;
    double target_center(double target_size) const;
    double effective_steepness() const noexcept;

private:
    std::string name_;
    LogisticFlow flow_;
    double delta_scale_;
    double source_size_per_target_;
    std::vector<double> target_sizes_;
    std::vector<double> size_probabilities_;
    double total_rfq_rate_ = 0.0;
};

class AggregatedFlow {
public:
    explicit AggregatedFlow(LogisticFlow direct_flow);
    explicit AggregatedFlow(std::vector<FlowSource> sources);

    const std::vector<FlowSource>& sources() const noexcept { return sources_; }
    void configure_target_sizes(const std::vector<double>& target_sizes);
    double rfq_arrival_rate(double target_size) const;
    double win_probability(double target_delta, double target_size) const;
    double arrival_rate(double target_delta, double target_size) const;
    double optimal_delta(double target_size, double spread, double additive_value,
                         double lower, double upper) const;

private:
    std::vector<FlowSource> sources_;
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
         double delta_min, double delta_max, double fee = 0.0,
         double rfq_size_step = 1.0);
    Tier(std::string name, std::vector<double> sizes, AggregatedFlow flow,
         SaturatingMarkout markout, bool use_markout,
         double delta_min, double delta_max, double fee = 0.0,
         double rfq_size_step = 1.0);

    const std::string& name() const noexcept { return name_; }
    const std::vector<double>& sizes() const noexcept { return sizes_; }
    const std::vector<double>& rfq_sizes() const noexcept { return rfq_sizes_; }
    double rfq_size_step() const noexcept { return rfq_size_step_; }
    const AggregatedFlow& flow() const noexcept { return flow_; }
    const SaturatingMarkout& markout() const noexcept { return markout_; }
    bool use_markout() const noexcept { return use_markout_; }
    double delta_min() const noexcept { return delta_min_; }
    double delta_max() const noexcept { return delta_max_; }
    double fee() const noexcept { return fee_; }

private:
    std::string name_;
    std::vector<double> sizes_;
    std::vector<double> rfq_sizes_;
    double rfq_size_step_ = 1.0;
    AggregatedFlow flow_;
    SaturatingMarkout markout_;
    bool use_markout_;
    double delta_min_;
    double delta_max_;
    double fee_;
};

class ArrivalDistribution {
public:
    virtual ~ArrivalDistribution() = default;
    virtual std::vector<std::pair<int, double>> fill_rates(int posted_size) const = 0;
    virtual int sample_incoming_size(std::mt19937_64& rng) const = 0;
    virtual double arrival_intensity() const noexcept = 0;
    virtual std::string name() const = 0;
};

class ZeroInflatedPoissonArrival final : public ArrivalDistribution {
public:
    ZeroInflatedPoissonArrival(double intensity, double mean, double zero_probability);

    std::vector<std::pair<int, double>> fill_rates(int posted_size) const override;
    int sample_incoming_size(std::mt19937_64& rng) const override;
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
    DarkPool(std::shared_ptr<ArrivalDistribution> arrivals,
             double fee,
             std::vector<double> posted_sizes,
             double min_fill_value = 0.0);

    const ArrivalDistribution& arrivals(Side side) const { (void)side; return *arrivals_; }
    double fee(Side side) const noexcept { (void)side; return fee_; }
    const std::vector<double>& posted_sizes() const noexcept { return posted_sizes_; }
    double min_fill_value() const noexcept { return min_fill_value_; }
    bool side_allowed(double inventory, Side side) const noexcept;
    bool risk_reducing(double inventory, Side side, double size) const noexcept;

private:
    std::shared_ptr<ArrivalDistribution> arrivals_;
    double fee_;
    std::vector<double> posted_sizes_;
    double min_fill_value_;
};


class ExponentialFlow {
public:
    ExponentialFlow(double a, double k);

    // ECN delta uses the same normalized price-improvement convention as tiers:
    // d=0 is the same-side touch and d=0.5 is mid.  If X=0.5-d is the quote
    // depth from mid measured in spread fractions, then
    // lambda_fill(d)=A exp(-k X)=A exp(-k(0.5-d)).
    double arrival_rate(double delta) const;
    double a() const noexcept { return a_; }
    double k() const noexcept { return k_; }

private:
    double a_;
    double k_;
};

class ECNFlowSource {
public:
    // ECN parent trade sizes are discrete source-base currency pillars:
    // 1.00M, 0.75M, 0.50M, 0.25M and 0.10M.
    ECNFlowSource(std::string name, ExponentialFlow flow,
                  double delta_scale = 1.0, double source_size_per_target = 1.0);
    ECNFlowSource(std::string name, ExponentialFlow flow,
                  double delta_scale, double source_size_per_target,
                  std::vector<double> trade_size_probabilities);
    // Backward-compatible constructor: discretize an exponential parent-size
    // model onto the fixed pillars using midpoint bins.
    ECNFlowSource(std::string name, ExponentialFlow flow,
                  double delta_scale, double source_size_per_target,
                  double legacy_mean_trade_size);

    const std::string& name() const noexcept { return name_; }
    const ExponentialFlow& flow() const noexcept { return flow_; }
    double delta_scale() const noexcept { return delta_scale_; }
    double source_size_per_target() const noexcept { return source_size_per_target_; }
    const std::vector<double>& trade_sizes() const noexcept { return trade_sizes_; }
    const std::vector<double>& trade_size_probabilities() const noexcept { return trade_size_probabilities_; }
    double mean_trade_size() const noexcept;
    double mean_target_trade_size() const noexcept {
        return mean_trade_size() / source_size_per_target_;
    }
    double sample_trade_size(std::mt19937_64& rng) const;

    // Map the target-pair master delta into the source-pair quote.  The same
    // affine convention is used by customer Tier flow sources:
    // d_source = 0.5 + alpha * (d_target - 0.5).
    double implied_delta(double target_delta) const noexcept;
    double arrival_rate(double target_delta) const;

    // Convert a source-pair trade reach X_source (spread fractions from mid)
    // into the equivalent target-pair reach X_target.
    double target_reach(double source_reach) const noexcept;

    // Parent ECN trade size is discrete and independent of price reach. Our
    // realized target-pair fill is the incoming source-size pillar capped by
    // the posted source-equivalent quantity and mapped back to target inventory.
    // fill_components() exposes the exact finite distribution used by the HJB.
    double full_fill_probability(double target_posted_size) const;
    double expected_fill_size(double target_posted_size) const;
    std::vector<std::pair<double, double>> fill_components(
        double target_posted_size,
        const std::vector<double>& target_breakpoints = {}) const;

private:
    std::string name_;
    ExponentialFlow flow_;
    double delta_scale_;
    double source_size_per_target_;
    std::vector<double> trade_sizes_;
    std::vector<double> trade_size_probabilities_;
};

class AggregatedECNFlow {
public:
    explicit AggregatedECNFlow(ExponentialFlow direct_flow);
    explicit AggregatedECNFlow(std::vector<ECNFlowSource> sources);

    const std::vector<ECNFlowSource>& sources() const noexcept { return sources_; }
    double arrival_rate(double target_delta) const;

private:
    std::vector<ECNFlowSource> sources_;
};

class PassiveECN {
public:
    PassiveECN(std::vector<double> deltas, ExponentialFlow flow,
               double quote_size, double maker_fee);
    PassiveECN(std::vector<double> deltas, AggregatedECNFlow flow,
               double quote_size, double maker_fee);

    const std::vector<double>& deltas() const noexcept { return deltas_; }
    const AggregatedECNFlow& flow() const noexcept { return flow_; }
    double quote_size() const noexcept { return quote_size_; }
    double maker_fee() const noexcept { return maker_fee_; }
    bool side_allowed(double inventory, Side side) const noexcept;
    bool risk_reducing(double inventory, Side side, double size) const noexcept;

private:
    std::vector<double> deltas_;
    AggregatedECNFlow flow_;
    double quote_size_;
    double maker_fee_;
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

struct PassiveECNPolicy {
    std::vector<double> q_grid;
    std::vector<double> bid_delta;
    std::vector<double> ask_delta;
    std::vector<bool> bid_active;
    std::vector<bool> ask_active;
};

struct Policy {
    std::vector<LadderPolicy> tiers;
    std::optional<DarkPoolPolicy> dark_pool;
    std::optional<PassiveECNPolicy> passive_ecn;
};

class PricingProblem {
public:
    PricingProblem(std::vector<double> operational_inventory_grid,
                   double spread,
                   double spot_drift,
                   QuadraticPenalty penalty,
                   InternalizationTime internalization_time,
                   std::vector<Tier> tiers,
                   std::optional<DarkPool> dark_pool = std::nullopt,
                   std::optional<PassiveECN> passive_ecn = std::nullopt);

    const InventoryGrid& grid() const noexcept { return grid_; }
    double spread() const noexcept { return spread_; }
    double spot_drift() const noexcept { return spot_drift_; }
    const QuadraticPenalty& penalty() const noexcept { return penalty_; }
    const InternalizationTime& internalization_time() const noexcept { return internalization_time_; }
    const std::vector<Tier>& tiers() const noexcept { return tiers_; }
    const std::optional<DarkPool>& dark_pool() const noexcept { return dark_pool_; }
    const std::optional<PassiveECN>& passive_ecn() const noexcept { return passive_ecn_; }


private:
    static double max_inventory_jump(const std::vector<Tier>& tiers,
                                     const std::optional<DarkPool>& dark_pool,
                                     const std::optional<PassiveECN>& passive_ecn);

    InventoryGrid grid_;
    double spread_;
    double spot_drift_;
    QuadraticPenalty penalty_;
    InternalizationTime internalization_time_;
    std::vector<Tier> tiers_;
    std::optional<DarkPool> dark_pool_;
    std::optional<PassiveECN> passive_ecn_;
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
    std::optional<PassiveECNPolicy> passive_ecn_policy;

    // Full hidden-grid policies are retained for simulation and diagnostics.
    std::vector<LadderPolicy> solve_tier_policies;
    std::optional<DarkPoolPolicy> solve_dark_pool_policy;
    std::optional<PassiveECNPolicy> solve_passive_ecn_policy;
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
    PassiveECNPolicy improve_passive_ecn(const std::vector<double>& value) const;

    const PricingProblem& problem_;
};


}  // namespace ladder_pricer
