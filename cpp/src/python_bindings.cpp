#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ladder_pricer/analytics.hpp"
#include "ladder_pricer/howard.hpp"
#include "ladder_pricer/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace ladder_pricer::python {
namespace {

double number(const py::dict& object, const char* key) {
    return py::cast<double>(object[py::str(key)]);
}

double optional_number(const py::dict& object, const char* key, double fallback) {
    const py::str k(key);
    return object.contains(k) ? py::cast<double>(object[k]) : fallback;
}

bool boolean(const py::dict& object, const char* key) {
    return py::cast<bool>(object[py::str(key)]);
}

std::string text(const py::dict& object, const char* key) {
    return py::cast<std::string>(object[py::str(key)]);
}

std::vector<double> numbers(const py::handle& value) {
    return py::cast<std::vector<double>>(value);
}

std::vector<double> optional_numbers(const py::dict& object, const char* key) {
    const py::str k(key);
    return object.contains(k) ? numbers(object[k]) : std::vector<double>{};
}

std::vector<double> build_grid(const py::dict& spec) {
    const std::string mode = text(spec, "mode");
    if (mode != "uniform") {
        throw std::invalid_argument("inventory grid mode must be 'uniform'");
    }

    const double qmax = number(spec, "maxAbs");
    const double step = number(spec, "step");
    if (qmax <= 0.0) throw std::invalid_argument("inventory maxAbs must be positive");
    if (std::abs(step - 1.0) > 1e-12) {
        throw std::invalid_argument("inventory grid step must be exactly 1.0");
    }

    const double rounded_qmax = std::round(qmax);
    if (std::abs(qmax - rounded_qmax) > 1e-12) {
        throw std::invalid_argument("inventory maxAbs must be an integer on the uniform 1M grid");
    }

    const int n = static_cast<int>(rounded_qmax);
    std::vector<double> states;
    states.reserve(static_cast<std::size_t>(2 * n + 1));
    for (int q = -n; q <= n; ++q) states.push_back(static_cast<double>(q));
    return states;
}



std::vector<double> rfq_size_support(const std::vector<double>& pricing_sizes, double step) {
    if (pricing_sizes.empty()) throw std::invalid_argument("tier sizes must not be empty");
    if (!(step > 0.0) || !std::isfinite(step)) {
        throw std::invalid_argument("RFQ size step must be positive");
    }
    const double lo = pricing_sizes.front();
    const double hi = pricing_sizes.back();
    std::vector<double> out;
    for (std::size_t n = 0;; ++n) {
        const double z = lo + static_cast<double>(n) * step;
        if (z > hi + 1e-10) break;
        out.push_back(std::min(z, hi));
    }
    if (out.empty() || std::abs(out.back() - hi) > 1e-10) out.push_back(hi);
    out.insert(out.end(), pricing_sizes.begin(), pricing_sizes.end());
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end(), [](double a, double b) {
        return std::abs(a - b) <= 1e-10;
    }), out.end());
    return out;
}

double a0_from_total_rfq_rate(const py::dict& flow,
                                  const std::vector<double>& target_sizes,
                                  double source_size_per_target) {
    // Compatibility with the short-lived totalRfqRate config format.  The
    // calibrated model uses lambda_side(z)=A0*z^(-theta-beta*z); therefore a
    // pooled two-sided total rate Lambda implies
    // A0 = Lambda / (2 * sum_j shape(z_j)).
    const double total_rate = number(flow, "totalRfqRate");
    const double theta = number(flow, "theta");
    const double beta = number(flow, "beta");
    double shape_mass = 0.0;
    for (double target_size : target_sizes) {
        const double source_size = target_size * source_size_per_target;
        shape_mass += std::pow(source_size, -theta - beta * source_size);
    }
    if (!(shape_mass > 0.0)) {
        throw std::invalid_argument("RFQ arrival-intensity shape must have positive mass");
    }
    return total_rate / (2.0 * shape_mass);
}

LogisticFlow build_logistic_flow(const py::dict& flow,
                                 const std::vector<double>& target_sizes,
                                 double source_size_per_target = 1.0) {
    const py::str a0_key("A0");
    const double a0 = flow.contains(a0_key)
        ? py::cast<double>(flow[a0_key])
        : a0_from_total_rfq_rate(flow, target_sizes, source_size_per_target);
    return LogisticFlow(a0, number(flow, "theta"), number(flow, "beta"),
                        number(flow, "shift"), number(flow, "steepness"), number(flow, "volumeShift"));
}

AggregatedFlow build_tier_flow(const py::dict& tier, double target_spread_pips) {
    const std::vector<double> pricing_sizes = numbers(tier[py::str("sizes")]);
    const std::vector<double> target_sizes = rfq_size_support(
        pricing_sizes, optional_number(tier, "rfqSizeStep", 1.0));
    const py::str sources_key("flowSources");
    if (!tier.contains(sources_key)) {
        const py::dict flow = py::cast<py::dict>(tier[py::str("flow")]);
        std::vector<FlowSource> direct;
        direct.emplace_back(
            "direct", build_logistic_flow(flow, target_sizes), 1.0, 1.0);
        return AggregatedFlow(std::move(direct));
    }

    std::vector<FlowSource> sources;
    const py::list source_cfgs = py::cast<py::list>(tier[sources_key]);
    sources.reserve(py::len(source_cfgs));
    for (const py::handle item : source_cfgs) {
        const py::dict source = py::cast<py::dict>(item);
        const py::dict flow = py::cast<py::dict>(source[py::str("flow")]);
        const std::string name = source.contains(py::str("name"))
            ? text(source, "name") : std::string("source");

        double delta_scale = 1.0;
        double source_size_per_target = 1.0;
        if (source.contains(py::str("mapping"))) {
            const py::dict mapping = py::cast<py::dict>(source[py::str("mapping")]);
            const std::string type = mapping.contains(py::str("type"))
                ? text(mapping, "type") : std::string("identity");
            if (type == "identity") {
                // defaults above
            } else if (type == "crossed") {
                const double cross_mid = number(mapping, "crossMid");
                const double source_spread_pips = number(mapping, "sourceSpreadPips");
                if (cross_mid <= 0.0 || source_spread_pips <= 0.0) {
                    throw std::invalid_argument("crossed flow mapping requires positive crossMid and sourceSpreadPips");
                }
                delta_scale = target_spread_pips / (cross_mid * source_spread_pips);
                source_size_per_target = cross_mid;
            } else if (type == "affine") {
                delta_scale = number(mapping, "deltaScale");
                source_size_per_target = optional_number(mapping, "sourceSizePerTarget", 1.0);
            } else {
                throw std::invalid_argument("flow source mapping type must be identity, crossed, or affine");
            }
        }
        sources.emplace_back(
            name, build_logistic_flow(flow, target_sizes, source_size_per_target),
            delta_scale, source_size_per_target);
    }
    return AggregatedFlow(std::move(sources));
}

std::optional<DarkPool> build_dark_pool(const py::dict& cfg) {
    if (!boolean(cfg, "enabled")) return std::nullopt;

    std::shared_ptr<ArrivalDistribution> arrivals = std::make_shared<ZeroInflatedPoissonArrival>(
        number(cfg, "lambda"), number(cfg, "mu"), number(cfg, "p0"));

    return DarkPool(
        std::move(arrivals), number(cfg, "feePips") / 10000.0,
        numbers(cfg[py::str("postedSizes")]));
}

AggregatedECNFlow build_ecn_flow(const py::dict& cfg, double target_spread_pips) {
    const py::str sources_key("flowSources");
    if (!cfg.contains(sources_key)) {
        const py::dict flow = py::cast<py::dict>(cfg[py::str("flow")]);
        std::vector<ECNFlowSource> direct;
        if (cfg.contains(py::str("tradeSizeProbabilities"))) {
            direct.emplace_back(
                "direct", ExponentialFlow(number(flow, "A"), number(flow, "k")),
                1.0, 1.0, numbers(cfg[py::str("tradeSizeProbabilities")]));
        } else {
            direct.emplace_back(
                "direct", ExponentialFlow(number(flow, "A"), number(flow, "k")),
                1.0, 1.0, optional_number(cfg, "meanTradeSize", 1.0));
        }
        return AggregatedECNFlow(std::move(direct));
    }

    std::vector<ECNFlowSource> sources;
    const py::list source_cfgs = py::cast<py::list>(cfg[sources_key]);
    sources.reserve(py::len(source_cfgs));
    for (const py::handle item : source_cfgs) {
        const py::dict source = py::cast<py::dict>(item);
        const py::dict flow = py::cast<py::dict>(source[py::str("flow")]);
        const std::string name = source.contains(py::str("name"))
            ? text(source, "name") : std::string("source");

        double delta_scale = 1.0;
        double source_size_per_target = 1.0;
        if (source.contains(py::str("mapping"))) {
            const py::dict mapping = py::cast<py::dict>(source[py::str("mapping")]);
            const std::string type = mapping.contains(py::str("type"))
                ? text(mapping, "type") : std::string("identity");
            if (type == "identity") {
                // defaults above
            } else if (type == "crossed") {
                const double cross_mid = number(mapping, "crossMid");
                const double source_spread_pips = number(mapping, "sourceSpreadPips");
                if (cross_mid <= 0.0 || source_spread_pips <= 0.0) {
                    throw std::invalid_argument("crossed ECN flow mapping requires positive crossMid and sourceSpreadPips");
                }
                delta_scale = target_spread_pips / (cross_mid * source_spread_pips);
                source_size_per_target = cross_mid;
            } else if (type == "affine") {
                delta_scale = number(mapping, "deltaScale");
                source_size_per_target = optional_number(mapping, "sourceSizePerTarget", 1.0);
            } else {
                throw std::invalid_argument("ECN flow source mapping type must be identity, crossed, or affine");
            }
        }
        if (source.contains(py::str("tradeSizeProbabilities"))) {
            sources.emplace_back(
                name, ExponentialFlow(number(flow, "A"), number(flow, "k")),
                delta_scale, source_size_per_target,
                numbers(source[py::str("tradeSizeProbabilities")]));
        } else {
            sources.emplace_back(
                name, ExponentialFlow(number(flow, "A"), number(flow, "k")),
                delta_scale, source_size_per_target,
                optional_number(source, "meanTradeSize", 1.0));
        }
    }
    return AggregatedECNFlow(std::move(sources));
}

std::optional<PassiveECN> build_passive_ecn(const py::dict& cfg, double target_spread_pips) {
    if (!boolean(cfg, "enabled")) return std::nullopt;
    return PassiveECN(
        numbers(cfg[py::str("deltas")]),
        build_ecn_flow(cfg, target_spread_pips),
        number(cfg, "quoteSize"), number(cfg, "makerFeePips") / 10000.0);
}

PricingProblem build_problem(const py::dict& cfg, std::optional<double> gamma_override = std::nullopt) {
    std::vector<Tier> tiers;
    const py::list tier_cfgs = py::cast<py::list>(cfg[py::str("tiers")]);
    for (const py::handle item : tier_cfgs) {
        const py::dict tier = py::cast<py::dict>(item);
        if (!boolean(tier, "enabled")) continue;
        const py::dict markout = py::cast<py::dict>(tier[py::str("markout")]);
        tiers.emplace_back(
            text(tier, "name"), numbers(tier[py::str("sizes")]),
            build_tier_flow(tier, number(cfg, "spreadPips")),
            SaturatingMarkout(number(markout, "impactScalePips") / 10000.0,
                              number(markout, "sizeExponent"), number(markout, "tauMinutes")),
            boolean(tier, "useMarkout"), number(tier, "deltaMin"), number(tier, "deltaMax"),
            optional_number(tier, "feePips", 0.0) / 10000.0,
            optional_number(tier, "rfqSizeStep", 1.0));
    }
    if (tiers.empty() && !boolean(py::cast<py::dict>(cfg[py::str("darkPool")]), "enabled")
        && !boolean(py::cast<py::dict>(cfg[py::str("passiveEcn")]), "enabled")) {
        throw std::invalid_argument("enable at least one pricing tier, dark pool, or passive ECN");
    }

    const py::dict internal = py::cast<py::dict>(cfg[py::str("internalization")]);
    const double sigma = number(cfg, "sigmaPips") / 10000.0;
    const double gamma = gamma_override.value_or(number(cfg, "gamma"));
    return PricingProblem(
        build_grid(py::cast<py::dict>(cfg[py::str("grid")])),
        number(cfg, "spreadPips") / 10000.0,
        number(cfg, "spotDrift"),
        QuadraticPenalty(gamma, sigma),
        InternalizationTime(number(internal, "tau0"), number(internal, "tau1"), number(internal, "tau2")),
        std::move(tiers),
        build_dark_pool(py::cast<py::dict>(cfg[py::str("darkPool")])),
        build_passive_ecn(py::cast<py::dict>(cfg[py::str("passiveEcn")]), number(cfg, "spreadPips")));
}

py::list matrix_value(const std::vector<std::vector<double>>& matrix) {
    py::list rows;
    for (const auto& row : matrix) rows.append(row);
    return rows;
}

py::dict solution_value(const Solution& solution, const PricingProblem& problem) {
    py::dict out;
    out["qGrid"] = solution.q_grid;
    out["value"] = solution.value;
    out["averageReward"] = solution.average_reward;
    out["hardInventoryLimit"] = solution.hard_inventory_limit;
    out["converged"] = solution.diagnostics.converged;
    out["iterations"] = solution.diagnostics.iterations;
    out["residual"] = solution.diagnostics.reward_upper_bound - solution.diagnostics.reward_lower_bound;
    out["valueChange"] = solution.diagnostics.value_change;
    out["bellmanResidual"] = solution.diagnostics.bellman_residual;

    py::list tier_values;
    for (std::size_t k = 0; k < solution.tier_policies.size(); ++k) {
        py::dict tier;
        tier["name"] = problem.tiers()[k].name();
        tier["sizes"] = solution.tier_policies[k].sizes;
        tier["bid"] = matrix_value(solution.tier_policies[k].bid);
        tier["ask"] = matrix_value(solution.tier_policies[k].ask);
        tier["feePips"] = problem.tiers()[k].fee() * 10000.0;
        tier_values.append(std::move(tier));
    }
    out["tiers"] = std::move(tier_values);

    if (solution.dark_pool_policy && problem.dark_pool()) {
        const auto& policy = *solution.dark_pool_policy;
        py::dict dark;
        dark["qGrid"] = policy.q_grid;
        dark["bidSize"] = policy.bid_size;
        dark["askSize"] = policy.ask_size;
        dark["bidActive"] = policy.bid_active;
        dark["askActive"] = policy.ask_active;
        dark["postedSizes"] = problem.dark_pool()->posted_sizes();
        dark["intensity"] = problem.dark_pool()->arrivals(Side::Bid).arrival_intensity();
        dark["distribution"] = problem.dark_pool()->arrivals(Side::Bid).name();
        dark["feePips"] = problem.dark_pool()->fee(Side::Bid) * 10000.0;
        out["darkPool"] = std::move(dark);
    } else {
        out["darkPool"] = py::none();
    }

    if (solution.passive_ecn_policy && problem.passive_ecn()) {
        const auto& policy = *solution.passive_ecn_policy;
        const auto& venue = *problem.passive_ecn();
        py::dict ecn;
        ecn["qGrid"] = policy.q_grid;
        ecn["bidDelta"] = policy.bid_delta;
        ecn["askDelta"] = policy.ask_delta;
        ecn["bidActive"] = policy.bid_active;
        ecn["askActive"] = policy.ask_active;
        ecn["deltas"] = venue.deltas();
        std::vector<double> fill_rates;
        fill_rates.reserve(venue.deltas().size());
        for (double delta : venue.deltas()) {
            fill_rates.push_back(venue.flow().arrival_rate(delta));
        }
        ecn["fillRates"] = fill_rates;
        py::list source_values;
        for (const auto& source : venue.flow().sources()) {
            py::dict item;
            item["name"] = source.name();
            item["A"] = source.flow().a();
            item["k"] = source.flow().k();
            item["deltaScale"] = source.delta_scale();
            item["sourceSizePerTarget"] = source.source_size_per_target();
            item["tradeSizes"] = source.trade_sizes();
            item["tradeSizeProbabilities"] = source.trade_size_probabilities();
            item["meanTradeSize"] = source.mean_trade_size();
            item["meanTargetTradeSize"] = source.mean_target_trade_size();
            item["fullFillProbability"] = source.full_fill_probability(venue.quote_size());
            item["expectedFillSize"] = source.expected_fill_size(venue.quote_size());
            std::vector<double> source_rates;
            source_rates.reserve(venue.deltas().size());
            for (double delta : venue.deltas()) source_rates.push_back(source.arrival_rate(delta));
            item["fillRates"] = std::move(source_rates);
            source_values.append(std::move(item));
        }
        ecn["flowSources"] = std::move(source_values);
        ecn["quoteSize"] = venue.quote_size();
        ecn["makerFeePips"] = venue.maker_fee() * 10000.0;
        out["passiveEcn"] = std::move(ecn);
    } else {
        out["passiveEcn"] = py::none();
    }
    return out;
}

py::dict stats_value(const PnlStatistics& statistics) {
    py::dict out;
    out["meanQuote"] = statistics.expected_quote_ccy;
    out["stdQuote"] = statistics.std_quote_ccy;
    out["meanBase"] = statistics.expected_base_ccy;
    out["stdBase"] = statistics.std_base_ccy;
    out["referenceSpot"] = statistics.reference_spot;
    return out;
}

py::dict fill_value(const FillEvent& fill) {
    py::dict out;
    out["time"] = fill.time_minutes;
    out["tier"] = fill.tier;
    out["side"] = fill.side;
    out["size"] = fill.size;
    out["price"] = fill.execution_price;
    out["inventoryBefore"] = fill.inventory_before;
    out["inventoryAfter"] = fill.inventory_after;
    return out;
}


py::dict rfq_event_value(const RfqEvent& event) {
    py::dict out;
    out["time"] = event.time_minutes;
    out["tier"] = event.tier;
    out["side"] = event.side;
    out["size"] = event.size;
    out["delta"] = event.delta;
    out["winProbability"] = event.win_probability;
    out["won"] = event.won;
    return out;
}

py::dict ecn_arrival_value(const EcnArrivalEvent& event) {
    py::dict out;
    out["time"] = event.time_minutes;
    out["source"] = event.source;
    out["side"] = event.side;
    out["referencePrice"] = event.reference_price;
    out["tradeDistancePips"] = event.trade_distance_pips;
    out["tradePrice"] = event.trade_price;
    out["quoteDepthPips"] = event.quote_depth_pips;
    // Backward-compatible alias used by older UI code.
    out["depthPips"] = event.quote_depth_pips;
    out["sourceTradeSize"] = event.source_trade_size;
    out["targetFillSize"] = event.target_fill_size;
    out["quoteActive"] = event.quote_active;
    out["won"] = event.won;
    return out;
}

py::dict fill_aggregate_value(const FillAggregate& aggregate) {
    py::dict out;
    out["tier"] = aggregate.tier;
    out["side"] = aggregate.side;
    out["tradeCount"] = aggregate.trade_count;
    out["volume"] = aggregate.volume;
    return out;
}

py::dict rfq_aggregate_value(const RfqAggregate& aggregate) {
    py::dict out;
    out["size"] = aggregate.size;
    out["wins"] = aggregate.wins;
    out["requests"] = aggregate.requests;
    return out;
}

py::dict rfq_tier_size_aggregate_value(const RfqTierSizeAggregate& aggregate) {
    py::dict out;
    out["tier"] = aggregate.tier;
    out["size"] = aggregate.size;
    out["wins"] = aggregate.wins;
    out["requests"] = aggregate.requests;
    return out;
}

py::dict rfq_delta_aggregate_value(const RfqDeltaAggregate& aggregate) {
    py::dict out;
    out["tier"] = aggregate.tier;
    out["size"] = aggregate.size;
    out["delta"] = aggregate.delta;
    out["wins"] = aggregate.wins;
    out["requests"] = aggregate.requests;
    out["admissibleRequests"] = aggregate.admissible_requests;
    out["expectedWins"] = aggregate.expected_wins;
    return out;
}

py::dict rfq_inventory_aggregate_value(const RfqInventoryAggregate& aggregate) {
    py::dict out;
    out["tier"] = aggregate.tier;
    out["side"] = aggregate.side;
    out["size"] = aggregate.size;
    out["inventory"] = aggregate.inventory;
    out["wins"] = aggregate.wins;
    out["requests"] = aggregate.requests;
    out["expectedWins"] = aggregate.expected_wins;
    return out;
}

py::dict path_value(const SamplePath& path) {
    py::dict out;
    out["times"] = path.times;
    out["spots"] = path.spots;
    out["inventories"] = path.inventories;
    out["cashes"] = path.cashes;
    py::list fills;
    for (const auto& fill : path.fills) fills.append(fill_value(fill));
    out["fills"] = std::move(fills);
    py::list rfq_events;
    for (const auto& event : path.rfq_events) rfq_events.append(rfq_event_value(event));
    out["rfqEvents"] = std::move(rfq_events);
    py::list ecn_arrivals;
    for (const auto& event : path.ecn_arrivals) ecn_arrivals.append(ecn_arrival_value(event));
    out["ecnArrivals"] = std::move(ecn_arrivals);
    return out;
}

py::dict monte_carlo_value(const MonteCarloResult& result, bool compact_result = false) {
    py::dict out;
    // Dashboard compact mode avoids materializing unused O(number_of_paths)
    // arrays as Python lists. The full API remains the default.
    if (!compact_result) out["pnlQuote"] = result.pnl_quote_ccy;
    out["pnlBase"] = result.pnl_base_ccy;
    if (!compact_result) {
        out["finalInventory"] = result.final_inventory;
        out["finalSpot"] = result.final_spot;
        out["tradeCount"] = result.trade_count;
    }
    out["times"] = result.sample_times;
    out["inventoryLower"] = result.inventory_lower;
    out["inventoryMedian"] = result.inventory_median;
    out["inventoryUpper"] = result.inventory_upper;
    py::list fill_aggregates;
    for (const auto& aggregate : result.fill_aggregates) {
        fill_aggregates.append(fill_aggregate_value(aggregate));
    }
    out["fillAggregates"] = std::move(fill_aggregates);
    py::list rfq_aggregates;
    for (const auto& aggregate : result.rfq_aggregates) {
        rfq_aggregates.append(rfq_aggregate_value(aggregate));
    }
    out["rfqAggregates"] = std::move(rfq_aggregates);
    py::list rfq_tier_size_aggregates;
    for (const auto& aggregate : result.rfq_tier_size_aggregates) {
        rfq_tier_size_aggregates.append(rfq_tier_size_aggregate_value(aggregate));
    }
    out["rfqTierSizeAggregates"] = std::move(rfq_tier_size_aggregates);
    py::list rfq_delta_aggregates;
    for (const auto& aggregate : result.rfq_delta_aggregates) {
        rfq_delta_aggregates.append(rfq_delta_aggregate_value(aggregate));
    }
    out["rfqDeltaAggregates"] = std::move(rfq_delta_aggregates);
    py::list rfq_inventory_aggregates;
    for (const auto& aggregate : result.rfq_inventory_aggregates) {
        rfq_inventory_aggregates.append(rfq_inventory_aggregate_value(aggregate));
    }
    out["rfqInventoryAggregates"] = std::move(rfq_inventory_aggregates);
    py::list paths;
    for (const auto& path : result.sample_paths) paths.append(path_value(path));
    out["samplePaths"] = std::move(paths);
    return out;
}

}  // namespace

class Engine {
public:
    explicit Engine(py::dict config)
        : config_(std::move(config)),
          reference_spot_(number(config_, "spot")),
          problem_(build_problem(config_)) {}

    py::dict solve() {
        {
            py::gil_scoped_release release;
            solution_ = HowardSolver(problem_).solve();
        }
        return solution_value(*solution_, problem_);
    }

    py::dict statistics(double horizon_minutes, double initial_inventory = 0.0) {
        ensure_solved();
        const double sigma = number(config_, "sigmaPips") / 10000.0;
        PnlStatistics statistics;
        {
            py::gil_scoped_release release;
            statistics = PnlAnalytics(problem_, *solution_, reference_spot_)
                             .statistics(horizon_minutes, sigma, initial_inventory);
        }
        return stats_value(statistics);
    }

    py::dict simulate(double horizon_minutes, int paths, double initial_inventory,
                      std::uint64_t seed, int retained_paths = 6, int sample_points = 191,
                      py::object progress_callback = py::none(), bool compact_result = false) {
        ensure_solved();
        const double sigma = number(config_, "sigmaPips") / 10000.0;
        MonteCarloResult result;

        std::function<void(int, int)> progress;
        if (!progress_callback.is_none()) {
            py::function callback = py::reinterpret_borrow<py::function>(progress_callback);
            progress = [callback](int completed, int total) {
                // The numerical loop runs without the GIL. Reacquire it only for
                // the sparse progress notifications exposed to the Dash worker.
                py::gil_scoped_acquire acquire;
                callback(completed, total);
            };
        }

        {
            py::gil_scoped_release release;
            result = MonteCarloSimulator(problem_, *solution_, reference_spot_)
                         .run(horizon_minutes, paths, sigma, initial_inventory, seed,
                              retained_paths, sample_points, progress);
        }
        return monte_carlo_value(result, compact_result);
    }

    py::list frontier(const std::vector<double>& gamma_values, double horizon_minutes,
                      double initial_inventory = 0.0) const {
        const double sigma = number(config_, "sigmaPips") / 10000.0;
        struct Point {
            double gamma;
            double mean;
            double std;
            double ratio;
            int iterations;
        };
        std::vector<Point> points;
        points.reserve(gamma_values.size());
        for (const double gamma : gamma_values) {
            // Reading the Python config requires the GIL. Heavy numerical work does not.
            PricingProblem problem = build_problem(config_, gamma);
            Solution solution;
            PnlStatistics statistics;
            {
                py::gil_scoped_release release;
                solution = HowardSolver(problem).solve();
                statistics = PnlAnalytics(problem, solution, reference_spot_)
                                 .statistics(horizon_minutes, sigma, initial_inventory);
            }
            points.push_back(Point{
                gamma,
                statistics.expected_base_ccy,
                statistics.std_base_ccy,
                statistics.std_base_ccy > 0.0
                    ? statistics.expected_base_ccy / statistics.std_base_ccy
                    : 0.0,
                solution.diagnostics.iterations,
            });
        }
        py::list out;
        for (const auto& point : points) {
            py::dict item;
            item["gamma"] = point.gamma;
            item["mean"] = point.mean;
            item["std"] = point.std;
            item["ratio"] = point.ratio;
            item["iterations"] = point.iterations;
            out.append(std::move(item));
        }
        return out;
    }

private:
    void ensure_solved() {
        if (solution_) return;
        py::gil_scoped_release release;
        solution_ = HowardSolver(problem_).solve();
    }

    py::dict config_;
    double reference_spot_;
    PricingProblem problem_;
    std::optional<Solution> solution_;
};

}  // namespace ladder_pricer::python

PYBIND11_MODULE(_native, module) {
    module.doc() = "C++ Howard pricing engine";
    module.attr("ECN_PARAMETERIZATION_VERSION") = 10;
    py::class_<ladder_pricer::python::Engine>(module, "Engine")
        .def(py::init<py::dict>())
        .def("solve", &ladder_pricer::python::Engine::solve)
        .def("statistics", &ladder_pricer::python::Engine::statistics,
             py::arg("horizon_minutes"), py::arg("initial_inventory") = 0.0)
        .def("simulate", &ladder_pricer::python::Engine::simulate,
             py::arg("horizon_minutes"), py::arg("paths"), py::arg("initial_inventory"),
             py::arg("seed"), py::arg("retained_paths") = 6, py::arg("sample_points") = 191,
             py::arg("progress_callback") = py::none(), py::arg("compact_result") = false)
        .def("frontier", &ladder_pricer::python::Engine::frontier,
             py::arg("gamma_values"), py::arg("horizon_minutes"),
             py::arg("initial_inventory") = 0.0);
}
