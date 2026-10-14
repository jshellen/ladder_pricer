#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ladder_pricer/solver.hpp"
#include "ladder_pricer/analytics.hpp"
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
    const std::vector<double>& target_sizes = pricing_sizes;
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

PricingProblem build_problem(const py::dict& cfg) {
    std::vector<Tier> tiers;
    // Each tier has its own signed adverse EUR/M markout curve, common to
    // every RFQ size inside that tier. Convert to positive quote/base price
    // magnitude: (EUR per traded EUR 1M) * (quote per EUR) / 1e6.
    const double reference_spot = number(cfg, "spot");
    if (!std::isfinite(reference_spot) || reference_spot <= 0.0) {
        throw std::invalid_argument("markout conversion requires a positive finite spot");
    }
    const py::list tier_cfgs = py::cast<py::list>(cfg[py::str("tiers")]);
    for (const py::handle item : tier_cfgs) {
        const py::dict tier = py::cast<py::dict>(item);
        if (!boolean(tier, "enabled")) continue;
        const py::dict markout = py::cast<py::dict>(tier[py::str("markout")]);
        const double m_inf = number(markout, "asymptoticEurPerMillion");
        const double tau_seconds = number(markout, "tauSeconds");
        if (!std::isfinite(m_inf) || m_inf > 0.0 || !std::isfinite(tau_seconds) || tau_seconds <= 0.0) {
            throw std::invalid_argument("tier markout requires nonpositive EUR/M and positive seconds");
        }
        const SaturatingMarkout tier_markout(-m_inf * reference_spot / 1e6, tau_seconds / 60.0);
        tiers.emplace_back(
            text(tier, "name"), numbers(tier[py::str("sizes")]),
            build_tier_flow(tier, number(cfg, "spreadPips")),
            tier_markout,
            boolean(tier, "useMarkout"), number(tier, "deltaMin"), number(tier, "deltaMax"),
            optional_number(tier, "feePips", 0.0) / 10000.0);
    }
    if (tiers.empty() && !boolean(py::cast<py::dict>(cfg[py::str("darkPool")]), "enabled")
        && !boolean(py::cast<py::dict>(cfg[py::str("passiveEcn")]), "enabled")) {
        throw std::invalid_argument("enable at least one pricing tier, dark pool, or passive ECN");
    }

    const double sigma = number(cfg, "sigmaPips") / 10000.0;
    const double gamma = number(cfg, "gamma");
    return PricingProblem(
        build_grid(py::cast<py::dict>(cfg[py::str("grid")])),
        number(cfg, "spreadPips") / 10000.0,
        QuadraticPenalty(gamma, sigma),
        std::move(tiers),
        build_dark_pool(py::cast<py::dict>(cfg[py::str("darkPool")])),
        build_passive_ecn(py::cast<py::dict>(cfg[py::str("passiveEcn")]), number(cfg, "spreadPips")));
}

py::list matrix_value(const std::vector<std::vector<double>>& matrix) {
    py::list rows;
    for (const auto& row : matrix) rows.append(row);
    return rows;
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
    out["regimes"] = path.regimes;
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
        out["finalRegime"] = result.final_regime;
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
        // A single production solver: joint inventory and volatility regimes.
        if (!config_.contains(py::str("volatilityRegimes"))) {
            throw std::invalid_argument("volatilityRegimes is required: the legacy single-volatility solver has been removed");
        }
        const py::dict cfg = py::cast<py::dict>(config_[py::str("volatilityRegimes")]);
        const auto sigma = numbers(cfg[py::str("sigmaPips")]);
        const auto rates = py::cast<std::vector<std::vector<double>>>(cfg[py::str("ratesPerMinute")]);
        ensure_solved();
        const SolverResult& result=*solution_;
        py::dict out;
        out["qGrid"] = result.q_grid;
        out["sigmaPips"] = result.sigma_pips;
        out["valueByRegime"] = matrix_value(result.values);
        out["averageReward"] = result.average_reward;
        out["iterations"] = result.iterations;
        out["converged"] = result.converged;
        out["evaluationResidual"] = result.evaluation_residual;
        py::list regimes;
        const auto& indices=problem_.grid().operational_indices();
        const auto& states=problem_.grid().states();
        const std::size_t hidden_n=states.size();
        for (std::size_t r=0;r<result.policies.size();++r) {
            const auto& policy=result.policies[r];
            py::dict item;
            item["sigmaPips"]=sigma[r];
            py::list tiers;
            for (std::size_t k=0;k<policy.tiers.size();++k) {
                const auto& ladder=policy.tiers[k];
                py::dict t;
                t["name"]=problem_.tiers()[k].name();
                t["sizes"]=ladder.sizes;
                t["feePips"]=problem_.tiers()[k].fee()*10000.0;
                std::vector<std::vector<double>> bid,ask;
                for (auto i:indices) {bid.push_back(ladder.bid[i]);ask.push_back(ladder.ask[i]);}
                t["bid"]=matrix_value(bid);
                t["ask"]=matrix_value(ask);
                tiers.append(t);
            }
            item["tiers"]=tiers;
            if (policy.passive_ecn) {
                const auto& e=*policy.passive_ecn;
                py::dict ecn;
                std::vector<double> bid,ask;
                std::vector<bool> bid_active,ask_active;
                for (auto i:indices) {
                    bid.push_back(e.bid_delta[i]); ask.push_back(e.ask_delta[i]);
                    bid_active.push_back(e.bid_active[i]); ask_active.push_back(e.ask_active[i]);
                }
                ecn["qGrid"]=result.q_grid;
                ecn["bidDelta"]=bid;
                ecn["askDelta"]=ask;
                ecn["bidActive"]=bid_active;
                ecn["askActive"]=ask_active;
                if (problem_.passive_ecn()) {
                    const auto& venue=*problem_.passive_ecn();
                    ecn["deltas"]=venue.deltas();
                    std::vector<double> rates;
                    for (auto delta:venue.deltas()) rates.push_back(venue.flow().arrival_rate(delta));
                    ecn["fillRates"]=rates;
                    ecn["quoteSize"]=venue.quote_size();
                    ecn["makerFeePips"]=venue.maker_fee()*10000.0;
                    py::list source_values;
                    for (const auto& source:venue.flow().sources()) {
                        py::dict source_value;
                        source_value["name"]=source.name();
                        source_value["A"]=source.flow().a();
                        source_value["k"]=source.flow().k();
                        source_value["deltaScale"]=source.delta_scale();
                        source_value["sourceSizePerTarget"]=source.source_size_per_target();
                        source_value["tradeSizes"]=source.trade_sizes();
                        source_value["tradeSizeProbabilities"]=source.trade_size_probabilities();
                        source_value["meanTradeSize"]=source.mean_trade_size();
                        source_value["meanTargetTradeSize"]=source.mean_target_trade_size();
                        source_value["fullFillProbability"]=source.full_fill_probability(venue.quote_size());
                        source_value["expectedFillSize"]=source.expected_fill_size(venue.quote_size());
                        std::vector<double> source_rates;
                        for (auto delta:venue.deltas()) source_rates.push_back(source.arrival_rate(delta));
                        source_value["fillRates"]=source_rates;
                        source_values.append(source_value);
                    }
                    ecn["flowSources"]=source_values;
                }
                item["passiveEcn"]=ecn;
            } else item["passiveEcn"]=py::none();
            if (policy.dark_pool && problem_.dark_pool()) {
                const auto& dp=*policy.dark_pool;
                const auto& venue=*problem_.dark_pool();
                py::dict dark;
                std::vector<double> bid,ask;
                std::vector<bool> ba,aa;
                for (auto i:indices) {
                    bid.push_back(dp.bid_size[i]);ask.push_back(dp.ask_size[i]);
                    ba.push_back(dp.bid_active[i]);aa.push_back(dp.ask_active[i]);
                }
                dark["qGrid"]=result.q_grid;
                dark["bidSize"]=bid;
                dark["askSize"]=ask;
                dark["bidActive"]=ba;
                dark["askActive"]=aa;
                dark["postedSizes"]=venue.posted_sizes();
                dark["intensity"]=venue.arrivals(Side::Bid).arrival_intensity();
                dark["distribution"]=venue.arrivals(Side::Bid).name();
                dark["feePips"]=venue.fee(Side::Bid)*10000.0;
                item["darkPool"]=dark;
            } else item["darkPool"]=py::none();

            // The simple RFQ markout remains a policy-dependent expected MTM
            // integral, not a third state variable. Include joint regime
            // switches in each tier's resolvent and expose the middle regime
            // cross-section in the legacy chart format.
            std::vector<std::vector<double>> weighted;
            for (const auto& joined:result.weighted_inventory) {
                weighted.emplace_back();
                if (!joined.empty())
                    weighted.back()={joined.begin()+static_cast<std::ptrdiff_t>(r*hidden_n),
                                      joined.begin()+static_cast<std::ptrdiff_t>((r+1)*hidden_n)};
            }
            ExpectedMtm mtm(problem_,std::move(weighted));
            py::list mtm_tiers;
            for (std::size_t k=0;k<problem_.tiers().size();++k) {
                const auto& tier=problem_.tiers()[k];
                py::dict m;
                m["name"]=tier.name();
                m["sizes"]=tier.sizes();
                std::vector<std::vector<double>> gb,ga,ib,ia;
                const double scale=1e6/reference_spot_;
                for (auto idx:indices) {
                    const double q=states[idx];
                    std::vector<double> b0,a0,b1,a1;
                    for (double z:tier.sizes()) {
                        b0.push_back(scale*mtm.trade_mtm(k, 1.,q+z));
                        a0.push_back(scale*mtm.trade_mtm(k,-1.,q-z));
                        b1.push_back(scale*mtm.win_increment(k, 1.,q,q+z));
                        a1.push_back(scale*mtm.win_increment(k,-1.,q,q-z));
                    }
                    gb.push_back(b0);ga.push_back(a0);ib.push_back(b1);ia.push_back(a1);
                }
                m["grossBidEur"]=matrix_value(gb);
                m["grossAskEur"]=matrix_value(ga);
                m["incrementalBidEur"]=matrix_value(ib);
                m["incrementalAskEur"]=matrix_value(ia);
                mtm_tiers.append(m);
            }
            item["expectedMtm"]=mtm_tiers;
            regimes.append(item);
        }
        out["regimes"]=regimes;
        // Normal-volatility slice for the existing policy and quote diagnostics.
        // The coupled solution, not a separate single-vol HJB, supplies it.
        py::dict middle=py::cast<py::dict>(regimes[py::int_(regimes.size()/2)]);
        out["value"]=result.values[result.values.size()/2];
        out["tiers"]=middle["tiers"];
        out["passiveEcn"]=middle["passiveEcn"];
        out["darkPool"]=middle["darkPool"];
        out["expectedMtm"]=middle["expectedMtm"];
        out["expectedMtmUnits"]="EUR per RFQ event";
        out["residual"]=result.evaluation_residual;
        out["valueChange"]=std::vector<double>{};
        out["bellmanResidual"]=std::vector<double>{};
        out["hardInventoryLimit"]=problem_.grid().hard_limit();
        out["stateVariables"]=std::vector<std::string>{"inventory", "volatility"};
        out["optimizationMethod"]="coupled (q,sigma) C++ damped expected-MTM best response";
        return out;
    }

    py::dict solve_volatility() { return solve(); }

    py::dict statistics(double horizon, double initial_inventory = 0.0,
                        int initial_regime = -1) {
        ensure_solved();
        const auto cfg=py::cast<py::dict>(config_[py::str("volatilityRegimes")]);
        const auto switches=py::cast<std::vector<std::vector<double>>>(cfg[py::str("ratesPerMinute")]);
        PnlStatistics result;
        {
            py::gil_scoped_release release;
            result=PnlAnalytics(problem_,*solution_,switches,reference_spot_)
                .statistics(horizon,initial_inventory,initial_regime);
        }
        py::dict out=stats_value(result);
        try {
            std::vector<double> times;
            { py::gil_scoped_release release;
              times=PnlAnalytics(problem_,*solution_,switches,reference_spot_)
                  .internalization_times(initial_regime); }
            py::dict times_data;
            times_data["qGrid"]=solution_->q_grid;
            times_data["minutes"]=times;
            out["internalizationTimes"]=times_data;
        } catch (const std::exception&) {
            // Some fixed policies have an infinite first-passage expectation.
            // That diagnostic must not block exact finite-horizon PnL moments.
        }
        return out;
    }

    py::dict simulate(double horizon, int paths, double initial_inventory, std::uint64_t seed,
                      int retained_paths = 6, int sample_points = 191,
                      py::object progress_callback = py::none(), bool compact_result = false,
                      int initial_regime = -1) {
        ensure_solved();
        const auto cfg=py::cast<py::dict>(config_[py::str("volatilityRegimes")]);
        const auto switches=py::cast<std::vector<std::vector<double>>>(cfg[py::str("ratesPerMinute")]);
        std::function<void(int,int)> progress;
        if(!progress_callback.is_none()) {
            py::function callback=py::reinterpret_borrow<py::function>(progress_callback);
            progress=[callback](int done,int total) {
                py::gil_scoped_acquire gil;
                callback(done,total);
            };
        }
        MonteCarloResult result;
        {
            py::gil_scoped_release release;
            result=MonteCarloSimulator(problem_,*solution_,switches,reference_spot_)
                .run(horizon,paths,initial_inventory,seed,retained_paths,sample_points,progress,initial_regime);
        }
        return monte_carlo_value(result,compact_result);
    }

    py::list frontier(const std::vector<double>& gamma_values, double horizon,
                      double initial_inventory = 0.0, int initial_regime = -1) const {
        const auto cfg=py::cast<py::dict>(config_[py::str("volatilityRegimes")]);
        const auto sigma=numbers(cfg[py::str("sigmaPips")]);
        const auto switches=py::cast<std::vector<std::vector<double>>>(cfg[py::str("ratesPerMinute")]);
        struct Point { double gamma, mean, std, ratio; int iterations; };
        std::vector<Point> points;
        points.reserve(gamma_values.size());
        for(const double gamma:gamma_values) {
            if(gamma<0.0 || !std::isfinite(gamma)) throw std::invalid_argument("invalid gamma");
            PricingProblem modified(problem_.grid().operational_states(),problem_.spread(),
                QuadraticPenalty(gamma,problem_.penalty().sigma()),
                problem_.tiers(),problem_.dark_pool(),problem_.passive_ecn());
            SolverResult result;
            PnlStatistics stats;
            {
                py::gil_scoped_release release;
                result=Solver(modified,sigma,switches).solve();
                stats=PnlAnalytics(modified,result,switches,reference_spot_)
                    .statistics(horizon,initial_inventory,initial_regime);
            }
            points.push_back({gamma,stats.expected_base_ccy,stats.std_base_ccy,
                stats.std_base_ccy>0.0 ? stats.expected_base_ccy/stats.std_base_ccy : 0.0,
                result.iterations});
        }
        py::list out;
        for(const auto& p:points) {
            py::dict item;
            item["gamma"]=p.gamma; item["mean"]=p.mean; item["std"]=p.std;
            item["ratio"]=p.ratio; item["iterations"]=p.iterations;
            out.append(item);
        }
        return out;
    }

private:
    py::dict config_;
    double reference_spot_;
    PricingProblem problem_;
    std::optional<SolverResult> solution_;
    void ensure_solved() {
        if(solution_) return;
        const auto cfg=py::cast<py::dict>(config_[py::str("volatilityRegimes")]);
        const auto sigma=numbers(cfg[py::str("sigmaPips")]);
        const auto switches=py::cast<std::vector<std::vector<double>>>(cfg[py::str("ratesPerMinute")]);
        SolverResult result;
        { py::gil_scoped_release release; result=Solver(problem_,sigma,switches).solve(); }
        solution_=std::move(result);
    }
};

}  // namespace ladder_pricer::python



PYBIND11_MODULE(_native, module) {
    module.doc() = "C++ coupled inventory-volatility policy iteration engine";
    module.attr("ECN_PARAMETERIZATION_VERSION") = 10;
    // Increment whenever the Python -> native configuration schema changes.
    // Version 2 expects a separate markout object inside each RFQ tier.
    module.attr("PRICING_CONFIG_SCHEMA_VERSION") = 11;
    py::class_<ladder_pricer::python::Engine>(module, "Engine")
        .def(py::init<py::dict>())
        .def("solve", &ladder_pricer::python::Engine::solve)
        .def("solve_volatility", &ladder_pricer::python::Engine::solve_volatility)
        .def("statistics", &ladder_pricer::python::Engine::statistics,
             py::arg("horizon_minutes"), py::arg("initial_inventory") = 0.0,
             py::arg("initial_regime") = -1)
        .def("simulate", &ladder_pricer::python::Engine::simulate,
             py::arg("horizon_minutes"), py::arg("paths"), py::arg("initial_inventory"),
             py::arg("seed"), py::arg("retained_paths") = 6, py::arg("sample_points") = 191,
             py::arg("progress_callback") = py::none(), py::arg("compact_result") = false,
             py::arg("initial_regime") = -1)
        .def("frontier", &ladder_pricer::python::Engine::frontier,
             py::arg("gamma_values"), py::arg("horizon_minutes"),
             py::arg("initial_inventory") = 0.0, py::arg("initial_regime") = -1);
}
