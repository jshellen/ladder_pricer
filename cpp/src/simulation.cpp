#include "ladder_pricer/simulation.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace ladder_pricer {
namespace {

constexpr double kTol = 1e-10;

double direction(Side side) noexcept { return side == Side::Bid ? 1.0 : -1.0; }
std::string side_name(Side side) { return side == Side::Bid ? "bid" : "ask"; }

struct Event {
    bool dark = false;
    bool ecn = false;
    bool tier_rfq = false;
    bool volatility_transition = false;
    bool ecn_quote_active = false;
    std::size_t tier = 0;
    Side side = Side::Bid;
    double size = 0.0;
    double delta = 0.0;
    double fee = 0.0;
    double rate = 0.0;
    std::size_t ecn_source = 0;
    std::size_t flow_source = 0;
    std::size_t volatility_target = 0;
};

std::size_t bracket_left(const std::vector<double>& grid, double q) {
    if (q <= grid.front() + kTol) return 0;
    if (q >= grid.back() - kTol) return grid.size() - 1;
    auto it = std::lower_bound(grid.begin(), grid.end(), q);
    const std::size_t right = static_cast<std::size_t>(std::distance(grid.begin(), it));
    if (std::abs(grid[right] - q) <= kTol) return right;
    return right - 1;
}

std::size_t nearest_index(const std::vector<double>& grid, double q) {
    if (q <= grid.front() + kTol) return 0;
    if (q >= grid.back() - kTol) return grid.size() - 1;
    auto it = std::lower_bound(grid.begin(), grid.end(), q);
    const std::size_t right = static_cast<std::size_t>(std::distance(grid.begin(), it));
    if (std::abs(grid[right] - q) <= kTol) return right;
    const std::size_t left = right - 1;
    return (q - grid[left] <= grid[right] - q) ? left : right;
}

struct RfqValidationStats {
    std::uint64_t wins = 0;
    std::uint64_t requests = 0;
    std::uint64_t admissible_requests = 0;
    double expected_wins = 0.0;
    double x_sum = 0.0;
};

double policy_delta(const LadderPolicy& policy, Side side, std::size_t size_idx, double q) {
    const auto& grid = policy.q_grid;
    const auto& mat = side == Side::Bid ? policy.bid : policy.ask;
    const std::size_t left = bracket_left(grid, q);
    if (left == grid.size() - 1 || std::abs(grid[left] - q) <= kTol) return mat[left][size_idx];
    const std::size_t right = left + 1;
    const double w = (q - grid[left]) / (grid[right] - grid[left]);
    return (1.0 - w) * mat[left][size_idx] + w * mat[right][size_idx];
}

double policy_delta_for_size(const LadderPolicy& policy, Side side, double size, double q) {
    const auto& sizes = policy.sizes;
    if (sizes.empty()) throw std::invalid_argument("ladder policy has no pricing sizes");
    if (size <= sizes.front() + kTol) return policy_delta(policy, side, 0, q);
    if (size >= sizes.back() - kTol) return policy_delta(policy, side, sizes.size() - 1, q);

    const auto right_it = std::upper_bound(sizes.begin(), sizes.end(), size);
    const std::size_t right = static_cast<std::size_t>(right_it - sizes.begin());
    const std::size_t left = right - 1;
    const double z0 = sizes[left];
    const double z1 = sizes[right];
    const double d0 = policy_delta(policy, side, left, q);
    const double d1 = policy_delta(policy, side, right, q);
    const double w = (size - z0) / (z1 - z0);

    // Execution price is affine in delta for a fixed side/reference spot, so
    // linear interpolation of delta is exactly linear interpolation of price.
    return (1.0 - w) * d0 + w * d1;
}

double dark_posted(const DarkPoolPolicy& policy, Side side, double q) {
    const auto& grid = policy.q_grid;
    const auto& sizes = side == Side::Bid ? policy.bid_size : policy.ask_size;
    const auto& active = side == Side::Bid ? policy.bid_active : policy.ask_active;
    const std::size_t i = bracket_left(grid, q);
    if (!active[i]) return 0.0;
    return sizes[i];
}

std::vector<Event> events(const PricingProblem& problem, const Solution& solution, double q,
                          std::size_t volatility_state, bool record_all_ecn_arrivals) {
    std::vector<Event> out;
    for (std::size_t k = 0; k < problem.tiers().size(); ++k) {
        const auto& tier = problem.tiers()[k];
        // One exogenous RFQ clock per customer-flow source.  When it rings,
        // customer direction is sampled 50/50 (the current symmetric-flow
        // assumption) and requested size is sampled from the source's
        // size distribution implied by the calibrated exogenous arrival-intensity curve.
        for (std::size_t s = 0; s < tier.flow().sources().size(); ++s) {
            const auto& source = tier.flow().sources()[s];
            const double rate = source.total_rfq_rate();
            if (rate <= 0.0) continue;
            Event event;
            event.tier_rfq = true;
            event.tier = k;
            event.fee = tier.fee();
            event.rate = rate;
            event.flow_source = s;
            out.push_back(event);
        }
    }
    const auto dark_policy = !solution.solve_dark_pool_policies_by_volatility.empty()
        ? solution.solve_dark_pool_policies_by_volatility.at(volatility_state)
        : solution.solve_dark_pool_policy;
    if (problem.dark_pool() && dark_policy) {
        const auto& venue = *problem.dark_pool();
        const auto& policy = *dark_policy;
        for (Side side : {Side::Bid, Side::Ask}) {
            const double dir = direction(side);
            const int u = static_cast<int>(std::llround(dark_posted(policy, side, q)));
            if (u <= 0 || !venue.risk_reducing(q, side, static_cast<double>(u)) ||
                !problem.grid().admissible(q, dir * u)) continue;

            // One exogenous market-order clock per side.  When it rings, the
            // incoming order size is sampled from the ZIP distribution and the
            // realized fill is min(incoming size, our posted size).
            const double rate = venue.arrivals(side).arrival_intensity();
            if (rate > 0.0) {
                Event event;
                event.dark = true;
                event.side = side;
                event.size = static_cast<double>(u);
                event.fee = venue.fee(side);
                event.rate = rate;
                out.push_back(std::move(event));
            }
        }
    }
    const auto ecn_policy = !solution.solve_passive_ecn_policies_by_volatility.empty()
        ? solution.solve_passive_ecn_policies_by_volatility.at(volatility_state)
        : solution.solve_passive_ecn_policy;
    if (problem.passive_ecn() && ecn_policy) {
        const auto& venue = *problem.passive_ecn();
        const auto& policy = *ecn_policy;
        const std::size_t i = bracket_left(policy.q_grid, q);
        const double z = venue.quote_size();

        // Each ECN source has its own exogenous market-trade clock and reach
        // distribution.  The policy chooses one master target-pair delta. A
        // crossed source inherits the same normalized delta because its quote
        // and theoretical spread are both obtained by crossing the target pair.
        // Keeping the parent clocks separate preserves the exact sum of the
        // source-specific fill curves and lets retained paths identify which
        // ECN pair generated an arrival/fill.
        for (Side side : {Side::Bid, Side::Ask}) {
            const bool policy_active = side == Side::Bid ? policy.bid_active[i] : policy.ask_active[i];
            const double delta = side == Side::Bid ? policy.bid_delta[i] : policy.ask_delta[i];
            const double dir = direction(side);
            const bool quote_active = policy_active && venue.risk_reducing(q, side, z) &&
                                      problem.grid().admissible(q, dir * z);
            if (!record_all_ecn_arrivals && !quote_active) continue;
            const auto& sources = venue.flow().sources();
            for (std::size_t s = 0; s < sources.size(); ++s) {
                const double rate = sources[s].flow().a();
                if (rate <= 0.0) continue;
                Event event;
                event.ecn = true;
                event.ecn_quote_active = quote_active;
                event.side = side;
                event.size = z;
                event.delta = delta;
                event.fee = venue.maker_fee();
                event.rate = rate;
                event.ecn_source = s;
                out.push_back(std::move(event));
            }
        }
    }

    // Exogenous volatility CTMC transitions.  Q is in 1/min, matching all
    // other event intensities in the simulator.
    const auto& qvol = problem.volatility().generator();
    for (std::size_t target = 0; target < problem.volatility().state_count(); ++target) {
        if (target == volatility_state) continue;
        const double rate = qvol[volatility_state][target];
        if (rate <= 0.0) continue;
        Event event;
        event.volatility_transition = true;
        event.volatility_target = target;
        event.rate = rate;
        out.push_back(std::move(event));
    }
    return out;
}

double advance_spot(double spot, double dt, double sigma, double base_drift,
                    std::map<double, double>& impacts, std::mt19937_64& rng) {
    if (dt <= 0.0) return spot;
    double deterministic = base_drift * dt;
    for (auto it = impacts.begin(); it != impacts.end();) {
        const double tau = it->first;
        const double decay = std::exp(-dt / tau);
        deterministic += it->second * (1.0 - decay);
        it->second *= decay;
        if (std::abs(it->second) < 1e-15) it = impacts.erase(it); else ++it;
    }
    double brownian = 0.0;
    if (sigma > 0.0) {
        std::normal_distribution<double> normal(0.0, 1.0);
        brownian = sigma * std::sqrt(dt) * normal(rng);
    }
    return spot + deterministic + brownian;
}

double percentile_select(std::vector<double>& values, double p) {
    // We only need six order statistics (lo/hi for 2.5%, 50%, 97.5%), not a
    // fully sorted cross-section. nth_element keeps the exact percentile
    // definition while reducing the tail from O(N log N) to expected O(N).
    if (values.empty()) return 0.0;
    const double x = p * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(x));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(x));
    const double w = x - static_cast<double>(lo);

    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lo), values.end());
    const double lower = values[lo];
    if (hi == lo) return lower;

    // After selecting lo, every element to its right is >= values[lo], so the
    // global hi-th order statistic can be selected entirely from that suffix.
    std::nth_element(values.begin() + static_cast<std::ptrdiff_t>(lo + 1),
                     values.begin() + static_cast<std::ptrdiff_t>(hi), values.end());
    const double upper = values[hi];
    return (1.0 - w) * lower + w * upper;
}

}  // namespace

MonteCarloResult MonteCarloSimulator::run(double horizon, int paths, double sigma,
                                           double q0, std::uint64_t seed,
                                           int retained_paths, int sample_points,
                                           const std::function<void(int, int)>& progress) const {
    (void)sigma;  // volatility now comes from PricingProblem::volatility()
    if (horizon < 0.0 || paths <= 0 || sample_points < 2) {
        throw std::invalid_argument("invalid Monte Carlo settings");
    }
    if (reference_spot_ <= 0.0) throw std::invalid_argument("reference spot must be positive");

    // Retained paths use a one-second market clock for the interactive price/quote
    // diagnostics. Non-retained paths can propagate spot exactly from event to event:
    // with the current Brownian + deterministic drift/markout dynamics, subdividing an
    // interval does not change its distribution and would only make large MC runs slower.
    // The clock is independent of the inventory/output sampling grid.
    constexpr double kMarketStepMinutes = 1.0 / 60.0;  // 1 second
    constexpr std::uint64_t kPathStride = 0x9e3779b97f4a7c15ULL;
    constexpr std::uint64_t kSpotSalt = 0xd1b54a32d192ed03ULL;

    MonteCarloResult result;
    result.pnl_quote_ccy.resize(paths);
    result.pnl_base_ccy.resize(paths);
    result.final_inventory.resize(paths);
    result.final_spot.resize(paths);
    result.trade_count.resize(paths);
    result.sample_times.resize(sample_points);
    for (int s = 0; s < sample_points; ++s) {
        result.sample_times[s] = horizon * static_cast<double>(s) / static_cast<double>(sample_points - 1);
    }
    std::vector<std::vector<double>> inventories(sample_points, std::vector<double>(paths));
    std::map<std::pair<std::string, std::string>, std::pair<std::uint64_t, double>> fill_aggregates;
    std::map<double, std::pair<std::uint64_t, std::uint64_t>> rfq_aggregates;  // wins, requests
    std::map<std::pair<std::string, double>, std::pair<std::uint64_t, std::uint64_t>>
        rfq_tier_size_aggregates;  // wins, requests
    // Delta is compacted into 1%-wide buckets.  The plotted x-coordinate is the
    // mean actual delta in the bucket, so this remains faithful when policy
    // interpolation produces non-grid quote deltas.
    constexpr double kDeltaDiagnosticBin = 0.01;
    std::map<std::tuple<std::string, double, long long>, RfqValidationStats>
        rfq_delta_aggregates;
    // Inventory can become continuous through ECN partial fills.  Bucket RFQs to
    // the nearest HJB inventory state while preserving the mean actual q in each
    // bucket for the diagnostic x-coordinate.
    std::map<std::tuple<std::string, std::string, double, std::size_t>, RfqValidationStats>
        rfq_inventory_aggregates;

    auto record_fill_aggregate = [&](const std::string& tier, Side side, double size) {
        auto& aggregate = fill_aggregates[{tier, side_name(side)}];
        ++aggregate.first;
        aggregate.second += size;
    };

    for (int path_idx = 0; path_idx < paths; ++path_idx) {
        const std::uint64_t path_seed = seed + kPathStride * static_cast<std::uint64_t>(path_idx + 1);
        std::mt19937_64 event_rng(path_seed);
        std::mt19937_64 spot_rng(path_seed ^ kSpotSalt);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        double t = 0.0;
        double q = q0;
        double spot = reference_spot_;
        double cash = 0.0;  // million quote currency
        int trades = 0;
        std::size_t vol_state = problem_.volatility().initial_state();
        std::map<double, double> impacts;
        int next_sample = 0;
        long long market_tick = 1;

        SamplePath detailed;
        const bool retain = path_idx < retained_paths;
        if (retain) {
            detailed.times.push_back(0.0);
            detailed.spots.push_back(spot);
            detailed.inventories.push_back(q);
            detailed.cashes.push_back(cash);
            detailed.volatility_states.push_back(static_cast<int>(vol_state));
        }

        // Inventory is piecewise constant, so confidence-band snapshots do not need
        // to drive the market-price clock.
        auto record_inventory_before = [&](double target_time) {
            while (next_sample < sample_points && result.sample_times[next_sample] < target_time - 1e-12) {
                inventories[next_sample][path_idx] = q;
                ++next_sample;
            }
        };
        auto record_inventory_through = [&](double target_time) {
            while (next_sample < sample_points && result.sample_times[next_sample] <= target_time + 1e-12) {
                inventories[next_sample][path_idx] = q;
                ++next_sample;
            }
        };
        record_inventory_through(0.0);

        while (t < horizon - 1e-12) {
            const auto ev = events(problem_, solution_, q, vol_state, retain);
            double total_rate = 0.0;
            for (const auto& e : ev) total_rate += e.rate;

            double event_time = horizon;
            bool has_event = false;
            if (total_rate > 0.0) {
                const double u = std::max(1e-16, uniform(event_rng));
                const double wait = -std::log(u) / total_rate;
                if (t + wait < horizon - 1e-12) {
                    event_time = t + wait;
                    has_event = true;
                }
            }

            // Evolve retained (plotted) paths on the one-second market clock until the
            // next asynchronous fill. Hidden MC paths skip these intermediate points and
            // propagate exactly to the event time below, which is equivalent in law.
            while (retain) {
                while (kMarketStepMinutes * static_cast<double>(market_tick) <= t + 1e-12) {
                    ++market_tick;
                }
                const double next_market_time = kMarketStepMinutes * static_cast<double>(market_tick);
                if (next_market_time >= event_time - 1e-12 || next_market_time > horizon + 1e-12) break;

                record_inventory_before(next_market_time);
                spot = advance_spot(spot, next_market_time - t, problem_.volatility().sigma(vol_state), problem_.spot_drift(), impacts, spot_rng);
                t = next_market_time;
                ++market_tick;
                record_inventory_through(t);
                if (retain) {
                    detailed.times.push_back(t);
                    detailed.spots.push_back(spot);
                    detailed.inventories.push_back(q);
                    detailed.cashes.push_back(cash);
                    detailed.volatility_states.push_back(static_cast<int>(vol_state));
                }
            }

            record_inventory_before(event_time);
            if (event_time > t + 1e-12) {
                spot = advance_spot(spot, event_time - t, problem_.volatility().sigma(vol_state), problem_.spot_drift(), impacts, spot_rng);
                t = event_time;
            } else {
                t = event_time;
            }

            if (!has_event) {
                record_inventory_through(t);
                if (retain && (detailed.times.empty() || std::abs(detailed.times.back() - t) > 1e-12)) {
                    detailed.times.push_back(t);
                    detailed.spots.push_back(spot);
                    detailed.inventories.push_back(q);
                    detailed.cashes.push_back(cash);
                    detailed.volatility_states.push_back(static_cast<int>(vol_state));
                }
                break;
            }

            double draw = uniform(event_rng) * total_rate;
            const Event* chosen = &ev.back();
            for (const auto& e : ev) {
                draw -= e.rate;
                if (draw <= 0.0) { chosen = &e; break; }
            }

            const double before = q;

            if (chosen->volatility_transition) {
                vol_state = chosen->volatility_target;
                if (retain) {
                    detailed.times.push_back(t);
                    detailed.spots.push_back(spot);
                    detailed.inventories.push_back(q);
                    detailed.cashes.push_back(cash);
                    detailed.volatility_states.push_back(static_cast<int>(vol_state));
                }
                record_inventory_through(t);
                continue;
            }

            if (chosen->tier_rfq) {
                const Side rfq_side = uniform(event_rng) < 0.5 ? Side::Bid : Side::Ask;
                const double dir = direction(rfq_side);
                const auto& tier = problem_.tiers()[chosen->tier];
                const auto& policy = !solution_.solve_tier_policies_by_volatility.empty()
                    ? solution_.solve_tier_policies_by_volatility.at(vol_state).at(chosen->tier)
                    : solution_.solve_tier_policies.at(chosen->tier);
                const auto& source = tier.flow().sources().at(chosen->flow_source);
                const std::size_t size_idx = source.sample_target_size_index(event_rng);
                const double rfq_size = source.target_sizes().at(size_idx);
                const double rfq_delta = policy_delta_for_size(policy, rfq_side, rfq_size, q);
                const bool admissible = problem_.grid().admissible(q, dir * rfq_size);
                const double p_win = admissible
                    ? source.win_probability(rfq_delta, rfq_size)
                    : 0.0;
                const bool won = uniform(event_rng) < p_win;

                auto& rfq_aggregate = rfq_aggregates[rfq_size];
                ++rfq_aggregate.second;
                if (won) ++rfq_aggregate.first;

                auto& tier_size_aggregate = rfq_tier_size_aggregates[{tier.name(), rfq_size}];
                ++tier_size_aggregate.second;
                if (won) ++tier_size_aggregate.first;

                const long long delta_bucket = static_cast<long long>(
                    std::llround(rfq_delta / kDeltaDiagnosticBin));
                auto& delta_aggregate = rfq_delta_aggregates[
                    {tier.name(), rfq_size, delta_bucket}];
                ++delta_aggregate.requests;
                if (admissible) ++delta_aggregate.admissible_requests;
                if (won) ++delta_aggregate.wins;
                delta_aggregate.expected_wins += p_win;
                delta_aggregate.x_sum += rfq_delta;

                const std::size_t inventory_bucket = nearest_index(policy.q_grid, q);
                auto& inventory_aggregate = rfq_inventory_aggregates[
                    {tier.name(), side_name(rfq_side), rfq_size, inventory_bucket}];
                ++inventory_aggregate.requests;
                if (won) ++inventory_aggregate.wins;
                inventory_aggregate.expected_wins += p_win;
                inventory_aggregate.x_sum += q;

                if (retain) {
                    detailed.rfq_events.push_back({
                        t, tier.name(), side_name(rfq_side), rfq_size,
                        rfq_delta, p_win, won
                    });
                }

                double execution = spot;
                if (won) {
                    // Inventory/cash are updated first for a won RFQ.  The markout
                    // shock below is then applied to the post-trade inventory state.
                    execution = rfq_side == Side::Bid
                        ? spot - problem_.spread() * (0.5 - rfq_delta)
                        : spot + problem_.spread() * (0.5 - rfq_delta);
                    if (rfq_side == Side::Bid) cash -= rfq_size * execution;
                    else cash += rfq_size * execution;
                    cash -= chosen->fee * rfq_size;
                    q += dir * rfq_size;
                    ++trades;
                    record_fill_aggregate(tier.name(), rfq_side, rfq_size);
                }

                // An RFQ is an information event whether or not we win it.
                if (tier.use_markout()) {
                    impacts[tier.markout().tau_minutes()] += -dir * tier.markout().asymptotic();
                }

                if (retain && won) {
                    detailed.fills.push_back({t, tier.name(), side_name(rfq_side),
                                              rfq_size, execution, before, q});
                }
            } else {
                const double dir = direction(chosen->side);
                double execution = spot;
                double executed_size = chosen->size;

                if (chosen->dark) {
                    const auto& venue = *problem_.dark_pool();
                    const int incoming = venue.arrivals(chosen->side).sample_incoming_size(event_rng);
                    const int posted = static_cast<int>(std::llround(chosen->size));
                    executed_size = static_cast<double>(std::min(incoming, posted));
                    if (executed_size <= 0.0) {
                        // A dark-pool market-order arrival occurred, but the ZIP
                        // size draw produced zero executable size.  Nothing is
                        // filled and inventory/cash stay unchanged.
                        record_inventory_through(t);
                        if (retain) {
                            detailed.times.push_back(t);
                            detailed.spots.push_back(spot);
                            detailed.inventories.push_back(q);
                            detailed.cashes.push_back(cash);
                            detailed.volatility_states.push_back(static_cast<int>(vol_state));
                        }
                        continue;
                    }
                } else if (chosen->ecn) {
                    const auto& venue = *problem_.passive_ecn();

                    // A parent trade arrived on one concrete ECN source. Sample
                    // reach in that source pair, then convert it back into the
                    // target-pair spread coordinate for diagnostics. For crossed
                    // sources delta_scale=1, so d_source == d_target exactly.
                    const auto& source = venue.flow().sources().at(chosen->ecn_source);
                    std::exponential_distribution<double> reach_distribution(source.flow().k());
                    const double source_reach = reach_distribution(event_rng);
                    const double source_trade_size = source.sample_trade_size(event_rng);
                    const double target_trade_reach = source.target_reach(source_reach);
                    const double target_quote_reach = 0.5 - chosen->delta;
                    const double signed_distance = problem_.spread() * target_trade_reach;
                    const double trade_distance_pips = signed_distance * 10000.0;
                    const double quote_depth_pips = problem_.spread() * target_quote_reach * 10000.0;
                    const double trade_price = chosen->side == Side::Bid
                        ? spot - signed_distance
                        : spot + signed_distance;
                    const bool won = chosen->ecn_quote_active &&
                        source_reach + 1e-12 >= source.delta_scale() * target_quote_reach;
                    const double source_posted_size = chosen->size * source.source_size_per_target();
                    const double target_fill_size =
                        std::min(source_trade_size, source_posted_size) / source.source_size_per_target();

                    if (retain) {
                        detailed.ecn_arrivals.push_back({
                            t, source.name(), side_name(chosen->side), spot,
                            trade_distance_pips, trade_price, quote_depth_pips,
                            source_trade_size, won ? target_fill_size : 0.0,
                            chosen->ecn_quote_active, won
                        });
                    }
                    if (!won) {
                        record_inventory_through(t);
                        if (retain) {
                            detailed.times.push_back(t);
                            detailed.spots.push_back(spot);
                            detailed.inventories.push_back(q);
                            detailed.cashes.push_back(cash);
                            detailed.volatility_states.push_back(static_cast<int>(vol_state));
                        }
                        continue;
                    }

                    executed_size = target_fill_size;
                    if (!venue.risk_reducing(q, chosen->side, executed_size)) {
                        throw std::runtime_error("sampled ECN fill is not risk reducing");
                    }

                    // We execute passively at our quote, not at the aggressor's limit.
                    const double quote_distance = problem_.spread() * (0.5 - chosen->delta);
                    execution = chosen->side == Side::Bid ? spot - quote_distance : spot + quote_distance;
                }

                if (chosen->dark && !problem_.dark_pool()->risk_reducing(q, chosen->side, executed_size)) {
                    throw std::runtime_error("sampled dark-pool fill is not risk reducing");
                }
                if (!problem_.grid().admissible(q, dir * executed_size)) {
                    throw std::runtime_error("sampled passive fill is not admissible");
                }
                if (chosen->side == Side::Bid) cash -= executed_size * execution;
                else cash += executed_size * execution;
                cash -= chosen->fee * executed_size;
                q += dir * executed_size;
                ++trades;

                std::string venue_name = chosen->dark ? "Dark pool" : "Passive ECN";
                if (chosen->ecn) {
                    const auto& source = problem_.passive_ecn()->flow().sources().at(chosen->ecn_source);
                    venue_name += " · " + source.name();
                }
                record_fill_aggregate(venue_name, chosen->side, executed_size);

                if (retain) {
                    detailed.fills.push_back({t, venue_name, side_name(chosen->side),
                                              executed_size, execution, before, q});
                }
            }

            if (retain) {
                // Keep an explicit event point for every RFQ/fill so the price path also
                // records markout shocks from RFQs that we did not win.
                detailed.times.push_back(t);
                detailed.spots.push_back(spot);
                detailed.inventories.push_back(q);
                detailed.cashes.push_back(cash);
                detailed.volatility_states.push_back(static_cast<int>(vol_state));
            }
            record_inventory_through(t);
        }

        while (next_sample < sample_points) {
            inventories[next_sample][path_idx] = q;
            ++next_sample;
        }

        const double pnl_million_quote = cash + q * spot - q0 * reference_spot_;
        result.pnl_quote_ccy[path_idx] = pnl_million_quote * 1'000'000.0;
        result.pnl_base_ccy[path_idx] = result.pnl_quote_ccy[path_idx] / spot;
        result.final_inventory[path_idx] = q;
        result.final_spot[path_idx] = spot;
        result.trade_count[path_idx] = trades;
        if (retain) result.sample_paths.push_back(std::move(detailed));

        // Report genuine completed-path progress at most about 100 times per run.
        // The callback is optional, so native analytics/tests pay essentially no
        // cost when the UI does not request progress updates.
        if (progress) {
            const int completed = path_idx + 1;
            const int report_every = std::max(1, paths / 100);
            if (completed == paths || completed % report_every == 0) {
                progress(completed, paths);
            }
        }
    }

    result.inventory_lower.resize(sample_points);
    result.inventory_median.resize(sample_points);
    result.inventory_upper.resize(sample_points);

    // Confidence-band construction used to dominate the apparent "94-97%"
    // stall for large runs: 381 independent cross-path vectors were fully
    // sorted after the final simulated path.  Exact quantiles do not require a
    // full sort, and the sample-time cross-sections are independent, so select
    // the required order statistics in parallel.  This preserves the exact
    // percentile values while making the end-of-run work close to linear in
    // the number of paths.
    auto compute_inventory_quantiles = [&](int s) {
        auto& snapshot = inventories[static_cast<std::size_t>(s)];
        result.inventory_lower[static_cast<std::size_t>(s)] = percentile_select(snapshot, 0.025);
        result.inventory_median[static_cast<std::size_t>(s)] = percentile_select(snapshot, 0.50);
        result.inventory_upper[static_cast<std::size_t>(s)] = percentile_select(snapshot, 0.975);
    };

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned worker_count = (paths >= 2'000 && sample_points >= 32)
        ? std::min<unsigned>(8u, std::min<unsigned>(hw, static_cast<unsigned>(sample_points)))
        : 1u;

    if (worker_count == 1u) {
        for (int s = 0; s < sample_points; ++s) compute_inventory_quantiles(s);
    } else {
        std::atomic<int> next_snapshot{0};
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (unsigned worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([&]() {
                for (;;) {
                    const int s = next_snapshot.fetch_add(1, std::memory_order_relaxed);
                    if (s >= sample_points) break;
                    compute_inventory_quantiles(s);
                }
            });
        }
        for (auto& worker : workers) worker.join();
    }

    // Emit fill summaries in stable business order rather than stochastic
    // first-hit order: customer tiers, dark pool, then each ECN source.
    auto append_fill_summary = [&](const std::string& tier_name, const std::string& side) {
        const auto it = fill_aggregates.find({tier_name, side});
        const std::uint64_t count = it == fill_aggregates.end() ? 0 : it->second.first;
        const double volume = it == fill_aggregates.end() ? 0.0 : it->second.second;
        result.fill_aggregates.push_back({tier_name, side, count, volume});
    };
    for (const auto& tier : problem_.tiers()) {
        append_fill_summary(tier.name(), "bid");
        append_fill_summary(tier.name(), "ask");
    }
    if (problem_.dark_pool()) {
        append_fill_summary("Dark pool", "bid");
        append_fill_summary("Dark pool", "ask");
    }
    if (problem_.passive_ecn()) {
        for (const auto& source : problem_.passive_ecn()->flow().sources()) {
            const std::string name = "Passive ECN · " + source.name();
            append_fill_summary(name, "bid");
            append_fill_summary(name, "ask");
        }
    }

    result.rfq_aggregates.reserve(rfq_aggregates.size());
    for (const auto& [size, aggregate] : rfq_aggregates) {
        result.rfq_aggregates.push_back({size, aggregate.first, aggregate.second});
    }

    result.rfq_tier_size_aggregates.reserve(rfq_tier_size_aggregates.size());
    for (const auto& [key, aggregate] : rfq_tier_size_aggregates) {
        const auto& [tier_name, size] = key;
        result.rfq_tier_size_aggregates.push_back(
            {tier_name, size, aggregate.first, aggregate.second});
    }

    result.rfq_delta_aggregates.reserve(rfq_delta_aggregates.size());
    for (const auto& [key, aggregate] : rfq_delta_aggregates) {
        const auto& [tier_name, size, bucket] = key;
        (void)bucket;
        const double mean_delta = aggregate.requests > 0
            ? aggregate.x_sum / static_cast<double>(aggregate.requests)
            : 0.0;
        result.rfq_delta_aggregates.push_back({
            tier_name, size, mean_delta, aggregate.wins, aggregate.requests,
            aggregate.admissible_requests, aggregate.expected_wins,
        });
    }

    result.rfq_inventory_aggregates.reserve(rfq_inventory_aggregates.size());
    for (const auto& [key, aggregate] : rfq_inventory_aggregates) {
        const auto& [tier_name, side, size, bucket] = key;
        (void)bucket;
        const double mean_inventory = aggregate.requests > 0
            ? aggregate.x_sum / static_cast<double>(aggregate.requests)
            : 0.0;
        result.rfq_inventory_aggregates.push_back({
            tier_name, side, size, mean_inventory, aggregate.wins,
            aggregate.requests, aggregate.expected_wins,
        });
    }
    return result;
}

}  // namespace ladder_pricer
