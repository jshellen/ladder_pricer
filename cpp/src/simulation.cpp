#include "ladder_pricer/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>

namespace ladder_pricer {
namespace {

constexpr double kTol = 1e-10;

double direction(Side side) noexcept { return side == Side::Bid ? 1.0 : -1.0; }
std::string side_name(Side side) { return side == Side::Bid ? "bid" : "ask"; }

struct Event {
    bool dark = false;
    bool ecn = false;
    bool tier_rfq = false;
    bool ecn_quote_active = false;
    std::size_t tier = 0;
    Side side = Side::Bid;
    double size = 0.0;
    double delta = 0.0;
    double fee = 0.0;
    double rate = 0.0;
};

std::size_t bracket_left(const std::vector<double>& grid, double q) {
    if (q <= grid.front() + kTol) return 0;
    if (q >= grid.back() - kTol) return grid.size() - 1;
    auto it = std::lower_bound(grid.begin(), grid.end(), q);
    const std::size_t right = static_cast<std::size_t>(std::distance(grid.begin(), it));
    if (std::abs(grid[right] - q) <= kTol) return right;
    return right - 1;
}

double policy_delta(const LadderPolicy& policy, Side side, std::size_t size_idx, double q) {
    const auto& grid = policy.q_grid;
    const auto& mat = side == Side::Bid ? policy.bid : policy.ask;
    const std::size_t left = bracket_left(grid, q);
    if (left == grid.size() - 1 || std::abs(grid[left] - q) <= kTol) return mat[left][size_idx];
    const std::size_t right = left + 1;
    const double w = (q - grid[left]) / (grid[right] - grid[left]);
    return (1.0 - w) * mat[left][size_idx] + w * mat[right][size_idx];
}

double dark_posted(const DarkPoolPolicy& policy, Side side, double q) {
    const auto& grid = policy.q_grid;
    const auto& sizes = side == Side::Bid ? policy.bid_size : policy.ask_size;
    const auto& active = side == Side::Bid ? policy.bid_active : policy.ask_active;
    const std::size_t i = bracket_left(grid, q);
    if (!active[i]) return 0.0;
    return sizes[i];
}

std::vector<Event> events(const PricingProblem& problem, const Solution& solution, double q, bool record_all_ecn_arrivals) {
    std::vector<Event> out;
    for (std::size_t k = 0; k < problem.tiers().size(); ++k) {
        const auto& tier = problem.tiers()[k];
        const auto& policy = solution.solve_tier_policies[k];
        for (Side side : {Side::Bid, Side::Ask}) {
            for (std::size_t j = 0; j < tier.sizes().size(); ++j) {
                const double z = tier.sizes()[j];
                // Customer RFQs arrive exogenously.  Our quote controls only the
                // conditional win probability; it must not thin the RFQ event clock.
                const double delta = policy_delta(policy, side, j, q);
                const double rate = tier.flow().rfq_arrival_rate(z);
                if (rate > 0.0) out.push_back({false, false, true, false, k, side, z, delta, tier.fee(), rate});
            }
        }
    }
    if (problem.dark_pool() && solution.solve_dark_pool_policy) {
        const auto& venue = *problem.dark_pool();
        const auto& policy = *solution.solve_dark_pool_policy;
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
                out.push_back({true, false, false, false, 0, side, static_cast<double>(u), 0.0, venue.fee(side), rate});
            }
        }
    }
    if (problem.passive_ecn() && solution.solve_passive_ecn_policy) {
        const auto& venue = *problem.passive_ecn();
        const auto& policy = *solution.solve_passive_ecn_policy;
        const std::size_t i = bracket_left(policy.q_grid, q);
        const double z = venue.quote_size();

        // ECN market trades are a marked Poisson process. Buy and sell trades each
        // arrive exogenously at rate A. Their distance D from mid is sampled later
        // from Exp(k). An active passive quote at depth delta fills iff D >= delta,
        // which implies lambda_fill(delta) = A * P(D >= delta) = A exp(-k delta).
        // For retained paths we keep both sides even when no quote is active so the
        // price chart can show market trades that we did not win. Hidden paths may
        // omit an inactive side because it cannot affect inventory or PnL.
        for (Side side : {Side::Bid, Side::Ask}) {
            const bool policy_active = side == Side::Bid ? policy.bid_active[i] : policy.ask_active[i];
            const double depth = side == Side::Bid ? policy.bid_depth[i] : policy.ask_depth[i];
            const double dir = direction(side);
            const bool quote_active = policy_active && problem.grid().admissible(q, dir * z);
            if (!record_all_ecn_arrivals && !quote_active) continue;
            const double rate = venue.flow().a();
            if (rate > 0.0) {
                out.push_back({false, true, false, quote_active, 0, side, z, depth, venue.maker_fee(), rate});
            }
        }
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

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double x = p * (values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(x));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(x));
    const double w = x - lo;
    return (1.0 - w) * values[lo] + w * values[hi];
}

}  // namespace

MonteCarloResult MonteCarloSimulator::run(double horizon, int paths, double sigma,
                                           double q0, std::uint64_t seed,
                                           int retained_paths, int sample_points) const {
    if (horizon < 0.0 || paths <= 0 || sigma < 0.0 || sample_points < 2) {
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
            const auto ev = events(problem_, solution_, q, retain);
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
                spot = advance_spot(spot, next_market_time - t, sigma, problem_.spot_drift(), impacts, spot_rng);
                t = next_market_time;
                ++market_tick;
                record_inventory_through(t);
                if (retain) {
                    detailed.times.push_back(t);
                    detailed.spots.push_back(spot);
                    detailed.inventories.push_back(q);
                    detailed.cashes.push_back(cash);
                }
            }

            record_inventory_before(event_time);
            if (event_time > t + 1e-12) {
                spot = advance_spot(spot, event_time - t, sigma, problem_.spot_drift(), impacts, spot_rng);
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
            const double dir = direction(chosen->side);

            if (chosen->tier_rfq) {
                const auto& tier = problem_.tiers()[chosen->tier];
                const bool admissible = problem_.grid().admissible(q, dir * chosen->size);
                const double p_win = admissible
                    ? tier.flow().win_probability(chosen->delta, chosen->size)
                    : 0.0;
                const bool won = uniform(event_rng) < p_win;

                double execution = spot;
                if (won) {
                    // Inventory/cash are updated first for a won RFQ.  The markout
                    // shock below is then applied to the post-trade inventory state.
                    execution = chosen->side == Side::Bid
                        ? spot - problem_.spread() * (0.5 - chosen->delta)
                        : spot + problem_.spread() * (0.5 - chosen->delta);
                    if (chosen->side == Side::Bid) cash -= chosen->size * execution;
                    else cash += chosen->size * execution;
                    cash -= chosen->fee * chosen->size;
                    q += dir * chosen->size;
                    ++trades;
                }

                // An RFQ is an information event whether or not we win it.
                if (tier.use_markout()) {
                    impacts[tier.markout().tau_minutes()] += -dir * tier.markout().asymptotic(chosen->size);
                }

                if (retain && won) {
                    detailed.fills.push_back({t, tier.name(), side_name(chosen->side),
                                              chosen->size, execution, before, q});
                }
            } else {
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
                        }
                        continue;
                    }
                } else if (chosen->ecn) {
                    const auto& venue = *problem_.passive_ecn();

                    // A parent ECN market trade has arrived. Sample its distance D
                    // from the contemporaneous mid/reference in pips. Because
                    // D ~ Exp(k), P(D >= delta) = exp(-k delta), so an active quote
                    // at depth delta has exactly the HJB fill intensity A exp(-k delta).
                    std::exponential_distribution<double> distance_distribution(venue.flow().k());
                    const double trade_distance_pips = distance_distribution(event_rng);
                    const double signed_distance = trade_distance_pips / 10000.0;
                    const double trade_price = chosen->side == Side::Bid
                        ? spot - signed_distance
                        : spot + signed_distance;
                    const bool won = chosen->ecn_quote_active && trade_distance_pips + 1e-12 >= chosen->delta;

                    if (retain) {
                        detailed.ecn_arrivals.push_back({
                            t, side_name(chosen->side), spot, trade_distance_pips, trade_price,
                            chosen->delta, chosen->ecn_quote_active, won
                        });
                    }
                    if (!won) {
                        record_inventory_through(t);
                        if (retain) {
                            detailed.times.push_back(t);
                            detailed.spots.push_back(spot);
                            detailed.inventories.push_back(q);
                            detailed.cashes.push_back(cash);
                        }
                        continue;
                    }

                    // We execute passively at our quote, not at the aggressor's limit.
                    const double quote_distance = chosen->delta / 10000.0;
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

                if (retain) {
                    detailed.fills.push_back({t, chosen->dark ? "Dark pool" : "Passive ECN",
                                              side_name(chosen->side), executed_size, execution, before, q});
                }
            }

            if (retain) {
                // Keep an explicit event point for every RFQ/fill so the price path also
                // records markout shocks from RFQs that we did not win.
                detailed.times.push_back(t);
                detailed.spots.push_back(spot);
                detailed.inventories.push_back(q);
                detailed.cashes.push_back(cash);
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
    }

    result.inventory_lower.resize(sample_points);
    result.inventory_median.resize(sample_points);
    result.inventory_upper.resize(sample_points);
    for (int s = 0; s < sample_points; ++s) {
        result.inventory_lower[s] = percentile(inventories[s], 0.025);
        result.inventory_median[s] = percentile(inventories[s], 0.50);
        result.inventory_upper[s] = percentile(inventories[s], 0.975);
    }
    return result;
}

}  // namespace ladder_pricer
