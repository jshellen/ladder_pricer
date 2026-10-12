#include <cmath>
#include "historical_test_flow.hpp"
#include "ladder_pricer/howard.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ladder_pricer;
static std::vector<double> legacy_ecn_pillars() {
    return {std::exp(-0.875), std::exp(-0.625)-std::exp(-0.875),
            std::exp(-0.375)-std::exp(-0.625), std::exp(-0.175)-std::exp(-0.375),
            1.0-std::exp(-0.175)};
}


std::vector<double> grid() {
    std::vector<double> pos;
    for (double q = 0.0; q <= 3.0 + 1e-12; q += 0.25) pos.push_back(q);
    for (double q = 4.0; q <= 20.0 + 1e-12; q += 1.0) pos.push_back(q);
    std::vector<double> out;
    for (auto it = pos.rbegin(); it != pos.rend() - 1; ++it) out.push_back(-*it);
    out.insert(out.end(), pos.begin(), pos.end());
    return out;
}

static ladder_pricer::DiscreteRfqDistribution explicit_fixture_distribution(
    const HistoricalLogisticFixture& flow, std::vector<double> sizes, double scale=1.) {
    std::vector<double> weights;
    double sum=0;
    for(double z:sizes) {const double w=flow.rfq_arrival_rate(z*scale); weights.push_back(w); sum+=w;}
    for(double& w:weights) w/=sum;
    for(double& z:sizes) z*=scale;
    return ladder_pricer::DiscreteRfqDistribution(2*sum,std::move(sizes),std::move(weights));
}

int main() {
    {
        // Normalized EUR/EURm markout maps to one size-independent price shock.
        SaturatingMarkout markout(2e-4, 0.5);
        assert(std::abs(markout.asymptotic() - 2e-4) < 1e-15);
        assert(markout.expected(0.0) == 0.0);
        assert(markout.expected(0.5) > 0.0);
        assert(markout.expected(0.5) < markout.asymptotic());
        assert(std::abs(markout.expected(100.0) - markout.asymptotic()) < 1e-12);
    }
    {
        const std::vector<double> direct_ecn_size_probabilities{
            std::exp(-0.875), std::exp(-0.625)-std::exp(-0.875),
            std::exp(-0.375)-std::exp(-0.625), std::exp(-0.175)-std::exp(-0.375),
            1.0-std::exp(-0.175)};
        PassiveECN ecn({-0.10, 0.0, 0.25, 0.50},
            AggregatedECNFlow(std::vector<ECNFlowSource>{
                ECNFlowSource("direct", ExponentialFlow(0.15, 2.4),
                              FXMapping{}, direct_ecn_size_probabilities)}),
            1.0, 0.0);
        assert(ecn.side_allowed(2.0, Side::Ask));
        assert(!ecn.side_allowed(2.0, Side::Bid));
        assert(ecn.side_allowed(-2.0, Side::Bid));
        assert(!ecn.side_allowed(-2.0, Side::Ask));
        assert(!ecn.side_allowed(0.0, Side::Bid));
        assert(!ecn.side_allowed(0.0, Side::Ask));
        assert(ecn.risk_reducing(2.0, Side::Ask, 1.0));
        assert(!ecn.risk_reducing(0.5, Side::Ask, 1.0));  // cannot cross through flat
        assert(ecn.risk_reducing(-2.0, Side::Bid, 1.0));
        assert(!ecn.risk_reducing(-0.5, Side::Bid, 1.0));
        // ECN uses the tier delta convention: d=0 touch, d=0.5 mid, with
        // lambda(d)=A exp(-k(0.5-d)).  A is therefore the intensity at mid.
        assert(std::abs(ecn.flow().arrival_rate(0.5) - 0.15) < 1e-15);
        assert(ecn.flow().arrival_rate(0.25) < ecn.flow().arrival_rate(0.5));
        assert(ecn.flow().arrival_rate(0.0) < ecn.flow().arrival_rate(0.25));
        assert(std::abs(ecn.flow().arrival_rate(0.0) - 0.15 * std::exp(-1.2)) < 1e-15);
    }
    {
        const double cross_mid = 1.20;
        const std::vector<double> direct_probs{0.40, 0.20, 0.15, 0.10, 0.15};
        const std::vector<double> crossed_probs{0.20, 0.20, 0.20, 0.20, 0.20};
        ECNFlowSource direct("EURSEK", ExponentialFlow(0.15, 2.4), FXMapping{}, direct_probs);
        ECNFlowSource crossed("USDSEK", ExponentialFlow(0.20, 3.0), FXMapping(1.0, cross_mid), crossed_probs);
        // A crossed quote inherits the master normalized delta exactly.
        assert(std::abs(crossed.implied_delta(0.30) - 0.30) < 1e-15);
        assert(std::abs(crossed.target_reach(0.25) - 0.25) < 1e-15);
        AggregatedECNFlow combined(std::vector<ECNFlowSource>{direct, crossed});
        const double expected = direct.arrival_rate(0.30) + crossed.arrival_rate(0.30);
        assert(std::abs(combined.arrival_rate(0.30) - expected) < 1e-15);
        assert(crossed.arrival_rate(0.30) > 0.0);

        // Trade sizes are discrete source-base pillars and map back to target-base
        // inventory before being capped by our posted target size.
        const double direct_mean = 1.0*0.40 + 0.75*0.20 + 0.50*0.15 + 0.25*0.10 + 0.10*0.15;
        const double crossed_mean = 1.0*0.20 + 0.75*0.20 + 0.50*0.20 + 0.25*0.20 + 0.10*0.20;
        assert(std::abs(direct.mean_target_trade_size() - direct_mean) < 1e-15);
        assert(std::abs(crossed.mean_target_trade_size() - crossed_mean / cross_mid) < 1e-15);
        const double posted = 1.0;
        assert(std::abs(direct.full_fill_probability(posted) - 0.40) < 1e-15);
        assert(std::abs(direct.expected_fill_size(posted) - direct_mean) < 1e-15);

        // The same discrete distribution is used by Monte Carlo sampling.
        std::mt19937_64 rng(123456);
        std::vector<int> sampled_counts(5, 0);
        constexpr int n_samples = 50000;
        const auto& pillars = direct.trade_sizes();
        for (int draw = 0; draw < n_samples; ++draw) {
            const double sampled = direct.sample_trade_size(rng);
            const auto it = std::find_if(pillars.begin(), pillars.end(), [&](double x) {
                return std::abs(x - sampled) < 1e-12;
            });
            assert(it != pillars.end());
            ++sampled_counts[static_cast<std::size_t>(std::distance(pillars.begin(), it))];
        }
        for (std::size_t j = 0; j < direct_probs.size(); ++j) {
            const double frequency = static_cast<double>(sampled_counts[j]) / n_samples;
            assert(std::abs(frequency - direct_probs[j]) < 0.01);
        }

        const auto components = direct.fill_components(posted, {0.25, 0.50, 0.75});
        double probability = 0.0;
        double expected_fill = 0.0;
        for (const auto& [fill, p] : components) {
            assert(fill > 0.0 && fill <= posted + 1e-15);
            probability += p;
            expected_fill += p * fill;
        }
        assert(std::abs(probability - 1.0) < 1e-12);
        assert(std::abs(expected_fill - direct.expected_fill_size(posted)) < 1e-12);
    }
    {
        std::vector<double> deltas;
        for (int i = 0; i <= 50; ++i) deltas.push_back(0.01 * i);
        const std::vector<double> ecn_sizes{1,2,3,5,10,20};
        std::vector<Tier> ecn_tiers;
        ecn_tiers.emplace_back("Tier 1", ecn_sizes, historical_test_flow(HistoricalLogisticFixture(0.0155,0.144,0.0857,0.52,8.42,0.026), ecn_sizes),
            SaturatingMarkout(1e-4, 0.5), true, -10, 100);
        ecn_tiers.emplace_back("Tier 2", ecn_sizes, historical_test_flow(HistoricalLogisticFixture(0.0232,0.303,0.122,0.48,2.86,0.02), ecn_sizes),
            SaturatingMarkout(1e-4, 0.5), true, -10, 100);
        PricingProblem ecn_problem(
            grid(), 0.002, 0.0,
            QuadraticPenalty(0.1, 0.002),
            InternalizationTime(4.0, 0.070, 0.0084),
            std::move(ecn_tiers), std::nullopt,
            PassiveECN(
                deltas,
                AggregatedECNFlow(std::vector<ECNFlowSource>{
                    ECNFlowSource("EURSEK", ExponentialFlow(2.5, 8.4), FXMapping{}, legacy_ecn_pillars()),
                    ECNFlowSource("USDSEK", ExponentialFlow(1.5, 7.0), FXMapping(0.75, 1.18), legacy_ecn_pillars()),
                }),
                1.0, 3e-4));
        const auto ecn_solution = HowardSolver(std::move(ecn_problem)).solve();
        assert(ecn_solution.diagnostics.converged);
        assert(ecn_solution.passive_ecn_policy.has_value());
        const auto& policy = *ecn_solution.passive_ecn_policy;
        auto at = [&](double q) {
            const auto it = std::find_if(policy.q_grid.begin(), policy.q_grid.end(),
                [&](double x) { return std::abs(x - q) < 1e-12; });
            assert(it != policy.q_grid.end());
            return static_cast<std::size_t>(std::distance(policy.q_grid.begin(), it));
        };
        const std::size_t pos = at(2.0);
        const std::size_t neg = at(-2.0);
        const std::size_t zero = at(0.0);
        assert(policy.ask_active[pos]);
        assert(!policy.bid_active[pos]);
        assert(policy.bid_active[neg]);
        assert(!policy.ask_active[neg]);
        assert(!policy.bid_active[zero] && !policy.ask_active[zero]);
        assert(policy.ask_delta[pos] >= 0.0 && policy.ask_delta[pos] <= 0.5);
        assert(policy.bid_delta[neg] >= 0.0 && policy.bid_delta[neg] <= 0.5);
    }
    {
        auto arrivals = std::make_shared<ZeroInflatedPoissonArrival>(0.0012, 2.9, 0.7172);
        DarkPool dark(arrivals, 3e-4, {1.0, 2.0, 3.0, 4.0, 5.0});
        assert(dark.side_allowed(2.0, Side::Ask));
        assert(!dark.side_allowed(2.0, Side::Bid));
        assert(dark.side_allowed(-2.0, Side::Bid));
        assert(!dark.side_allowed(-2.0, Side::Ask));
        assert(!dark.side_allowed(0.0, Side::Bid));
        assert(!dark.side_allowed(0.0, Side::Ask));
        assert(dark.risk_reducing(2.0, Side::Ask, 2.0));
        assert(!dark.risk_reducing(2.0, Side::Ask, 3.0));
        assert(dark.risk_reducing(-2.0, Side::Bid, 2.0));
        assert(!dark.risk_reducing(-2.0, Side::Bid, 3.0));
        assert(std::abs(dark.fee(Side::Bid) - dark.fee(Side::Ask)) < 1e-15);
        assert(std::abs(dark.arrivals(Side::Bid).arrival_intensity() -
                        dark.arrivals(Side::Ask).arrival_intensity()) < 1e-15);
    }
    {
        Tier fee_tier("Fee tier", {1.0},
            historical_test_flow(HistoricalLogisticFixture(0.1,0.1,0.1,0.5,5.0,0.0), {1.0}),
            SaturatingMarkout(0.0, 0.5), false, -10.0, 100.0, 2e-4);
        assert(std::abs(fee_tier.fee() - 2e-4) < 1e-15);
    }
    {
        // RFQ markout is a portfolio mark-to-market on the inventory that
        // remains in each branch: q' on a win, q on a loss.  RFQs blocked by
        // the hard inventory limit still contribute the loss information shock.
        const double spread = 0.002;
        const double delta = 0.20;
        const double fee = 0.0;
        const double impact = 2e-4;
        const double tau_markout = 0.75;
        const InternalizationTime tau_inventory(1.0, 0.2, 0.05);
        const HistoricalLogisticFixture flow(0.5, 0.0, 0.0, 0.30, 5.0, 0.0);
        std::vector<Tier> payoff_tiers;
        payoff_tiers.emplace_back(
            "RFQ payoff", std::vector<double>{1.0}, historical_test_flow(flow, {1.0}),
            SaturatingMarkout(impact, tau_markout), true,
            -1.0, 1.0, fee);
        PricingProblem payoff_problem(
            std::vector<double>{-2.0,-1.0,0.0,1.0,2.0}, spread, 0.0,
            QuadraticPenalty(0.0, 0.0), tau_inventory, std::move(payoff_tiers));

        Policy policy = PolicyBuilder(payoff_problem).initial_policy();
        for (auto& row : policy.tiers[0].bid) row[0] = delta;
        for (auto& row : policy.tiers[0].ask) row[0] = delta;
        const std::vector<double> h(payoff_problem.grid().states().size(), 0.0);
        const auto& states = payoff_problem.grid().states();

        // Synthetic policy-resolvent curve for this branch-payoff unit test:
        // only half of the current inventory remains exposed, in the
        // exponentially weighted sense, while the markout develops.
        MarkoutResolvent resolvent;
        MarkoutExposureCurve curve;
        curve.tau_minutes = tau_markout;
        for (double q : states) curve.effective_inventory.push_back(0.5 * q);
        resolvent.curves.push_back(std::move(curve));
        const auto rhs = BellmanModel(payoff_problem, &resolvent).rhs(h, policy);
        const auto idx = [&](double q) {
            const auto it = std::find_if(states.begin(), states.end(),
                [&](double x) { return std::abs(x-q) < 1e-12; });
            assert(it != states.end());
            return static_cast<std::size_t>(std::distance(states.begin(), it));
        };
        const double rfq_rate = flow.rfq_arrival_rate(1.0);
        const double pwin = flow.win_probability(delta, 1.0);
        const double win_rate = rfq_rate * pwin;
        const double loss_rate = rfq_rate - win_rate;
        const auto markout_pnl = [&](double q_after, double sign) {
            return sign * impact * (0.5 * q_after);
        };
        const double edge = spread * (0.5-delta) - fee;

        // q=+1: Bid RFQ leaves q'=+2 on a win and predicts a down move;
        // Ask RFQ leaves q'=0 and predicts an up move.  The payoff uses the
        // resolvent exposure r_tau(q), not an exogenous tau_internalization(q).
        {
            const double q = 1.0;
            const double bid_win = edge + markout_pnl(2.0, -1.0);
            const double bid_loss = markout_pnl(q, -1.0);
            const double ask_win = edge + markout_pnl(0.0, +1.0);
            const double ask_loss = markout_pnl(q, +1.0);
            const double expected =
                win_rate * bid_win + loss_rate * bid_loss
              + win_rate * ask_win + loss_rate * ask_loss;
            assert(std::abs(rhs[idx(q)] - expected) < 1e-13);
        }

        // q=+3 is the padded hard limit.  A Bid fill is inadmissible, but the
        // customer-sell RFQ still predicts a down move on the unchanged q=+3.
        {
            const double q = payoff_problem.grid().hard_limit();
            assert(std::abs(q - 3.0) < 1e-12);
            const double blocked_bid_loss = markout_pnl(q, -1.0);
            const double ask_q_next = q - 1.0;
            const double ask_win = edge + markout_pnl(ask_q_next, +1.0);
            const double ask_loss = markout_pnl(q, +1.0);
            const double expected =
                rfq_rate * blocked_bid_loss
              + win_rate * ask_win + loss_rate * ask_loss;
            assert(std::abs(rhs[idx(q)] - expected) < 1e-13);
        }
    }
    {
        // The resolvent HJB must be invariant to the legacy exogenous
        // InternalizationTime parameters. They are retained only as a
        // benchmark/diagnostic and no longer price RFQ markout.
        auto make_problem = [](InternalizationTime legacy_tau) {
            std::vector<Tier> tiers;
            tiers.emplace_back(
                "Resolvent tier", std::vector<double>{1.0}, historical_test_flow(HistoricalLogisticFixture(0.35, 0.0, 0.0, 0.32, 5.0, 0.0), std::vector<double>{1.0}),
                SaturatingMarkout(2e-4, 0.8), true,
                -1.0, 1.0, 0.0);
            return PricingProblem(
                std::vector<double>{-2.0,-1.0,0.0,1.0,2.0}, 0.002, 0.0,
                QuadraticPenalty(0.1, 0.002), legacy_tau, std::move(tiers));
        };
        const auto a = HowardSolver(make_problem(InternalizationTime(0.1, 0.0, 0.0))).solve();
        const auto b = HowardSolver(make_problem(InternalizationTime(100.0, 5.0, 1.0))).solve();
        assert(a.diagnostics.converged && b.diagnostics.converged);
        assert(std::abs(a.average_reward - b.average_reward) < 1e-12);
        assert(a.markout_exposure.size() == 1 && b.markout_exposure.size() == 1);
        for (std::size_t i = 0; i < a.value.size(); ++i) {
            assert(std::abs(a.value[i] - b.value[i]) < 1e-11);
            assert(std::abs(a.markout_exposure[0].effective_inventory[i] -
                            b.markout_exposure[0].effective_inventory[i]) < 1e-11);
        }
        const auto& r = a.markout_exposure[0].effective_inventory;
        assert(std::abs(r[r.size()/2]) < 1e-9);
        for (std::size_t i = 0; i < r.size()/2; ++i) {
            assert(std::abs(r[i] + r[r.size()-1-i]) < 1e-9);
        }
    }
    {
        const HistoricalLogisticFixture eur_flow(0.10, 0.2, 0.01, 0.50, 8.0, 0.0);
        const HistoricalLogisticFixture usd_flow(0.20, 0.2, 0.01, 0.50, 6.0, 0.0);
        const double cross_mid = 1.20;
        FlowSource eur("EURSEK", explicit_fixture_distribution(eur_flow, {1.,2.,3.,5.}), historical_win(eur_flow));
        FlowSource usd("USDSEK", explicit_fixture_distribution(usd_flow, {1.,2.,3.,5.}, cross_mid), historical_win(usd_flow), FXMapping(1.,cross_mid));
        const std::vector<double> rfq_sizes{1.0, 2.0, 3.0, 5.0};
        
        
        assert(std::abs(usd.implied_delta(0.30) - 0.30) < 1e-15);
        assert(std::abs(usd.source_size(2.0) - 2.4) < 1e-15);
        AggregatedFlow combined(std::vector<FlowSource>{eur, usd});
        // The calibrated exogenous intensity curve lambda_side(z)=A0*shape(z)
        // defines both the HJB pricing-knot intensities and, after normalization
        // on the separately configured customer RFQ support, the Monte Carlo
        // size distribution.  The MC source clock pools both sides.
        double eur_rfq_total = 0.0;
        double usd_rfq_total = 0.0;
        for (double z : rfq_sizes) {
            eur_rfq_total += eur.rfq_arrival_rate(z);
            usd_rfq_total += usd.rfq_arrival_rate(z);
        }
        assert(std::abs(2.0 * eur_rfq_total - eur.total_rfq_rate()) < 1e-14);
        assert(std::abs(2.0 * usd_rfq_total - usd.total_rfq_rate()) < 1e-14);
        const double expected = eur.arrival_rate(0.30, 2.0) + usd.arrival_rate(0.30, 2.0);
        assert(std::abs(combined.arrival_rate(0.30, 2.0) - expected) < 1e-15);
        const double rfq = eur.rfq_arrival_rate(2.0) + usd.rfq_arrival_rate(2.0);
        assert(std::abs(combined.rfq_arrival_rate(2.0) - rfq) < 1e-15);
        assert(std::abs(combined.win_probability(0.30, 2.0) - expected / rfq) < 1e-15);

        // The RFQ-size PMF is not a second calibration input: it is exactly the
        // normalized exogenous arrival-intensity curve on the configured RFQ support.
        double curve_mass = 0.0;
        for (double z : rfq_sizes) curve_mass += eur_flow.rfq_arrival_rate(z);
        for (std::size_t j = 0; j < rfq_sizes.size(); ++j) {
            const double expected_probability = eur_flow.rfq_arrival_rate(rfq_sizes[j]) / curve_mass;
            assert(std::abs(eur.rfq_distribution().probabilities()[j] - expected_probability) < 1e-15);
            assert(std::abs(eur.rfq_arrival_rate(rfq_sizes[j])
                            - eur_flow.rfq_arrival_rate(rfq_sizes[j])) < 1e-15);
        }

        const double d = combined.optimal_delta(2.0, 0.002, 0.0, -10.0, 100.0);
        assert(std::isfinite(d));
        assert(d >= -10.0 && d <= 100.0);
        auto objective = [&](double x) {
            return combined.arrival_rate(x, 2.0) * (2.0 * 0.002 * (0.5 - x));
        };
        double brute_best = -10.0;
        for (int i = 0; i <= 20000; ++i) {
            const double x = -10.0 + 10.5 * static_cast<double>(i) / 20000.0;
            if (objective(x) > objective(brute_best)) brute_best = x;
        }
        assert(objective(d) + 1e-10 >= objective(brute_best));
    }
    {
        // RFQ-refactor step 1: empirical RFQ size calibration is stored
        // independently of pricing knots.  Reconfiguring the legacy pricing/RFQ
        // support must not mutate Lambda or the calibrated source-size PMF.
        const HistoricalLogisticFixture legacy_win_and_arrival(0.10, 0.2, 0.01, 0.50, 8.0, 0.0);
        DiscreteRfqDistribution calibrated(
            0.72, {1.0, 2.0, 5.0, 10.0}, {0.50, 0.25, 0.15, 0.10});
        FlowSource source("EURSEK", calibrated, historical_win(legacy_win_and_arrival));

        const auto before_sizes = source.rfq_distribution().source_sizes();
        const auto before_probabilities = source.rfq_distribution().probabilities();
        const double before_lambda = source.rfq_distribution().total_rfq_rate();

        assert(source.rfq_distribution().source_sizes() == before_sizes);
        assert(source.rfq_distribution().probabilities() == before_probabilities);
        assert(std::abs(source.rfq_distribution().total_rfq_rate() - before_lambda) < 1e-15);
        double one_side_total = 0.0;
        for (std::size_t i = 0; i < before_probabilities.size(); ++i) {
            one_side_total += source.rfq_distribution().one_side_arrival_rate(i);
        }
        assert(std::abs(one_side_total - 0.5 * before_lambda) < 1e-15);
    }
    {
        // RFQ-refactor step 2: fixed-policy HJB evaluation integrates over the
        // exogenous RFQ distribution and interpolates the quote from pricing
        // knots.  Adding internal pricing knots must therefore leave the HJB
        // value unchanged when the represented quote curve is unchanged.
        auto make_problem = [](std::vector<double> pricing_knots) {
            HistoricalLogisticFixture win_model(0.17, 0.2, 0.01, 0.50, 7.0, 0.01);
            DiscreteRfqDistribution rfq_dist(
                0.80, {1.0, 2.0, 4.0, 5.0}, {0.40, 0.30, 0.20, 0.10});
            AggregatedFlow flow(std::vector<FlowSource>{
                FlowSource("EURSEK", rfq_dist, historical_win(win_model))
            });
            std::vector<Tier> tiers;
            tiers.emplace_back(
                "Tier", std::move(pricing_knots), std::move(flow),
                SaturatingMarkout(0.0, 1.0), false,
                -1.0, 1.0, 0.0, 1.0);
            return PricingProblem(
                std::vector<double>{-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0},
                0.002, 0.0, QuadraticPenalty(0.0, 0.0),
                InternalizationTime(1.0, 0.0, 0.0), std::move(tiers));
        };

        auto coarse = make_problem({1.0, 3.0, 5.0});
        auto fine = make_problem({1.0, 2.0, 3.0, 4.0, 5.0});
        Policy coarse_policy = PolicyBuilder(coarse).initial_policy();
        Policy fine_policy = PolicyBuilder(fine).initial_policy();
        constexpr double d = 0.23;
        for (auto& r : coarse_policy.tiers[0].bid) std::fill(r.begin(), r.end(), d);
        for (auto& r : coarse_policy.tiers[0].ask) std::fill(r.begin(), r.end(), d);
        for (auto& r : fine_policy.tiers[0].bid) std::fill(r.begin(), r.end(), d);
        for (auto& r : fine_policy.tiers[0].ask) std::fill(r.begin(), r.end(), d);

        const std::vector<double> h_coarse(coarse.grid().states().size(), 0.0);
        const std::vector<double> h_fine(fine.grid().states().size(), 0.0);
        const auto rhs_coarse = BellmanModel(coarse).rhs(h_coarse, coarse_policy);
        const auto rhs_fine = BellmanModel(fine).rhs(h_fine, fine_policy);
        assert(rhs_coarse.size() == rhs_fine.size());
        for (std::size_t i = 0; i < rhs_coarse.size(); ++i) {
            assert(std::abs(rhs_coarse[i] - rhs_fine[i]) < 1e-13);
        }
    }

    {
        // RFQ-refactor step 3: pricing knots are optimized jointly through the
        // interpolated quote curve.  Put all RFQ mass at 4M while the pricing
        // curve has knots only at 1M, 3M and 5M.  The optimized interpolated
        // 4M quote must match the single-RFQ optimum even though 4M is not a
        // pricing knot.
        const HistoricalLogisticFixture win_model(0.10, 0.2, 0.01, 0.50, 8.0, 0.0);
        const DiscreteRfqDistribution rfq_dist(0.40, {4.0}, {1.0});
        AggregatedFlow flow(std::vector<FlowSource>{
            FlowSource("EURSEK", rfq_dist, historical_win(win_model))
        });
        std::vector<Tier> local_tiers;
        local_tiers.emplace_back(
            "Tier", std::vector<double>{1.0, 3.0, 5.0}, std::move(flow),
            SaturatingMarkout(0.0, 1.0), false,
            -1.0, 1.0, 0.0, 1.0);
        PricingProblem local_problem(
            std::vector<double>{-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0},
            0.002, 0.0, QuadraticPenalty(0.0, 0.0),
            InternalizationTime(1.0, 0.0, 0.0), std::move(local_tiers));
        PolicyBuilder builder(local_problem);
        Policy previous = builder.initial_policy();
        const std::vector<double> zero(local_problem.grid().states().size(), 0.0);
        const Policy improved = builder.improve(zero, previous);
        const std::size_t q0 = local_problem.grid().zero_index();
        const auto& row = improved.tiers[0].bid[q0];
        const double implied_4m = 0.5 * (row[1] + row[2]);
        const double expected = win_model.optimal_delta(4.0, 0.002, 0.0);
        assert(std::abs(implied_4m - expected) < 2e-5);
    }

    const std::vector<double> sizes{1,2,3,5,10,20};
    std::vector<Tier> tiers;
    tiers.emplace_back("Tier 1", sizes, historical_test_flow(HistoricalLogisticFixture(0.0155,0.144,0.0857,0.52,8.42,0.026), sizes),
        SaturatingMarkout(1e-4, 0.5), true, -10, 100);
    tiers.emplace_back("Tier 2", sizes, historical_test_flow(HistoricalLogisticFixture(0.0232,0.303,0.122,0.48,2.86,0.02), sizes),
        SaturatingMarkout(1e-4, 0.5), true, -10, 100);
    tiers.emplace_back("Tier 3", std::vector<double>{1}, historical_test_flow(HistoricalLogisticFixture(1.0,0.144,0.0857,0.52,20.0,0.026), std::vector<double>{1}),
        SaturatingMarkout(4e-4, 0.10), true, -10, 100);

    PricingProblem problem(grid(), 0.002, 0.0,
        QuadraticPenalty(0.1, 0.002),
        InternalizationTime(4.0,0.070,0.0084), tiers);
    const auto solution = HowardSolver(std::move(problem)).solve();
    std::cout << "converged=" << solution.diagnostics.converged
              << " iterations=" << solution.diagnostics.iterations
              << " rho=" << solution.average_reward << "\n";
    assert(solution.diagnostics.converged);
    // The final diagnostic is evaluated after operational policy snapping.
    const auto& diagnostic = solution.diagnostics;
    assert(diagnostic.final_bellman_residual_by_state.size() ==
           solution.solve_q_grid.size() * solution.volatility_states.size());
    double measured_max = 0.0;
    for (double e : diagnostic.final_bellman_residual_by_state) {
        assert(std::isfinite(e));
        measured_max = std::max(measured_max, std::abs(e));
    }
    assert(std::abs(measured_max - diagnostic.final_bellman_residual) < 1e-10);
    // Runtime check also executes in Release builds (where assert is disabled).
    const auto max_it = std::max_element(
        diagnostic.final_bellman_residual_by_state.begin(),
        diagnostic.final_bellman_residual_by_state.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    const std::size_t worst = static_cast<std::size_t>(
        std::distance(diagnostic.final_bellman_residual_by_state.begin(), max_it));
    const std::size_t nq = solution.solve_q_grid.size();
    if (diagnostic.final_bellman_residual_by_state.empty() ||
        !std::isfinite(diagnostic.final_bellman_residual) ||
        std::abs(measured_max - diagnostic.final_bellman_residual) > 1e-10 ||
        diagnostic.worst_inventory != solution.solve_q_grid[worst % nq] ||
        diagnostic.worst_volatility_index != worst / nq) {
        throw std::runtime_error("Howard final Bellman residual diagnostics inconsistent");
    }
    assert(std::find(solution.solve_q_grid.begin(), solution.solve_q_grid.end(),
                     diagnostic.worst_inventory) != solution.solve_q_grid.end());
    assert(diagnostic.worst_volatility_index < solution.volatility_states.size());
    assert(std::isfinite(solution.average_reward));
    assert(solution.value.size() == grid().size());
    assert(std::abs(solution.value[solution.value.size()/2]) < 1e-8);

    {
        // Joint (inventory, volatility) Howard regression.  Volatility is an
        // exogenous CTMC state and must expand values/policies without changing
        // the inventory grid representation exposed to pricing.
        std::vector<Tier> vol_tiers;
        vol_tiers.emplace_back("Vol tier", std::vector<double>{1,2,3,5}, historical_test_flow(HistoricalLogisticFixture(0.03,0.15,0.03,0.50,7.0,0.01), std::vector<double>{1,2,3,5}),
            SaturatingMarkout(1e-4, 0.5), true, -10, 100);
        const std::vector<double> sigmas{0.0006, 0.0010, 0.0018};
        const std::vector<std::vector<double>> qvol{
            {-0.04, 0.04, 0.00},
            { 0.02,-0.05, 0.03},
            { 0.00, 0.05,-0.05},
        };
        PricingProblem vol_problem(
            grid(), 0.002, 0.0, QuadraticPenalty(0.75, sigmas[1]),
            VolatilityModel(sigmas, qvol, 1),
            InternalizationTime(4.0,0.070,0.0084), std::move(vol_tiers));
        const auto vol_solution = HowardSolver(std::move(vol_problem)).solve();
        assert(vol_solution.diagnostics.converged);
        assert(vol_solution.volatility_states.size() == 3);
        assert(vol_solution.value_by_volatility.size() == 3);
        assert(vol_solution.tier_policies_by_volatility.size() == 3);
        assert(vol_solution.solve_tier_policies_by_volatility.size() == 3);
        for (const auto& values : vol_solution.value_by_volatility) {
            assert(values.size() == grid().size());
        }
        assert(vol_solution.value == vol_solution.value_by_volatility[1]);
        assert(!vol_solution.markout_exposure_joint.empty());
        assert(vol_solution.markout_exposure_joint.front().effective_inventory.size()
               == 3 * grid().size());
    }
    return 0;
}
