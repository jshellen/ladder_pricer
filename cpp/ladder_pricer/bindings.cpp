#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hjb_ladder.hpp"

namespace py = pybind11;
using namespace hjb;

namespace {

Side parse_side(const std::string& side) {
    if (side == "bid" || side == "Bid") return Side::Bid;
    if (side == "ask" || side == "Ask") return Side::Ask;
    throw std::invalid_argument("Unknown side: " + side);
}

QuotePolicy build_policy_for_tier(
    const PriceTier& tier,
    const SolverConfig& config,
    const std::vector<double>& h_vec
) {
    tier.validate();
    config.validate();

    if (h_vec.size() != config.q_grid.size()) {
        throw std::invalid_argument(
            "build_policy_for_tier: h_vec size must match config.q_grid size."
        );
    }

    LinearInterpolator1D h_view{&config.q_grid, &h_vec};
    RungBuilder builder(
        tier,
        config.spread,
        config.golden_tol,
        config.golden_max_iter,
        h_view
    );

    QuotePolicy policy;
    policy.reset_shape(config.q_grid, tier.sizes);
    builder.build_policy_inplace(policy);
    return policy;
}

void build_policy_inplace_for_tier(
    PriceTier& tier,
    const SolverConfig& config,
    const std::vector<double>& h_vec
) {
    tier.validate();
    config.validate();

    if (h_vec.size() != config.q_grid.size()) {
        throw std::invalid_argument(
            "build_policy_inplace_for_tier: h_vec size must match config.q_grid size."
        );
    }

    LinearInterpolator1D h_view{&config.q_grid, &h_vec};
    RungBuilder builder(
        tier,
        config.spread,
        config.golden_tol,
        config.golden_max_iter,
        h_view
    );

    tier.policy.reset_shape(config.q_grid, tier.sizes);
    builder.build_policy_inplace(tier.policy);
}

} // namespace

PYBIND11_MODULE(ladder_pricer, m) {
    m.doc() = "Inventory-based HJB ladder pricer";

    py::enum_<Side>(m, "Side")
        .value("Bid", Side::Bid)
        .value("Ask", Side::Ask)
        .export_values();

    py::class_<FlowCurve, std::shared_ptr<FlowCurve>>(m, "FlowCurve")
        .def("hit_ratio", &FlowCurve::hit_ratio)
        .def("arrival_rate", &FlowCurve::arrival_rate);

    py::class_<MarkoutModel, std::shared_ptr<MarkoutModel>>(m, "MarkoutModel")
        .def("expected_markout", &MarkoutModel::expected_markout);

    py::class_<InventoryPenalty, std::shared_ptr<InventoryPenalty>>(m, "InventoryPenalty")
        .def("value", &InventoryPenalty::value);

    py::class_<LogisticFlowCurve, FlowCurve, std::shared_ptr<LogisticFlowCurve>>(
        m, "LogisticFlowCurve"
    )
        .def(
            py::init<double, double, double, double, double, double>(),
            py::arg("A0") = 1.0,
            py::arg("theta") = 0.0,
            py::arg("shift") = 0.20,
            py::arg("steepness") = 10.0,
            py::arg("volume_shift") = 0.05,
            py::arg("z_floor") = 1e-8
        )
        .def_readwrite("A0", &LogisticFlowCurve::A0)
        .def_readwrite("theta", &LogisticFlowCurve::theta)
        .def_readwrite("shift", &LogisticFlowCurve::shift)
        .def_readwrite("steepness", &LogisticFlowCurve::steepness)
        .def_readwrite("volume_shift", &LogisticFlowCurve::volume_shift)
        .def_readwrite("z_floor", &LogisticFlowCurve::z_floor)
        .def("A", &LogisticFlowCurve::A)
        .def("hit_ratio", &LogisticFlowCurve::hit_ratio)
        .def("arrival_rate", &LogisticFlowCurve::arrival_rate);

    py::class_<SqrtMarkoutModel, MarkoutModel, std::shared_ptr<SqrtMarkoutModel>>(
        m, "SqrtMarkoutModel"
    )
        .def(
            py::init<double, double>(),
            py::arg("base") = 0.005,
            py::arg("coeff") = 0.003
        )
        .def_readwrite("base", &SqrtMarkoutModel::base)
        .def_readwrite("coeff", &SqrtMarkoutModel::coeff)
        .def("expected_markout", &SqrtMarkoutModel::expected_markout);

    py::class_<
        PolynomialInventoryPenalty,
        InventoryPenalty,
        std::shared_ptr<PolynomialInventoryPenalty>
    >(m, "PolynomialInventoryPenalty")
        .def(
            py::init<double, double, double, double, double>(),
            py::arg("risk_aversion") = 2.0,
            py::arg("sigma") = 0.25,
            py::arg("tau0") = 2.0,
            py::arg("cubic_coeff") = 0.1,
            py::arg("quartic_coeff") = 0.0015
        )
        .def_readwrite("risk_aversion", &PolynomialInventoryPenalty::risk_aversion)
        .def_readwrite("sigma", &PolynomialInventoryPenalty::sigma)
        .def_readwrite("tau0", &PolynomialInventoryPenalty::tau0)
        .def_readwrite("cubic_coeff", &PolynomialInventoryPenalty::cubic_coeff)
        .def_readwrite("quartic_coeff", &PolynomialInventoryPenalty::quartic_coeff)
        .def("value", &PolynomialInventoryPenalty::value);

    py::class_<QuotePolicy>(m, "QuotePolicy")
        .def(py::init<>())
        .def(
            py::init<std::vector<double>, std::vector<double>, Matrix, Matrix>(),
            py::arg("q_grid"),
            py::arg("sizes"),
            py::arg("bid"),
            py::arg("ask")
        )
        .def_readwrite("q_grid", &QuotePolicy::q_grid)
        .def_readwrite("sizes", &QuotePolicy::sizes)
        .def_readwrite("bid", &QuotePolicy::bid)
        .def_readwrite("ask", &QuotePolicy::ask)
        .def("validate", &QuotePolicy::validate)
        .def("reset_shape", &QuotePolicy::reset_shape, py::arg("q_grid"), py::arg("sizes"))

        .def(
            "delta_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, Side side) {
                return self.delta_at_index(i, j, side);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side")
        )
        .def(
            "delta_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, const std::string& side) {
                return self.delta_at_index(i, j, parse_side(side));
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side")
        )

        .def(
            "quote",
            [](const QuotePolicy& self, double q, double z, Side side) {
                return self.quote(q, z, side);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side")
        )
        .def(
            "quote",
            [](const QuotePolicy& self, double q, double z, const std::string& side) {
                return self.quote(q, z, parse_side(side));
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side")
        )

        .def(
            "reference_delta_at_index",
            [](const QuotePolicy& self, std::size_t i, Side side) {
                return self.reference_delta_at_index(i, side);
            },
            py::arg("i"),
            py::arg("side")
        )
        .def(
            "reference_delta_at_index",
            [](const QuotePolicy& self, std::size_t i, const std::string& side) {
                return self.reference_delta_at_index(i, parse_side(side));
            },
            py::arg("i"),
            py::arg("side")
        )
        .def(
            "reference_delta",
            [](const QuotePolicy& self, double q, Side side) {
                return self.reference_delta(q, side);
            },
            py::arg("q"),
            py::arg("side")
        )
        .def(
            "reference_delta",
            [](const QuotePolicy& self, double q, const std::string& side) {
                return self.reference_delta(q, parse_side(side));
            },
            py::arg("q"),
            py::arg("side")
        )

        .def_static(
            "price_improvement_pips_from_delta",
            &QuotePolicy::price_improvement_pips_from_delta,
            py::arg("delta"),
            py::arg("spread")
        )
        .def_static(
            "quote_relative_to_mid_pips_from_delta",
            [](double delta, Side side, double spread) {
                return QuotePolicy::quote_relative_to_mid_pips_from_delta(delta, side, spread);
            },
            py::arg("delta"),
            py::arg("side"),
            py::arg("spread")
        )
        .def_static(
            "quote_relative_to_mid_pips_from_delta",
            [](double delta, const std::string& side, double spread) {
                return QuotePolicy::quote_relative_to_mid_pips_from_delta(
                    delta, parse_side(side), spread
                );
            },
            py::arg("delta"),
            py::arg("side"),
            py::arg("spread")
        )
        .def_static(
            "distance_to_mid_pips_from_delta",
            &QuotePolicy::distance_to_mid_pips_from_delta,
            py::arg("delta"),
            py::arg("spread")
        )
        .def_static(
            "quote_price_from_delta",
            [](double mid, double delta, Side side, double spread) {
                return QuotePolicy::quote_price_from_delta(mid, delta, side, spread);
            },
            py::arg("mid"),
            py::arg("delta"),
            py::arg("side"),
            py::arg("spread")
        )
        .def_static(
            "quote_price_from_delta",
            [](double mid, double delta, const std::string& side, double spread) {
                return QuotePolicy::quote_price_from_delta(mid, delta, parse_side(side), spread);
            },
            py::arg("mid"),
            py::arg("delta"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "price_improvement_pips_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, Side side, double spread) {
                return self.price_improvement_pips_at_index(i, j, side, spread);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "price_improvement_pips_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, const std::string& side, double spread) {
                return self.price_improvement_pips_at_index(i, j, parse_side(side), spread);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "quote_relative_to_mid_pips_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, Side side, double spread) {
                return self.quote_relative_to_mid_pips_at_index(i, j, side, spread);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "quote_relative_to_mid_pips_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, const std::string& side, double spread) {
                return self.quote_relative_to_mid_pips_at_index(i, j, parse_side(side), spread);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "distance_to_mid_pips_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, Side side, double spread) {
                return self.distance_to_mid_pips_at_index(i, j, side, spread);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "distance_to_mid_pips_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, const std::string& side, double spread) {
                return self.distance_to_mid_pips_at_index(i, j, parse_side(side), spread);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "volume_premium_pips_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, Side side, double spread) {
                return self.volume_premium_pips_at_index(i, j, side, spread);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "volume_premium_pips_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, const std::string& side, double spread) {
                return self.volume_premium_pips_at_index(i, j, parse_side(side), spread);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "quote_price_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, Side side, double mid, double spread) {
                return self.quote_price_at_index(i, j, side, mid, spread);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side"),
            py::arg("mid"),
            py::arg("spread")
        )
        .def(
            "quote_price_at_index",
            [](const QuotePolicy& self, std::size_t i, std::size_t j, const std::string& side, double mid, double spread) {
                return self.quote_price_at_index(i, j, parse_side(side), mid, spread);
            },
            py::arg("i"),
            py::arg("j"),
            py::arg("side"),
            py::arg("mid"),
            py::arg("spread")
        )

        .def(
            "price_improvement_pips",
            [](const QuotePolicy& self, double q, double z, Side side, double spread) {
                return self.price_improvement_pips(q, z, side, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "price_improvement_pips",
            [](const QuotePolicy& self, double q, double z, const std::string& side, double spread) {
                return self.price_improvement_pips(q, z, parse_side(side), spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "quote_relative_to_mid_pips",
            [](const QuotePolicy& self, double q, double z, Side side, double spread) {
                return self.quote_relative_to_mid_pips(q, z, side, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "quote_relative_to_mid_pips",
            [](const QuotePolicy& self, double q, double z, const std::string& side, double spread) {
                return self.quote_relative_to_mid_pips(q, z, parse_side(side), spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "distance_to_mid_pips",
            [](const QuotePolicy& self, double q, double z, Side side, double spread) {
                return self.distance_to_mid_pips(q, z, side, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "distance_to_mid_pips",
            [](const QuotePolicy& self, double q, double z, const std::string& side, double spread) {
                return self.distance_to_mid_pips(q, z, parse_side(side), spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "volume_premium_pips",
            [](const QuotePolicy& self, double q, double z, Side side, double spread) {
                return self.volume_premium_pips(q, z, side, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "volume_premium_pips",
            [](const QuotePolicy& self, double q, double z, const std::string& side, double spread) {
                return self.volume_premium_pips(q, z, parse_side(side), spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "quote_price",
            [](const QuotePolicy& self, double q, double z, Side side, double mid, double spread) {
                return self.quote_price(q, z, side, mid, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("mid"),
            py::arg("spread")
        )
        .def(
            "quote_price",
            [](const QuotePolicy& self, double q, double z, const std::string& side, double mid, double spread) {
                return self.quote_price(q, z, parse_side(side), mid, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("mid"),
            py::arg("spread")
        );

    py::class_<PriceTier, std::shared_ptr<PriceTier>>(m, "PriceTier")
        .def(py::init<>())
        .def(
            py::init<
                std::string,
                std::vector<double>,
                std::shared_ptr<FlowCurve>,
                std::shared_ptr<MarkoutModel>,
                double,
                double
            >(),
            py::arg("name"),
            py::arg("sizes"),
            py::arg("flow_curve"),
            py::arg("markout_model"),
            py::arg("delta_min") = -5.0,
            py::arg("delta_max") = 5.0
        )
        .def_readwrite("name", &PriceTier::name)
        .def_readwrite("sizes", &PriceTier::sizes)
        .def_readwrite("flow_curve", &PriceTier::flow_curve)
        .def_readwrite("markout_model", &PriceTier::markout_model)
        .def_readwrite("delta_min", &PriceTier::delta_min)
        .def_readwrite("delta_max", &PriceTier::delta_max)
        .def_readwrite("policy", &PriceTier::policy)
        .def("validate", &PriceTier::validate)
        .def("arrival_rate", &PriceTier::arrival_rate, py::arg("delta"), py::arg("z"))
        .def("expected_markout", &PriceTier::expected_markout, py::arg("z"))

        .def(
            "quote",
            [](const PriceTier& self, double q, double z, Side side) {
                return self.quote(q, z, side);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side")
        )
        .def(
            "quote",
            [](const PriceTier& self, double q, double z, const std::string& side) {
                return self.quote(q, z, parse_side(side));
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side")
        )

        .def(
            "quote_price",
            [](const PriceTier& self, double q, double z, Side side, double mid, double spread) {
                return self.quote_price(q, z, side, mid, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("mid"),
            py::arg("spread")
        )
        .def(
            "quote_price",
            [](const PriceTier& self, double q, double z, const std::string& side, double mid, double spread) {
                return self.quote_price(q, z, parse_side(side), mid, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("mid"),
            py::arg("spread")
        )

        .def(
            "price_improvement_pips",
            [](const PriceTier& self, double q, double z, Side side, double spread) {
                return self.price_improvement_pips(q, z, side, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "price_improvement_pips",
            [](const PriceTier& self, double q, double z, const std::string& side, double spread) {
                return self.price_improvement_pips(q, z, parse_side(side), spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "quote_relative_to_mid_pips",
            [](const PriceTier& self, double q, double z, Side side, double spread) {
                return self.quote_relative_to_mid_pips(q, z, side, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "quote_relative_to_mid_pips",
            [](const PriceTier& self, double q, double z, const std::string& side, double spread) {
                return self.quote_relative_to_mid_pips(q, z, parse_side(side), spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "distance_to_mid_pips",
            [](const PriceTier& self, double q, double z, Side side, double spread) {
                return self.distance_to_mid_pips(q, z, side, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "distance_to_mid_pips",
            [](const PriceTier& self, double q, double z, const std::string& side, double spread) {
                return self.distance_to_mid_pips(q, z, parse_side(side), spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )

        .def(
            "volume_premium_pips",
            [](const PriceTier& self, double q, double z, Side side, double spread) {
                return self.volume_premium_pips(q, z, side, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        )
        .def(
            "volume_premium_pips",
            [](const PriceTier& self, double q, double z, const std::string& side, double spread) {
                return self.volume_premium_pips(q, z, parse_side(side), spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("spread")
        );

    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("q_grid", &SolverConfig::q_grid)
        .def_readwrite("dt", &SolverConfig::dt)
        .def_readwrite("n_iter", &SolverConfig::n_iter)
        .def_readwrite("spot_drift", &SolverConfig::spot_drift)
        .def_readwrite("spread", &SolverConfig::spread)
        .def_readwrite("golden_tol", &SolverConfig::golden_tol)
        .def_readwrite("golden_max_iter", &SolverConfig::golden_max_iter)
        .def_readwrite("early_stop", &SolverConfig::early_stop)
        .def_readwrite("tol_h", &SolverConfig::tol_h)
        .def_readwrite("tol_rhs", &SolverConfig::tol_rhs)
        .def_readwrite("min_iter", &SolverConfig::min_iter)
        .def_readwrite("consecutive_passes_required", &SolverConfig::consecutive_passes_required)
        .def("validate", &SolverConfig::validate);

    py::class_<SolverDiagnostics>(m, "SolverDiagnostics")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("n_iter"))
        .def_readwrite("converged", &SolverDiagnostics::converged)
        .def_readwrite("iterations_used", &SolverDiagnostics::iterations_used)
        .def_readwrite("final_max_h_change", &SolverDiagnostics::final_max_h_change)
        .def_readwrite("final_max_rhs", &SolverDiagnostics::final_max_rhs)
        .def_readwrite("consecutive_passes", &SolverDiagnostics::consecutive_passes)
        .def_readwrite("history_max_h_change", &SolverDiagnostics::history_max_h_change)
        .def_readwrite("history_max_rhs", &SolverDiagnostics::history_max_rhs)
        .def("reserve", &SolverDiagnostics::reserve, py::arg("n_iter"))
        .def(
            "record_iteration",
            &SolverDiagnostics::record_iteration,
            py::arg("iteration"),
            py::arg("max_h_change"),
            py::arg("max_rhs")
        )
        .def(
            "update_stopping_state",
            &SolverDiagnostics::update_stopping_state,
            py::arg("passes_now"),
            py::arg("early_stop"),
            py::arg("consecutive_passes_required")
        );

    py::class_<HJBSolution>(m, "HJBSolution")
        .def(py::init<>())
        .def_readwrite("h", &HJBSolution::h)
        .def_readwrite("q_grid", &HJBSolution::q_grid)
        .def_readwrite("tiers", &HJBSolution::tiers)
        .def_readwrite("diagnostics", &HJBSolution::diagnostics);

    py::class_<HJBLadderSolver>(m, "HJBLadderSolver")
        .def(
            py::init<
                SolverConfig,
                std::shared_ptr<InventoryPenalty>,
                std::vector<std::shared_ptr<PriceTier>>
            >(),
            py::arg("config"),
            py::arg("penalty"),
            py::arg("tiers")
        )
        .def_readwrite("config", &HJBLadderSolver::config)
        .def_readwrite("penalty", &HJBLadderSolver::penalty)
        .def_readwrite("tiers", &HJBLadderSolver::tiers)
        .def("update_policies", &HJBLadderSolver::update_policies, py::arg("h_vec"))
        .def(
            "bellman_rhs_from_policies",
            [](const HJBLadderSolver& self, const std::vector<double>& h_vec) {
                std::vector<double> rhs;
                self.bellman_rhs_from_policies(h_vec, rhs);
                return rhs;
            },
            py::arg("h_vec")
        )
        .def("solve", &HJBLadderSolver::solve);

    m.def(
        "build_policy_for_tier",
        &build_policy_for_tier,
        py::arg("tier"),
        py::arg("config"),
        py::arg("h_vec")
    );

    m.def(
        "build_policy_inplace_for_tier",
        &build_policy_inplace_for_tier,
        py::arg("tier"),
        py::arg("config"),
        py::arg("h_vec")
    );
}