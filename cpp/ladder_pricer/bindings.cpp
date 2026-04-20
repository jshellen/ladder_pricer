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

TierRole parse_tier_role(const std::string& role) {
    if (role == "customer" || role == "Customer") return TierRole::Customer;
    if (role == "ecn" || role == "ECN" || role == "Ecn") return TierRole::ECN;
    throw std::invalid_argument("Unknown tier role: " + role);
}

} // namespace

PYBIND11_MODULE(ladder_pricer, m) {
    m.doc() = "Inventory-based HJB ladder pricer";

    py::enum_<Side>(m, "Side")
        .value("Bid", Side::Bid)
        .value("Ask", Side::Ask)
        .export_values();

    py::enum_<TierRole>(m, "TierRole")
        .value("Customer", TierRole::Customer)
        .value("ECN", TierRole::ECN)
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

    py::class_<ExponentialECNPolicy>(m, "ExponentialECNPolicy")
        .def(py::init<>())
        .def(
            py::init<double, double, double, double, double>(),
            py::arg("delta_start"),
            py::arg("delta_target"),
            py::arg("decay"),
            py::arg("decay_min"),
            py::arg("decay_max")
        )
        .def_readwrite("delta_start", &ExponentialECNPolicy::delta_start)
        .def_readwrite("delta_target", &ExponentialECNPolicy::delta_target)
        .def_readwrite("decay", &ExponentialECNPolicy::decay)
        .def_readwrite("decay_min", &ExponentialECNPolicy::decay_min)
        .def_readwrite("decay_max", &ExponentialECNPolicy::decay_max)
        .def("validate", &ExponentialECNPolicy::validate)
        .def(
            "delta_at_abs_inventory",
            &ExponentialECNPolicy::delta_at_abs_inventory,
            py::arg("q_abs"),
            py::arg("decay_override") = -1.0
        );

    py::class_<QuoteSummary>(m, "QuoteSummary")
        .def(py::init<>())
        .def_readwrite("delta", &QuoteSummary::delta)
        .def_readwrite("price_improvement_frac", &QuoteSummary::price_improvement_frac)
        .def_readwrite("price_improvement_pct_of_spread", &QuoteSummary::price_improvement_pct_of_spread)
        .def_readwrite("price_improvement_pips", &QuoteSummary::price_improvement_pips)
        .def_readwrite("quote_relative_to_mid", &QuoteSummary::quote_relative_to_mid)
        .def_readwrite("quote_relative_to_mid_pips", &QuoteSummary::quote_relative_to_mid_pips)
        .def_readwrite("distance_to_mid", &QuoteSummary::distance_to_mid)
        .def_readwrite("distance_to_mid_pips", &QuoteSummary::distance_to_mid_pips)
        .def_readwrite("reference_delta", &QuoteSummary::reference_delta)
        .def_readwrite("volume_premium_pips", &QuoteSummary::volume_premium_pips)
        .def_readwrite("quote_price", &QuoteSummary::quote_price);

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
            "delta",
            [](const QuotePolicy& self, double q, double z, Side side) {
                return self.delta(q, z, side);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side")
        )
        .def(
            "delta",
            [](const QuotePolicy& self, double q, double z, const std::string& side) {
                return self.delta(q, z, parse_side(side));
            },
            py::arg("q"),
            py::arg("z"),
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
        .def(
            "quote_summary",
            [](const QuotePolicy& self, double q, double z, Side side, double mid, double spread) {
                return self.quote_summary(q, z, side, mid, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("mid"),
            py::arg("spread")
        )
        .def(
            "quote_summary",
            [](const QuotePolicy& self, double q, double z, const std::string& side, double mid, double spread) {
                return self.quote_summary(q, z, parse_side(side), mid, spread);
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
                double,
                TierRole,
                ExponentialECNPolicy
            >(),
            py::arg("name"),
            py::arg("sizes"),
            py::arg("flow_curve"),
            py::arg("markout_model"),
            py::arg("delta_min") = -5.0,
            py::arg("delta_max") = 5.0,
            py::arg("role") = TierRole::Customer,
            py::arg("ecn_policy") = ExponentialECNPolicy{}
        )
        .def(
            py::init([](
                const std::string& name,
                const std::vector<double>& sizes,
                const std::shared_ptr<FlowCurve>& flow_curve,
                const std::shared_ptr<MarkoutModel>& markout_model,
                double delta_min,
                double delta_max,
                const std::string& role,
                const ExponentialECNPolicy& ecn_policy
            ) {
                return PriceTier(
                    name,
                    sizes,
                    flow_curve,
                    markout_model,
                    delta_min,
                    delta_max,
                    parse_tier_role(role),
                    ecn_policy
                );
            }),
            py::arg("name"),
            py::arg("sizes"),
            py::arg("flow_curve"),
            py::arg("markout_model"),
            py::arg("delta_min") = -5.0,
            py::arg("delta_max") = 5.0,
            py::arg("role") = std::string("customer"),
            py::arg("ecn_policy") = ExponentialECNPolicy{}
        )
        .def_readwrite("name", &PriceTier::name)
        .def_readwrite("role", &PriceTier::role)
        .def_readwrite("sizes", &PriceTier::sizes)
        .def_readwrite("flow_curve", &PriceTier::flow_curve)
        .def_readwrite("markout_model", &PriceTier::markout_model)
        .def_readwrite("delta_min", &PriceTier::delta_min)
        .def_readwrite("delta_max", &PriceTier::delta_max)
        .def_readwrite("ecn_policy", &PriceTier::ecn_policy)
        .def_readwrite("policy", &PriceTier::policy)
        .def("validate", &PriceTier::validate)
        .def("is_customer", &PriceTier::is_customer)
        .def("is_ecn", &PriceTier::is_ecn)
        .def("arrival_rate", &PriceTier::arrival_rate, py::arg("delta"), py::arg("z"))
        .def("expected_markout", &PriceTier::expected_markout, py::arg("z"))
        .def(
            "ecn_active_delta",
            &PriceTier::ecn_active_delta,
            py::arg("q"),
            py::arg("decay_override") = -1.0
        )
        .def_static(
            "next_inventory",
            &PriceTier::next_inventory,
            py::arg("q"),
            py::arg("z"),
            py::arg("side")
        )
        .def_static(
            "next_inventory",
            [](double q, double z, const std::string& side) {
                return PriceTier::next_inventory(q, z, parse_side(side));
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side")
        )
        .def_static(
            "is_inventory_reducing_side",
            &PriceTier::is_inventory_reducing_side,
            py::arg("q"),
            py::arg("side"),
            py::arg("tol") = 1e-12
        )
        .def_static(
            "is_inventory_reducing_side",
            [](double q, const std::string& side, double tol) {
                return PriceTier::is_inventory_reducing_side(q, parse_side(side), tol);
            },
            py::arg("q"),
            py::arg("side"),
            py::arg("tol") = 1e-12
        )
        .def_static(
            "size_does_not_cross_flat",
            &PriceTier::size_does_not_cross_flat,
            py::arg("q"),
            py::arg("z"),
            py::arg("tol") = 1e-12
        )
        .def(
            "is_admissible",
            [](const PriceTier& self, double q, double z, Side side, double tol) {
                return self.is_admissible(q, z, side, tol);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("tol") = 1e-12
        )
        .def(
            "is_admissible",
            [](const PriceTier& self, double q, double z, const std::string& side, double tol) {
                return self.is_admissible(q, z, parse_side(side), tol);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("tol") = 1e-12
        )
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
            "quote_summary",
            [](const PriceTier& self, double q, double z, Side side, double mid, double spread) {
                return self.quote_summary(q, z, side, mid, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("mid"),
            py::arg("spread")
        )
        .def(
            "quote_summary",
            [](const PriceTier& self, double q, double z, const std::string& side, double mid, double spread) {
                return self.quote_summary(q, z, parse_side(side), mid, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("mid"),
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

    py::class_<GoldenSectionSearch>(m, "GoldenSectionSearch")
        .def(py::init<>())
        .def(py::init<double, int>(), py::arg("tol"), py::arg("max_iter"))
        .def_readwrite("tol", &GoldenSectionSearch::tol)
        .def_readwrite("max_iter", &GoldenSectionSearch::max_iter)
        .def("validate", &GoldenSectionSearch::validate);

    py::class_<LadderBoundsPolicy>(m, "LadderBoundsPolicy")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("delta_min"), py::arg("delta_max"))
        .def_readwrite("delta_min", &LadderBoundsPolicy::delta_min)
        .def_readwrite("delta_max", &LadderBoundsPolicy::delta_max)
        .def("validate", &LadderBoundsPolicy::validate)
        .def_static("is_shrink_mode", &LadderBoundsPolicy::is_shrink_mode, py::arg("q"), py::arg("side"))
        .def_static("is_expand_mode", &LadderBoundsPolicy::is_expand_mode, py::arg("q"), py::arg("side"))
        .def_static(
            "is_shrink_mode",
            [](double q, const std::string& side) {
                return LadderBoundsPolicy::is_shrink_mode(q, parse_side(side));
            },
            py::arg("q"),
            py::arg("side")
        )
        .def_static(
            "is_expand_mode",
            [](double q, const std::string& side) {
                return LadderBoundsPolicy::is_expand_mode(q, parse_side(side));
            },
            py::arg("q"),
            py::arg("side")
        );

    py::class_<SolverDiagnostics>(m, "SolverDiagnostics")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("n_iter"))
        .def_readwrite("converged", &SolverDiagnostics::converged)
        .def_readwrite("iterations_used", &SolverDiagnostics::iterations_used)
        .def_readwrite("final_max_h_change", &SolverDiagnostics::final_max_h_change)
        .def_readwrite("final_max_rhs", &SolverDiagnostics::final_max_rhs)
        .def_readwrite("consecutive_passes", &SolverDiagnostics::consecutive_passes)
        .def_readwrite("history_max_h_change", &SolverDiagnostics::history_max_h_change)
        .def_readwrite("history_max_rhs", &SolverDiagnostics::history_max_rhs);

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
        .def(
            "bellman_rhs_from_policies",
            [](const HJBLadderSolver& self, const std::vector<double>& h_vec) {
                if (h_vec.size() != self.config.q_grid.size()) {
                    throw std::invalid_argument(
                        "bellman_rhs_from_policies: h_vec size must match config.q_grid size."
                    );
                }
                std::vector<double> rhs(self.config.q_grid.size(), 0.0);
                self.bellman_rhs_from_policies_unchecked(h_vec, rhs);
                return rhs;
            },
            py::arg("h_vec")
        )
        .def("solve", &HJBLadderSolver::solve);
}