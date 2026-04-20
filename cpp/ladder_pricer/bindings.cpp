#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "hjb_ladder.hpp"

namespace py = pybind11;
using namespace hjb;

namespace {

Side parse_side_string(const std::string& side) {
    if (side == "bid" || side == "Bid" || side == "BID") {
        return Side::Bid;
    }
    if (side == "ask" || side == "Ask" || side == "ASK") {
        return Side::Ask;
    }
    throw std::invalid_argument("side must be 'bid' or 'ask'.");
}

Side parse_side_object(const py::object& side_obj) {
    if (py::isinstance<py::str>(side_obj)) {
        return parse_side_string(side_obj.cast<std::string>());
    }
    return side_obj.cast<Side>();
}

inline void validate_h_vec_against_solver(
    const HJBLadderSolver& solver,
    const std::vector<double>& h_vec,
    const char* caller_name
) {
    if (h_vec.size() != solver.config.q_grid.size()) {
        throw std::invalid_argument(
            std::string(caller_name) +
            ": h_vec size must match solver.config.q_grid size."
        );
    }
}

inline OptionalDeltaRow make_optional_delta_row(
    const std::optional<std::vector<double>>& prev_delta
) {
    if (prev_delta) {
        return std::cref(*prev_delta);
    }
    return std::nullopt;
}

} // namespace

PYBIND11_MODULE(ladder_pricer, m) {
    m.doc() = "Pybind11 bindings for hjb ladder solver";

    py::enum_<Side>(m, "Side")
        .value("Bid", Side::Bid)
        .value("Ask", Side::Ask)
        .export_values();

    py::class_<LogisticFlowCurve>(m, "LogisticFlowCurve")
        .def(py::init<>())
        .def(
            py::init<double, double, double, double, double, double>(),
            py::arg("A0"),
            py::arg("theta"),
            py::arg("shift"),
            py::arg("steepness"),
            py::arg("volume_shift"),
            py::arg("z_floor") = 1e-8
        )
        .def_readwrite("A0", &LogisticFlowCurve::A0)
        .def_readwrite("theta", &LogisticFlowCurve::theta)
        .def_readwrite("shift", &LogisticFlowCurve::shift)
        .def_readwrite("steepness", &LogisticFlowCurve::steepness)
        .def_readwrite("volume_shift", &LogisticFlowCurve::volume_shift)
        .def_readwrite("z_floor", &LogisticFlowCurve::z_floor)
        .def("A", &LogisticFlowCurve::A, py::arg("z"))
        .def("hit_ratio", &LogisticFlowCurve::hit_ratio, py::arg("delta"), py::arg("z"))
        .def("arrival_rate", &LogisticFlowCurve::arrival_rate, py::arg("delta"), py::arg("z"));

    py::class_<SqrtMarkoutModel>(m, "SqrtMarkoutModel")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("base"), py::arg("coeff"))
        .def_readwrite("base", &SqrtMarkoutModel::base)
        .def_readwrite("coeff", &SqrtMarkoutModel::coeff)
        .def("expected_markout", &SqrtMarkoutModel::expected_markout, py::arg("z"));

    py::class_<PolynomialInventoryPenalty>(m, "PolynomialInventoryPenalty")
        .def(py::init<>())
        .def(
            py::init<double, double, double, double, double>(),
            py::arg("risk_aversion"),
            py::arg("sigma"),
            py::arg("tau0"),
            py::arg("tau1"),
            py::arg("tau2")
        )
        .def_readwrite("risk_aversion", &PolynomialInventoryPenalty::risk_aversion)
        .def_readwrite("sigma", &PolynomialInventoryPenalty::sigma)
        .def_readwrite("tau0", &PolynomialInventoryPenalty::tau0)
        .def_readwrite("tau1", &PolynomialInventoryPenalty::tau1)
        .def_readwrite("tau2", &PolynomialInventoryPenalty::tau2)
        .def("value", &PolynomialInventoryPenalty::value, py::arg("q"));

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
            "delta_at_abs_inventory_with_decay",
            &ExponentialECNPolicy::delta_at_abs_inventory_with_decay,
            py::arg("q_abs"),
            py::arg("decay")
        );

    py::class_<QuoteSummary>(m, "QuoteSummary")
        .def(py::init<>())
        .def_readwrite("delta", &QuoteSummary::delta)
        .def_readwrite("price_improvement_frac", &QuoteSummary::price_improvement_frac)
        .def_readwrite(
            "price_improvement_pct_of_spread",
            &QuoteSummary::price_improvement_pct_of_spread
        )
        .def_readwrite("price_improvement_pips", &QuoteSummary::price_improvement_pips)
        .def_readwrite("quote_relative_to_mid", &QuoteSummary::quote_relative_to_mid)
        .def_readwrite("quote_relative_to_mid_pips", &QuoteSummary::quote_relative_to_mid_pips)
        .def_readwrite("distance_to_mid", &QuoteSummary::distance_to_mid)
        .def_readwrite("distance_to_mid_pips", &QuoteSummary::distance_to_mid_pips)
        .def_readwrite("reference_delta", &QuoteSummary::reference_delta)
        .def_readwrite("volume_premium_pips", &QuoteSummary::volume_premium_pips)
        .def_readwrite("quote_price", &QuoteSummary::quote_price);

    py::class_<QuoteMetrics>(m, "QuoteMetrics")
        .def(py::init<>())
        .def_readonly_static("pips_per_unit", &QuoteMetrics::pips_per_unit)
        .def_static(
            "price_improvement",
            &QuoteMetrics::price_improvement,
            py::arg("delta"),
            py::arg("spread")
        )
        .def_static(
            "price_improvement_pct_of_spread",
            &QuoteMetrics::price_improvement_pct_of_spread,
            py::arg("delta")
        )
        .def_static(
            "price_improvement_pips",
            &QuoteMetrics::price_improvement_pips,
            py::arg("delta"),
            py::arg("spread")
        )
        .def_static(
            "quote_relative_to_mid",
            &QuoteMetrics::quote_relative_to_mid,
            py::arg("delta"),
            py::arg("side"),
            py::arg("spread")
        )
        .def_static(
            "quote_relative_to_mid_pips",
            &QuoteMetrics::quote_relative_to_mid_pips,
            py::arg("delta"),
            py::arg("side"),
            py::arg("spread")
        )
        .def_static(
            "distance_to_mid",
            &QuoteMetrics::distance_to_mid,
            py::arg("delta"),
            py::arg("spread")
        )
        .def_static(
            "distance_to_mid_pips",
            &QuoteMetrics::distance_to_mid_pips,
            py::arg("delta"),
            py::arg("spread")
        )
        .def_static(
            "volume_premium_pips",
            &QuoteMetrics::volume_premium_pips,
            py::arg("delta_ref"),
            py::arg("delta_cur"),
            py::arg("spread")
        )
        .def_static(
            "quote_price",
            &QuoteMetrics::quote_price,
            py::arg("mid"),
            py::arg("delta"),
            py::arg("side"),
            py::arg("spread")
        )
        .def_static(
            "make_summary",
            &QuoteMetrics::make_summary,
            py::arg("delta"),
            py::arg("delta_ref"),
            py::arg("side"),
            py::arg("mid"),
            py::arg("spread")
        );

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
        .def_static(
            "resize_matrix",
            [](Matrix mat, std::size_t rows, std::size_t cols) {
                QuotePolicy::resize_matrix(mat, rows, cols);
                return mat;
            },
            py::arg("mat"),
            py::arg("rows"),
            py::arg("cols")
        )
        .def("reset_shape", &QuotePolicy::reset_shape, py::arg("q_grid"), py::arg("sizes"))
        .def("validate", &QuotePolicy::validate)
        .def("delta", &QuotePolicy::delta, py::arg("q"), py::arg("z"), py::arg("side"))
        .def("reference_delta", &QuotePolicy::reference_delta, py::arg("q"), py::arg("side"))
        .def(
            "quote_summary",
            &QuotePolicy::quote_summary,
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
        .def_readwrite(
            "consecutive_passes_required",
            &SolverConfig::consecutive_passes_required
        )
        .def("validate", &SolverConfig::validate);

    py::class_<SolverGridMeta>(m, "SolverGridMeta")
        .def(py::init<>())
        .def_readwrite("nq", &SolverGridMeta::nq)
        .def_readwrite("q0_idx", &SolverGridMeta::q0_idx)
        .def_readwrite("q_min", &SolverGridMeta::q_min)
        .def_readwrite("q_max", &SolverGridMeta::q_max);

    m.def("build_solver_grid_meta", &build_solver_grid_meta, py::arg("q_grid"));

    py::class_<GoldenSectionSearch>(m, "GoldenSectionSearch")
        .def(py::init<>())
        .def(py::init<double, int>(), py::arg("tol"), py::arg("max_iter"))
        .def_readwrite("tol", &GoldenSectionSearch::tol)
        .def_readwrite("max_iter", &GoldenSectionSearch::max_iter)
        .def("validate", &GoldenSectionSearch::validate);

    py::class_<Tier>(m, "Tier")
        .def_readwrite("name", &Tier::name)
        .def_readwrite("flow_curve", &Tier::flow_curve)
        .def_readwrite("markout_model", &Tier::markout_model)
        .def_readwrite("policy", &Tier::policy)
        .def("tier_type_name", &Tier::tier_type_name)
        .def("sizes", &Tier::sizes, py::return_value_policy::reference_internal)

        .def(
            "A",
            [](const Tier& self, double z) {
                return self.flow_curve.A(z);
            },
            py::arg("z")
        )
        .def(
            "hit_ratio",
            [](const Tier& self, double delta, double z) {
                return self.flow_curve.hit_ratio(delta, z);
            },
            py::arg("delta"),
            py::arg("z")
        )
        .def(
            "arrival_rate",
            [](const Tier& self, double delta, double z) {
                return self.flow_curve.arrival_rate(delta, z);
            },
            py::arg("delta"),
            py::arg("z")
        )
        .def(
            "expected_markout",
            [](const Tier& self, double z) {
                return self.markout_model.expected_markout(z);
            },
            py::arg("z")
        )

        .def(
            "is_admissible",
            [](const Tier& self, double q, double z, const py::object& side) {
                return self.is_admissible(q, z, parse_side_object(side));
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side")
        )
        .def("validate", &Tier::validate)
        .def("reset_policy_shape", &Tier::reset_policy_shape, py::arg("q_grid"))
        .def(
            "quote",
            [](const Tier& self, double q, double z, const py::object& side) {
                return self.quote(q, z, parse_side_object(side));
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side")
        )
        .def(
            "quote_summary",
            [](const Tier& self,
               double q,
               double z,
               const py::object& side,
               double mid,
               double spread) {
                return self.quote_summary(q, z, parse_side_object(side), mid, spread);
            },
            py::arg("q"),
            py::arg("z"),
            py::arg("side"),
            py::arg("mid"),
            py::arg("spread")
        );

    py::class_<MDPTier, Tier>(m, "MDPTier")
        .def(py::init<>())
        .def(
            py::init<
                std::string,
                std::vector<double>,
                LogisticFlowCurve,
                SqrtMarkoutModel,
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
        .def_readwrite("sizes_", &MDPTier::sizes_)
        .def_readwrite("delta_min", &MDPTier::delta_min)
        .def_readwrite("delta_max", &MDPTier::delta_max);

    py::class_<ECNTier, Tier>(m, "ECNTier")
        .def(py::init<>())
        .def(
            py::init<
                std::string,
                LogisticFlowCurve,
                SqrtMarkoutModel,
                double,
                double,
                ExponentialECNPolicy
            >(),
            py::arg("name"),
            py::arg("flow_curve"),
            py::arg("markout_model"),
            py::arg("delta_min") = -5.0,
            py::arg("delta_max") = 5.0,
            py::arg("ecn_policy") = ExponentialECNPolicy{}
        )
        .def_readwrite("delta_min", &ECNTier::delta_min)
        .def_readwrite("delta_max", &ECNTier::delta_max)
        .def_readwrite("ecn_policy", &ECNTier::ecn_policy)
        .def_static(
            "is_active_side",
            [](double q, const py::object& side, double tol) {
                return ECNTier::is_active_side(q, parse_side_object(side), tol);
            },
            py::arg("q"),
            py::arg("side"),
            py::arg("tol") = 1e-12
        )
        .def(
            "active_delta_with_decay",
            &ECNTier::active_delta_with_decay,
            py::arg("q"),
            py::arg("decay")
        )
        .def(
            "active_delta",
            [](const ECNTier& self, double q) {
                return self.active_delta_with_decay(q, self.ecn_policy.decay);
            },
            py::arg("q")
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
        .def_readwrite("mdp_tiers", &HJBSolution::mdp_tiers)
        .def_readwrite("ecn_tiers", &HJBSolution::ecn_tiers)
        .def_readwrite("diagnostics", &HJBSolution::diagnostics);

    py::class_<HJBLadderSolver>(m, "HJBLadderSolver")
        .def(
            py::init<
                SolverConfig,
                PolynomialInventoryPenalty,
                std::vector<MDPTier>,
                std::vector<ECNTier>
            >(),
            py::arg("config"),
            py::arg("penalty"),
            py::arg("mdp_tiers") = std::vector<MDPTier>{},
            py::arg("ecn_tiers") = std::vector<ECNTier>{}
        )
        .def_readwrite("config", &HJBLadderSolver::config)
        .def_readwrite("penalty", &HJBLadderSolver::penalty)
        .def_readwrite("mdp_tiers", &HJBLadderSolver::mdp_tiers)
        .def_readwrite("ecn_tiers", &HJBLadderSolver::ecn_tiers)
        .def_readwrite("optimizer", &HJBLadderSolver::optimizer)
        .def_readwrite("grid_meta_", &HJBLadderSolver::grid_meta_)

        .def_static(
            "validate_mdp_bounds_request",
            [](std::size_t rung_idx,
               const std::vector<double>& current_delta,
               const std::optional<std::vector<double>>& prev_delta) {
                HJBLadderSolver::validate_mdp_bounds_request(
                    rung_idx,
                    current_delta,
                    make_optional_delta_row(prev_delta)
                );
            },
            py::arg("rung_idx"),
            py::arg("current_delta"),
            py::arg("prev_delta") = std::nullopt
        )
        .def_static(
            "repair_or_throw_bounds",
            [](double lower,
               double upper,
               const std::string& tier_name,
               const char* side_name) {
                HJBLadderSolver::repair_or_throw_bounds(lower, upper, tier_name, side_name);
                return std::make_pair(lower, upper);
            },
            py::arg("lower"),
            py::arg("upper"),
            py::arg("tier_name"),
            py::arg("side_name")
        )

        .def("validate_problem_definition", &HJBLadderSolver::validate_problem_definition)
        .def("initialize_policy_shapes", &HJBLadderSolver::initialize_policy_shapes)
        .def("prepare_solve_context", &HJBLadderSolver::prepare_solve_context)

        .def(
            "mdp_bid_bounds_for_rung",
            [](const HJBLadderSolver& self,
               const MDPTier& tier,
               std::size_t rung_idx,
               const std::vector<double>& current_delta,
               double q,
               const std::optional<std::vector<double>>& prev_delta) {
                return self.mdp_bid_bounds_for_rung(
                    tier,
                    rung_idx,
                    current_delta,
                    q,
                    make_optional_delta_row(prev_delta)
                );
            },
            py::arg("tier"),
            py::arg("rung_idx"),
            py::arg("current_delta"),
            py::arg("q"),
            py::arg("prev_delta") = std::nullopt
        )
        .def(
            "mdp_ask_bounds_for_rung",
            [](const HJBLadderSolver& self,
               const MDPTier& tier,
               std::size_t rung_idx,
               const std::vector<double>& current_delta,
               double q,
               const std::optional<std::vector<double>>& prev_delta) {
                return self.mdp_ask_bounds_for_rung(
                    tier,
                    rung_idx,
                    current_delta,
                    q,
                    make_optional_delta_row(prev_delta)
                );
            },
            py::arg("tier"),
            py::arg("rung_idx"),
            py::arg("current_delta"),
            py::arg("q"),
            py::arg("prev_delta") = std::nullopt
        )

        .def(
            "build_ecn_policy_from_decay",
            &HJBLadderSolver::build_ecn_policy_from_decay,
            py::arg("tier"),
            py::arg("decay")
        )

        .def(
            "optimize_ecn_decay",
            [](const HJBLadderSolver& self,
               const ECNTier& tier,
               const std::vector<double>& h_vec) {
                validate_h_vec_against_solver(self, h_vec, "optimize_ecn_decay");
                const LinearInterpolator1D h{self.config.q_grid, h_vec};
                return self.optimize_ecn_decay(tier, h);
            },
            py::arg("tier"),
            py::arg("h_vec")
        )
        .def(
            "mdp_bid_bellman_contribution",
            [](const HJBLadderSolver& self,
               const MDPTier& tier,
               double q,
               std::size_t q_index,
               const std::vector<double>& h_vec) {
                validate_h_vec_against_solver(self, h_vec, "mdp_bid_bellman_contribution");
                const LinearInterpolator1D h{self.config.q_grid, h_vec};
                return self.mdp_bid_bellman_contribution(tier, q, q_index, h);
            },
            py::arg("tier"),
            py::arg("q"),
            py::arg("q_index"),
            py::arg("h_vec")
        )
        .def(
            "mdp_ask_bellman_contribution",
            [](const HJBLadderSolver& self,
               const MDPTier& tier,
               double q,
               std::size_t q_index,
               const std::vector<double>& h_vec) {
                validate_h_vec_against_solver(self, h_vec, "mdp_ask_bellman_contribution");
                const LinearInterpolator1D h{self.config.q_grid, h_vec};
                return self.mdp_ask_bellman_contribution(tier, q, q_index, h);
            },
            py::arg("tier"),
            py::arg("q"),
            py::arg("q_index"),
            py::arg("h_vec")
        )
        .def(
            "ecn_bellman_contribution",
            [](const HJBLadderSolver& self,
               const ECNTier& tier,
               double q,
               std::size_t q_index,
               const std::vector<double>& h_vec) {
                validate_h_vec_against_solver(self, h_vec, "ecn_bellman_contribution");
                const LinearInterpolator1D h{self.config.q_grid, h_vec};
                return self.ecn_bellman_contribution(tier, q, q_index, h);
            },
            py::arg("tier"),
            py::arg("q"),
            py::arg("q_index"),
            py::arg("h_vec")
        )
        .def(
            "update_policies",
            [](HJBLadderSolver& self, const std::vector<double>& h_vec) {
                validate_h_vec_against_solver(self, h_vec, "update_policies");
                const LinearInterpolator1D h{self.config.q_grid, h_vec};
                self.update_policies(h);
            },
            py::arg("h_vec")
        )
        .def(
            "bellman_rhs_from_policies",
            &HJBLadderSolver::bellman_rhs_from_policies,
            py::arg("h_vec")
        )
        .def("solve", &HJBLadderSolver::solve);
}