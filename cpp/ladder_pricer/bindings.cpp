#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
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
            py::init<double, double, double, double, double, double, double>(),
            py::arg("A0"),
            py::arg("theta"),
            py::arg("beta"),
            py::arg("shift"),
            py::arg("steepness"),
            py::arg("volume_shift"),
            py::arg("z_floor") = 1e-8
        )
        .def_readwrite("A0", &LogisticFlowCurve::A0)
        .def_readwrite("theta", &LogisticFlowCurve::theta)
        .def_readwrite("beta", &LogisticFlowCurve::beta)
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
        .def("expected_markout", &SqrtMarkoutModel::expected_markout, py::arg("z"), py::arg("t"));

    py::class_<SqrtTimeMarkoutModel>(m, "SqrtTimeMarkoutModel")
        .def(py::init<>())
        .def(
            py::init<double, double, double, double, double, double>(),
            py::arg("a0"), py::arg("a1"), py::arg("a2"),
            py::arg("b0"), py::arg("b1"), py::arg("b2")
        )
        .def_readwrite("a0", &SqrtTimeMarkoutModel::a0)
        .def_readwrite("a1", &SqrtTimeMarkoutModel::a1)
        .def_readwrite("a2", &SqrtTimeMarkoutModel::a2)
        .def_readwrite("b0", &SqrtTimeMarkoutModel::b0)
        .def_readwrite("b1", &SqrtTimeMarkoutModel::b1)
        .def_readwrite("b2", &SqrtTimeMarkoutModel::b2)
        .def("alpha", &SqrtTimeMarkoutModel::alpha, py::arg("z"))
        .def("beta", &SqrtTimeMarkoutModel::beta, py::arg("z"))
        .def("expected_markout", &SqrtTimeMarkoutModel::expected_markout, py::arg("z"), py::arg("t"));

    py::class_<CarryCost>(m, "CarryCost")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("risk_aversion"), py::arg("sigma"))
        .def_readwrite("risk_aversion", &CarryCost::risk_aversion)
        .def_readwrite("sigma", &CarryCost::sigma)
        .def("value", &CarryCost::value);

    py::class_<PolynomialInternalizationTime>(m, "PolynomialInternalizationTime")
        .def(py::init<>())
        .def(py::init<double, double, double>(), py::arg("tau0"), py::arg("tau1"), py::arg("tau2"))
        .def_readwrite("tau0", &PolynomialInternalizationTime::tau0)
        .def_readwrite("tau1", &PolynomialInternalizationTime::tau1)
        .def_readwrite("tau2", &PolynomialInternalizationTime::tau2)
        .def("value", &PolynomialInternalizationTime::value, py::arg("q"));

    py::class_<PolynomialInventoryPenalty>(m, "PolynomialInventoryPenalty")
        .def(py::init<>())
        .def(
            py::init<CarryCost, PolynomialInternalizationTime>(),
            py::arg("carry_cost"),
            py::arg("internalization_time")
        )
        .def_readwrite("carry_cost", &PolynomialInventoryPenalty::carry_cost)
        .def_readwrite("internalization_time", &PolynomialInventoryPenalty::internalization_time)
        .def("value", &PolynomialInventoryPenalty::value, py::arg("q"));


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

    py::class_<QuoteMetrics>(m, "QuoteMetrics")
        .def(py::init<>())
        .def_property_readonly_static(
            "pips_per_unit",
            [](py::object) { return QuoteMetrics::pips_per_unit; }
        )
        .def_static("price_improvement", &QuoteMetrics::price_improvement, py::arg("delta"), py::arg("spread"))
        .def_static("price_improvement_pct_of_spread", &QuoteMetrics::price_improvement_pct_of_spread, py::arg("delta"))
        .def_static("price_improvement_pips", &QuoteMetrics::price_improvement_pips, py::arg("delta"), py::arg("spread"))
        .def_static("quote_relative_to_mid", &QuoteMetrics::quote_relative_to_mid, py::arg("delta"), py::arg("side"), py::arg("spread"))
        .def_static("quote_relative_to_mid_pips", &QuoteMetrics::quote_relative_to_mid_pips, py::arg("delta"), py::arg("side"), py::arg("spread"))
        .def_static("distance_to_mid", &QuoteMetrics::distance_to_mid, py::arg("delta"), py::arg("spread"))
        .def_static("distance_to_mid_pips", &QuoteMetrics::distance_to_mid_pips, py::arg("delta"), py::arg("spread"))
        .def_static("volume_premium_pips", &QuoteMetrics::volume_premium_pips, py::arg("delta_ref"), py::arg("delta_cur"), py::arg("spread"))
        .def_static("quote_price", &QuoteMetrics::quote_price, py::arg("mid"), py::arg("delta"), py::arg("side"), py::arg("spread"))
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

    py::class_<DarkPoolPolicy>(m, "DarkPoolPolicy")
        .def(py::init<>())
        .def(py::init<std::vector<double>>(), py::arg("q_grid"))
        .def_readwrite("q_grid", &DarkPoolPolicy::q_grid)
        .def_readwrite("bid_size", &DarkPoolPolicy::bid_size)
        .def_readwrite("ask_size", &DarkPoolPolicy::ask_size)
        .def_readwrite("bid_active", &DarkPoolPolicy::bid_active)
        .def_readwrite("ask_active", &DarkPoolPolicy::ask_active)
        .def("reset_shape", &DarkPoolPolicy::reset_shape, py::arg("q_grid"))
        .def("validate", &DarkPoolPolicy::validate)
        .def(
            "posted_size",
            [](const DarkPoolPolicy& self, double q, const py::object& side) {
                return self.posted_size(q, parse_side_object(side));
            },
            py::arg("q"),
            py::arg("side")
        )
        .def(
            "is_active",
            [](const DarkPoolPolicy& self, std::size_t q_index, const py::object& side) {
                return self.is_active(q_index, parse_side_object(side));
            },
            py::arg("q_index"),
            py::arg("side")
        )
        .def(
            "set_posted_size",
            [](DarkPoolPolicy& self,
               std::size_t q_index,
               const py::object& side,
               double size,
               bool active) {
                self.set_posted_size(q_index, parse_side_object(side), size, active);
            },
            py::arg("q_index"),
            py::arg("side"),
            py::arg("size"),
            py::arg("active")
        );

    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("q_grid", &SolverConfig::q_grid)
        .def_readwrite("dt", &SolverConfig::dt)
        .def_readwrite("n_iter", &SolverConfig::n_iter)
        .def_readwrite("spot", &SolverConfig::spot)
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
        .def_readwrite("use_markout", &Tier::use_markout)
        .def("tier_type_name", &Tier::tier_type_name)
        .def("sizes", &Tier::sizes, py::return_value_policy::reference_internal)
        .def("A", [](const Tier& self, double z) { return self.flow_curve.A(z); }, py::arg("z"))
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
            [](const Tier& self, double z, double t) {
                return expected_markout(self.markout_model, z, t);
            },
            py::arg("z"),
            py::arg("t")
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
            py::init<std::string, std::vector<double>, LogisticFlowCurve, MarkoutModel, double, double, bool>(),
            py::arg("name"),
            py::arg("sizes"),
            py::arg("flow_curve"),
            py::arg("markout_model"),
            py::arg("delta_min") = -5.0,
            py::arg("delta_max") = 5.0,
            py::arg("use_markout") = true
        )
        .def_readwrite("sizes_", &MDPTier::sizes_)
        .def_readwrite("delta_min", &MDPTier::delta_min)
        .def_readwrite("delta_max", &MDPTier::delta_max);

    py::class_<ArrivalDistribution, std::shared_ptr<ArrivalDistribution>>(m, "ArrivalDistribution")
        .def("validate", &ArrivalDistribution::validate)
        .def("name", &ArrivalDistribution::name);

    py::class_<GeometricArrivalDist, ArrivalDistribution, std::shared_ptr<GeometricArrivalDist>>(m, "GeometricArrivalDist")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("lambda"), py::arg("p"))
        .def_readwrite("lambda", &GeometricArrivalDist::lambda)
        .def_readwrite("p", &GeometricArrivalDist::p)
        .def("validate", &GeometricArrivalDist::validate)
        .def("name", &GeometricArrivalDist::name);

    py::class_<ZeroInflatedPoissonArrivalDist, ArrivalDistribution, std::shared_ptr<ZeroInflatedPoissonArrivalDist>>(m, "ZeroInflatedPoissonArrivalDist")
        .def(py::init<>())
        .def(py::init<double, double, double>(), py::arg("lambda_arr"), py::arg("mu"), py::arg("p0"))
        .def_readwrite("lambda_arr", &ZeroInflatedPoissonArrivalDist::lambda_arr)
        .def_readwrite("mu", &ZeroInflatedPoissonArrivalDist::mu)
        .def_readwrite("p0", &ZeroInflatedPoissonArrivalDist::p0)
        .def("validate", &ZeroInflatedPoissonArrivalDist::validate)
        .def("name", &ZeroInflatedPoissonArrivalDist::name);

    py::class_<DarkPoolVenue>(m, "DarkPoolVenue")
        .def(py::init<>())
        .def(
            py::init<double, double, double, double, double, double, std::vector<double>, bool, double>(),
            py::arg("lambda_bid"),
            py::arg("lambda_ask"),
            py::arg("p_bid"),
            py::arg("p_ask"),
            py::arg("fee_per_unit_bid"),
            py::arg("fee_per_unit_ask"),
            py::arg("posted_sizes"),
            py::arg("allow_both_sides") = false,
            py::arg("min_fill_value") = 0.0
        )
        .def(
            py::init<std::shared_ptr<ArrivalDistribution>, std::shared_ptr<ArrivalDistribution>, double, double, std::vector<double>, bool, double>(),
            py::arg("dist_bid"),
            py::arg("dist_ask"),
            py::arg("fee_per_unit_bid"),
            py::arg("fee_per_unit_ask"),
            py::arg("posted_sizes"),
            py::arg("allow_both_sides") = false,
            py::arg("min_fill_value") = 0.0
        )
        .def_readwrite("dist_bid", &DarkPoolVenue::dist_bid)
        .def_readwrite("dist_ask", &DarkPoolVenue::dist_ask)
        .def_readwrite("fee_per_unit_bid", &DarkPoolVenue::fee_per_unit_bid)
        .def_readwrite("fee_per_unit_ask", &DarkPoolVenue::fee_per_unit_ask)
        .def_readwrite("posted_sizes", &DarkPoolVenue::posted_sizes)
        .def_readwrite("allow_both_sides", &DarkPoolVenue::allow_both_sides)
        .def_readwrite("min_fill_value", &DarkPoolVenue::min_fill_value)
        .def_readwrite("policy", &DarkPoolVenue::policy)
        .def_property_readonly("lambda_bid", [](const DarkPoolVenue& v) {
            if (auto* g = dynamic_cast<const GeometricArrivalDist*>(v.dist_bid.get()))
                return g->lambda;
            if (auto* z = dynamic_cast<const ZeroInflatedPoissonArrivalDist*>(v.dist_bid.get()))
                return z->lambda_arr;
            throw std::runtime_error("lambda_bid: unsupported distribution type");
        })
        .def_property_readonly("lambda_ask", [](const DarkPoolVenue& v) {
            if (auto* g = dynamic_cast<const GeometricArrivalDist*>(v.dist_ask.get()))
                return g->lambda;
            if (auto* z = dynamic_cast<const ZeroInflatedPoissonArrivalDist*>(v.dist_ask.get()))
                return z->lambda_arr;
            throw std::runtime_error("lambda_ask: unsupported distribution type");
        })
        .def_property_readonly("p_bid", [](const DarkPoolVenue& v) {
            if (auto* g = dynamic_cast<const GeometricArrivalDist*>(v.dist_bid.get()))
                return g->p;
            throw std::runtime_error("p_bid is only defined for GeometricArrivalDist");
        })
        .def_property_readonly("p_ask", [](const DarkPoolVenue& v) {
            if (auto* g = dynamic_cast<const GeometricArrivalDist*>(v.dist_ask.get()))
                return g->p;
            throw std::runtime_error("p_ask is only defined for GeometricArrivalDist");
        })
        .def("validate", &DarkPoolVenue::validate)
        .def("reset_policy_shape", &DarkPoolVenue::reset_policy_shape, py::arg("q_grid"))
        .def_static("is_integer_like", &DarkPoolVenue::is_integer_like, py::arg("x"), py::arg("tol") = 1e-10)
        .def(
            "is_admissible",
            [](const DarkPoolVenue& self, double q, const py::object& side, double tol) {
                return self.is_admissible(q, parse_side_object(side), tol);
            },
            py::arg("q"),
            py::arg("side"),
            py::arg("tol") = 1e-12
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
        .def_readwrite("dark_pool", &HJBSolution::dark_pool)
        .def_readwrite("diagnostics", &HJBSolution::diagnostics);

    py::class_<HJBLadderSolver>(m, "HJBLadderSolver")
        .def(
            py::init<SolverConfig, PolynomialInventoryPenalty, std::vector<MDPTier>, std::optional<DarkPoolVenue>>(),
            py::arg("config"),
            py::arg("penalty"),
            py::arg("mdp_tiers") = std::vector<MDPTier>{},
            py::arg("dark_pool") = std::nullopt
        )
        .def_readwrite("config", &HJBLadderSolver::config)
        .def_readwrite("penalty", &HJBLadderSolver::penalty)
        .def_readwrite("mdp_tiers", &HJBLadderSolver::mdp_tiers)
        .def_readwrite("dark_pool", &HJBLadderSolver::dark_pool)
        .def_readwrite("optimizer", &HJBLadderSolver::optimizer)
        .def_readwrite("grid_meta_", &HJBLadderSolver::grid_meta_)
        .def_static(
            "repair_or_throw_bounds",
            [](double lower, double upper, const std::string& tier_name, const char* side_name) {
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
            "mdp_bounds_for_rung",
            [](const HJBLadderSolver& self,
               const MDPTier& tier,
               std::size_t rung_idx,
               const std::vector<double>& current_delta,
               double q,
               const py::object& side,
               const std::optional<std::vector<double>>& prev_delta) {
                return self.mdp_bounds_for_rung(
                    tier,
                    rung_idx,
                    current_delta,
                    q,
                    parse_side_object(side),
                    make_optional_delta_row(prev_delta)
                );
            },
            py::arg("tier"),
            py::arg("rung_idx"),
            py::arg("current_delta"),
            py::arg("q"),
            py::arg("side"),
            py::arg("prev_delta") = std::nullopt
        )
        .def(
            "build_dark_pool_policy",
            [](const HJBLadderSolver& self, DarkPoolVenue& venue, const std::vector<double>& h_vec) {
                validate_h_vec_against_solver(self, h_vec, "build_dark_pool_policy");
                const LinearInterpolator1D h{self.config.q_grid, h_vec};
                self.build_dark_pool_policy(venue, h);
            },
            py::arg("venue"),
            py::arg("h_vec")
        )
        .def(
            "mdp_bellman_contribution_side",
            [](const HJBLadderSolver& self,
               const MDPTier& tier,
               double q,
               std::size_t q_index,
               const std::vector<double>& h_vec,
               const py::object& side) {
                validate_h_vec_against_solver(self, h_vec, "mdp_bellman_contribution_side");
                const LinearInterpolator1D h{self.config.q_grid, h_vec};
                return self.mdp_bellman_contribution_side(
                    tier,
                    q,
                    q_index,
                    h,
                    parse_side_object(side)
                );
            },
            py::arg("tier"),
            py::arg("q"),
            py::arg("q_index"),
            py::arg("h_vec"),
            py::arg("side")
        )
        .def(
            "dark_pool_bellman_contribution",
            [](const HJBLadderSolver& self,
               const DarkPoolVenue& venue,
               double q,
               std::size_t q_index,
               const std::vector<double>& h_vec) {
                validate_h_vec_against_solver(self, h_vec, "dark_pool_bellman_contribution");
                const LinearInterpolator1D h{self.config.q_grid, h_vec};
                return self.dark_pool_bellman_contribution(venue, q, q_index, h);
            },
            py::arg("venue"),
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
