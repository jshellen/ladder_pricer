#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <vector>

#include "hjb_ladder.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ladder_pricer, m) {
    using namespace hjb;

    m.doc() = "HJB ladder pricer bindings";

    // ============================================================
    // Abstract base classes
    // ============================================================

    py::class_<FlowCurve, std::shared_ptr<FlowCurve>>(m, "FlowCurve")
        .def("hit_ratio", &FlowCurve::hit_ratio)
        .def("arrival_rate", &FlowCurve::arrival_rate);

    py::class_<MarkoutModel, std::shared_ptr<MarkoutModel>>(m, "MarkoutModel")
        .def("expected_markout", &MarkoutModel::expected_markout);

    py::class_<InventoryPenalty, std::shared_ptr<InventoryPenalty>>(m, "InventoryPenalty")
        .def("value", &InventoryPenalty::value);

    // ============================================================
    // Concrete model components
    // ============================================================

    py::class_<LogisticFlowCurve, FlowCurve, std::shared_ptr<LogisticFlowCurve>>(m, "LogisticFlowCurve")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("A0"),
             py::arg("theta"),
             py::arg("shift"),
             py::arg("steepness"),
             py::arg("volume_shift"),
             py::arg("z_floor") = 1e-8)
        .def_readwrite("A0", &LogisticFlowCurve::A0)
        .def_readwrite("theta", &LogisticFlowCurve::theta)
        .def_readwrite("shift", &LogisticFlowCurve::shift)
        .def_readwrite("steepness", &LogisticFlowCurve::steepness)
        .def_readwrite("volume_shift", &LogisticFlowCurve::volume_shift)
        .def_readwrite("z_floor", &LogisticFlowCurve::z_floor)
        .def("A", &LogisticFlowCurve::A)
        .def("hit_ratio", &LogisticFlowCurve::hit_ratio)
        .def("arrival_rate", &LogisticFlowCurve::arrival_rate);

    py::class_<SqrtMarkoutModel, MarkoutModel, std::shared_ptr<SqrtMarkoutModel>>(m, "SqrtMarkoutModel")
        .def(py::init<>())
        .def(py::init<double, double>(),
             py::arg("base"),
             py::arg("coeff"))
        .def_readwrite("base", &SqrtMarkoutModel::base)
        .def_readwrite("coeff", &SqrtMarkoutModel::coeff)
        .def("expected_markout", &SqrtMarkoutModel::expected_markout);

    py::class_<PolynomialInventoryPenalty, InventoryPenalty, std::shared_ptr<PolynomialInventoryPenalty>>(
        m, "PolynomialInventoryPenalty"
    )
        .def(py::init<>())
        .def(py::init<double, double, double, double, double>(),
             py::arg("risk_aversion"),
             py::arg("sigma"),
             py::arg("tau0"),
             py::arg("cubic_coeff"),
             py::arg("quartic_coeff"))
        .def_readwrite("risk_aversion", &PolynomialInventoryPenalty::risk_aversion)
        .def_readwrite("sigma", &PolynomialInventoryPenalty::sigma)
        .def_readwrite("tau0", &PolynomialInventoryPenalty::tau0)
        .def_readwrite("cubic_coeff", &PolynomialInventoryPenalty::cubic_coeff)
        .def_readwrite("quartic_coeff", &PolynomialInventoryPenalty::quartic_coeff)
        .def("value", &PolynomialInventoryPenalty::value);

    // ============================================================
    // Quote policy
    // ============================================================

    py::class_<QuotePolicy>(m, "QuotePolicy")
        .def(py::init<>())
        .def(py::init<
                 std::vector<double>,
                 std::vector<double>,
                 std::vector<std::vector<double>>,
                 std::vector<std::vector<double>>>(),
             py::arg("q_grid"),
             py::arg("sizes"),
             py::arg("bid"),
             py::arg("ask"))
        .def_readwrite("q_grid", &QuotePolicy::q_grid)
        .def_readwrite("sizes", &QuotePolicy::sizes)
        .def_readwrite("bid", &QuotePolicy::bid)
        .def_readwrite("ask", &QuotePolicy::ask)
        .def("validate", &QuotePolicy::validate)
        .def("delta_at_index", &QuotePolicy::delta_at_index,
             py::arg("i"), py::arg("j"), py::arg("side"))
        .def("quote", &QuotePolicy::quote,
             py::arg("q"), py::arg("z"), py::arg("side"));

    // ============================================================
    // Price tier
    // ============================================================

    py::class_<PriceTier, std::shared_ptr<PriceTier>>(m, "PriceTier")
        .def(py::init<>())
        .def(py::init<
                 std::string,
                 std::vector<double>,
                 std::shared_ptr<FlowCurve>,
                 std::shared_ptr<MarkoutModel>,
                 double,
                 double,
                 double,
                 int>(),
             py::arg("name"),
             py::arg("sizes"),
             py::arg("flow_curve"),
             py::arg("markout_model"),
             py::arg("delta_min") = -0.5,
             py::arg("delta_max") = 4.0,
             py::arg("golden_tol") = 1e-4,
             py::arg("golden_max_iter") = 32)
        .def_readwrite("name", &PriceTier::name)
        .def_readwrite("sizes", &PriceTier::sizes)
        .def_readwrite("flow_curve", &PriceTier::flow_curve)
        .def_readwrite("markout_model", &PriceTier::markout_model)
        .def_readwrite("delta_min", &PriceTier::delta_min)
        .def_readwrite("delta_max", &PriceTier::delta_max)
        .def_readwrite("golden_tol", &PriceTier::golden_tol)
        .def_readwrite("golden_max_iter", &PriceTier::golden_max_iter)
        .def_readwrite("policy", &PriceTier::policy)
        .def("validate", &PriceTier::validate)
        .def_static("next_inventory", &PriceTier::next_inventory,
                    py::arg("q"), py::arg("z"), py::arg("side"))
        .def_static("is_shrink_mode", &PriceTier::is_shrink_mode,
                    py::arg("q"), py::arg("side"))
        .def_static("is_expand_mode", &PriceTier::is_expand_mode,
                    py::arg("q"), py::arg("side"))
        .def("arrival_rate", &PriceTier::arrival_rate,
             py::arg("delta"), py::arg("z"))
        .def("expected_markout", &PriceTier::expected_markout,
             py::arg("z"))
        .def("build_policy", &PriceTier::build_policy,
             py::arg("spread"),
             py::arg("h_fn"),
             py::arg("q_grid"))
        .def("quote", &PriceTier::quote,
             py::arg("q"), py::arg("z"), py::arg("side"));

    // ============================================================
    // Solver config
    // ============================================================

    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("q_grid", &SolverConfig::q_grid)
        .def_readwrite("dt", &SolverConfig::dt)
        .def_readwrite("n_iter", &SolverConfig::n_iter)
        .def_readwrite("spot_drift", &SolverConfig::spot_drift)
        .def_readwrite("spread", &SolverConfig::spread)
        .def_readwrite("early_stop", &SolverConfig::early_stop)
        .def_readwrite("tol_h", &SolverConfig::tol_h)
        .def_readwrite("tol_rhs", &SolverConfig::tol_rhs)
        .def_readwrite("min_iter", &SolverConfig::min_iter)
        .def_readwrite("consecutive_passes_required", &SolverConfig::consecutive_passes_required)
        .def("validate", &SolverConfig::validate);

    // ============================================================
    // Solution
    // ============================================================

    py::class_<HJBSolution>(m, "HJBSolution")
        .def(py::init<>())
        .def_readwrite("h", &HJBSolution::h)
        .def_readwrite("q_grid", &HJBSolution::q_grid)
        .def_readwrite("tiers", &HJBSolution::tiers)
        .def_readwrite("converged", &HJBSolution::converged)
        .def_readwrite("iterations_used", &HJBSolution::iterations_used)
        .def_readwrite("final_max_h_change", &HJBSolution::final_max_h_change)
        .def_readwrite("final_max_rhs", &HJBSolution::final_max_rhs)
        .def_readwrite("history_max_h_change", &HJBSolution::history_max_h_change)
        .def_readwrite("history_max_rhs", &HJBSolution::history_max_rhs);

    // ============================================================
    // Solver
    // ============================================================

    py::class_<HJBLadderSolver>(m, "HJBLadderSolver")
        .def(py::init<
                 SolverConfig,
                 std::shared_ptr<InventoryPenalty>,
                 std::vector<std::shared_ptr<PriceTier>>>(),
             py::arg("config"),
             py::arg("penalty"),
             py::arg("tiers"))
        .def_readwrite("config", &HJBLadderSolver::config)
        .def_readwrite("penalty", &HJBLadderSolver::penalty)
        .def_readwrite("tiers", &HJBLadderSolver::tiers)
        .def("update_policies", &HJBLadderSolver::update_policies,
             py::arg("h_vec"))
        .def("bellman_rhs_from_policies", &HJBLadderSolver::bellman_rhs_from_policies,
             py::arg("h_vec"))
        .def("solve", &HJBLadderSolver::solve);
}