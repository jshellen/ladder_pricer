#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hjb_ladder.hpp"

namespace py = pybind11;
using namespace hjb;

PYBIND11_MODULE(ladder_pricer, m) {
    m.doc() = "C++ HJB ladder solver with pybind11 bindings";

    py::class_<FlowCurve, std::shared_ptr<FlowCurve>>(m, "FlowCurve");
    py::class_<MarkoutModel, std::shared_ptr<MarkoutModel>>(m, "MarkoutModel");
    py::class_<InventoryPenalty, std::shared_ptr<InventoryPenalty>>(m, "InventoryPenalty");

    py::class_<LogisticFlowCurve, FlowCurve, std::shared_ptr<LogisticFlowCurve>>(m, "LogisticFlowCurve")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("A0") = 1.0,
             py::arg("theta") = 0.0,
             py::arg("k") = 2.0,
             py::arg("m0") = 0.2,
             py::arg("m_alpha") = 0.0,
             py::arg("z_floor") = 1e-8)
        .def_readwrite("A0", &LogisticFlowCurve::A0)
        .def_readwrite("theta", &LogisticFlowCurve::theta)
        .def_readwrite("k", &LogisticFlowCurve::k)
        .def_readwrite("m0", &LogisticFlowCurve::m0)
        .def_readwrite("m_alpha", &LogisticFlowCurve::m_alpha)
        .def_readwrite("z_floor", &LogisticFlowCurve::z_floor)
        .def("A", &LogisticFlowCurve::A)
        .def("m", &LogisticFlowCurve::m)
        .def("arrival_rate", &LogisticFlowCurve::arrival_rate);

    py::class_<SqrtMarkoutModel, MarkoutModel, std::shared_ptr<SqrtMarkoutModel>>(m, "SqrtMarkoutModel")
        .def(py::init<double, double>(),
             py::arg("base") = 0.005,
             py::arg("coeff") = 0.003)
        .def_readwrite("base", &SqrtMarkoutModel::base)
        .def_readwrite("coeff", &SqrtMarkoutModel::coeff)
        .def("expected_markout", &SqrtMarkoutModel::expected_markout);

    py::class_<PolynomialInventoryPenalty, InventoryPenalty, std::shared_ptr<PolynomialInventoryPenalty>>(m, "PolynomialInventoryPenalty")
        .def(py::init<double, double, double, double, double>(),
             py::arg("risk_aversion"),
             py::arg("sigma"),
             py::arg("tau0"),
             py::arg("cubic_coeff") = 0.1,
             py::arg("quartic_coeff") = 0.0015)
        .def_readwrite("risk_aversion", &PolynomialInventoryPenalty::risk_aversion)
        .def_readwrite("sigma", &PolynomialInventoryPenalty::sigma)
        .def_readwrite("tau0", &PolynomialInventoryPenalty::tau0)
        .def_readwrite("cubic_coeff", &PolynomialInventoryPenalty::cubic_coeff)
        .def_readwrite("quartic_coeff", &PolynomialInventoryPenalty::quartic_coeff)
        .def("value", &PolynomialInventoryPenalty::value);

    py::class_<QuotePolicy>(m, "QuotePolicy")
        .def(py::init<>())
        .def_readwrite("q_grid", &QuotePolicy::q_grid)
        .def_readwrite("sizes", &QuotePolicy::sizes)
        .def_readwrite("bid", &QuotePolicy::bid)
        .def_readwrite("ask", &QuotePolicy::ask)
        .def("delta_at_index", &QuotePolicy::delta_at_index)
        .def("quote", &QuotePolicy::quote);

    py::class_<PriceTier, std::shared_ptr<PriceTier>>(m, "PriceTier")
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
        .def_readwrite("delta_min", &PriceTier::delta_min)
        .def_readwrite("delta_max", &PriceTier::delta_max)
        .def_readwrite("golden_tol", &PriceTier::golden_tol)
        .def_readwrite("golden_max_iter", &PriceTier::golden_max_iter)
        .def_readwrite("policy", &PriceTier::policy)
        .def("arrival_rate", &PriceTier::arrival_rate)
        .def("expected_markout", &PriceTier::expected_markout)
        .def("quote", &PriceTier::quote)
        .def("build_policy",
             [](PriceTier& self,
                const std::function<double(double)>& h_fn,
                const std::vector<double>& q_grid) {
                 return self.build_policy(h_fn, q_grid);
             });

    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("q_grid", &SolverConfig::q_grid)
        .def_readwrite("dt", &SolverConfig::dt)
        .def_readwrite("n_iter", &SolverConfig::n_iter)
        .def_readwrite("spot_drift", &SolverConfig::spot_drift)
        .def_readwrite("early_stop", &SolverConfig::early_stop)
        .def_readwrite("tol_h", &SolverConfig::tol_h)
        .def_readwrite("tol_rhs", &SolverConfig::tol_rhs)
        .def_readwrite("min_iter", &SolverConfig::min_iter)
        .def_readwrite("consecutive_passes_required", &SolverConfig::consecutive_passes_required)
        .def("validate", &SolverConfig::validate);

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

    py::class_<HJBLadderSolver>(m, "HJBLadderSolver")
        .def(py::init<
                 SolverConfig,
                 std::shared_ptr<InventoryPenalty>,
                 std::vector<std::shared_ptr<PriceTier>>>(),
             py::arg("config"),
             py::arg("penalty"),
             py::arg("tiers"))
        .def("solve", &HJBLadderSolver::solve)
        .def("update_policies", &HJBLadderSolver::update_policies)
        .def("bellman_rhs_from_policies", &HJBLadderSolver::bellman_rhs_from_policies);
}