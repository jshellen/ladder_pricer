#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hjb_ladder.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ladder_pricer, m) {
    m.doc() = "HJB ladder solver bindings";

    // ============================================================
    // Enums
    // ============================================================

    py::enum_<hjb::Side>(m, "Side")
        .value("Bid", hjb::Side::Bid)
        .value("Ask", hjb::Side::Ask)
        .export_values();

    // ============================================================
    // Flat storage containers
    // ============================================================

    py::class_<hjb::FlatCube3D>(m, "FlatCube3D")
        .def(py::init<>())
        .def(py::init<std::size_t, std::size_t, std::size_t, double>(),
             py::arg("nq"),
             py::arg("ny"),
             py::arg("nnu"),
             py::arg("init") = 0.0)
        .def_readwrite("nq", &hjb::FlatCube3D::nq)
        .def_readwrite("ny", &hjb::FlatCube3D::ny)
        .def_readwrite("nnu", &hjb::FlatCube3D::nnu)
        .def("reset", &hjb::FlatCube3D::reset,
             py::arg("nq"),
             py::arg("ny"),
             py::arg("nnu"),
             py::arg("init") = 0.0)
        .def("index", &hjb::FlatCube3D::index,
             py::arg("iq"), py::arg("iy"), py::arg("inu"))
        .def("at",
             [](const hjb::FlatCube3D& self,
                std::size_t iq,
                std::size_t iy,
                std::size_t inu) {
                 return self.at(iq, iy, inu);
             },
             py::arg("iq"), py::arg("iy"), py::arg("inu"))
        .def("empty", &hjb::FlatCube3D::empty)
        .def_property_readonly(
            "shape",
            [](const hjb::FlatCube3D& self) {
                return py::make_tuple(self.nq, self.ny, self.nnu);
            })
        .def("numpy",
             [](hjb::FlatCube3D& self) {
                 py::object owner =
                     py::cast(&self, py::return_value_policy::reference_internal);
                 return py::array_t<double>(
                     {
                         static_cast<py::ssize_t>(self.nq),
                         static_cast<py::ssize_t>(self.ny),
                         static_cast<py::ssize_t>(self.nnu)
                     },
                     {
                         static_cast<py::ssize_t>(sizeof(double) * self.ny * self.nnu),
                         static_cast<py::ssize_t>(sizeof(double) * self.nnu),
                         static_cast<py::ssize_t>(sizeof(double))
                     },
                     self.data.data(),
                     owner
                 );
             });

    py::class_<hjb::FlatCube4D>(m, "FlatCube4D")
        .def(py::init<>())
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, double>(),
             py::arg("nq"),
             py::arg("ny"),
             py::arg("nnu"),
             py::arg("nz"),
             py::arg("init") = 0.0)
        .def_readwrite("nq", &hjb::FlatCube4D::nq)
        .def_readwrite("ny", &hjb::FlatCube4D::ny)
        .def_readwrite("nnu", &hjb::FlatCube4D::nnu)
        .def_readwrite("nz", &hjb::FlatCube4D::nz)
        .def("ensure_shape", &hjb::FlatCube4D::ensure_shape,
             py::arg("nq"),
             py::arg("ny"),
             py::arg("nnu"),
             py::arg("nz"))
        .def("index", &hjb::FlatCube4D::index,
             py::arg("iq"), py::arg("iy"), py::arg("inu"), py::arg("iz"))
        .def("at",
             [](const hjb::FlatCube4D& self,
                std::size_t iq,
                std::size_t iy,
                std::size_t inu,
                std::size_t iz) {
                 return self.at(iq, iy, inu, iz);
             },
             py::arg("iq"), py::arg("iy"), py::arg("inu"), py::arg("iz"))
        .def("empty", &hjb::FlatCube4D::empty)
        .def_property_readonly(
            "shape",
            [](const hjb::FlatCube4D& self) {
                return py::make_tuple(self.nq, self.ny, self.nnu, self.nz);
            })
        .def("numpy",
             [](hjb::FlatCube4D& self) {
                 py::object owner =
                     py::cast(&self, py::return_value_policy::reference_internal);
                 return py::array_t<double>(
                     {
                         static_cast<py::ssize_t>(self.nq),
                         static_cast<py::ssize_t>(self.ny),
                         static_cast<py::ssize_t>(self.nnu),
                         static_cast<py::ssize_t>(self.nz)
                     },
                     {
                         static_cast<py::ssize_t>(sizeof(double) * self.ny * self.nnu * self.nz),
                         static_cast<py::ssize_t>(sizeof(double) * self.nnu * self.nz),
                         static_cast<py::ssize_t>(sizeof(double) * self.nz),
                         static_cast<py::ssize_t>(sizeof(double))
                     },
                     self.data.data(),
                     owner
                 );
             });

    // ============================================================
    // Abstract base classes
    // ============================================================

    py::class_<hjb::FlowCurve, std::shared_ptr<hjb::FlowCurve>>(m, "FlowCurve");
    py::class_<hjb::DriftJumpModel, std::shared_ptr<hjb::DriftJumpModel>>(m, "DriftJumpModel");
    py::class_<hjb::InventoryPenalty, std::shared_ptr<hjb::InventoryPenalty>>(m, "InventoryPenalty");

    // ============================================================
    // Concrete model components
    // ============================================================

    py::class_<hjb::LogisticFlowCurve, hjb::FlowCurve, std::shared_ptr<hjb::LogisticFlowCurve>>(
        m, "LogisticFlowCurve")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("A0"),
             py::arg("theta"),
             py::arg("k"),
             py::arg("m0"),
             py::arg("m_alpha"),
             py::arg("z_floor") = 1e-8)
        .def_readwrite("A0", &hjb::LogisticFlowCurve::A0)
        .def_readwrite("theta", &hjb::LogisticFlowCurve::theta)
        .def_readwrite("k", &hjb::LogisticFlowCurve::k)
        .def_readwrite("m0", &hjb::LogisticFlowCurve::m0)
        .def_readwrite("m_alpha", &hjb::LogisticFlowCurve::m_alpha)
        .def_readwrite("z_floor", &hjb::LogisticFlowCurve::z_floor)
        .def("A", &hjb::LogisticFlowCurve::A, py::arg("z"))
        .def("m", &hjb::LogisticFlowCurve::m, py::arg("z"))
        .def("arrival_rate", &hjb::LogisticFlowCurve::arrival_rate,
             py::arg("delta"), py::arg("z"));

    py::class_<hjb::SqrtDriftJumpModel, hjb::DriftJumpModel, std::shared_ptr<hjb::SqrtDriftJumpModel>>(
        m, "SqrtDriftJumpModel")
        .def(py::init<>())
        .def(py::init<double, double>(),
             py::arg("base"),
             py::arg("coeff"))
        .def_readwrite("base", &hjb::SqrtDriftJumpModel::base)
        .def_readwrite("coeff", &hjb::SqrtDriftJumpModel::coeff)
        .def("jump_size", &hjb::SqrtDriftJumpModel::jump_size, py::arg("z"));

    py::class_<hjb::PolynomialInventoryPenalty,
               hjb::InventoryPenalty,
               std::shared_ptr<hjb::PolynomialInventoryPenalty>>(
        m, "PolynomialInventoryPenalty")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double>(),
             py::arg("risk_aversion"),
             py::arg("sigma_ref"),
             py::arg("tau0"),
             py::arg("cubic_coeff"),
             py::arg("quartic_coeff"))
        .def_readwrite("risk_aversion", &hjb::PolynomialInventoryPenalty::risk_aversion)
        .def_readwrite("sigma_ref", &hjb::PolynomialInventoryPenalty::sigma_ref)
        .def_readwrite("tau0", &hjb::PolynomialInventoryPenalty::tau0)
        .def_readwrite("cubic_coeff", &hjb::PolynomialInventoryPenalty::cubic_coeff)
        .def_readwrite("quartic_coeff", &hjb::PolynomialInventoryPenalty::quartic_coeff)
        .def("value", &hjb::PolynomialInventoryPenalty::value, py::arg("q"));

    // ============================================================
    // Quote policy
    // ============================================================

    py::class_<hjb::QuotePolicy>(m, "QuotePolicy")
        .def(py::init<>())
        .def(py::init<
                 std::vector<double>,
                 std::vector<double>,
                 std::vector<double>,
                 std::vector<double>,
                 hjb::FlatCube4D,
                 hjb::FlatCube4D>(),
             py::arg("q_grid"),
             py::arg("y_grid"),
             py::arg("nu_grid"),
             py::arg("sizes"),
             py::arg("bid"),
             py::arg("ask"))
        .def_readwrite("q_grid", &hjb::QuotePolicy::q_grid)
        .def_readwrite("y_grid", &hjb::QuotePolicy::y_grid)
        .def_readwrite("nu_grid", &hjb::QuotePolicy::nu_grid)
        .def_readwrite("sizes", &hjb::QuotePolicy::sizes)
        .def_readwrite("bid", &hjb::QuotePolicy::bid)
        .def_readwrite("ask", &hjb::QuotePolicy::ask)
        .def("set_axes", &hjb::QuotePolicy::set_axes,
             py::arg("q_grid"),
             py::arg("y_grid"),
             py::arg("nu_grid"),
             py::arg("sizes"))
        .def("ensure_storage_shape", &hjb::QuotePolicy::ensure_storage_shape)
        .def("validate", &hjb::QuotePolicy::validate)
        .def("delta_at_index", &hjb::QuotePolicy::delta_at_index,
             py::arg("iq"),
             py::arg("iy"),
             py::arg("inu"),
             py::arg("iz"),
             py::arg("side"))
        .def("quote", &hjb::QuotePolicy::quote,
             py::arg("q"),
             py::arg("y"),
             py::arg("nu"),
             py::arg("z"),
             py::arg("side"));

    // ============================================================
    // Price tier
    // ============================================================

    py::class_<hjb::PriceTier, std::shared_ptr<hjb::PriceTier>>(m, "PriceTier")
        .def(py::init<>())
        .def(py::init<
                 std::string,
                 std::vector<double>,
                 std::shared_ptr<hjb::FlowCurve>,
                 std::shared_ptr<hjb::DriftJumpModel>,
                 double,
                 double>(),
             py::arg("name"),
             py::arg("sizes"),
             py::arg("flow_curve"),
             py::arg("jump_model"),
             py::arg("delta_min") = -0.5,
             py::arg("delta_max") = 4.0)
        .def_readwrite("name", &hjb::PriceTier::name)
        .def_readwrite("sizes", &hjb::PriceTier::sizes)
        .def_readwrite("flow_curve", &hjb::PriceTier::flow_curve)
        .def_readwrite("jump_model", &hjb::PriceTier::jump_model)
        .def_readwrite("delta_min", &hjb::PriceTier::delta_min)
        .def_readwrite("delta_max", &hjb::PriceTier::delta_max)
        .def_readwrite("policy", &hjb::PriceTier::policy)
        .def("validate", &hjb::PriceTier::validate)
        .def("arrival_rate", &hjb::PriceTier::arrival_rate,
             py::arg("delta"), py::arg("z"))
        .def("jump_size", &hjb::PriceTier::jump_size, py::arg("z"))
        .def("quote", &hjb::PriceTier::quote,
             py::arg("q"),
             py::arg("y"),
             py::arg("nu"),
             py::arg("z"),
             py::arg("side"));

    // ============================================================
    // Solver config
    // ============================================================

    py::class_<hjb::SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("q_grid", &hjb::SolverConfig::q_grid)
        .def_readwrite("y_grid", &hjb::SolverConfig::y_grid)
        .def_readwrite("nu_grid", &hjb::SolverConfig::nu_grid)
        .def_readwrite("dt", &hjb::SolverConfig::dt)
        .def_readwrite("n_iter", &hjb::SolverConfig::n_iter)
        .def_readwrite("kappa_y", &hjb::SolverConfig::kappa_y)
        .def_readwrite("kappa_nu", &hjb::SolverConfig::kappa_nu)
        .def_readwrite("nu_bar", &hjb::SolverConfig::nu_bar)
        .def_readwrite("eta_nu", &hjb::SolverConfig::eta_nu)
        .def_readwrite("golden_tol", &hjb::SolverConfig::golden_tol)
        .def_readwrite("golden_max_iter", &hjb::SolverConfig::golden_max_iter)
        .def_readwrite("early_stop", &hjb::SolverConfig::early_stop)
        .def_readwrite("tol_h", &hjb::SolverConfig::tol_h)
        .def_readwrite("tol_rhs", &hjb::SolverConfig::tol_rhs)
        .def_readwrite("min_iter", &hjb::SolverConfig::min_iter)
        .def_readwrite("consecutive_passes_required",
                       &hjb::SolverConfig::consecutive_passes_required)
        .def("validate", &hjb::SolverConfig::validate);

    // ============================================================
    // Solution
    // ============================================================

    py::class_<hjb::HJBSolution>(m, "HJBSolution")
        .def(py::init<>())
        .def_readwrite("h", &hjb::HJBSolution::h)
        .def_readwrite("q_grid", &hjb::HJBSolution::q_grid)
        .def_readwrite("y_grid", &hjb::HJBSolution::y_grid)
        .def_readwrite("nu_grid", &hjb::HJBSolution::nu_grid)
        .def_readwrite("tiers", &hjb::HJBSolution::tiers)
        .def_readwrite("converged", &hjb::HJBSolution::converged)
        .def_readwrite("iterations_used", &hjb::HJBSolution::iterations_used)
        .def_readwrite("final_max_h_change", &hjb::HJBSolution::final_max_h_change)
        .def_readwrite("final_max_rhs", &hjb::HJBSolution::final_max_rhs)
        .def_readwrite("history_max_h_change", &hjb::HJBSolution::history_max_h_change)
        .def_readwrite("history_max_rhs", &hjb::HJBSolution::history_max_rhs);

    // ============================================================
    // Solver
    // ============================================================

    py::class_<hjb::HJBLadderSolver>(m, "HJBLadderSolver")
        .def(py::init<
                 hjb::SolverConfig,
                 std::shared_ptr<hjb::InventoryPenalty>,
                 std::vector<std::shared_ptr<hjb::PriceTier>>>(),
             py::arg("config"),
             py::arg("penalty"),
             py::arg("tiers"))
        .def_readwrite("config", &hjb::HJBLadderSolver::config)
        .def_readwrite("penalty", &hjb::HJBLadderSolver::penalty)
        .def_readwrite("tiers", &hjb::HJBLadderSolver::tiers)
        .def("initialize_policies", &hjb::HJBLadderSolver::initialize_policies)
        .def("update_policies",
             py::overload_cast<const hjb::FlatCube3D&>(
                 &hjb::HJBLadderSolver::update_policies),
             py::arg("h_mat"))
        .def("solve", &hjb::HJBLadderSolver::solve);

    // ============================================================
    // Free helper functions
    // ============================================================

    m.def("clamp", &hjb::clamp,
          py::arg("x"), py::arg("lo"), py::arg("hi"));

    m.def("parse_side", &hjb::parse_side, py::arg("side"));
    m.def("side_name", &hjb::side_name, py::arg("side"));

    m.def("interp_linear", &hjb::interp_linear,
          py::arg("grid"), py::arg("vals"), py::arg("x"));

    m.def("interp_trilinear", &hjb::interp_trilinear,
          py::arg("q_grid"),
          py::arg("y_grid"),
          py::arg("nu_grid"),
          py::arg("vals"),
          py::arg("q"),
          py::arg("y"),
          py::arg("nu"));

    m.def("interp_trilinear_rung", &hjb::interp_trilinear_rung,
          py::arg("q_grid"),
          py::arg("y_grid"),
          py::arg("nu_grid"),
          py::arg("vals"),
          py::arg("q"),
          py::arg("y"),
          py::arg("nu"),
          py::arg("iz"));

    m.def("max_abs_vec", &hjb::max_abs_vec, py::arg("x"));
    m.def("max_abs_3d", &hjb::max_abs_3d, py::arg("x"));
}