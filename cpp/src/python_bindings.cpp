#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ladder_pricer/analytics.hpp"
#include "ladder_pricer/howard.hpp"
#include "ladder_pricer/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace ladder_pricer::python {
namespace {

double number(const py::dict& object, const char* key) {
    return py::cast<double>(object[py::str(key)]);
}

bool boolean(const py::dict& object, const char* key) {
    return py::cast<bool>(object[py::str(key)]);
}

std::string text(const py::dict& object, const char* key) {
    return py::cast<std::string>(object[py::str(key)]);
}

std::vector<double> numbers(const py::handle& value) {
    return py::cast<std::vector<double>>(value);
}

std::vector<double> build_grid(const py::dict& spec) {
    const std::string mode = text(spec, "mode");
    const double qmax = number(spec, "maxAbs");
    if (qmax <= 0.0) throw std::invalid_argument("inventory maxAbs must be positive");

    std::vector<double> positive;
    if (mode == "uniform") {
        const double step = number(spec, "step");
        if (step <= 0.0) throw std::invalid_argument("inventory step must be positive");
        for (double q = 0.0; q <= qmax + 0.5 * step; q += step) {
            if (q <= qmax + 1e-10) positive.push_back(q);
        }
    } else if (mode == "piecewise") {
        const double half = number(spec, "fineHalfWidth");
        const double fine = number(spec, "fineStep");
        const double coarse = number(spec, "coarseStep");
        if (fine <= 0.0 || coarse <= 0.0 || half < 0.0 || half > qmax) {
            throw std::invalid_argument("invalid piecewise inventory grid");
        }
        for (double q = 0.0; q <= half + 0.5 * fine; q += fine) {
            if (q <= half + 1e-10) positive.push_back(q);
        }
        for (double q = half + coarse; q <= qmax + 0.5 * coarse; q += coarse) {
            if (q <= qmax + 1e-10) positive.push_back(q);
        }
    } else {
        throw std::invalid_argument("inventory grid mode must be 'uniform' or 'piecewise'");
    }

    if (positive.empty() || std::abs(positive.back() - qmax) > 1e-10) positive.push_back(qmax);

    std::vector<double> states;
    states.reserve(2 * positive.size() - 1);
    for (auto it = positive.rbegin(); it != positive.rend() - 1; ++it) states.push_back(-*it);
    states.insert(states.end(), positive.begin(), positive.end());
    return states;
}

std::optional<DarkPool> build_dark_pool(const py::dict& cfg) {
    if (!boolean(cfg, "enabled")) return std::nullopt;

    const std::string distribution = text(cfg, "distribution");
    std::shared_ptr<ArrivalDistribution> bid;
    std::shared_ptr<ArrivalDistribution> ask;
    if (distribution == "geometric") {
        bid = std::make_shared<GeometricArrival>(number(cfg, "lambdaBid"), number(cfg, "pBid"));
        ask = std::make_shared<GeometricArrival>(number(cfg, "lambdaAsk"), number(cfg, "pAsk"));
    } else if (distribution == "zip") {
        bid = std::make_shared<ZeroInflatedPoissonArrival>(number(cfg, "lambdaBid"), number(cfg, "muBid"), number(cfg, "p0Bid"));
        ask = std::make_shared<ZeroInflatedPoissonArrival>(number(cfg, "lambdaAsk"), number(cfg, "muAsk"), number(cfg, "p0Ask"));
    } else {
        throw std::invalid_argument("dark-pool distribution must be 'geometric' or 'zip'");
    }

    return DarkPool(
        std::move(bid), std::move(ask), number(cfg, "feeBid"), number(cfg, "feeAsk"),
        numbers(cfg[py::str("postedSizes")]), boolean(cfg, "allowBothSides"));
}

PricingProblem build_problem(const py::dict& cfg, std::optional<double> gamma_override = std::nullopt) {
    std::vector<Tier> tiers;
    const py::list tier_cfgs = py::cast<py::list>(cfg[py::str("tiers")]);
    for (const py::handle item : tier_cfgs) {
        const py::dict tier = py::cast<py::dict>(item);
        if (!boolean(tier, "enabled")) continue;
        const py::dict flow = py::cast<py::dict>(tier[py::str("flow")]);
        const py::dict markout = py::cast<py::dict>(tier[py::str("markout")]);
        tiers.emplace_back(
            text(tier, "name"), numbers(tier[py::str("sizes")]),
            LogisticFlow(number(flow, "A0"), number(flow, "theta"), number(flow, "beta"),
                         number(flow, "shift"), number(flow, "steepness"), number(flow, "volumeShift")),
            SaturatingMarkout(number(markout, "impactScalePips") / 10000.0,
                              number(markout, "sizeExponent"), number(markout, "tauMinutes")),
            boolean(tier, "useMarkout"), number(tier, "deltaMin"), number(tier, "deltaMax"));
    }
    if (tiers.empty() && !boolean(py::cast<py::dict>(cfg[py::str("darkPool")]), "enabled")) {
        throw std::invalid_argument("enable at least one pricing tier or the dark pool");
    }

    const py::dict internal = py::cast<py::dict>(cfg[py::str("internalization")]);
    const double sigma = number(cfg, "sigmaPips") / 10000.0;
    const double gamma = gamma_override.value_or(number(cfg, "gamma"));
    return PricingProblem(
        build_grid(py::cast<py::dict>(cfg[py::str("grid")])),
        number(cfg, "spreadPips") / 10000.0,
        number(cfg, "spotDrift"),
        QuadraticPenalty(gamma, sigma),
        InternalizationTime(number(internal, "tau0"), number(internal, "tau1"), number(internal, "tau2")),
        std::move(tiers),
        build_dark_pool(py::cast<py::dict>(cfg[py::str("darkPool")])));
}

py::list matrix_value(const std::vector<std::vector<double>>& matrix) {
    py::list rows;
    for (const auto& row : matrix) rows.append(row);
    return rows;
}

py::dict solution_value(const Solution& solution, const PricingProblem& problem) {
    py::dict out;
    out["qGrid"] = solution.q_grid;
    out["value"] = solution.value;
    out["averageReward"] = solution.average_reward;
    out["hardInventoryLimit"] = solution.hard_inventory_limit;
    out["converged"] = solution.diagnostics.converged;
    out["iterations"] = solution.diagnostics.iterations;
    out["residual"] = solution.diagnostics.reward_upper_bound - solution.diagnostics.reward_lower_bound;
    out["valueChange"] = solution.diagnostics.value_change;
    out["bellmanResidual"] = solution.diagnostics.bellman_residual;

    py::list tier_values;
    for (std::size_t k = 0; k < solution.tier_policies.size(); ++k) {
        py::dict tier;
        tier["name"] = problem.tiers()[k].name();
        tier["sizes"] = solution.tier_policies[k].sizes;
        tier["bid"] = matrix_value(solution.tier_policies[k].bid);
        tier["ask"] = matrix_value(solution.tier_policies[k].ask);
        tier_values.append(std::move(tier));
    }
    out["tiers"] = std::move(tier_values);

    if (solution.dark_pool_policy && problem.dark_pool()) {
        const auto& policy = *solution.dark_pool_policy;
        py::dict dark;
        dark["qGrid"] = policy.q_grid;
        dark["bidSize"] = policy.bid_size;
        dark["askSize"] = policy.ask_size;
        dark["bidActive"] = policy.bid_active;
        dark["askActive"] = policy.ask_active;
        dark["postedSizes"] = problem.dark_pool()->posted_sizes();
        dark["allowBothSides"] = problem.dark_pool()->allow_both_sides();
        dark["bidIntensity"] = problem.dark_pool()->arrivals(Side::Bid).arrival_intensity();
        dark["askIntensity"] = problem.dark_pool()->arrivals(Side::Ask).arrival_intensity();
        dark["bidDistribution"] = problem.dark_pool()->arrivals(Side::Bid).name();
        dark["askDistribution"] = problem.dark_pool()->arrivals(Side::Ask).name();
        dark["bidFee"] = problem.dark_pool()->fee(Side::Bid);
        dark["askFee"] = problem.dark_pool()->fee(Side::Ask);
        out["darkPool"] = std::move(dark);
    } else {
        out["darkPool"] = py::none();
    }
    return out;
}

py::dict stats_value(const PnlStatistics& statistics) {
    py::dict out;
    out["meanQuote"] = statistics.expected_quote_ccy;
    out["stdQuote"] = statistics.std_quote_ccy;
    out["meanBase"] = statistics.expected_base_ccy;
    out["stdBase"] = statistics.std_base_ccy;
    out["referenceSpot"] = statistics.reference_spot;
    return out;
}

py::dict fill_value(const FillEvent& fill) {
    py::dict out;
    out["time"] = fill.time_minutes;
    out["tier"] = fill.tier;
    out["side"] = fill.side;
    out["size"] = fill.size;
    out["price"] = fill.execution_price;
    out["inventoryBefore"] = fill.inventory_before;
    out["inventoryAfter"] = fill.inventory_after;
    return out;
}

py::dict path_value(const SamplePath& path) {
    py::dict out;
    out["times"] = path.times;
    out["spots"] = path.spots;
    out["inventories"] = path.inventories;
    py::list fills;
    for (const auto& fill : path.fills) fills.append(fill_value(fill));
    out["fills"] = std::move(fills);
    return out;
}

py::dict monte_carlo_value(const MonteCarloResult& result) {
    py::dict out;
    out["pnlQuote"] = result.pnl_quote_ccy;
    out["pnlBase"] = result.pnl_base_ccy;
    out["finalInventory"] = result.final_inventory;
    out["finalSpot"] = result.final_spot;
    out["tradeCount"] = result.trade_count;
    out["times"] = result.sample_times;
    out["inventoryLower"] = result.inventory_lower;
    out["inventoryMedian"] = result.inventory_median;
    out["inventoryUpper"] = result.inventory_upper;
    py::list paths;
    for (const auto& path : result.sample_paths) paths.append(path_value(path));
    out["samplePaths"] = std::move(paths);
    return out;
}

}  // namespace

class Engine {
public:
    explicit Engine(py::dict config)
        : config_(std::move(config)),
          reference_spot_(number(config_, "spot")),
          problem_(build_problem(config_)) {}

    py::dict solve() {
        {
            py::gil_scoped_release release;
            solution_ = HowardSolver(problem_).solve();
        }
        return solution_value(*solution_, problem_);
    }

    py::dict statistics(double horizon_minutes, double initial_inventory = 0.0) {
        ensure_solved();
        const double sigma = number(config_, "sigmaPips") / 10000.0;
        PnlStatistics statistics;
        {
            py::gil_scoped_release release;
            statistics = PnlAnalytics(problem_, *solution_, reference_spot_)
                             .statistics(horizon_minutes, sigma, initial_inventory);
        }
        return stats_value(statistics);
    }

    py::dict simulate(double horizon_minutes, int paths, double initial_inventory,
                      std::uint64_t seed, int retained_paths = 6, int sample_points = 191) {
        ensure_solved();
        const double sigma = number(config_, "sigmaPips") / 10000.0;
        MonteCarloResult result;
        {
            py::gil_scoped_release release;
            result = MonteCarloSimulator(problem_, *solution_, reference_spot_)
                         .run(horizon_minutes, paths, sigma, initial_inventory, seed,
                              retained_paths, sample_points);
        }
        return monte_carlo_value(result);
    }

    py::list frontier(const std::vector<double>& gamma_values, double horizon_minutes,
                      double initial_inventory = 0.0) const {
        const double sigma = number(config_, "sigmaPips") / 10000.0;
        struct Point {
            double gamma;
            double mean;
            double std;
            double ratio;
            int iterations;
        };
        std::vector<Point> points;
        points.reserve(gamma_values.size());
        for (const double gamma : gamma_values) {
            // Reading the Python config requires the GIL. Heavy numerical work does not.
            PricingProblem problem = build_problem(config_, gamma);
            Solution solution;
            PnlStatistics statistics;
            {
                py::gil_scoped_release release;
                solution = HowardSolver(problem).solve();
                statistics = PnlAnalytics(problem, solution, reference_spot_)
                                 .statistics(horizon_minutes, sigma, initial_inventory);
            }
            points.push_back(Point{
                gamma,
                statistics.expected_base_ccy,
                statistics.std_base_ccy,
                statistics.std_base_ccy > 0.0
                    ? statistics.expected_base_ccy / statistics.std_base_ccy
                    : 0.0,
                solution.diagnostics.iterations,
            });
        }
        py::list out;
        for (const auto& point : points) {
            py::dict item;
            item["gamma"] = point.gamma;
            item["mean"] = point.mean;
            item["std"] = point.std;
            item["ratio"] = point.ratio;
            item["iterations"] = point.iterations;
            out.append(std::move(item));
        }
        return out;
    }

private:
    void ensure_solved() {
        if (solution_) return;
        py::gil_scoped_release release;
        solution_ = HowardSolver(problem_).solve();
    }

    py::dict config_;
    double reference_spot_;
    PricingProblem problem_;
    std::optional<Solution> solution_;
};

}  // namespace ladder_pricer::python

PYBIND11_MODULE(_native, module) {
    module.doc() = "Trinity 2.0 C++ Howard pricing engine";
    py::class_<ladder_pricer::python::Engine>(module, "Engine")
        .def(py::init<py::dict>())
        .def("solve", &ladder_pricer::python::Engine::solve)
        .def("statistics", &ladder_pricer::python::Engine::statistics,
             py::arg("horizon_minutes"), py::arg("initial_inventory") = 0.0)
        .def("simulate", &ladder_pricer::python::Engine::simulate,
             py::arg("horizon_minutes"), py::arg("paths"), py::arg("initial_inventory"),
             py::arg("seed"), py::arg("retained_paths") = 6, py::arg("sample_points") = 191)
        .def("frontier", &ladder_pricer::python::Engine::frontier,
             py::arg("gamma_values"), py::arg("horizon_minutes"),
             py::arg("initial_inventory") = 0.0);
}
