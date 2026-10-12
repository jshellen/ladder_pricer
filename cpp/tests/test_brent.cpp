#include "ladder_pricer/detail/brent.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using ladder_pricer::detail::bounded_brent_minimize;

int main() {
    int evaluations = 0;
    auto quadratic = [&](double x) { ++evaluations; return (x - 0.137) * (x - 0.137) + 2; };
    const auto q = bounded_brent_minimize(quadratic, -1., 1., 30);
    assert(std::abs(q.first - 0.137) < 1e-7 && std::abs(q.second - 2) < 1e-12);
    assert(evaluations < 40);
    const auto shifted = bounded_brent_minimize([](double x) { return std::pow(x - 1000.42, 2); }, 1000., 1001., 30);
    assert(std::abs(shifted.first - 1000.42) < 1e-5);
    const auto boundary = bounded_brent_minimize([](double x) { return x; }, -2., 3., 25);
    assert(std::abs(boundary.first + 2) < 1e-5);
    const auto flat = bounded_brent_minimize([](double) { return 7.; }, 0., 1.);
    assert(flat.second == 7.);
    const auto singleton = bounded_brent_minimize([](double x) { return x*x; }, 2., 2.);
    assert(singleton.first == 2. && singleton.second == 4.);
    bool threw = false;
    try { bounded_brent_minimize([](double x) { return x; }, 2., 1.); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    threw = false;
    try { bounded_brent_minimize([](double) { return std::nan(""); }, 0., 1.); } catch (const std::domain_error&) { threw = true; }
    assert(threw);
    std::cout << "Brent tests passed; quadratic evaluations=" << evaluations << "\n";
}
