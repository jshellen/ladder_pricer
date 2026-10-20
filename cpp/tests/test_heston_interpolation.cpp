#include "ladder_pricer/heston_interpolation.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    using ladder_pricer::bracket_variance;
    const std::vector<double> sigma{2.0, 4.0, 8.0};
    const auto below = bracket_variance(sigma, 0.0);
    assert(below.lower == 0 && below.upper == 0 && below.blend(3.0, 100.0) == 3.0);
    const auto above = bracket_variance(sigma, 200.0);
    assert(above.lower == 2 && above.upper == 2);
    const auto exact = bracket_variance(sigma, 16.0);
    assert(exact.blend(5.0, 9.0) == 9.0);
    const auto midpoint = bracket_variance(sigma, 10.0);
    assert(midpoint.lower == 0 && midpoint.upper == 1);
    assert(std::abs(midpoint.upper_weight - .5) < 1e-12);
    assert(midpoint.blend(2.0, 10.0) == 6.0);
    assert(midpoint.choose(.49) == 1 && midpoint.choose(.51) == 0);
    // Sigma interpolation would give a DIFFERENT weight: must interpolate in v.
    const auto skew = bracket_variance(sigma, 28.0);
    assert(skew.lower == 1 && skew.upper == 2);
    assert(std::abs(skew.upper_weight - .25) < 1e-12);
    bool threw = false;
    try { (void)bracket_variance(sigma, std::nan("")); }
    catch(const std::invalid_argument&) { threw = true; }
    assert(threw);
}
