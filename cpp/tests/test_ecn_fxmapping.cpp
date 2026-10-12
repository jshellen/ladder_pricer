#include <algorithm>
#include "ladder_pricer/core.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ladder_pricer;

static void check(double a, double b, const char* label) {
    if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b)>1e-12*std::max({1.0,std::abs(a),std::abs(b)})) {
        std::cerr << label << ": " << a << " vs " << b << '\n';
        throw std::runtime_error("FX mapping regression failed");
    }
}
int main() {
    const std::vector<double> probs{0.2,0.25,0.3,0.15,0.1};
    for (double scale: {1.0, 0.75, 1.25}) {
        for (double size_scale: {1.0, 1.18, 0.83}) {
            FXMapping mapping(scale,size_scale);

            ECNFlowSource component("USDSEK", ExponentialFlow(0.15,2.4),mapping,probs);
            for(double d: {-0.2,0.0,0.15,0.35,0.5}) {
                check(component.implied_delta(d),mapping.implied_delta(d),"shared delta");
            }
            for(double size: {0.1,0.25,0.5,1.0,2.0}) {
                const auto a=component.fill_components(size);
                for (std::size_t i=0;i<a.size();++i) {
                    if (a[i].first <= 0 || a[i].second < 0) throw std::runtime_error("invalid fill component");
                }
            }
            for(double r: {0.0,0.2,0.5,1.2}) {
                check(component.target_reach(r),mapping.target_reach(r),"shared reach");
            }
        }
    }
    std::cout << "ECN FXMapping checks: PASS (9 mappings)\n";
}
