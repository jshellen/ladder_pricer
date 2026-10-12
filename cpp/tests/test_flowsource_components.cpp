#include "ladder_pricer/core.hpp"
#include <stdexcept>
#define assert(condition) do { if (!(condition)) throw std::runtime_error("check failed: " #condition); } while (false)
#include <cmath>
#include <iostream>
#include <random>
using namespace ladder_pricer;
int main() {
    const DiscreteRfqDistribution d(0.30, {1.,2.,4.}, {0.2,0.3,0.5});
    const LogisticWinProbability win(0.52, 8., 0.02);
    const FXMapping mapping(1.25, 1.20);
    FlowSource source("crossed", d, win, mapping);
    assert(source.total_rfq_rate() == .30);
    auto atoms = source.rfq_atoms();
    assert(atoms.size() == 3);
    for (std::size_t i=0; i<3; ++i) {
        assert(std::abs(atoms[i].target_size - d.source_sizes()[i]/1.2) < 1e-12);
        assert(std::abs(atoms[i].one_side_arrival_rate - .15*d.probabilities()[i]) < 1e-12);
    }
    const double delta=.27, size=2./1.2;
    const double expected = win.win_probability(mapping.implied_delta(delta), 2.);
    assert(std::abs(source.win_probability(delta,size)-expected)<1e-14);
    const auto before=source.rfq_atoms();
    // Pricing grids are no longer an input to FlowSource.
    assert(source.total_rfq_rate()==.30);
    const auto after=source.rfq_atoms();
    assert(before.size()==after.size());
    for(std::size_t i=0;i<before.size();++i){
        assert(before[i].target_size==after[i].target_size);
        assert(before[i].one_side_arrival_rate==after[i].one_side_arrival_rate);
    }
    std::mt19937_64 a(1234),b(1234);
    FlowSource other("crossed",d,win,mapping);
    for(int i=0;i<10000;++i)
        assert(source.sample_rfq_target_size(a)==other.sample_rfq_target_size(b));
    bool rejects=false;
    try { FXMapping invalid(-1., 1.); } catch (const std::invalid_argument&) { rejects=true; }
    assert(rejects);
    std::cout << "FlowSource components and pricing-grid invariance passed\n";
}
