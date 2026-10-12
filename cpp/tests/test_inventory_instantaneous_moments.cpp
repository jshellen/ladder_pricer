#include "ladder_pricer/core.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ladder_pricer;

// Small-time independent compound-Poisson experiment. Each draw models a
// single exogenous event using the same size/reach samplers as simulation.cpp.
// Compare generator coefficients and exact finite-dt compound-Poisson moments, not the variance
// of inventory over a long horizon (where policies and inventory change).
struct Moments { double first=0, second=0; };
void check(const std::string& name, Moments expected, Moments actual,
           double n, double tolerance_sigma=6.0) {
    // Conservative stochastic bound based on sample fourth moments is handled
    // by the caller using its empirical standard errors below.
    (void)tolerance_sigma;
    std::cout << name << " generator=(" << expected.first << "," << expected.second
              << ") sampled=(" << actual.first << "," << actual.second << ") n=" << n << "\n";
}
void close(const std::string& name, double observed, double theory,
           double sample_variance, int n) {
    const double se=std::sqrt(std::max(0.0,sample_variance)/n);
    const double tol=7.0*se+1e-4;
    if(std::abs(observed-theory)>tol) {
        throw std::runtime_error(name+" expected="+std::to_string(theory)+
                                 " observed="+std::to_string(observed)+
                                 " tolerance="+std::to_string(tol));
    }
}
struct Samples {
    double m1=0,m2=0,m3=0,m4=0;
    void add(double x) {m1+=x;m2+=x*x;m3+=x*x*x;m4+=x*x*x*x;}
    void verify(const std::string& title, double e1, double e2, int n) {
        m1/=n;m2/=n;m3/=n;m4/=n;
        close(title+" drift",m1,e1,m2-m1*m1,n);
        close(title+" second raw",m2,e2,m4-m2*m2,n);
        check(title,{e1,e2},{m1,m2},n);
    }
};
int main() {
    constexpr int n=250000;
    std::mt19937_64 rng(20261009);
    std::uniform_real_distribution<double> u(0,1);
    // Fixed q=0, delta=0.30, dt=0.01 minute. Both bid and ask are
    // admissible. RFQ requests arrive at pooled rate Lambda, split 50/50.
    const FlowSource rfq("USDSEK",DiscreteRfqDistribution(4.0,{1,2,4},{.2,.5,.3}),
                         LogisticWinProbability(.5,6.0,.03),FXMapping(1,1.2));
    const double dt=.01, delta=.30, lambda=rfq.total_rfq_rate();
    double r1=0,r2=0;
    for(const auto& atom:rfq.rfq_atoms()) {
        const double hit=rfq.win_probability(delta,atom.target_size);
        // Equal bid/ask intensity; odd inventory drift must cancel.
        r2+=2*atom.one_side_arrival_rate*hit*atom.target_size*atom.target_size;
    }
    Samples rfq_samples;
    std::poisson_distribution<int> rfq_clock(lambda*dt);
    for(int i=0;i<n;++i) {
        double dq=0;
        const int count=rfq_clock(rng);
        for(int k=0;k<count;++k) {
            const double z=rfq.sample_rfq_target_size(rng);
            const double side=u(rng)<.5?1.0:-1.0;
            if(u(rng)<rfq.win_probability(delta,z))dq+=side*z;
        }
        rfq_samples.add(dq);
    }
    rfq_samples.verify("RFQ crossed",r1*dt,r2*dt+(r1*dt)*(r1*dt),n);

    // q=+5, risk-reducing ask ECN posted z=1. Parent arrivals at A;
    // sampled reach determines hit, and source size is capped before mapping.
    const ECNFlowSource ecn("USDSEK",ExponentialFlow(8.0,2.4),FXMapping(.8,1.2),
                            {0.15,0.20,0.25,0.20,0.20});
    const double depth=.5-delta, rate=ecn.flow().a();
    const double hit=std::exp(-ecn.flow().k()*ecn.delta_scale()*depth);
    double e1=0,e2=0;
    for(auto [z,prob]:ecn.fill_components(1.0)) {
        e1-=rate*hit*prob*z;
        e2+=rate*hit*prob*z*z;
    }
    Samples ecn_samples;
    std::poisson_distribution<int> ecn_clock(rate*dt);
    std::exponential_distribution<double> reach(ecn.flow().k());
    for(int i=0;i<n;++i) {
        double dq=0;
        for(int k=ecn_clock(rng);k>0;--k){
            const double source_reach=reach(rng);
            const double source_size=ecn.sample_trade_size(rng);
            if(source_reach+1e-12>=ecn.delta_scale()*depth)
                dq-=ecn.mapping().target_size(std::min(source_size,ecn.mapping().source_size(1.0)));
        }
        ecn_samples.add(dq);
    }
    ecn_samples.verify("ECN crossed partial fills",e1*dt,e2*dt+(e1*dt)*(e1*dt),n);

    // q=+5, risk-reducing ask darkpool posted z=2. The zero-inflated
    // incoming clock remains exogenous even when no fill occurs.
    const ZeroInflatedPoissonArrival dark(10.0,2.9,.35);
    double d1=0,d2=0;
    for(auto [size,arrival]:dark.fill_rates(2)) {
        d1-=arrival*size;
        d2+=arrival*size*size;
    }
    Samples dark_samples;
    std::poisson_distribution<int> dark_clock(dark.arrival_intensity()*dt);
    for(int i=0;i<n;++i){
        double dq=0;
        for(int k=dark_clock(rng);k>0;--k)
            dq-=std::min(2,dark.sample_incoming_size(rng));
        dark_samples.add(dq);
    }
    dark_samples.verify("Darkpool ZIP partial fills",d1*dt,d2*dt+(d1*dt)*(d1*dt),n);
    std::cout<<"All channel-wise instantaneous moment comparisons passed\n";
}
