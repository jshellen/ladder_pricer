#include "ladder_pricer/analytics.hpp"
#include "ladder_pricer/policy_operator.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

namespace ladder_pricer {
namespace {

constexpr double kTol = 1e-10;

double direction(Side side) noexcept { return side == Side::Bid ? 1.0 : -1.0; }

const std::vector<double>& row(const LadderPolicy& p, Side side, std::size_t i) {
    return side == Side::Bid ? p.bid[i] : p.ask[i];
}


struct Weights { std::size_t left, right; double wl, wr; };
Weights weights(const std::vector<double>& grid, double q) {
    if (q <= grid.front() + kTol) return {0,0,1,0};
    if (q >= grid.back() - kTol) { const auto i=grid.size()-1; return {i,i,1,0}; }
    auto it=std::lower_bound(grid.begin(),grid.end(),q);
    const auto r=static_cast<std::size_t>(std::distance(grid.begin(),it));
    if (std::abs(grid[r]-q)<=kTol) return {r,r,1,0};
    const auto l=r-1; const double wr=(q-grid[l])/(grid[r]-grid[l]);
    return {l,r,1-wr,wr};
}

using Exponent = std::vector<int>;

std::vector<Exponent> monomials(int d) {
    std::vector<Exponent> out;
    out.push_back(Exponent(d,0));
    for(int r=0;r<d;++r){ Exponent e(d,0); e[r]=1; out.push_back(e); }
    for(int r=0;r<d;++r) for(int u=r;u<d;++u){ Exponent e(d,0); ++e[r]; ++e[u]; out.push_back(e); }
    return out;
}

int monomial_index(const std::vector<Exponent>& mons, const Exponent& target) {
    for(std::size_t i=0;i<mons.size();++i) if(mons[i]==target) return static_cast<int>(i);
    throw std::runtime_error("moment monomial not found");
}

std::vector<std::pair<Exponent,double>> shifted_expansion(const Exponent& exponent,
                                                          const std::vector<double>& shift) {
    std::vector<std::pair<Exponent,double>> terms{{Exponent(exponent.size(),0),1.0}};
    for(std::size_t r=0;r<exponent.size();++r){
        const int power=exponent[r];
        if(power==0) continue;
        std::vector<std::pair<Exponent,double>> next;
        for(const auto& [base,c0]:terms){
            for(int k=0;k<=power;++k){
                Exponent e=base; e[r]+=k;
                const double comb = power==2 && k==1 ? 2.0 : 1.0;
                const double coeff=c0*comb*std::pow(shift[r],power-k);
                bool merged=false;
                for(auto& [existing,c]:next){ if(existing==e){ c+=coeff; merged=true; break; } }
                if(!merged) next.push_back({std::move(e),coeff});
            }
        }
        terms=std::move(next);
    }
    return terms;
}

struct MomentEvent { double target_q; double rate; std::vector<double> shift; };

std::vector<double> fill_breakpoints(const std::vector<double>& grid, std::size_t i,
                                     double dir, double posted_size) {
    std::vector<double> out;
    const double q = grid[i];
    if (dir > 0.0) {
        for (std::size_t j = i + 1; j < grid.size(); ++j) {
            const double x = grid[j] - q;
            if (x >= posted_size - kTol) break;
            if (x > kTol) out.push_back(x);
        }
    } else {
        for (std::size_t j = i; j-- > 0;) {
            const double x = q - grid[j];
            if (x >= posted_size - kTol) break;
            if (x > kTol) out.push_back(x);
        }
    }
    return out;
}

}  // namespace

PnlStatistics PnlAnalytics::statistics(double horizon, double q0, int initial_regime) const {
    if(horizon<0.0) throw std::invalid_argument("invalid PnL analytics settings");
    if(reference_spot_<=0.0) throw std::invalid_argument("reference spot must be positive");
    const auto& qgrid=problem_.grid().states();
    const std::size_t n=qgrid.size();
    const std::size_t R=solution_.policies.size();
    if(R<2 || solution_.sigma_pips.size()!=R || switches_.size()!=R)
        throw std::runtime_error("invalid volatility regime analytics configuration");
    if(initial_regime<0) initial_regime=static_cast<int>(R/2);
    if(initial_regime>=static_cast<int>(R)) throw std::invalid_argument("initial regime out of range");
    for(const auto& policy:solution_.policies)
        if(policy.tiers.size()!=problem_.tiers().size()) throw std::runtime_error("solution/model mismatch");

    std::vector<double> taus;
    for(const auto& tier:problem_.tiers()) if(tier.use_markout()) taus.push_back(tier.markout().tau_minutes());
    std::sort(taus.begin(),taus.end());
    taus.erase(std::unique(taus.begin(),taus.end(),[](double a,double b){return std::abs(a-b)<1e-12;}),taus.end());
    const int nimpact=static_cast<int>(taus.size());
    const int pnlvar=nimpact;
    const int nvars=nimpact+1;
    const auto mons=monomials(nvars);
    const std::size_t nm=mons.size();
    const std::size_t dim=R*n*nm;
    SparseMatrix A(dim,dim);
    auto smi=[&](std::size_t qi,int mi,std::size_t r){return (r*n+qi)*nm+static_cast<std::size_t>(mi);};

    std::map<double,int> tau_to_var;
    for(int k=0;k<nimpact;++k) tau_to_var[taus[static_cast<std::size_t>(k)]]=k;

    for(std::size_t r=0;r<R;++r) {
    const auto& regime_policy=solution_.policies[r];
    const double sigma=solution_.sigma_pips[r]/10000.0;
    if(switches_[r].size()!=R) throw std::invalid_argument("invalid regime transition matrix");
    for(std::size_t i=0;i<n;++i){
        const double q=qgrid[i];
        std::vector<std::vector<double>> drift_linear(nvars,std::vector<double>(nvars,0.0));
        for(int k=0;k<nimpact;++k) drift_linear[k][k]=-1.0/taus[static_cast<std::size_t>(k)];
        for(int k=0;k<nimpact;++k) drift_linear[pnlvar][k]=q/taus[static_cast<std::size_t>(k)];
        const double pnl_diff_var=(sigma*q)*(sigma*q);

        for(std::size_t tm=0;tm<nm;++tm){
            const auto& exponent=mons[tm];
            for(int var=0;var<nvars;++var){
                const int power=exponent[static_cast<std::size_t>(var)];
                if(power==0) continue;
                Exponent reduced=exponent; --reduced[static_cast<std::size_t>(var)];
                for(int u=0;u<nvars;++u){
                    const double coeff=drift_linear[static_cast<std::size_t>(var)][static_cast<std::size_t>(u)];
                    if(coeff==0.0) continue;
                    Exponent source=reduced; ++source[static_cast<std::size_t>(u)];
                    const int sm=monomial_index(mons,source);
                    A.add(smi(i,sm,r),smi(i,static_cast<int>(tm),r),power*coeff);
                }
            }
            const int pp=exponent[static_cast<std::size_t>(pnlvar)];
            if(pp>=2 && pnl_diff_var>0.0){
                Exponent source=exponent; source[static_cast<std::size_t>(pnlvar)]-=2;
                const int sm=monomial_index(mons,source);
                A.add(smi(i,sm,r),smi(i,static_cast<int>(tm),r),0.5*pp*(pp-1)*pnl_diff_var);
            }
        }

        std::vector<MomentEvent> events;
        for(std::size_t k=0;k<problem_.tiers().size();++k){
            const auto& tier=problem_.tiers()[k];
            const auto& policy=regime_policy.tiers[k];
            for(Side side:{Side::Bid,Side::Ask}){
                const double dir=direction(side); const auto& deltas=row(policy,side,i);
                for(std::size_t j=0;j<tier.sizes().size();++j){
                    const double z=tier.sizes()[j];
                    // MC generates information shocks for every exogenous RFQ,
                    // whether we win it or not, and even at the inventory cap.
                    const double lambda=tier.flow().rfq_arrival_rate(z);
                    const bool admissible=problem_.grid().admissible(q,dir*z);
                    const double delta=deltas[j];
                    const double rate=admissible?tier.flow().arrival_rate(delta,z):0.0;
                    const double lost_rate=std::max(0.0,lambda-rate);
                    if(rate>0.0){
                        std::vector<double> shift(nvars,0.0);
                        shift[static_cast<std::size_t>(pnlvar)]=z*(problem_.spread()*(0.5-delta)-tier.fee());
                        if(tier.use_markout()) shift[static_cast<std::size_t>(tau_to_var[tier.markout().tau_minutes()])]=-dir*tier.markout().asymptotic();
                        events.push_back({q+dir*z,rate,std::move(shift)});
                    }
                    if(lost_rate>0.0 && tier.use_markout()){
                        std::vector<double> shift(nvars,0.0);
                        shift[static_cast<std::size_t>(tau_to_var[tier.markout().tau_minutes()])]=-dir*tier.markout().asymptotic();
                        events.push_back({q,lost_rate,std::move(shift)});
                    }
                }
            }
        }
        if(problem_.dark_pool() && regime_policy.dark_pool){
            const auto& venue=*problem_.dark_pool(); const auto& p=*regime_policy.dark_pool;
            for(Side side:{Side::Bid,Side::Ask}){
                const bool active=side==Side::Bid?p.bid_active[i]:p.ask_active[i];
                const double posted=side==Side::Bid?p.bid_size[i]:p.ask_size[i];
                if(!active || posted<=0.0 || !venue.risk_reducing(q, side, posted)) continue;
                const int u=static_cast<int>(std::llround(posted)); const double dir=direction(side);
                for(const auto& [fill,rate]:venue.arrivals(side).fill_rates(u)){
                    if(rate<=0.0) continue;
                    std::vector<double> shift(nvars,0.0); shift[static_cast<std::size_t>(pnlvar)]=-venue.fee(side)*fill;
                    events.push_back({q+dir*fill,rate,std::move(shift)});
                }
            }
        }
        if(problem_.passive_ecn() && regime_policy.passive_ecn){
            const auto& venue=*problem_.passive_ecn(); const auto& p=*regime_policy.passive_ecn;
            const double z=venue.quote_size();
            for(Side side:{Side::Bid,Side::Ask}){
                const bool active=side==Side::Bid?p.bid_active[i]:p.ask_active[i];
                const double delta=side==Side::Bid?p.bid_delta[i]:p.ask_delta[i];
                if(!active) continue;
                const double dir=direction(side);
                auto breaks=fill_breakpoints(qgrid,i,dir,z);
                // ECN parent sizes are discrete; fill_components() returns the
                // exact finite fill distribution (breaks are retained only for
                // API compatibility with the shared continuation logic).
                for(const auto& source:venue.flow().sources()){
                    const double hit_rate=source.arrival_rate(delta);
                    if(hit_rate<=0.0) continue;
                    for(const auto& [fill,probability]:source.fill_components(z,breaks)){
                        const double rate=hit_rate*probability;
                        if(rate<=0.0) continue;
                        std::vector<double> shift(nvars,0.0);
                        shift[static_cast<std::size_t>(pnlvar)]=fill*(problem_.spread()*(0.5-delta)-venue.maker_fee());
                        events.push_back({q+dir*fill,rate,std::move(shift)});
                    }
                }
            }
        }

        for(const auto& event:events){
            const auto w=weights(qgrid,event.target_q);
            for(std::size_t tm=0;tm<nm;++tm){
                A.add(smi(i,static_cast<int>(tm),r),smi(i,static_cast<int>(tm),r),-event.rate);
                const auto expansion=shifted_expansion(mons[tm],event.shift);
                for(const auto& [source_exp,coeff]:expansion){
                    const int sm=monomial_index(mons,source_exp);
                    if(w.wl!=0.0) A.add(smi(i,sm,r),smi(w.left,static_cast<int>(tm),r),event.rate*w.wl*coeff);
                    if(w.wr!=0.0) A.add(smi(i,sm,r),smi(w.right,static_cast<int>(tm),r),event.rate*w.wr*coeff);
                }
            }
        }
        // Regime switches preserve inventory, price and accumulated markout/PnL.
        // Only the instantaneous Brownian variance and quote policy change.
        for(std::size_t other=0;other<R;++other) if(other!=r) {
            const double rate=switches_[r][other];
            if(rate<0.0 || !std::isfinite(rate)) throw std::invalid_argument("invalid switching rate");
            for(std::size_t tm=0;tm<nm;++tm) {
                A.add(smi(i,static_cast<int>(tm),r),smi(i,static_cast<int>(tm),r),-rate);
                A.add(smi(i,static_cast<int>(tm),r),smi(i,static_cast<int>(tm),other),rate);
            }
        }
    }
    }

    const Exponent zero_exp(nvars,0);
    Exponent pnl_exp(nvars,0); pnl_exp[static_cast<std::size_t>(pnlvar)]=1;
    Exponent pnl2_exp(nvars,0); pnl2_exp[static_cast<std::size_t>(pnlvar)]=2;
    const int zero_m=monomial_index(mons,zero_exp);
    const int pnl_m=monomial_index(mons,pnl_exp);
    const int pnl2_m=monomial_index(mons,pnl2_exp);

    std::vector<double> y0(dim,0.0);
    const auto iw=weights(qgrid,q0);
    y0[smi(iw.left,zero_m,static_cast<std::size_t>(initial_regime))]+=iw.wl; y0[smi(iw.right,zero_m,static_cast<std::size_t>(initial_regime))]+=iw.wr;
    const auto yT=horizon==0.0?y0:A.exp_action_row(y0,horizon);
    double mean_mq=0.0, second_mq2=0.0;
    for(std::size_t r=0;r<R;++r) for(std::size_t i=0;i<n;++i){ mean_mq+=yT[smi(i,pnl_m,r)]; second_mq2+=yT[smi(i,pnl2_m,r)]; }
    const double variance_mq2=std::max(0.0,second_mq2-mean_mq*mean_mq);
    const double mean_quote=mean_mq*1'000'000.0;
    const double std_quote=std::sqrt(variance_mq2)*1'000'000.0;
    return {horizon,q0,reference_spot_,mean_quote,std_quote,mean_quote/reference_spot_,std_quote/reference_spot_};
}

std::vector<double> PnlAnalytics::internalization_times(int initial_regime) const {
    const std::size_t R=solution_.policies.size();
    const auto& grid=problem_.grid();
    const auto& states=grid.states();
    const std::size_t n=states.size(), N=R*n;
    if(initial_regime<0) initial_regime=static_cast<int>(R/2);
    if(initial_regime<0 || static_cast<std::size_t>(initial_regime)>=R)
        throw std::invalid_argument("initial volatility regime out of range");
    DenseMatrix generator(N,N);
    for(std::size_t r=0;r<R;++r) {
        const auto fixed=fixed_policy_operator(problem_,solution_.policies[r]);
        for(std::size_t i=0;i<n;++i) {
            const std::size_t row=r*n+i;
            for(std::size_t j=0;j<n;++j) generator(row,r*n+j)+=fixed.generator(i,j);
            for(std::size_t other=0;other<R;++other) if(other!=r) {
                const double rate=switches_[r][other];
                generator(row,other*n+i)+=rate;
                generator(row,row)-=rate;
            }
        }
    }
    std::vector<std::size_t> transient;
    transient.reserve(N-R);
    for(std::size_t r=0;r<R;++r) for(std::size_t i=0;i<n;++i)
        if(std::abs(states[i])>1e-10) transient.push_back(r*n+i);
    DenseMatrix A(transient.size(),transient.size());
    std::vector<double> rhs(transient.size(),1.0);
    for(std::size_t i=0;i<transient.size();++i)
        for(std::size_t j=0;j<transient.size();++j)
            A(i,j)=-generator(transient[i],transient[j]);
    const auto tau=A.solve(rhs);
    std::vector<double> times;
    times.reserve(grid.operational_indices().size());
    for(const std::size_t i:grid.operational_indices()) {
        if(std::abs(states[i])<1e-10) {times.push_back(0.0);continue;}
        const auto key=static_cast<std::size_t>(initial_regime)*n+i;
        const auto index=std::find(transient.begin(),transient.end(),key);
        const double t=tau[static_cast<std::size_t>(index-transient.begin())];
        times.push_back(std::max(0.0,t));
    }
    return times;
}

}  // namespace ladder_pricer
