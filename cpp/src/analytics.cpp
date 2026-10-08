#include "ladder_pricer/analytics.hpp"

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

double interpolate_size_row(const std::vector<double>& sizes,
                            const std::vector<double>& values, double z) {
    if (sizes.size() != values.size() || sizes.empty())
        throw std::invalid_argument("pricing-size row has inconsistent dimensions");
    if (z <= sizes.front() + kTol) return values.front();
    if (z >= sizes.back() - kTol) return values.back();
    auto upper = std::upper_bound(sizes.begin(), sizes.end(), z);
    const auto right = static_cast<std::size_t>(std::distance(sizes.begin(), upper));
    const auto left = right - 1;
    const double w = (z - sizes[left]) / (sizes[right] - sizes[left]);
    return (1.0 - w) * values[left] + w * values[right];
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

PnlStatistics PnlAnalytics::statistics(double horizon, double sigma, double q0) const {
    if(horizon<0.0 || sigma<0.0) throw std::invalid_argument("invalid PnL analytics settings");
    if(reference_spot_<=0.0) throw std::invalid_argument("reference spot must be positive");
    const auto& qgrid=problem_.grid().states();
    const std::size_t n=qgrid.size();
    if(solution_.solve_tier_policies.size()!=problem_.tiers().size()) throw std::runtime_error("solution/model mismatch");

    std::vector<double> taus;
    for(const auto& tier:problem_.tiers()) if(tier.use_markout()) taus.push_back(tier.markout().tau_minutes());
    std::sort(taus.begin(),taus.end());
    taus.erase(std::unique(taus.begin(),taus.end(),[](double a,double b){return std::abs(a-b)<1e-12;}),taus.end());
    const int nimpact=static_cast<int>(taus.size());
    const int pnlvar=nimpact;
    const int nvars=nimpact+1;
    const auto mons=monomials(nvars);
    const std::size_t nm=mons.size();
    const std::size_t dim=n*nm;
    SparseMatrix A(dim,dim);
    auto smi=[&](std::size_t qi,int mi){return qi*nm+static_cast<std::size_t>(mi);};

    std::map<double,int> tau_to_var;
    for(int k=0;k<nimpact;++k) tau_to_var[taus[static_cast<std::size_t>(k)]]=k;

    for(std::size_t i=0;i<n;++i){
        const double q=qgrid[i];
        std::vector<double> drift_const(nvars,0.0);
        std::vector<std::vector<double>> drift_linear(nvars,std::vector<double>(nvars,0.0));
        for(int k=0;k<nimpact;++k) drift_linear[k][k]=-1.0/taus[static_cast<std::size_t>(k)];
        drift_const[pnlvar]=problem_.spot_drift()*q;
        for(int k=0;k<nimpact;++k) drift_linear[pnlvar][k]=q/taus[static_cast<std::size_t>(k)];
        const double pnl_diff_var=(sigma*q)*(sigma*q);

        for(std::size_t tm=0;tm<nm;++tm){
            const auto& exponent=mons[tm];
            for(int r=0;r<nvars;++r){
                const int power=exponent[static_cast<std::size_t>(r)];
                if(power==0) continue;
                Exponent reduced=exponent; --reduced[static_cast<std::size_t>(r)];
                if(drift_const[static_cast<std::size_t>(r)]!=0.0){
                    const int sm=monomial_index(mons,reduced);
                    A.add(smi(i,sm),smi(i,static_cast<int>(tm)),power*drift_const[static_cast<std::size_t>(r)]);
                }
                for(int u=0;u<nvars;++u){
                    const double coeff=drift_linear[static_cast<std::size_t>(r)][static_cast<std::size_t>(u)];
                    if(coeff==0.0) continue;
                    Exponent source=reduced; ++source[static_cast<std::size_t>(u)];
                    const int sm=monomial_index(mons,source);
                    A.add(smi(i,sm),smi(i,static_cast<int>(tm)),power*coeff);
                }
            }
            const int pp=exponent[static_cast<std::size_t>(pnlvar)];
            if(pp>=2 && pnl_diff_var>0.0){
                Exponent source=exponent; source[static_cast<std::size_t>(pnlvar)]-=2;
                const int sm=monomial_index(mons,source);
                A.add(smi(i,sm),smi(i,static_cast<int>(tm)),0.5*pp*(pp-1)*pnl_diff_var);
            }
        }

        std::vector<MomentEvent> events;
        for(std::size_t k=0;k<problem_.tiers().size();++k){
            const auto& tier=problem_.tiers()[k];
            const auto& policy=solution_.solve_tier_policies[k];
            for(Side side:{Side::Bid,Side::Ask}){
                const double dir=direction(side); const auto& deltas=row(policy,side,i);
                for(double z:tier.rfq_sizes()){
                    const double rfq_rate=tier.flow().rfq_arrival_rate(z);
                    if(rfq_rate<=0.0) continue;

                    const bool admissible=problem_.grid().admissible(q,dir*z);
                    const double delta=interpolate_size_row(tier.sizes(),deltas,z);
                    const double win_rate=admissible?tier.flow().arrival_rate(delta,z):0.0;
                    const double loss_rate=std::max(0.0,rfq_rate-win_rate);

                    // Won RFQ: execution cashflow + inventory transition.  The
                    // information shock is the same RFQ markout shock used by
                    // Monte Carlo and subsequently acts on the post-trade
                    // inventory through the mark-to-market dynamics.
                    if(win_rate>0.0){
                        std::vector<double> shift(nvars,0.0);
                        shift[static_cast<std::size_t>(pnlvar)]=z*(problem_.spread()*(0.5-delta)-tier.fee());
                        if(tier.use_markout()) shift[static_cast<std::size_t>(tau_to_var[tier.markout().tau_minutes()])]=-dir*tier.markout().asymptotic();
                        events.push_back({q+dir*z,win_rate,std::move(shift)});
                    }

                    // Lost (or hard-limit blocked) RFQ: inventory is unchanged,
                    // but the RFQ is still an information event.  Do not add a
                    // no-op event when markout is disabled.
                    if(loss_rate>0.0 && tier.use_markout()){
                        std::vector<double> shift(nvars,0.0);
                        shift[static_cast<std::size_t>(tau_to_var[tier.markout().tau_minutes()])]=-dir*tier.markout().asymptotic();
                        events.push_back({q,loss_rate,std::move(shift)});
                    }
                }
            }
        }
        if(problem_.dark_pool() && solution_.solve_dark_pool_policy){
            const auto& venue=*problem_.dark_pool(); const auto& p=*solution_.solve_dark_pool_policy;
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
        if(problem_.passive_ecn() && solution_.solve_passive_ecn_policy){
            const auto& venue=*problem_.passive_ecn(); const auto& p=*solution_.solve_passive_ecn_policy;
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
                A.add(smi(i,static_cast<int>(tm)),smi(i,static_cast<int>(tm)),-event.rate);
                const auto expansion=shifted_expansion(mons[tm],event.shift);
                for(const auto& [source_exp,coeff]:expansion){
                    const int sm=monomial_index(mons,source_exp);
                    if(w.wl!=0.0) A.add(smi(i,sm),smi(w.left,static_cast<int>(tm)),event.rate*w.wl*coeff);
                    if(w.wr!=0.0) A.add(smi(i,sm),smi(w.right,static_cast<int>(tm)),event.rate*w.wr*coeff);
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
    y0[smi(iw.left,zero_m)]+=iw.wl; y0[smi(iw.right,zero_m)]+=iw.wr;
    const auto yT=horizon==0.0?y0:A.exp_action_row(y0,horizon);
    double mean_mq=0.0, second_mq2=0.0;
    for(std::size_t i=0;i<n;++i){ mean_mq+=yT[smi(i,pnl_m)]; second_mq2+=yT[smi(i,pnl2_m)]; }
    const double variance_mq2=std::max(0.0,second_mq2-mean_mq*mean_mq);
    const double mean_quote=mean_mq*1'000'000.0;
    const double std_quote=std::sqrt(variance_mq2)*1'000'000.0;
    return {horizon,q0,reference_spot_,mean_quote,std_quote,mean_quote/reference_spot_,std_quote/reference_spot_};
}

}  // namespace ladder_pricer
