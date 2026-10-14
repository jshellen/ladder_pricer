#include "ladder_pricer/solver.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ladder_pricer {
namespace {
double direction(Side s) { return s == Side::Bid ? 1.0 : -1.0; }
const std::vector<double>& deltas(const LadderPolicy& p, Side s, std::size_t i) {
    return s == Side::Bid ? p.bid[i] : p.ask[i];
}
std::vector<double> segment(const std::vector<double>& vector, std::size_t start, std::size_t size) {
    return {vector.begin() + static_cast<std::ptrdiff_t>(start),
            vector.begin() + static_cast<std::ptrdiff_t>(start+size)};
}
void blend_tiers(Policy& trial, const Policy& proposal, double alpha) {
    for (std::size_t k=0; k<trial.tiers.size(); ++k) {
        for (std::size_t i=0; i<trial.tiers[k].bid.size(); ++i) {
            for (std::size_t j=0; j<trial.tiers[k].bid[i].size(); ++j) {
                trial.tiers[k].bid[i][j] += alpha*(proposal.tiers[k].bid[i][j] - trial.tiers[k].bid[i][j]);
                trial.tiers[k].ask[i][j] += alpha*(proposal.tiers[k].ask[i][j] - trial.tiers[k].ask[i][j]);
            }
        }
    }
}
}

Solver::Solver(const PricingProblem& base,
                  std::vector<double> sigma_pips,
                  std::vector<std::vector<double>> switch_rates)
    : sigmas_(std::move(sigma_pips)), switch_rates_(std::move(switch_rates)) {
    const auto R=sigmas_.size();
    if (R < 2 || R > 12 || switch_rates_.size()!=R) {
        throw std::invalid_argument("volatility requires 2-12 regimes and square switching matrix");
    }
    for (std::size_t i=0; i<R; ++i) {
        if (!(sigmas_[i] > 0.0) || !std::isfinite(sigmas_[i]))
            throw std::invalid_argument("sigmaPips must be positive and finite");
        if (i && !(sigmas_[i] > sigmas_[i-1]))
            throw std::invalid_argument("sigmaPips must increase across regimes");
        if (switch_rates_[i].size()!=R)
            throw std::invalid_argument("volatility transition matrix must be square");
        double outgoing=0.0;
        for (std::size_t j=0;j<R;++j) {
            if (!std::isfinite(switch_rates_[i][j])) throw std::invalid_argument("invalid volatility transition rate");
            if (i!=j) {
                if (switch_rates_[i][j]<0) throw std::invalid_argument("negative transition rate");
                outgoing+=switch_rates_[i][j];
            }
        }
        if (outgoing<=0.0) throw std::invalid_argument("each regime must have outgoing transitions");
        // Diagonal is implied, not supplied; require zero to catch common confusion.
        if (std::abs(switch_rates_[i][i]) > 1e-14)
            throw std::invalid_argument("volatility input matrix diagonal must be zero");
        problems_.emplace_back(base.grid().operational_states(), base.spread(),
            QuadraticPenalty(base.penalty().risk_aversion(), sigmas_[i] / 10000.0),
            base.tiers(), base.dark_pool(), base.passive_ecn());
    }
    // A single ergodic-average reward is meaningful only when the exogenous
    // volatility chain can communicate between all regimes.
    for (std::size_t start=0;start<R;++start) {
        std::vector<bool> reachable(R,false);
        reachable[start]=true;
        bool changed=true;
        while (changed) {
            changed=false;
            for (std::size_t i=0;i<R;++i) if (reachable[i])
                for (std::size_t j=0;j<R;++j)
                    if (switch_rates_[i][j]>0.0 && !reachable[j]) {
                        reachable[j]=true;
                        changed=true;
                    }
        }
        if (std::find(reachable.begin(),reachable.end(),false)!=reachable.end())
            throw std::invalid_argument("volatility regimes must form an irreducible Markov chain");
    }
}

Solver::Evaluation Solver::evaluate(const std::vector<Policy>& policies) const {
    const auto R = problems_.size();
    const auto& states = problems_[0].grid().states();
    const auto n=states.size(), N=R*n;
    if (policies.size()!=R) throw std::invalid_argument("incorrect regime policy count");
    DenseMatrix generator(N,N);
    std::vector<double> reward(N,0.0), inventory(N,0.0);
    for (std::size_t r=0;r<R;++r) {
        const auto op = fixed_policy_operator(problems_[r], policies[r]);
        for (std::size_t i=0;i<n;++i) {
            const auto row=r*n+i;
            inventory[row] = states[i];
            reward[row] = op.reward[i];
            for (std::size_t j=0;j<n;++j) generator(row,r*n+j) += op.generator(i,j);
            double total=0;
            for (std::size_t s=0;s<R;++s) if (s!=r) {
                const auto rate=switch_rates_[r][s];
                generator(row,s*n+i) += rate;
                total+=rate;
            }
            generator(row,row) -= total;
        }
    }

    Evaluation result;
    result.weighted_inventory.resize(problems_[0].tiers().size());
    // One joint resolvent for each DISTINCT tier time constant.  All regime
    // switches are included during the RFQ's unfolding markout curve.
    std::vector<std::pair<double,std::vector<double>>> cache;
    for (std::size_t k=0;k<problems_[0].tiers().size();++k) {
        const auto& tier=problems_[0].tiers()[k];
        if (!tier.use_markout() || tier.markout().asymptotic()==0.0) continue;
        const double tau=tier.markout().tau_minutes();
        auto it=std::find_if(cache.begin(),cache.end(),[tau](const auto& p){return p.first==tau;});
        if (it==cache.end()) {
            DenseMatrix A(N,N);
            for (std::size_t i=0;i<N;++i)
                for (std::size_t j=0;j<N;++j)
                    A(i,j)=(i==j ? 1.0 : 0.0)-tau*generator(i,j);
            cache.emplace_back(tau,A.solve(inventory));
            it=cache.end()-1;
        }
        result.weighted_inventory[k]=it->second;
    }
    const auto mtms=mtm_views(result);
    for (std::size_t r=0;r<R;++r) {
        for (std::size_t i=0;i<n;++i) {
            const double q=states[i];
            for (std::size_t k=0;k<problems_[r].tiers().size();++k) {
                const auto& t=problems_[r].tiers()[k];
                for (auto side:{Side::Bid, Side::Ask}) {
                    const double dir=direction(side);
                    for (std::size_t j=0;j<t.sizes().size();++j) {
                        const double z=t.sizes()[j];
                        if (!problems_[r].grid().admissible(q,dir*z)) continue;
                        const double rate=t.flow().arrival_rate(deltas(policies[r].tiers[k],side,i)[j],z);
                        reward[r*n+i] += rate*mtms[r].win_increment(k,dir,q,q+dir*z);
                    }
                }
            }
        }
    }
    // Joint ergodic Poisson equation: G h + r = rho, h(q=0,v=middle)=0.
    DenseMatrix A(N+1,N+1);
    std::vector<double> b(N+1,0.0);
    for (std::size_t i=0;i<N;++i) {
        for (std::size_t j=0;j<N;++j) A(i,j)=generator(i,j);
        A(i,N)=-1.0;
        b[i]=-reward[i];
    }
    A(N,(R/2)*n+problems_[0].grid().zero_index())=1.0;
    auto solution=A.solve(b);
    result.value=segment(solution,0,N);
    result.rho=solution[N];
    for (std::size_t i=0;i<N;++i) {
        double residual=reward[i]-result.rho;
        for (std::size_t j=0;j<N;++j) residual+=generator(i,j)*result.value[j];
        result.residual=std::max(result.residual,std::abs(residual));
    }
    return result;
}

std::vector<ExpectedMtm> Solver::mtm_views(const Evaluation& e) const {
    std::vector<ExpectedMtm> views;
    const auto n=problems_[0].grid().states().size();
    for (std::size_t r=0;r<problems_.size();++r) {
        std::vector<std::vector<double>> m;
        for (const auto& joint:e.weighted_inventory)
            m.push_back(joint.empty()? std::vector<double>{}:segment(joint,r*n,n));
        views.emplace_back(problems_[r],std::move(m));
    }
    return views;
}

SolverResult Solver::solve() const {
    const auto R=problems_.size(), n=problems_[0].grid().states().size();
    std::vector<PolicyBuilder> builders;
    for (const auto& problem:problems_) builders.emplace_back(problem);
    std::vector<Policy> policy;
    for (const auto& builder:builders) policy.emplace_back(builder.initial_policy());
    constexpr int max_iter=90;
    constexpr double abs_tol=1e-12;
    int iterations=0;
    bool converged=false;
    Evaluation current=evaluate(policy);
    for (int it=1;it<=max_iter;++it) {
        iterations=it;
        const auto mtms=mtm_views(current);
        std::vector<Policy> proposal;
        for (std::size_t r=0;r<R;++r)
            proposal.emplace_back(builders[r].improve(segment(current.value,r*n,n),policy[r],mtms[r]));
        Evaluation best_evaluation=evaluate(proposal);
        double best=best_evaluation.rho;
        std::vector<Policy> chosen=proposal;
        if (best<=current.rho+abs_tol) {
            // Frozen-MTM greedy improvement is not globally monotone.  Follow
            // the existing damped best-response implementation but evaluate
            // every proposal on the JOINT true average reward.
            for (double alpha: {0.5,0.25,0.125,0.0625,0.03125,0.015625}) {
                auto trial=policy;
                for (std::size_t r=0;r<R;++r) blend_tiers(trial[r],proposal[r],alpha);
                auto trial_evaluation=evaluate(trial);
                if (trial_evaluation.rho>best) {
                    best=trial_evaluation.rho;
                    chosen=std::move(trial);
                    best_evaluation=std::move(trial_evaluation);
                }
            }
        }
        if (best<=current.rho+abs_tol) {
            converged=true; // stalled best response, NOT global optimality
            break;
        }
        policy=std::move(chosen);
        current=std::move(best_evaluation);
        if (!std::isfinite(current.rho)) throw std::runtime_error("nonfinite joint-volatility reward");
    }
    for (std::size_t r=0;r<R;++r) policy[r]=builders[r].snap_operational_constraints(policy[r]);
    current=evaluate(policy);
    SolverResult out;
    out.q_grid=problems_[0].grid().operational_states();
    out.sigma_pips=sigmas_;
    out.average_reward=current.rho;
    out.iterations=iterations;
    out.converged=converged;
    out.evaluation_residual=current.residual;
    out.policies=std::move(policy);
    out.weighted_inventory=std::move(current.weighted_inventory);
    const auto& indices=problems_[0].grid().operational_indices();
    for (std::size_t r=0;r<R;++r) {
        out.values.emplace_back();
        for (auto i:indices) out.values.back().push_back(current.value[r*n+i]);
    }
    return out;
}
} // namespace ladder_pricer
