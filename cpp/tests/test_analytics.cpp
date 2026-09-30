#include "ladder_pricer/howard.hpp"
#include "ladder_pricer/analytics.hpp"
#include <iostream>
#include <vector>
using namespace ladder_pricer;
std::vector<double> grid(){std::vector<double> p;for(double q=0;q<=3.0+1e-12;q+=0.25)p.push_back(q);for(double q=4;q<=20+1e-12;q+=1)p.push_back(q);std::vector<double> o;for(auto it=p.rbegin();it!=p.rend()-1;++it)o.push_back(-*it);o.insert(o.end(),p.begin(),p.end());return o;}
int main(){std::vector<double> sizes{1,2,3,5,10,20};std::vector<Tier> tiers;
tiers.emplace_back("Tier 1",sizes,LogisticFlow(.0155,.144,.0857,.52,8.42,.026),SaturatingMarkout(1e-4,.5,.5),true,-10,100);
tiers.emplace_back("Tier 2",sizes,LogisticFlow(.0232,.303,.122,.48,2.86,.02),SaturatingMarkout(1e-4,.5,.5),true,-10,100);
tiers.emplace_back("Tier 3",std::vector<double>{1},LogisticFlow(1,.144,.0857,.52,20,.026),SaturatingMarkout(4e-4,.5,.1),true,-10,100);
PricingProblem p(grid(),.002,0,QuadraticPenalty(.1,.002),InternalizationTime(4,.07,.0084),tiers);auto s=HowardSolver(p).solve();auto st=PnlAnalytics(p,s,11.5).statistics(570,.002,0);std::cout<<st.expected_base_ccy<<" "<<st.std_base_ccy<<"\n";}
