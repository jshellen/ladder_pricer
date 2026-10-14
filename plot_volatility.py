"""Run the coupled HJB and save ECN-volatility diagnostics.

Usage: PYTHONPATH=python python plot_volatility.py
"""
from __future__ import annotations

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from trinity.defaults import default_config
from trinity.engine import Engine

c=default_config()
r=Engine(c).solve_volatility()
with open('volatility_experiment_result.json','w',encoding='utf-8') as f:
    json.dump(r,f,indent=2)
q=np.asarray(r['qGrid'])

fig,axes=plt.subplots(1,2,figsize=(13,4.6),sharey=True,constrained_layout=True)
for i,reg in enumerate(r['regimes']):
    sigma=reg['sigmaPips']
    ecn=reg['passiveEcn']
    for ax,side,mask in [(axes[0],'bid',q<0),(axes[1],'ask',q>0)]:
        delta=np.asarray(ecn[side+'Delta'])
        active=np.asarray(ecn[side+'Active'],dtype=bool)
        y=np.where(active & mask,delta,np.nan)
        ax.plot(q,y,marker='o',markersize=2,linewidth=1.7,label=f'σ = {sigma:g} pips/√min')
for ax,side in zip(axes,['Bid (reduces short inventory)','Ask (reduces long inventory)']):
    ax.set_title(side)
    ax.grid(alpha=.22)
    ax.set_xlabel('Inventory (EUR M)')
    ax.set_ylim(0,.51)
axes[0].set_ylabel('ECN price-improvement delta (0 = touch; 0.5 = mid)')
axes[1].legend(loc='upper left',fontsize=9)
fig.suptitle('Joint Markov-regime HJB: optimal passive ECN policy',fontsize=14)
fig.savefig('ecn_policy_by_volatility.png',dpi=185,bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4.8),constrained_layout=True)
for reg in r['regimes']:
    ecn=reg['passiveEcn']; d=np.asarray(ecn['askDelta']); active=np.asarray(ecn['askActive'])
    dist=c['spreadPips']*(.5-d)
    ax.plot(q,np.where((q>0)&active,dist,np.nan),marker='o',markersize=2.5,linewidth=1.8,label=f"σ = {reg['sigmaPips']:g} pips/√min")
ax.grid(alpha=.22)
ax.legend()
ax.set_xlabel('Positive inventory (EUR M)')
ax.set_ylabel('ECN ask quote depth from mid (pips)')
ax.set_title('Higher volatility → more aggressive risk-reducing ECN quote')
fig.savefig('ecn_quote_depth_by_volatility.png',dpi=185,bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4.8),constrained_layout=True)
for sigma,v in zip(r['sigmaPips'],r['valueByRegime']):
    ax.plot(q,np.array(v),linewidth=1.8,label=f'σ = {sigma:g} pips/√min')
ax.grid(alpha=.22)
ax.legend()
ax.set_xlabel('Inventory (EUR M)')
ax.set_ylabel('Relative value (quote currency M × price)')
ax.set_title('Joint-regime relative value functions')
fig.savefig('value_function_by_volatility.png',dpi=185,bbox_inches='tight')
plt.close(fig)
print('Iterations',r['iterations'],'converged',r['converged'],'residual',r['evaluationResidual'],'reward',r['averageReward'])
for x in [1,5,10,15,20]:
    i=r['qGrid'].index(float(x))
    print(f'+{x} M ask:',[(rr['sigmaPips'],rr['passiveEcn']['askDelta'][i]) for rr in r['regimes']])
