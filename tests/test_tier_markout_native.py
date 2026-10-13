"""Integration regression: Python configuration reaches the native tier-level HJB.

Skipped if the optional compiled extension is not built in the checkout.
"""
import pytest

pytest.importorskip('trinity._native')

from trinity import Engine, default_config


def _q0_bid_one_m(solution, tier_index):
    q0 = solution['qGrid'].index(0.0)
    return solution['tiers'][tier_index]['bid'][q0][0]


def test_distinct_tier_markouts_reach_native_hjb():
    cfg = default_config()
    cfg['passiveEcn']['enabled'] = False
    baseline = Engine(cfg).solve()
    assert baseline['converged']
    assert len(baseline['tiers']) == 2

    cfg['tiers'][0]['markout'] = {
        'asymptoticEurPerMillion': -15.0, 'tauSeconds': 120.0
    }
    tier1_changed = Engine(cfg).solve()
    assert tier1_changed['converged']
    assert abs(_q0_bid_one_m(tier1_changed, 0) - _q0_bid_one_m(baseline, 0)) > 0.01

    cfg['tiers'][0]['markout'] = default_config()['tiers'][0]['markout']
    cfg['tiers'][1]['markout'] = {
        'asymptoticEurPerMillion': -15.0, 'tauSeconds': 120.0
    }
    tier2_changed = Engine(cfg).solve()
    assert tier2_changed['converged']
    assert abs(_q0_bid_one_m(tier2_changed, 1) - _q0_bid_one_m(baseline, 1)) > 0.01

    # We changed only one tier's markout parameters at a time; the other
    # tier's configuration has not been mutated via a shared dict.
    assert cfg['tiers'][0]['markout'] == default_config()['tiers'][0]['markout']
