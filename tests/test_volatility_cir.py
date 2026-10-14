import math
import pytest
from trinity.volatility import cir_volatility_regimes


def test_cir_generator_has_positive_rates_and_correct_interior_drift():
    cfg=cir_volatility_regimes(levels=11)
    sigma=cfg["sigmaPips"]
    rates=cfg["ratesPerMinute"]
    assert len(sigma)==11
    assert sigma==sorted(sigma)
    for i,row in enumerate(rates):
        assert row[i]==0.0
        assert all(x>=0 for x in row)
        assert sum(x>0 for x in row)<=2
        if 0<i<len(sigma)-1:
            v=sigma[i]**2
            drift=sum(rate * (sigma[j]**2-v) for j,rate in enumerate(row))
            assert drift == pytest.approx(.04*(20**2-v),abs=1e-9)


def test_invalid_cir_parameters():
    with pytest.raises(ValueError):cir_volatility_regimes(levels=13)
    with pytest.raises(ValueError):cir_volatility_regimes(vol_of_vol=-1)
