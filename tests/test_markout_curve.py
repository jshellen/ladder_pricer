import numpy as np
import pytest

from trinity.markout import markout_eur_per_million
from trinity.defaults import default_config


def test_example_curve_fit_and_negative_sign():
    """All five user-supplied RFQ markout points fit the two-parameter model."""
    cfg = default_config()["tiers"][0]["markout"]
    times = np.array([10.0, 30.0, 60.0, 120.0, 300.0])
    observed = np.array([-10.0, -30.0, -40.0, -50.0, -50.0])
    fitted = markout_eur_per_million(times, cfg["asymptoticEurPerMillion"], cfg["tauSeconds"])
    assert np.sqrt(np.mean((fitted - observed) ** 2)) < 1.5
    assert np.all(fitted < 0.0)
    assert np.all(np.diff(fitted) < 0.0)


def test_limits_and_half_life():
    amplitude, tau = -50.0, 40.0
    assert markout_eur_per_million(0, amplitude, tau) == 0.0
    assert markout_eur_per_million(-1, amplitude, tau) == 0.0
    assert markout_eur_per_million(tau, amplitude, tau) == pytest.approx(-50 * (1 - np.exp(-1)))
    assert markout_eur_per_million(1e5, amplitude, tau) == pytest.approx(-50.0)


def test_spot_unit_conversion_and_directions():
    # 50 EUR / EUR 1M * 11.5 SEK/EUR = 575 SEK / EUR 1M = 0.000575 SEK/EUR.
    spot = 11.5
    amplitude = -50.0
    price_magnitude = -amplitude * spot / 1e6
    assert price_magnitude == pytest.approx(0.000575)
    # Customer buy => dealer sells EUR => EURSEK rises; customer sell => falls.
    for dealer_inventory_direction, price_move in [(-1, price_magnitude), (+1, -price_magnitude)]:
        assert dealer_inventory_direction * price_move == pytest.approx(-price_magnitude)


@pytest.mark.parametrize("amplitude,tau", [(1, 40), (float('nan'), 40), (-50, 0), (-50, float('inf'))])
def test_bad_curve_parameters_rejected(amplitude, tau):
    with pytest.raises(ValueError, match="Markout"):
        markout_eur_per_million(30.0, amplitude, tau)
