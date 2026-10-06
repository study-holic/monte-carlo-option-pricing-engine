import pytest

from monte_carlo_option_pricing import (
    bs_call_price,
    mc_antithetic,
    mc_control_variate,
    mc_european_call,
)

PARAMS = [
    (100, 100, 0.05, 0.2, 1.0),
    (100, 90, 0.03, 0.3, 0.5),
    (100, 120, 0.01, 0.25, 2.0),
]


@pytest.mark.parametrize("estimator", [mc_european_call, mc_antithetic, mc_control_variate])
@pytest.mark.parametrize("S0, K, r, sigma, T", PARAMS)
def test_price_matches_black_scholes(estimator, S0, K, r, sigma, T):
    analytical = bs_call_price(S0, K, r, sigma, T)
    price, stderr = estimator(S0, K, r, sigma, T, 200_000, seed=0)
    assert abs(price - analytical) < 4 * stderr
