"""Tests for the Black-Scholes pricing module."""

from __future__ import annotations

import numpy as np
import pytest

from option_pricing.black_scholes import (
    black_scholes_call,
    black_scholes_d1_d2,
    black_scholes_put,
)


class TestBlackScholesCall:
    """Tests for the European call pricing formula."""

    def test_standard_test_case(self):
        """Verify the canonical ATM test case from Hull's textbook."""
        # S=100, K=100, T=1, r=5%, sigma=20%, q=0
        # Expected value from Hull (2017), Chapter 15.
        price = black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=0.20)
        assert price == pytest.approx(10.4506, abs=1e-4)

    def test_deep_itm_call(self):
        """Deep ITM call should approach intrinsic value."""
        # S=200, K=100, T=1, r=5%, sigma=20%
        # Intrinsic value: S - K*exp(-rT) = 200 - 95.123 = 104.877
        price = black_scholes_call(S=200, K=100, T=1, r=0.05, sigma=0.20)
        intrinsic = 200 - 100 * np.exp(-0.05)
        assert price == pytest.approx(intrinsic, abs=0.01)

    def test_deep_otm_call(self):
        """Deep OTM call should be near zero."""
        # S=50, K=100, T=0.1, r=5%, sigma=20%
        price = black_scholes_call(S=50, K=100, T=0.1, r=0.05, sigma=0.20)
        assert price == pytest.approx(0.0, abs=1e-4)

    def test_zero_volatility_limit(self):
        """As sigma -> 0, call approaches max(0, S - K*exp(-rT))."""
        # We can't actually test sigma=0 (would raise), but very small sigma
        # should give the discounted intrinsic value.
        S, K, T, r = 110, 100, 1, 0.05
        price = black_scholes_call(S=S, K=K, T=T, r=r, sigma=1e-6)
        forward_intrinsic = max(0, S - K * np.exp(-r * T))
        assert price == pytest.approx(forward_intrinsic, abs=1e-4)

    def test_dividend_reduces_call_price(self):
        """Adding a dividend yield should reduce the call price."""
        no_div = black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=0.20, q=0)
        with_div = black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=0.20, q=0.03)
        assert with_div < no_div


class TestBlackScholesPut:
    """Tests for the European put pricing formula."""

    def test_standard_test_case(self):
        """Verify the canonical ATM put price."""
        price = black_scholes_put(S=100, K=100, T=1, r=0.05, sigma=0.20)
        assert price == pytest.approx(5.5735, abs=1e-4)

    def test_deep_itm_put(self):
        """Deep ITM put should approach K*exp(-rT) - S."""
        price = black_scholes_put(S=50, K=100, T=1, r=0.05, sigma=0.20)
        intrinsic = 100 * np.exp(-0.05) - 50
        assert price == pytest.approx(intrinsic, abs=0.01)

    def test_deep_otm_put(self):
        """Deep OTM put should be near zero."""
        price = black_scholes_put(S=200, K=100, T=0.1, r=0.05, sigma=0.20)
        assert price == pytest.approx(0.0, abs=1e-4)


class TestPutCallParity:
    """Verify that call and put prices satisfy put-call parity."""

    @pytest.mark.parametrize(
        "S,K,T,r,sigma,q",
        [
            (100, 100, 1, 0.05, 0.20, 0),
            (100, 110, 0.5, 0.03, 0.30, 0),
            (50, 100, 2, 0.04, 0.25, 0.02),
            (200, 100, 0.25, 0.06, 0.15, 0.01),
        ],
    )
    def test_parity_holds(self, S, K, T, r, sigma, q):
        """C - P = S*exp(-qT) - K*exp(-rT) for any consistent inputs."""
        call = black_scholes_call(S, K, T, r, sigma, q)
        put = black_scholes_put(S, K, T, r, sigma, q)
        parity_lhs = call - put
        parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert parity_lhs == pytest.approx(parity_rhs, abs=1e-10)


class TestInputValidation:
    """Tests for input validation."""

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="Spot price S must be positive"):
            black_scholes_call(S=-100, K=100, T=1, r=0.05, sigma=0.20)

    def test_zero_strike_raises(self):
        with pytest.raises(ValueError, match="Strike price K must be positive"):
            black_scholes_call(S=100, K=0, T=1, r=0.05, sigma=0.20)

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="Time to expiration T must be positive"):
            black_scholes_call(S=100, K=100, T=-1, r=0.05, sigma=0.20)

    def test_zero_volatility_raises(self):
        with pytest.raises(ValueError, match="Volatility sigma must be positive"):
            black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=0)


class TestD1D2:
    """Tests for the d1, d2 helper function."""

    def test_d1_minus_d2_equals_sigma_sqrt_T(self):
        """The defining relationship: d1 - d2 = sigma*sqrt(T)."""
        S, K, T, r, sigma = 100, 110, 0.5, 0.05, 0.25
        d1, d2 = black_scholes_d1_d2(S, K, T, r, sigma)
        assert d1 - d2 == pytest.approx(sigma * np.sqrt(T), abs=1e-12)

    def test_atm_with_zero_drift(self):
        """For ATM with r=q=0, d1 = sigma*sqrt(T)/2 and d2 = -sigma*sqrt(T)/2."""
        d1, d2 = black_scholes_d1_d2(S=100, K=100, T=1, r=0, sigma=0.20, q=0)
        assert d1 == pytest.approx(0.1, abs=1e-12)
        assert d2 == pytest.approx(-0.1, abs=1e-12)