"""Tests for implied volatility calibration."""

from __future__ import annotations

import numpy as np
import pytest

from option_pricing.black_scholes import black_scholes_call, black_scholes_put
from option_pricing.calibration import (
    implied_volatility_binomial,
    implied_volatility_bs,
)
from option_pricing.parameterizations import (
    crr_parameters,
    leisen_reimer_parameters,
)
from option_pricing.pricers import binomial_price


class TestBlackScholesIV:
    """Tests for Black-Scholes implied volatility inversion."""

    @pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_recovers_atm_call_iv(self, sigma_true):
        """Inversion should recover the original sigma for ATM calls."""
        price = black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=sigma_true)
        iv = implied_volatility_bs(
            market_price=price, S=100, K=100, T=1, r=0.05, option_type="call"
        )
        assert iv == pytest.approx(sigma_true, abs=1e-7)

    @pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.30])
    def test_recovers_otm_put_iv(self, sigma_true):
        """Inversion should work for OTM puts."""
        price = black_scholes_put(S=100, K=90, T=0.5, r=0.04, sigma=sigma_true)
        iv = implied_volatility_bs(
            market_price=price, S=100, K=90, T=0.5, r=0.04, option_type="put"
        )
        assert iv == pytest.approx(sigma_true, abs=1e-7)

    def test_dividend_yield(self):
        """Inversion should work with a non-zero dividend yield."""
        sigma_true = 0.25
        price = black_scholes_call(
            S=100, K=105, T=0.75, r=0.05, sigma=sigma_true, q=0.03
        )
        iv = implied_volatility_bs(
            market_price=price, S=100, K=105, T=0.75, r=0.05,
            option_type="call", q=0.03
        )
        assert iv == pytest.approx(sigma_true, abs=1e-7)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="Market price must be positive"):
            implied_volatility_bs(
                market_price=-1.0, S=100, K=100, T=1, r=0.05, option_type="call"
            )

    def test_arbitrage_violation_raises(self):
        """Price below intrinsic value should trigger informative error."""
        # An ATM call cannot trade below intrinsic - K*exp(-rT)
        # A trivially low price will fail to bracket
        with pytest.raises(ValueError, match="below model price"):
            implied_volatility_bs(
                market_price=0.001, S=100, K=100, T=1, r=0.05, option_type="call",
                sigma_lo=0.05,
            )


class TestBinomialIV:
    """Tests for binomial implied volatility inversion."""

    @pytest.mark.parametrize("scheme", ["crr", "lr"])
    def test_recovers_iv_european_call(self, scheme):
        """Inversion should recover the original sigma."""
        sigma_true = 0.20
        N = 501  # odd N for LR consistency
        if scheme == "crr":
            params = crr_parameters(T=1, N=N, r=0.05, sigma=sigma_true)
        else:
            params = leisen_reimer_parameters(
                S=100, K=100, T=1, N=N, r=0.05, sigma=sigma_true
            )
        price = binomial_price(
            S=100, K=100, T=1, r=0.05, N=N, params=params,
            option_type="call", exercise_style="european"
        )
        iv = implied_volatility_binomial(
            market_price=price, S=100, K=100, T=1, r=0.05, N=N,
            scheme=scheme, option_type="call"
        )
        assert iv == pytest.approx(sigma_true, abs=1e-6)

    def test_lr_iv_close_to_bs_iv(self):
        """For European options at high N, LR and BS implied vol should agree."""
        sigma_true = 0.25
        # Compute a "market" price at the true sigma using BS
        market_price = black_scholes_call(
            S=100, K=100, T=1, r=0.05, sigma=sigma_true
        )
        # Invert using LR with large N
        iv = implied_volatility_binomial(
            market_price=market_price, S=100, K=100, T=1, r=0.05, N=501,
            scheme="lr", option_type="call"
        )
        # Should match BS sigma to high precision (LR is O(1/N^2))
        assert iv == pytest.approx(sigma_true, abs=1e-4)

    def test_american_put_iv(self):
        """Inversion of American put under CRR."""
        sigma_true = 0.25
        N = 200
        params = crr_parameters(T=0.5, N=N, r=0.05, sigma=sigma_true)
        price = binomial_price(
            S=95, K=100, T=0.5, r=0.05, N=N, params=params,
            option_type="put", exercise_style="american"
        )
        iv = implied_volatility_binomial(
            market_price=price, S=95, K=100, T=0.5, r=0.05, N=N,
            scheme="crr", option_type="put", exercise_style="american"
        )
        assert iv == pytest.approx(sigma_true, abs=1e-5)

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="scheme must be"):
            implied_volatility_binomial(
                market_price=10.0, S=100, K=100, T=1, r=0.05, N=100,
                scheme="invalid", option_type="call"
            )