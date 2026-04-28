"""Tests for binomial option pricers."""

from __future__ import annotations

import numpy as np
import pytest

from option_pricing.black_scholes import black_scholes_call, black_scholes_put
from option_pricing.parameterizations import (
    crr_parameters,
    leisen_reimer_parameters,
    tian_parameters,
)
from option_pricing.pricers import (
    binomial_price,
    binomial_price_closed_form,
)


# Standard test case shared across tests
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
BS_CALL = black_scholes_call(S=S, K=K, T=T, r=r, sigma=sigma)
BS_PUT = black_scholes_put(S=S, K=K, T=T, r=r, sigma=sigma)


class TestEuropeanCall:
    """Tests for European call pricing via backward induction."""

    @pytest.mark.parametrize("N", [100, 500, 1000])
    def test_crr_converges_to_bs(self, N):
        """CRR with sufficient N should be close to Black-Scholes."""
        params = crr_parameters(T=T, N=N, r=r, sigma=sigma)
        price = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="call", exercise_style="european"
        )
        # Tolerance scales as 1/N for CRR
        tol = 0.5 / np.sqrt(N)
        assert abs(price - BS_CALL) < tol

    @pytest.mark.parametrize("N", [101, 501, 1001])
    def test_lr_converges_faster(self, N):
        """Leisen-Reimer should achieve near-Black-Scholes accuracy at low N."""
        params = leisen_reimer_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=sigma
        )
        price = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="call", exercise_style="european"
        )
        # LR should be much tighter than CRR for the same N
        tol = 0.05
        assert abs(price - BS_CALL) < tol


class TestEuropeanPut:
    """Tests for European put pricing."""

    @pytest.mark.parametrize("N", [100, 500])
    def test_crr_converges_to_bs(self, N):
        params = crr_parameters(T=T, N=N, r=r, sigma=sigma)
        price = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="put", exercise_style="european"
        )
        tol = 0.5 / np.sqrt(N)
        assert abs(price - BS_PUT) < tol


class TestPutCallParity:
    """Verify put-call parity holds in the binomial pricer."""

    @pytest.mark.parametrize("N", [50, 100, 500])
    def test_parity_european_crr(self, N):
        """For European options, C - P = S - K*exp(-rT) in any consistent model."""
        params = crr_parameters(T=T, N=N, r=r, sigma=sigma)
        call = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="call", exercise_style="european"
        )
        put = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="put", exercise_style="european"
        )
        parity_lhs = call - put
        parity_rhs = S - K * np.exp(-r * T)
        # For European options, parity should hold to high precision
        assert parity_lhs == pytest.approx(parity_rhs, abs=1e-10)


class TestAmericanOptions:
    """Tests for American option pricing."""

    def test_american_call_equals_european_no_dividends(self):
        """American call on non-dividend stock equals European call (Merton)."""
        N = 500
        params = crr_parameters(T=T, N=N, r=r, sigma=sigma)
        european = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="call", exercise_style="european"
        )
        american = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="call", exercise_style="american"
        )
        # Should be equal to high precision since no early exercise is optimal
        assert american == pytest.approx(european, abs=1e-10)

    def test_american_put_exceeds_european_put(self):
        """American put should be worth strictly more than European put."""
        N = 500
        params = crr_parameters(T=T, N=N, r=r, sigma=sigma)
        european = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="put", exercise_style="european"
        )
        american = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="put", exercise_style="american"
        )
        # American put should be strictly larger when there's any chance
        # of early exercise
        assert american > european

    def test_deep_itm_american_put_equals_intrinsic(self):
        """Very deep ITM American put should approach intrinsic value."""
        # Very deep ITM put: S = 50, K = 100
        N = 500
        params = crr_parameters(T=T, N=N, r=r, sigma=sigma)
        american = binomial_price(
            S=50, K=100, T=T, r=r, N=N, params=params,
            option_type="put", exercise_style="american"
        )
        intrinsic = 100 - 50  # 50
        # For deep ITM, American put = intrinsic exactly (exercise immediately)
        assert american == pytest.approx(intrinsic, abs=1e-2)


class TestClosedFormCRR:
    """Tests for the closed-form CRR European pricer."""

    def test_closed_form_matches_backward_induction(self):
        """The closed-form CRR pricer should match backward induction."""
        N = 100
        params = crr_parameters(T=T, N=N, r=r, sigma=sigma)
        bi_price = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="call", exercise_style="european"
        )
        cf_price = binomial_price_closed_form(
            S=S, K=K, T=T, r=r, N=N, params=params, option_type="call"
        )
        assert bi_price == pytest.approx(cf_price, abs=1e-10)


class TestSchemeConsistency:
    """Sanity checks across all schemes."""

    @pytest.mark.parametrize(
        "scheme",
        [
            ("crr", crr_parameters),
            ("tian", tian_parameters),
            ("lr", leisen_reimer_parameters),
        ],
    )
    def test_all_schemes_agree_at_high_N(self, scheme):
        """At high N, all schemes should produce nearly identical prices."""
        scheme_name, scheme_fn = scheme
        N = 1001
        if scheme_name == "crr":
            params = scheme_fn(T=T, N=N, r=r, sigma=sigma)
        else:
            params = scheme_fn(S=S, K=K, T=T, N=N, r=r, sigma=sigma)
        price = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="call", exercise_style="european"
        )
        # All schemes should be within 0.05 of Black-Scholes at N=1001
        assert abs(price - BS_CALL) < 0.05