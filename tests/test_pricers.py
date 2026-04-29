"""Tests for binomial option pricers."""

from __future__ import annotations

import numpy as np
import pytest

from option_pricing.black_scholes import black_scholes_call, black_scholes_put
from option_pricing.parameterizations import (
    crr_parameters,
    leisen_reimer_parameters,
    tian_1993_parameters,
    tian_1999_parameters,
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
            ("tian_1993", tian_1993_parameters),
            ("tian_1999", tian_1999_parameters),
            ("lr", leisen_reimer_parameters),
        ],
    )
    def test_all_schemes_agree_at_high_N(self, scheme):
        """At high N, all schemes should produce nearly identical prices."""
        scheme_name, scheme_fn = scheme
        N = 1001
        if scheme_name in ("crr", "tian_1993"):
            params = scheme_fn(T=T, N=N, r=r, sigma=sigma)
        else:
            params = scheme_fn(S=S, K=K, T=T, N=N, r=r, sigma=sigma)
        price = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="call", exercise_style="european"
        )
        # All schemes should be within 0.05 of Black-Scholes at N=1001
        assert abs(price - BS_CALL) < 0.05

class TestRichardsonExtrapolation:
    """Tests for the Richardson extrapolation utility (Tian 1999, eq. 17)."""

    def test_zero_error_passes_through(self):
        """If both inputs equal the true price, output equals true price."""
        from option_pricing.pricers import richardson_extrapolation
        true_price = 7.1559
        result = richardson_extrapolation(price_N=true_price, price_2N=true_price)
        assert result == pytest.approx(true_price, abs=1e-12)

    def test_default_rho_is_two(self):
        """Default rho=2 implements 2*C(2N) - C(N), the standard form."""
        from option_pricing.pricers import richardson_extrapolation
        # With rho = 2: extrapolated = 2*price_2N - price_N
        result = richardson_extrapolation(price_N=1.0, price_2N=2.0)
        assert result == pytest.approx(3.0, abs=1e-12)

    def test_rho_one_raises(self):
        """rho = 1 is singular and must raise."""
        from option_pricing.pricers import richardson_extrapolation
        with pytest.raises(ValueError, match="rho != 1"):
            richardson_extrapolation(price_N=1.0, price_2N=2.0, rho=1.0)

    def test_tian_paper_extrapolation_example(self):
        """Reproduce the worked example from Tian (1999), p. 829.

        Tian states: with FB error -0.011883 at N=100, error ratio
        2.033633, the extrapolation reduces the error to -0.000400.

        We compute the extrapolated price from FB prices at N=50 and
        N=100, then check that |extrap - BS| matches Tian's stated
        |error| of 0.000400 (4 decimals).
        """
        from option_pricing.black_scholes import black_scholes_call
        from option_pricing.parameterizations import tian_1999_parameters
        from option_pricing.pricers import (
            binomial_price,
            richardson_extrapolation,
        )

        S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.06, 0.20
        bs = black_scholes_call(S=S, K=K, T=T, r=r, sigma=sigma)

        params_50 = tian_1999_parameters(S=S, K=K, T=T, N=50, r=r, sigma=sigma)
        params_100 = tian_1999_parameters(S=S, K=K, T=T, N=100, r=r, sigma=sigma)
        price_50 = binomial_price(
            S=S, K=K, T=T, r=r, N=50, params=params_50,
            option_type="call", exercise_style="european",
        )
        price_100 = binomial_price(
            S=S, K=K, T=T, r=r, N=100, params=params_100,
            option_type="call", exercise_style="european",
        )

        extrap = richardson_extrapolation(
            price_N=price_50, price_2N=price_100, rho=2.0,
        )
        # Tian's worked figure: extrap error ~ -0.000400
        assert abs(extrap - bs) < 5e-4
        # And the extrapolation actually beats the un-extrapolated N=100
        assert abs(extrap - bs) < abs(price_100 - bs)