"""Tests for tree parameterization schemes."""

from __future__ import annotations

import numpy as np
import pytest

from option_pricing.parameterizations import (
    TreeParameters,
    crr_parameters,
    flexible_binomial_parameters,
    leisen_reimer_parameters,
    tian_1993_parameters,
    tian_1999_parameters,
)
from option_pricing.peizer_pratt import peizer_pratt_inversion


class TestTreeParameters:
    """Tests for the TreeParameters dataclass."""

    def test_valid_construction(self):
        params = TreeParameters(u=1.1, d=0.9, p=0.5, dt=0.01)
        assert params.u == 1.1
        assert params.d == 0.9

    def test_invalid_u_le_d_raises(self):
        with pytest.raises(ValueError, match="Require u > d"):
            TreeParameters(u=0.9, d=1.1, p=0.5, dt=0.01)

    def test_invalid_p_outside_unit_raises(self):
        with pytest.raises(ValueError, match=r"Require p in \(0, 1\)"):
            TreeParameters(u=1.1, d=0.9, p=1.5, dt=0.01)


class TestCRRParameters:
    """Tests for Cox-Ross-Rubinstein parameterization."""

    def test_standard_case(self):
        """Verify CRR parameters on the canonical test case."""
        params = crr_parameters(T=1, N=100, r=0.05, sigma=0.20)
        dt = 1 / 100
        expected_u = np.exp(0.20 * np.sqrt(dt))
        expected_d = 1 / expected_u
        expected_p = (np.exp(0.05 * dt) - expected_d) / (expected_u - expected_d)
        assert params.u == pytest.approx(expected_u, abs=1e-12)
        assert params.d == pytest.approx(expected_d, abs=1e-12)
        assert params.p == pytest.approx(expected_p, abs=1e-12)
        assert params.dt == pytest.approx(dt, abs=1e-12)

    def test_symmetry_u_d(self):
        """CRR scheme has u * d = 1 by construction."""
        params = crr_parameters(T=1, N=100, r=0.05, sigma=0.20)
        assert params.u * params.d == pytest.approx(1.0, abs=1e-12)

    def test_no_arbitrage_holds(self):
        """The no-arbitrage condition d < exp(r*dt) < u must hold."""
        params = crr_parameters(T=1, N=100, r=0.05, sigma=0.20)
        R = np.exp(0.05 * params.dt)
        assert params.d < R < params.u

    def test_p_increases_with_rate(self):
        """Higher riskless rate increases the up-probability."""
        low_r = crr_parameters(T=1, N=100, r=0.02, sigma=0.20)
        high_r = crr_parameters(T=1, N=100, r=0.10, sigma=0.20)
        assert high_r.p > low_r.p

    def test_dividend_decreases_p(self):
        """Higher dividend yield decreases the up-probability."""
        no_div = crr_parameters(T=1, N=100, r=0.05, sigma=0.20, q=0)
        with_div = crr_parameters(T=1, N=100, r=0.05, sigma=0.20, q=0.03)
        assert with_div.p < no_div.p


class TestTian1993Parameters:
    """Tests for the Tian (1993) moment-matching parameterization."""

    def test_returns_valid_tree(self):
        params = tian_1993_parameters(T=1, N=100, r=0.05, sigma=0.20)
        assert isinstance(params, TreeParameters)

    def test_no_arbitrage_holds(self):
        params = tian_1993_parameters(T=1, N=100, r=0.05, sigma=0.20)
        R = np.exp(0.05 * params.dt)
        assert params.d < R < params.u

    def test_does_not_depend_on_strike(self):
        """The Tian (1993) parameterization should not depend on K.

        ``tian_1993_parameters`` does not even take a strike argument.
        This test pins down that intent: scheme parameters are
        determined entirely by (T, N, r, sigma, q).
        """
        import inspect
        sig = inspect.signature(tian_1993_parameters)
        assert "K" not in sig.parameters
        assert "S" not in sig.parameters

    def test_first_moment_matches(self):
        """Tian (1993) matches the first moment: p*u + (1-p)*d = exp((r-q)*dt)."""
        T, N, r, sigma, q = 1.0, 100, 0.05, 0.20, 0.02
        params = tian_1993_parameters(T=T, N=N, r=r, sigma=sigma, q=q)
        forward = params.p * params.u + (1 - params.p) * params.d
        expected = np.exp((r - q) * params.dt)
        assert forward == pytest.approx(expected, abs=1e-12)

    def test_second_moment_matches(self):
        """Tian (1993) matches the second moment of S(t+dt)/S(t).

        Under the lognormal target, E[(S(t+dt)/S(t))^2] = exp((2(r-q) + sigma^2) * dt).
        The binomial second moment is p*u^2 + (1-p)*d^2.
        """
        T, N, r, sigma = 1.0, 100, 0.05, 0.20
        params = tian_1993_parameters(T=T, N=N, r=r, sigma=sigma)
        binom_m2 = params.p * params.u**2 + (1 - params.p) * params.d**2
        expected_m2 = np.exp((2 * r + sigma**2) * params.dt)
        assert binom_m2 == pytest.approx(expected_m2, abs=1e-12)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            tian_1993_parameters(T=1, N=100, r=0.05, sigma=-0.1)


class TestFlexibleBinomial:
    """Tests for the Tian (1999) flexible binomial parameterization."""

    def test_lambda_zero_recovers_crr(self):
        """At lambda = 0 the flexible binomial coincides with CRR."""
        T, N, r, sigma = 1.0, 100, 0.05, 0.20
        fb = flexible_binomial_parameters(T=T, N=N, r=r, sigma=sigma, lam=0.0)
        crr = crr_parameters(T=T, N=N, r=r, sigma=sigma)
        assert fb.u == pytest.approx(crr.u, abs=1e-14)
        assert fb.d == pytest.approx(crr.d, abs=1e-14)
        assert fb.p == pytest.approx(crr.p, abs=1e-14)
        assert fb.dt == pytest.approx(crr.dt, abs=1e-14)

    def test_positive_lambda_tilts_up(self):
        """Positive lambda shifts the tree upward (u*d > 1)."""
        T, N, r, sigma = 1.0, 100, 0.05, 0.20
        fb = flexible_binomial_parameters(T=T, N=N, r=r, sigma=sigma, lam=1.0)
        assert fb.u * fb.d > 1.0

    def test_negative_lambda_tilts_down(self):
        """Negative lambda shifts the tree downward (u*d < 1)."""
        T, N, r, sigma = 1.0, 100, 0.05, 0.20
        fb = flexible_binomial_parameters(T=T, N=N, r=r, sigma=sigma, lam=-1.0)
        assert fb.u * fb.d < 1.0

    def test_no_arbitrage_holds_for_moderate_lambda(self):
        """For bounded lambda and reasonable N, no-arbitrage holds."""
        T, N, r, sigma = 1.0, 100, 0.05, 0.20
        for lam in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            params = flexible_binomial_parameters(
                T=T, N=N, r=r, sigma=sigma, lam=lam,
            )
            R = np.exp(r * params.dt)
            assert params.d < R < params.u, f"failed at lam={lam}"

    def test_dividend_yield_supported(self):
        """Dividend yield enters via p but not u, d."""
        T, N, r, sigma, lam = 1.0, 100, 0.05, 0.20, 0.5
        fb_no_q = flexible_binomial_parameters(
            T=T, N=N, r=r, sigma=sigma, lam=lam, q=0.0,
        )
        fb_with_q = flexible_binomial_parameters(
            T=T, N=N, r=r, sigma=sigma, lam=lam, q=0.03,
        )
        # u and d depend only on sigma, lam, dt -- not on r or q
        assert fb_no_q.u == pytest.approx(fb_with_q.u, abs=1e-14)
        assert fb_no_q.d == pytest.approx(fb_with_q.d, abs=1e-14)
        # p does change
        assert fb_no_q.p != fb_with_q.p


class TestTian1999Parameters:
    """Tests for the strike-aligned Tian (1999) parameterization."""

    def test_returns_valid_tree(self):
        params = tian_1999_parameters(
            S=100, K=100, T=1, N=100, r=0.05, sigma=0.20,
        )
        assert isinstance(params, TreeParameters)

    def test_atm_strike_aligned_node_equals_strike(self):
        """At ATM, the strike-aligned terminal node equals K exactly."""
        S, K, T, N, r, sigma = 100.0, 100.0, 1.0, 100, 0.05, 0.20
        params = tian_1999_parameters(S=S, K=K, T=T, N=N, r=r, sigma=sigma)
        # Reconstruct j_0 the same way the function does.
        sqrt_dt = np.sqrt(params.dt)
        u_0 = np.exp(sigma * sqrt_dt)
        d_0 = np.exp(-sigma * sqrt_dt)
        eta = (np.log(K / S) - N * np.log(d_0)) / np.log(u_0 / d_0)
        j_0 = int(round(eta))
        terminal = S * (params.u ** j_0) * (params.d ** (N - j_0))
        assert terminal == pytest.approx(K, abs=1e-9)

    @pytest.mark.parametrize("K,N", [
        (90.0, 50), (95.0, 100), (105.0, 200), (110.0, 500),
        (120.0, 100), (80.0, 250),
    ])
    def test_strike_aligned_node_equals_strike_otm_itm(self, K, N):
        """For OTM/ITM strikes, alignment holds at machine precision."""
        S, T, r, sigma = 100.0, 1.0, 0.05, 0.20
        params = tian_1999_parameters(S=S, K=K, T=T, N=N, r=r, sigma=sigma)
        sqrt_dt = np.sqrt(params.dt)
        u_0 = np.exp(sigma * sqrt_dt)
        d_0 = np.exp(-sigma * sqrt_dt)
        eta = (np.log(K / S) - N * np.log(d_0)) / np.log(u_0 / d_0)
        j_0 = int(round(eta))
        terminal = S * (params.u ** j_0) * (params.d ** (N - j_0))
        assert abs(terminal - K) < 1e-9

    def test_no_arbitrage_holds_typical_inputs(self):
        S, K, T, N, r, sigma = 100.0, 100.0, 1.0, 100, 0.05, 0.20
        params = tian_1999_parameters(S=S, K=K, T=T, N=N, r=r, sigma=sigma)
        R = np.exp(r * params.dt)
        assert params.d < R < params.u

    def test_atm_with_zero_log_ratio_recovers_crr_at_even_N(self):
        """For S = K and N even, j_0 = N/2 and lambda = 0 (CRR).

        The strike-alignment formula gives
            lambda = (log(K/S) - (2 j_0 - N) sigma sqrt(dt)) / (N sigma^2 dt).
        At K = S, log(K/S) = 0. For even N, eta = N/2 exactly, so
        j_0 = N/2 and (2 j_0 - N) = 0, giving lambda = 0.
        """
        S, K, T, sigma = 100.0, 100.0, 1.0, 0.20
        N = 100  # even, so j_0 = 50
        tian = tian_1999_parameters(S=S, K=K, T=T, N=N, r=0.05, sigma=sigma)
        crr = crr_parameters(T=T, N=N, r=0.05, sigma=sigma)
        assert tian.u == pytest.approx(crr.u, abs=1e-14)
        assert tian.d == pytest.approx(crr.d, abs=1e-14)
        assert tian.p == pytest.approx(crr.p, abs=1e-14)


class TestTian1999PaperReproduction:
    """Reproduce numerical results from Tian (1999), JFM 19(7), 817-843.

    These tests pin the implementation against the paper's published
    tables. They use the same setup the paper uses: S = K = 100,
    sigma = 0.2, T = 0.5, r = 0.06.
    """

    def _european_call(self, params, S, K, T, r, N):
        from option_pricing.pricers import binomial_price
        return binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="call", exercise_style="european",
        )

    @pytest.mark.parametrize("N,price_expected", [
        (50,   7.1829),
        (100,  7.1648),
        (150,  7.1553),
        (200,  7.1491),
        (400,  7.1591),
        (800,  7.1542),
        (5000, 7.1556),
    ])
    def test_table_I_lambda_positive_one(self, N, price_expected):
        """Tian (1999) Table I, lambda = +1 column."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.06, 0.20
        params = flexible_binomial_parameters(
            T=T, N=N, r=r, sigma=sigma, lam=1.0,
        )
        price = self._european_call(params, S, K, T, r, N)
        # Paper reports prices to 4 decimals
        assert price == pytest.approx(price_expected, abs=5e-5)

    @pytest.mark.parametrize("N,price_expected", [
        (50,   7.1276),
        (100,  7.1417),
        (150,  7.1464),
        (200,  7.1488),
        (400,  7.1524),
        (800,  7.1541),
        (5000, 7.1556),
    ])
    def test_table_I_lambda_zero(self, N, price_expected):
        """Tian (1999) Table I, lambda = 0 column (CRR baseline)."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.06, 0.20
        params = flexible_binomial_parameters(
            T=T, N=N, r=r, sigma=sigma, lam=0.0,
        )
        price = self._european_call(params, S, K, T, r, N)
        assert price == pytest.approx(price_expected, abs=5e-5)

    @pytest.mark.parametrize("N,price_expected", [
        (50,   7.1785),
        (100,  7.1624),
        (150,  7.1537),
        (200,  7.1480),
        (400,  7.1585),
        (800,  7.1539),
        (5000, 7.1556),
    ])
    def test_table_I_lambda_negative_one(self, N, price_expected):
        """Tian (1999) Table I, lambda = -1 column."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.06, 0.20
        params = flexible_binomial_parameters(
            T=T, N=N, r=r, sigma=sigma, lam=-1.0,
        )
        price = self._european_call(params, S, K, T, r, N)
        assert price == pytest.approx(price_expected, abs=5e-5)

    @pytest.mark.parametrize("N,price_expected", [
        (25,   10.139765),
        (50,   10.165893),
        (100,  10.178175),
        (200,  10.184097),
        (400,  10.187085),
        (800,  10.188570),
        (1600, 10.189314),
        (3200, 10.189686),
        (6400, 10.189873),
    ])
    def test_table_II_strike_aligned_K95(self, N, price_expected):
        """Tian (1999) Table II top, strike-aligned FB with K = 95."""
        S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.06, 0.20
        params = tian_1999_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=sigma,
        )
        price = self._european_call(params, S, K, T, r, N)
        # Paper reports to 6 decimals
        assert price == pytest.approx(price_expected, abs=5e-7)


class TestLeisenReimerParameters:
    """Tests for Leisen-Reimer (1996) parameterization."""

    def test_returns_valid_tree(self):
        params = leisen_reimer_parameters(
            S=100, K=100, T=1, N=101, r=0.05, sigma=0.20
        )
        assert isinstance(params, TreeParameters)

    def test_even_n_promoted_to_odd(self):
        """LR with even N should use N+1 internally; dt reflects that."""
        params = leisen_reimer_parameters(
            S=100, K=100, T=1, N=100, r=0.05, sigma=0.20
        )
        # dt = T / 101, not T / 100
        assert params.dt == pytest.approx(1 / 101, abs=1e-12)

    def test_no_arbitrage_holds(self):
        params = leisen_reimer_parameters(
            S=100, K=100, T=1, N=101, r=0.05, sigma=0.20
        )
        R = np.exp(0.05 * params.dt)
        assert params.d < R < params.u

    def test_p_close_to_half_atm(self):
        """For ATM (d2 close to 0), LR probability is close to 1/2."""
        params = leisen_reimer_parameters(
            S=100, K=100, T=1, N=101, r=0.0, sigma=0.20
        )
        # With r=0 and ATM, d2 = -sigma*sqrt(T)/2 which is mildly negative
        # So p should be slightly below 1/2
        assert 0.4 < params.p < 0.5


class TestPeizerPratt:
    """Tests for the Peizer-Pratt inversion function."""

    def test_h_at_zero(self):
        """h(0, N) = 1/2 by symmetry."""
        for N in [11, 21, 51, 101, 1001]:
            assert peizer_pratt_inversion(0, N) == pytest.approx(0.5, abs=1e-12)

    def test_h_in_unit_interval(self):
        """h(z, N) is always in [0, 1]."""
        for z in [-3, -1, -0.1, 0.1, 1, 3]:
            for N in [11, 51, 101, 501]:
                h = peizer_pratt_inversion(z, N)
                assert 0 <= h <= 1

    def test_h_symmetry(self):
        """h(z, N) + h(-z, N) = 1 (symmetry around 1/2)."""
        for z in [0.5, 1.0, 1.5, 2.0]:
            for N in [11, 51, 101]:
                h_pos = peizer_pratt_inversion(z, N)
                h_neg = peizer_pratt_inversion(-z, N)
                assert h_pos + h_neg == pytest.approx(1.0, abs=1e-12)

    def test_h_monotonic_in_z(self):
        """h(z, N) is monotonically increasing in z."""
        N = 101
        z_values = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
        h_values = [peizer_pratt_inversion(z, N) for z in z_values]
        for i in range(len(h_values) - 1):
            assert h_values[i] < h_values[i + 1]

    def test_h_correct_for_LR_calibration(self):
        """Verify LR calibration produces a tree pricing correctly.

        The defining property of Peizer-Pratt within Leisen-Reimer is that
        the resulting tree, when used to price a European call, agrees with
        Black-Scholes to high accuracy. We test this indirectly by verifying
        that the LR tree parameters with these probabilities satisfy
        risk-neutral pricing identities.
        """
        from option_pricing.parameterizations import leisen_reimer_parameters

        params = leisen_reimer_parameters(
            S=100, K=100, T=1, N=101, r=0.05, sigma=0.20
        )
        # The LR scheme defines u, d such that p*u + (1-p)*d = exp((r-q)*dt).
        M = np.exp(0.05 * params.dt)
        forward_price = params.p * params.u + (1 - params.p) * params.d
        assert forward_price == pytest.approx(M, abs=1e-12)

    def test_even_N_raises(self):
        with pytest.raises(ValueError, match="N must be odd"):
            peizer_pratt_inversion(0.5, 100)

    def test_zero_N_raises(self):
        with pytest.raises(ValueError, match="N must be positive"):
            peizer_pratt_inversion(0.5, 0)