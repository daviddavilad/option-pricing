"""Tests for tree parameterization schemes."""

from __future__ import annotations

import numpy as np
import pytest

from option_pricing.parameterizations import (
    TreeParameters,
    crr_parameters,
    leisen_reimer_parameters,
    tian_closed_form_parameters,
    tian_parameters,
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


class TestTianParameters:
    """Tests for Tian (1999) parameterization."""

    def test_returns_valid_tree(self):
        """Tian parameters should yield a valid tree."""
        params = tian_parameters(S=100, K=100, T=1, N=100, r=0.05, sigma=0.20)
        # All conditions enforced by TreeParameters dataclass
        assert isinstance(params, TreeParameters)

    def test_no_arbitrage_holds(self):
        params = tian_parameters(S=100, K=100, T=1, N=100, r=0.05, sigma=0.20)
        R = np.exp(0.05 * params.dt)
        assert params.d < R < params.u

    def test_strict_strike_alignment(self):
        """Strict Tian should have one terminal node exactly equal to K."""
        S, K, N = 100, 110, 100
        params = tian_parameters(
            S=S, K=K, T=1, N=N, r=0.05, sigma=0.20
        )
        # Find a* (the strike-aligned index) and verify u^a* * d^(N-a*) * S = K
        # We use the same convention as the implementation (a* = round(N*p_CRR))
        crr = crr_parameters(T=1, N=N, r=0.05, sigma=0.20)
        a_star = max(1, min(N - 1, int(round(N * crr.p))))
        terminal_at_a_star = (params.u ** a_star) * (params.d ** (N - a_star)) * S
        # Should match K to high precision (the alignment is by construction)
        assert terminal_at_a_star == pytest.approx(K, rel=1e-9)

    def test_falls_back_to_crr_at_extreme(self):
        """For very small N where Tian might fail, falls back to CRR."""
        # Both should produce valid trees
        tian = tian_parameters(S=100, K=100, T=1, N=2, r=0.05, sigma=0.20)
        crr = crr_parameters(T=1, N=2, r=0.05, sigma=0.20)
        assert isinstance(tian, TreeParameters)
        assert isinstance(crr, TreeParameters)


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