"""Tree parameterizations for binomial option pricing schemes.

This module provides the up-factor, down-factor, and risk-neutral
probability for several binomial pricing schemes:

    1. Cox-Ross-Rubinstein (CRR, 1979): the standard symmetric tree
       with u = exp(sigma * sqrt(dt)) and d = 1/u.

    2. Tian (1993): a closed-form parameterization matching the first
       three moments of the lognormal distribution per period.

    3. Tian (1999): the flexible binomial model with a free tilt
       parameter lambda. A specific choice of lambda places one terminal
       node exactly at the strike, yielding smooth (monotone) convergence.

    4. Leisen-Reimer (1996): a scheme calibrated via inversion of the
       Peizer-Pratt approximation to the normal CDF, achieving O(1/N^2)
       convergence without oscillation.

Each function returns a TreeParameters dataclass with fields u, d, p
(risk-neutral up-probability), and dt (time step).

References:
    Cox, J. C., Ross, S. A., and Rubinstein, M. (1979). Option Pricing:
    A Simplified Approach. Journal of Financial Economics, 7(3), 229-263.

    Tian, Y. (1993). A Modified Lattice Approach to Option Pricing.
    Journal of Futures Markets, 13(5), 563-577.

    Tian, Y. (1999). A Flexible Binomial Option Pricing Model. Journal
    of Futures Markets, 19(7), 817-843.

    Leisen, D. P. J., and Reimer, M. (1996). Binomial Models for Option
    Valuation -- Examining and Improving Convergence. Applied
    Mathematical Finance, 3(4), 319-346.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from option_pricing.black_scholes import black_scholes_d1_d2
from option_pricing.peizer_pratt import peizer_pratt_inversion

__all__ = [
    "TreeParameters",
    "crr_parameters",
    "tian_1993_parameters",
    "tian_1999_parameters",
    "flexible_binomial_parameters",
    "leisen_reimer_parameters",
]


@dataclass(frozen=True)
class TreeParameters:
    """Parameters defining a binomial tree.

    Attributes:
        u: Up-factor; gross return on the underlying in the up state.
        d: Down-factor; gross return on the underlying in the down state.
        p: Risk-neutral probability of an up-move per period.
        dt: Length of one time step in years.
    """

    u: float
    d: float
    p: float
    dt: float

    def __post_init__(self):
        """Verify the no-arbitrage condition d < exp(r*dt) < u is satisfiable."""
        if self.u <= self.d:
            raise ValueError(f"Require u > d, got u={self.u}, d={self.d}")
        if not 0 < self.p < 1:
            raise ValueError(f"Require p in (0, 1), got p={self.p}")
        if self.dt <= 0:
            raise ValueError(f"Require dt > 0, got dt={self.dt}")


def crr_parameters(
    T: float,
    N: int,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> TreeParameters:
    """Compute Cox-Ross-Rubinstein tree parameters.

    The standard CRR parameterization uses a symmetric tree with
        u = exp(sigma * sqrt(dt)),  d = 1/u
    and the risk-neutral probability
        p = (exp((r - q) * dt) - d) / (u - d).

    Args:
        T: Time to expiration in years. Must be positive.
        N: Number of time steps. Must be positive.
        r: Continuously compounded riskless rate.
        sigma: Volatility of the underlying. Must be positive.
        q: Continuous dividend yield. Defaults to 0.

    Returns:
        TreeParameters dataclass with u, d, p, and dt.

    Raises:
        ValueError: If inputs are invalid or no-arbitrage condition fails.
    """
    if T <= 0:
        raise ValueError(f"Time T must be positive, got {T}")
    if N <= 0:
        raise ValueError(f"Number of steps N must be positive, got {N}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    return TreeParameters(u=u, d=d, p=p, dt=dt)


def tian_1993_parameters(
    T: float,
    N: int,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> TreeParameters:
    """Compute Tian (1993) third-moment-matching tree parameters.

    Tian's 1993 "modified lattice" parameterization matches the first
    three moments of the lognormal distribution of S(t+dt)/S(t) per
    period. The closed-form solution is

        M = exp((r - q) * dt),   V = exp(sigma^2 * dt)
        u = (M * V / 2) * (V + 1 + sqrt(V^2 + 2V - 3))
        d = (M * V / 2) * (V + 1 - sqrt(V^2 + 2V - 3))
        p = (M - d) / (u - d).

    The discriminant V^2 + 2V - 3 = (V - 1)(V + 3) is non-negative for
    all V >= 1, which holds whenever sigma > 0, so the closed form is
    always well-defined here.

    Note: this parameterization does NOT depend on the strike K. It is
    not a strike-aligned scheme; the strike-aligned construction in the
    spirit of "place a node on K" is in Tian (1999) -- see
    ``tian_1999_parameters``.

    Args:
        T: Time to expiration in years. Must be positive.
        N: Number of time steps. Must be positive.
        r: Continuously compounded riskless rate.
        sigma: Volatility of the underlying. Must be positive.
        q: Continuous dividend yield. Defaults to 0.

    Returns:
        TreeParameters dataclass with u, d, p, and dt.

    Raises:
        ValueError: If inputs are invalid.

    References:
        Tian, Y. (1993). A Modified Lattice Approach to Option Pricing.
        Journal of Futures Markets, 13(5), 563-577.
    """
    if T <= 0:
        raise ValueError(f"Time T must be positive, got {T}")
    if N <= 0:
        raise ValueError(f"Number of steps N must be positive, got {N}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")

    dt = T / N
    M = np.exp((r - q) * dt)
    V = np.exp(sigma**2 * dt)

    # discriminant = (V - 1)(V + 3) >= 0 for V >= 1, i.e. sigma^2 * dt >= 0
    discriminant = V**2 + 2 * V - 3
    sqrt_disc = np.sqrt(max(discriminant, 0.0))

    u = (M * V / 2) * (V + 1 + sqrt_disc)
    d = (M * V / 2) * (V + 1 - sqrt_disc)
    p = (M - d) / (u - d)

    return TreeParameters(u=u, d=d, p=p, dt=dt)


def flexible_binomial_parameters(
    T: float,
    N: int,
    r: float,
    sigma: float,
    lam: float,
    q: float = 0.0,
) -> TreeParameters:
    """Compute Tian (1999) flexible binomial tree parameters for a given tilt.

    Implements equation (6) of Tian (1999):

        u = exp(sigma * sqrt(dt) + lam * sigma^2 * dt)
        d = exp(-sigma * sqrt(dt) + lam * sigma^2 * dt)
        p = (exp((r - q) * dt) - d) / (u - d).

    The tilt parameter ``lam`` (Tian's lambda) is free. ``lam = 0``
    recovers the CRR parameterization. The no-arbitrage condition,
    equation (7) of the paper, is

        |lam - (r - q) / sigma^2| <= 1 / (sigma * sqrt(dt)),

    which holds for any bounded ``lam`` once dt is sufficiently small.
    For a finite N this condition can fail at extreme tilts, in which
    case p falls outside (0, 1) and the TreeParameters dataclass will
    raise a ValueError.

    This function exposes the full family. For the strike-aligned
    choice of ``lam`` defined by Tian (1999) eq. (13), use
    ``tian_1999_parameters``.

    Args:
        T: Time to expiration in years. Must be positive.
        N: Number of time steps. Must be positive.
        r: Continuously compounded riskless rate.
        sigma: Volatility of the underlying. Must be positive.
        lam: Tilt parameter (lambda in the paper). Real-valued.
        q: Continuous dividend yield. Defaults to 0.

    Returns:
        TreeParameters dataclass with u, d, p, and dt.

    Raises:
        ValueError: If inputs are invalid, or if (u, d, p) violate the
            no-arbitrage / probability conditions enforced by
            TreeParameters.

    References:
        Tian, Y. (1999). A Flexible Binomial Option Pricing Model.
        Journal of Futures Markets, 19(7), 817-843. See eq. (6).
    """
    if T <= 0:
        raise ValueError(f"Time T must be positive, got {T}")
    if N <= 0:
        raise ValueError(f"Number of steps N must be positive, got {N}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")

    dt = T / N
    sqrt_dt = np.sqrt(dt)
    sigma2_dt = sigma * sigma * dt

    u = np.exp(sigma * sqrt_dt + lam * sigma2_dt)
    d = np.exp(-sigma * sqrt_dt + lam * sigma2_dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    return TreeParameters(u=u, d=d, p=p, dt=dt)


def tian_1999_parameters(
    S: float,
    K: float,
    T: float,
    N: int,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> TreeParameters:
    """Compute strike-aligned Tian (1999) flexible binomial parameters.

    Implements the strike-alignment choice of the tilt parameter from
    Tian (1999), eqs. (11) and (13). The construction places the
    terminal node (N, j_0) exactly at the strike K, which yields smooth
    (essentially monotone) convergence in N and admits Richardson-type
    extrapolation.

    The procedure:

        1. Compute CRR baseline factors u_0, d_0 (eq. 8).
        2. Compute eta = [log(K/S) - N log d_0] / log(u_0 / d_0)
           (eq. 10), giving the real-valued "ideal" up-count for which
           u_0^eta * d_0^(N-eta) * S = K.
        3. Round to the nearest integer j_0 (eq. 11).
        4. Solve for lambda in eq. (13):
           lambda = [log(K/S) - (2 j_0 - N) sigma sqrt(dt)] / (N sigma^2 dt)
           which makes u^{j_0} d^{N - j_0} S = K exactly under the
           flexible parameterization eq. (6).
        5. Return (u, d, p, dt) from ``flexible_binomial_parameters``
           with that lambda.

    Note on dividends: Tian's paper is written for a non-dividend-paying
    asset (drift r). Here we follow the standard convention of replacing
    r by r - q in the risk-neutral probability and in the implicit
    forward growth, but we keep the strike-alignment formula (which
    only involves sigma, sqrt(dt), and log(K/S)) unchanged. The
    strike-aligned node is therefore positioned by the same equation as
    in the q = 0 case; only p depends on q.

    Args:
        S: Current spot price. Must be positive.
        K: Strike price. Must be positive.
        T: Time to expiration in years. Must be positive.
        N: Number of time steps. Must be positive.
        r: Continuously compounded riskless rate.
        sigma: Volatility of the underlying. Must be positive.
        q: Continuous dividend yield. Defaults to 0.

    Returns:
        TreeParameters dataclass with u, d, p, and dt.

    Raises:
        ValueError: If inputs are invalid, or if the resulting
            (u, d, p) violate the no-arbitrage / probability conditions
            enforced by TreeParameters (which can occur at very small
            N where the no-arbitrage bound, eq. 7, is binding).

    References:
        Tian, Y. (1999). A Flexible Binomial Option Pricing Model.
        Journal of Futures Markets, 19(7), 817-843. See eqs. (6),
        (11), and (13).
    """
    if S <= 0:
        raise ValueError(f"Spot S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time T must be positive, got {T}")
    if N <= 0:
        raise ValueError(f"Number of steps N must be positive, got {N}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")

    dt = T / N
    sqrt_dt = np.sqrt(dt)

    # CRR baseline factors (eq. 8 of Tian 1999)
    u_0 = np.exp(sigma * sqrt_dt)
    d_0 = np.exp(-sigma * sqrt_dt)

    # Real-valued ideal up-count (eq. 10) and nearest-integer j_0 (eq. 11)
    log_K_over_S = np.log(K / S)
    eta = (log_K_over_S - N * np.log(d_0)) / np.log(u_0 / d_0)
    j_0 = int(round(eta))

    # Solve eq. (13) for the strike-aligning tilt
    lam = (log_K_over_S - (2 * j_0 - N) * sigma * sqrt_dt) / (N * sigma**2 * dt)

    return flexible_binomial_parameters(
        T=T, N=N, r=r, sigma=sigma, lam=lam, q=q,
    )


def leisen_reimer_parameters(
    S: float,
    K: float,
    T: float,
    N: int,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> TreeParameters:
    """Compute Leisen-Reimer (1996) tree parameters via Peizer-Pratt inversion.

    The Leisen-Reimer scheme chooses the binomial probability p such that
    the binomial CDF agrees with the normal CDF at the d2 quantile to
    O(1/N^2) accuracy, using an inversion of the Peizer-Pratt approximation
    to the standard normal distribution. The result is an O(1/N^2)
    convergence rate to Black-Scholes (as opposed to O(1/N) for CRR), with
    no oscillation.

    The construction requires N to be odd; even N is handled by using N+1.

    Args:
        S: Current spot price. Must be positive.
        K: Strike price. Must be positive.
        T: Time to expiration in years. Must be positive.
        N: Number of time steps. Must be positive. Even N internally
            uses N+1.
        r: Continuously compounded riskless rate.
        sigma: Volatility of the underlying. Must be positive.
        q: Continuous dividend yield. Defaults to 0.

    Returns:
        TreeParameters dataclass with u, d, p, and dt.
        The dt field uses T / N_effective where N_effective is the
        actual (odd) number of steps used.

    Raises:
        ValueError: If inputs are invalid.
    """
    if S <= 0:
        raise ValueError(f"Spot S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time T must be positive, got {T}")
    if N <= 0:
        raise ValueError(f"Number of steps N must be positive, got {N}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")

    # Force N to be odd
    N_eff = N if N % 2 == 1 else N + 1
    dt = T / N_eff

    d1, d2 = black_scholes_d1_d2(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
    p = peizer_pratt_inversion(d2, N_eff)
    p_prime = peizer_pratt_inversion(d1, N_eff)

    # Guard against numerical edge cases where p saturates to 0 or 1
    # for very small sigma. In such cases LR's parameterization breaks down;
    # fall back to CRR which handles low-volatility cases more robustly.
    eps = 1e-9
    if p < eps or p > 1 - eps:
        return crr_parameters(T=T, N=N_eff, r=r, sigma=sigma, q=q)

    M = np.exp((r - q) * dt)
    u = M * p_prime / p
    d = M * (1 - p_prime) / (1 - p)

    return TreeParameters(u=u, d=d, p=p, dt=dt)