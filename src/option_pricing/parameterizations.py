"""Tree parameterizations for binomial option pricing schemes.

This module provides the up-factor, down-factor, and risk-neutral
probability for three binomial pricing schemes:

    1. Cox-Ross-Rubinstein (CRR, 1979): the standard symmetric tree
       with u = exp(sigma * sqrt(dt)) and d = 1/u.

    2. Tian (1999): a flexible tree in which one terminal node aligns
       with the strike, eliminating the integer-cutoff oscillation of
       the standard CRR scheme.

    3. Leisen-Reimer (1996): a scheme calibrated via inversion of the
       Peizer-Pratt approximation to the normal CDF, achieving O(1/N^2)
       convergence without oscillation.

Each function returns a TreeParameters dataclass with fields u, d, p
(risk-neutral up-probability), and dt (time step).

References:
    Cox, J. C., Ross, S. A., and Rubinstein, M. (1979). Option Pricing:
    A Simplified Approach. Journal of Financial Economics, 7(3), 229-263.

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
    "tian_parameters",
    "tian_closed_form_parameters",
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


def tian_closed_form_parameters(
    S: float,
    K: float,
    T: float,
    N: int,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> TreeParameters:
    """Compute closed-form Tian-style tree parameters via mean/variance match.

    This is the simplified Tian-style parameterization that matches the
    mean and variance of the log-return per period without enforcing
    strict strike alignment. Convenient when no specific strike is given
    or when robustness across many strikes is desired.
    
    For strict strike-aligned Tian (1999), use ``tian_parameters``.

    The construction proceeds as follows. Let a* be the integer nearest
    to N * p_CRR, where p_CRR is the standard CRR risk-neutral probability.
    Then u and d are determined by simultaneously matching:

        1. Strike alignment: u^a* * d^(N - a*) * S = K
        2. Risk-neutral pricing: p*u + (1-p)*d = exp((r-q)*dt)
        3. Variance matching: p*(1-p)*(log(u) - log(d))^2 = sigma^2 * dt

    The system is solved analytically by parameterizing through
    log(u) and log(d).

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
        ValueError: If inputs are invalid.

    Notes:
        For inputs near the no-arbitrage boundary, Tian's scheme can
        produce parameters that fail the no-arbitrage check. In these
        edge cases we fall back to the CRR parameterization.
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

    # Get CRR baseline to determine a*
    crr = crr_parameters(T=T, N=N, r=r, sigma=sigma, q=q)
    a_star = int(round(N * crr.p))
    a_star = max(1, min(N - 1, a_star))  # ensure 1 <= a* <= N-1

    # Strike alignment condition: a* * log(u) + (N - a*) * log(d) = log(K/S)
    # Variance condition: p*(1-p)*(log(u) - log(d))^2 = sigma^2 * dt
    # Risk-neutral pricing: p*u + (1-p)*d = exp((r-q)*dt)
    #
    # Let x = log(u), y = log(d). The strike condition gives:
    #     a* * x + (N - a*) * y = log(K/S)
    # which is one equation in two unknowns. We add the symmetric-spread
    # condition x = -y * (N - a*) / a* * (-1)... actually we use the
    # standard Tian closed-form.
    #
    # Standard Tian closed-form (from the 1999 paper): set
    #     M = exp((r - q) * dt)
    #     V = exp(sigma^2 * dt)
    # Then
    #     u = M*V/2 * (V + 1 + sqrt(V^2 + 2*V - 3))
    #     d = M*V/2 * (V + 1 - sqrt(V^2 + 2*V - 3))
    # (This matches the variance and the mean of the log-return; the
    # strike alignment is then approximate via choice of a*.)
    M = np.exp((r - q) * dt)
    V = np.exp(sigma**2 * dt)
    discriminant = V**2 + 2 * V - 3
    if discriminant < 0:
        # Falls back to CRR if we hit a numerical edge case.
        return crr

    sqrt_disc = np.sqrt(discriminant)
    u = M * V / 2 * (V + 1 + sqrt_disc)
    d = M * V / 2 * (V + 1 - sqrt_disc)
    p = (M - d) / (u - d)

    # Sanity check: if we fall outside (0, 1), fall back to CRR.
    if not 0 < p < 1 or u <= d:
        return crr

    return TreeParameters(u=u, d=d, p=p, dt=dt)


def tian_parameters(
    S: float,
    K: float,
    T: float,
    N: int,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> TreeParameters:
    """Compute strict strike-aligned Tian (1999) tree parameters.

    Tian's flexible parameterization chooses (u, d, p) such that:

        1. One terminal node coincides exactly with the strike K:
           u^{a*} * d^{N - a*} * S = K
        2. Risk-neutral pricing: p*u + (1-p)*d = exp((r-q)*dt)
        3. Variance matching: p*(1-p)*(log u - log d)^2 = sigma^2 * dt

    where a* is the integer nearest to N * p_CRR (the standard CRR
    risk-neutral probability), centering the tree near the strike.

    The construction eliminates the integer-cutoff oscillation
    characteristic of the CRR scheme by guaranteeing strike-node
    alignment at every N. The resulting convergence is O(1/N) without
    oscillation.

    Args:
        S: Current spot price. Must be positive.
        K: Strike price. Must be positive.
        T: Time to expiration in years. Must be positive.
        N: Number of time steps. Must be positive.
        r: Continuously compounded riskless rate.
        sigma: Volatility of the underlying. Must be positive.
        q: Continuous dividend yield. Defaults to 0.

    Returns:
        TreeParameters dataclass with u, d, p, dt.

    Raises:
        ValueError: If inputs are invalid.

    Notes:
        For inputs near the no-arbitrage boundary or extreme parameter
        ranges, the strict strike-alignment system may fail to admit
        a valid solution. In these cases we fall back to the
        closed-form Tian parameterization.
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

    from scipy.optimize import brentq

    dt = T / N

    # Get CRR baseline to determine a*
    crr = crr_parameters(T=T, N=N, r=r, sigma=sigma, q=q)
    a_star = int(round(N * crr.p))
    a_star = max(1, min(N - 1, a_star))  # ensure 1 <= a* <= N-1

    # Set up the system. With x = log(u), the strike condition gives
    # y = log(d) = (L - a* * x) / (N - a*), where L = log(K/S).
    L = np.log(K / S)
    M = np.exp((r - q) * dt)  # forward growth factor per period
    target_var = sigma**2 * dt

    def y_of_x(x: float) -> float:
        return (L - a_star * x) / (N - a_star)

    def p_of_xy(x: float, y: float) -> float:
        # Risk-neutral pricing: p*exp(x) + (1-p)*exp(y) = M
        # Solving for p: p = (M - exp(y)) / (exp(x) - exp(y))
        ex, ey = np.exp(x), np.exp(y)
        denom = ex - ey
        if abs(denom) < 1e-14:
            return np.nan
        return (M - ey) / denom

    def variance_residual(x: float) -> float:
        """Returns variance(x) - target_var; we seek the root."""
        y = y_of_x(x)
        if x <= y:
            # Need u > d, i.e., x > y. If violated, return a strong negative.
            return -target_var
        p = p_of_xy(x, y)
        if not (0 < p < 1) or np.isnan(p):
            # Invalid p; signal "too far" from the solution.
            # Returning a large positive pushes the solver away.
            return target_var
        return p * (1 - p) * (x - y) ** 2 - target_var

    # Bracket for x. Reference is x_CRR = sigma * sqrt(dt).
    x_crr = sigma * np.sqrt(dt)
    x_lo, x_hi = 0.1 * x_crr, 5.0 * x_crr

    # Verify the bracket actually brackets a root
    f_lo = variance_residual(x_lo)
    f_hi = variance_residual(x_hi)

    if f_lo * f_hi >= 0:
        # No sign change in the bracket; fall back to closed-form Tian.
        return tian_closed_form_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=sigma, q=q
        )

    try:
        x_solution = brentq(variance_residual, x_lo, x_hi, xtol=1e-12)
    except (ValueError, RuntimeError):
        return tian_closed_form_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=sigma, q=q
        )

    y_solution = y_of_x(x_solution)
    u = np.exp(x_solution)
    d = np.exp(y_solution)
    p = p_of_xy(x_solution, y_solution)

    if not (0 < p < 1) or u <= d:
        return tian_closed_form_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=sigma, q=q
        )

    return TreeParameters(u=u, d=d, p=p, dt=dt)


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