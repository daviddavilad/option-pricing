"""Implied volatility calibration via numerical inversion.

Computes the implied volatility that, when input to a given pricing model
(Black-Scholes or any binomial scheme), reproduces a target market price.
The inversion is performed by Brent's method on the volatility, exploiting
the monotonicity of European option prices in volatility.

The module provides:
    1. implied_volatility_bs: inversion against the Black-Scholes formula.
    2. implied_volatility_binomial: inversion against any binomial scheme.

References:
    Brent, R. P. (1973). Algorithms for Minimization Without Derivatives.
    Englewood Cliffs, NJ: Prentice-Hall.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy.optimize import brentq

from option_pricing.black_scholes import black_scholes_call, black_scholes_put
from option_pricing.parameterizations import (
    crr_parameters,
    leisen_reimer_parameters,
    tian_parameters,
)
from option_pricing.pricers import binomial_price

__all__ = [
    "implied_volatility_bs",
    "implied_volatility_binomial",
]

OptionType = Literal["call", "put"]
SchemeName = Literal["crr", "tian", "lr"]


def _implied_volatility_brent(
    pricer: Callable[[float], float],
    market_price: float,
    sigma_lo: float = 0.01,
    sigma_hi: float = 5.0,
    tol: float = 1e-8,
) -> float:
    """Invert a monotonic price function via Brent's method.

    Args:
        pricer: A function taking sigma and returning a model price.
            Must be monotonically increasing in sigma over [sigma_lo, sigma_hi].
        market_price: Target price to match.
        sigma_lo: Lower bound for the volatility search interval.
        sigma_hi: Upper bound for the volatility search interval.
        tol: Convergence tolerance on the implied volatility.

    Returns:
        The implied volatility solving pricer(sigma) = market_price.

    Raises:
        ValueError: If the market price lies outside the bracket
            [pricer(sigma_lo), pricer(sigma_hi)], suggesting either an
            arbitrage violation or a bracket too narrow for the input.
    """
    f_lo = pricer(sigma_lo) - market_price
    f_hi = pricer(sigma_hi) - market_price

    if f_lo > 0:
        raise ValueError(
            f"Market price {market_price:.4f} is below model price at "
            f"sigma_lo={sigma_lo}: {pricer(sigma_lo):.4f}. "
            "Possible arbitrage violation or input error."
        )
    if f_hi < 0:
        raise ValueError(
            f"Market price {market_price:.4f} exceeds model price at "
            f"sigma_hi={sigma_hi}: {pricer(sigma_hi):.4f}. "
            "Try widening the upper bound or check inputs."
        )

    sigma = brentq(
        lambda s: pricer(s) - market_price,
        sigma_lo,
        sigma_hi,
        xtol=tol,
    )
    return sigma


def implied_volatility_bs(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    q: float = 0.0,
    sigma_lo: float = 0.01,
    sigma_hi: float = 5.0,
    tol: float = 1e-8,
) -> float:
    """Compute Black-Scholes implied volatility from a market option price.

    Args:
        market_price: Observed option price to match. Must be positive.
        S: Spot price of the underlying. Must be positive.
        K: Strike price. Must be positive.
        T: Time to expiration in years. Must be positive.
        r: Continuously compounded riskless rate.
        option_type: "call" or "put".
        q: Continuous dividend yield. Defaults to 0.
        sigma_lo: Lower bound for the volatility search.
        sigma_hi: Upper bound for the volatility search.
        tol: Convergence tolerance on sigma.

    Returns:
        The Black-Scholes implied volatility.

    Raises:
        ValueError: If inputs are invalid or no solution exists in the bracket.

    Examples:
        >>> # Recover sigma=0.20 from a price computed at sigma=0.20
        >>> from option_pricing.black_scholes import black_scholes_call
        >>> price = black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=0.20)
        >>> iv = implied_volatility_bs(
        ...     market_price=price, S=100, K=100, T=1, r=0.05, option_type="call"
        ... )
        >>> abs(iv - 0.20) < 1e-6
        True
    """
    if market_price <= 0:
        raise ValueError(f"Market price must be positive, got {market_price}")

    if option_type == "call":
        pricer = lambda sigma: black_scholes_call(
            S=S, K=K, T=T, r=r, sigma=sigma, q=q
        )
    elif option_type == "put":
        pricer = lambda sigma: black_scholes_put(
            S=S, K=K, T=T, r=r, sigma=sigma, q=q
        )
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return _implied_volatility_brent(pricer, market_price, sigma_lo, sigma_hi, tol)


def implied_volatility_binomial(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    N: int,
    scheme: SchemeName,
    option_type: OptionType,
    exercise_style: Literal["european", "american"] = "european",
    q: float = 0.0,
    sigma_lo: float = 0.01,
    sigma_hi: float = 5.0,
    tol: float = 1e-8,
) -> float:
    """Compute binomial implied volatility from a market option price.

    Inverts a binomial pricing scheme to recover the volatility that
    reproduces a target market price. Supports CRR, Tian, and
    Leisen-Reimer parameterizations.

    Args:
        market_price: Observed option price to match. Must be positive.
        S: Spot price of the underlying.
        K: Strike price.
        T: Time to expiration in years.
        r: Continuously compounded riskless rate.
        N: Number of binomial steps.
        scheme: Tree parameterization, one of "crr", "tian", "lr".
        option_type: "call" or "put".
        exercise_style: "european" or "american".
        q: Continuous dividend yield.
        sigma_lo: Lower bound for the volatility search.
        sigma_hi: Upper bound for the volatility search.
        tol: Convergence tolerance on sigma.

    Returns:
        The implied volatility under the specified binomial scheme.

    Raises:
        ValueError: If inputs are invalid or scheme is unrecognized.

    Notes:
        For high-N inversions on large samples, the LR scheme is
        substantially faster than CRR for equivalent accuracy because
        LR achieves O(1/N^2) convergence while CRR is O(1/N).
    """
    if market_price <= 0:
        raise ValueError(f"Market price must be positive, got {market_price}")

    def _build_params(sigma: float):
        if scheme == "crr":
            return crr_parameters(T=T, N=N, r=r, sigma=sigma, q=q)
        elif scheme == "tian":
            return tian_parameters(S=S, K=K, T=T, N=N, r=r, sigma=sigma, q=q)
        elif scheme == "lr":
            return leisen_reimer_parameters(
                S=S, K=K, T=T, N=N, r=r, sigma=sigma, q=q
            )
        else:
            raise ValueError(
                f"scheme must be 'crr', 'tian', or 'lr', got {scheme!r}"
            )

    def pricer(sigma: float) -> float:
        params = _build_params(sigma)
        return binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type=option_type, exercise_style=exercise_style, q=q,
        )

    return _implied_volatility_brent(pricer, market_price, sigma_lo, sigma_hi, tol)