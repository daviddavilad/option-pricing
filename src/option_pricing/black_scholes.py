"""Black-Scholes-Merton option pricing.

Closed-form pricing formulas for European options on assets paying a
continuous dividend yield. Implements the standard Black-Scholes-Merton
formula and provides analytical Greeks.

References:
    Black, F., and Scholes, M. (1973). The Pricing of Options and
    Corporate Liabilities. Journal of Political Economy, 81(3), 637-654.

    Merton, R. C. (1973). Theory of Rational Option Pricing. Bell Journal
    of Economics and Management Science, 4(1), 141-183.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_d1_d2",
]


def black_scholes_d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> tuple[float, float]:
    """Compute the d1 and d2 quantities of the Black-Scholes formula.

    Args:
        S: Current spot price of the underlying asset. Must be positive.
        K: Strike price of the option. Must be positive.
        T: Time to expiration in years. Must be positive.
        r: Continuously compounded riskless interest rate.
        sigma: Volatility of the underlying asset returns. Must be positive.
        q: Continuous dividend yield. Defaults to 0 (non-dividend-paying).

    Returns:
        A tuple (d1, d2) of the Black-Scholes auxiliary quantities, where
        d1 = [log(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T).

    Raises:
        ValueError: If any of S, K, T, or sigma is non-positive.
    """
    if S <= 0:
        raise ValueError(f"Spot price S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time to expiration T must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")

    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Price a European call option using the Black-Scholes-Merton formula.

    Computes the no-arbitrage price of a European call option on an asset
    paying a continuous dividend yield, under the Black-Scholes-Merton
    assumptions of geometric Brownian motion with constant drift and
    volatility.

    Args:
        S: Current spot price of the underlying. Must be positive.
        K: Strike price of the option. Must be positive.
        T: Time to expiration in years. Must be positive.
        r: Continuously compounded riskless interest rate.
        sigma: Volatility of the underlying. Must be positive.
        q: Continuous dividend yield. Defaults to 0.

    Returns:
        The Black-Scholes-Merton price of the call option.

    Examples:
        >>> # Standard test case: ATM, 1-year expiry, 5% rate, 20% vol
        >>> price = black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=0.20)
        >>> round(price, 4)
        10.4506
    """
    d1, d2 = black_scholes_d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Price a European put option using the Black-Scholes-Merton formula.

    Computes the no-arbitrage price of a European put option on an asset
    paying a continuous dividend yield. Internally uses the call price
    and put-call parity, ensuring numerical consistency between the two.

    Args:
        S: Current spot price of the underlying. Must be positive.
        K: Strike price of the option. Must be positive.
        T: Time to expiration in years. Must be positive.
        r: Continuously compounded riskless interest rate.
        sigma: Volatility of the underlying. Must be positive.
        q: Continuous dividend yield. Defaults to 0.

    Returns:
        The Black-Scholes-Merton price of the put option.

    Examples:
        >>> # Standard test case: ATM, 1-year expiry, 5% rate, 20% vol
        >>> price = black_scholes_put(S=100, K=100, T=1, r=0.05, sigma=0.20)
        >>> round(price, 4)
        5.5735
    """
    d1, d2 = black_scholes_d1_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)