"""Binomial tree pricers for European and American options.

This module provides the pricing engine for binomial option valuation.
The engine is scheme-agnostic: it accepts any TreeParameters dataclass
(from CRR, Tian, Leisen-Reimer, or any future scheme) and prices European
or American options via backward induction.

The implementation is fully vectorized using NumPy: at each time step,
the value array is updated via a single broadcast operation rather than
an explicit Python loop over nodes. This brings practical performance
close to compiled code for moderate N (up to ~5000).

For European options, an alternative closed-form pricer based on the
Cox-Ross-Rubinstein formula is also provided. This is primarily useful
as a cross-check on the backward-induction pricer.

References:
    Cox, J. C., Ross, S. A., and Rubinstein, M. (1979). Option Pricing:
    A Simplified Approach. Journal of Financial Economics, 7(3), 229-263.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy.stats import binom

from option_pricing.parameterizations import TreeParameters

__all__ = [
    "OptionType",
    "ExerciseStyle",
    "binomial_price",
    "binomial_price_closed_form",
    "american_exercise_boundary",
    "richardson_extrapolation",
]

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


def _payoff_function(
    option_type: OptionType, K: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the payoff function for a given option type and strike.

    Args:
        option_type: Either "call" or "put".
        K: Strike price.

    Returns:
        A function that maps an array of stock prices to an array of payoffs.
    """
    if option_type == "call":
        return lambda S: np.maximum(S - K, 0.0)
    elif option_type == "put":
        return lambda S: np.maximum(K - S, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    N: int,
    params: TreeParameters,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    q: float = 0.0,
) -> float:
    """Price an option via backward induction on a binomial tree.

    Args:
        S: Current spot price. Must be positive.
        K: Strike price. Must be positive.
        T: Time to expiration in years. Must be positive.
        r: Continuously compounded riskless rate.
        N: Number of time steps. Must be positive.
        params: TreeParameters with u, d, p, dt. dt should equal T/N.
        option_type: "call" or "put".
        exercise_style: "european" or "american".
        q: Continuous dividend yield. Defaults to 0.

    Returns:
        The time-0 price of the option.

    Raises:
        ValueError: If inputs are invalid or params are inconsistent with T, N.

    Notes:
        For Leisen-Reimer parameterizations with even N (which internally
        uses N+1), pass the original N here; we use params.dt to infer
        the actual step count.
    """
    if S <= 0:
        raise ValueError(f"Spot price S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time T must be positive, got {T}")
    if N <= 0:
        raise ValueError(f"Number of steps N must be positive, got {N}")

    # Use the actual N from params.dt to handle LR's odd-N adjustment
    N_eff = int(round(T / params.dt))
    payoff = _payoff_function(option_type, K)

    # Terminal stock prices: S_N(j) = S * u^j * d^(N-j) for j = 0, ..., N
    j = np.arange(N_eff + 1)
    terminal_prices = S * (params.u ** j) * (params.d ** (N_eff - j))

    # Terminal option values
    V = payoff(terminal_prices)

    # Discount factor per period
    discount = np.exp(-r * params.dt)

    # Backward induction
    if exercise_style == "european":
        for n in range(N_eff - 1, -1, -1):
            V = discount * (params.p * V[1 : n + 2] + (1 - params.p) * V[0 : n + 1])
    elif exercise_style == "american":
        for n in range(N_eff - 1, -1, -1):
            j_n = np.arange(n + 1)
            stock_prices_n = S * (params.u ** j_n) * (params.d ** (n - j_n))
            continuation = discount * (
                params.p * V[1 : n + 2] + (1 - params.p) * V[0 : n + 1]
            )
            exercise = payoff(stock_prices_n)
            V = np.maximum(continuation, exercise)
    else:
        raise ValueError(
            f"exercise_style must be 'european' or 'american', got {exercise_style!r}"
        )

    return float(V[0])


def binomial_price_closed_form(
    S: float,
    K: float,
    T: float,
    r: float,
    N: int,
    params: TreeParameters,
    option_type: OptionType,
    q: float = 0.0,
) -> float:
    """Price a European option using the closed-form CRR formula.

    Implements equation (3.10) of the accompanying note:

        C_0 = S * Phi(a; N, p') - K * R^{-N} * Phi(a; N, p)

    where Phi is the complementary binomial CDF and p' = (u/R) * p.

    This pricer is primarily useful as a cross-check on the backward-
    induction pricer for European options. Backward induction is
    preferred in practice because it generalizes to American options.

    Args:
        S: Current spot price. Must be positive.
        K: Strike price. Must be positive.
        T: Time to expiration in years. Must be positive.
        r: Continuously compounded riskless rate.
        N: Number of time steps.
        params: TreeParameters with u, d, p, dt.
        option_type: "call" or "put".
        q: Continuous dividend yield. Defaults to 0.

    Returns:
        The European option price computed via the closed-form formula.

    Raises:
        ValueError: If inputs are invalid.
    """
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    N_eff = int(round(T / params.dt))
    R = np.exp(r * params.dt)
    p_prime = (params.u / R) * params.p

    # a = smallest non-negative integer such that u^a * d^(N-a) * S > K
    if option_type == "call":
        if S * params.u**N_eff <= K:
            return 0.0
        log_ratio = np.log(K / S) - N_eff * np.log(params.d)
        log_ud = np.log(params.u / params.d)
        a = int(np.ceil(log_ratio / log_ud))
        a = max(0, a)
    elif option_type == "put":
        # For puts, a is the largest j such that u^j * d^(N-j) * S < K
        # i.e., the put is in the money for j < a (strictly)
        if S * params.d**N_eff >= K:
            return 0.0
        log_ratio = np.log(K / S) - N_eff * np.log(params.d)
        log_ud = np.log(params.u / params.d)
        a = int(np.floor(log_ratio / log_ud))
        a = min(N_eff, a)
    else:
        raise ValueError(f"option_type must be 'call' or 'put'")

    # Complementary binomial CDFs
    if option_type == "call":
        Phi_p = 1 - binom.cdf(a - 1, N_eff, params.p)
        Phi_p_prime = 1 - binom.cdf(a - 1, N_eff, p_prime)
        return S * Phi_p_prime - K * np.exp(-r * T) * Phi_p
    else:  # put
        Phi_p = binom.cdf(a, N_eff, params.p)
        Phi_p_prime = binom.cdf(a, N_eff, p_prime)
        return K * np.exp(-r * T) * Phi_p - S * Phi_p_prime


def american_exercise_boundary(
    S: float,
    K: float,
    T: float,
    r: float,
    N: int,
    params: TreeParameters,
    option_type: OptionType,
    q: float = 0.0,
) -> np.ndarray:
    """Extract the American exercise boundary from the binomial tree.

    For American puts, returns the largest stock price at each time step
    n at which immediate exercise is optimal. For American calls on
    dividend-paying stocks, returns the smallest such price.

    Args:
        S: Current spot price.
        K: Strike price.
        T: Time to expiration.
        r: Riskless rate.
        N: Number of time steps.
        params: TreeParameters.
        option_type: "call" or "put".
        q: Continuous dividend yield.

    Returns:
        Array of length N+1 giving the exercise boundary at each time step.
        Entries are np.nan at time steps where no exercise is optimal.
    """
    N_eff = int(round(T / params.dt))
    payoff = _payoff_function(option_type, K)

    j = np.arange(N_eff + 1)
    terminal_prices = S * (params.u**j) * (params.d ** (N_eff - j))
    V = payoff(terminal_prices)

    discount = np.exp(-r * params.dt)
    boundary = np.full(N_eff + 1, np.nan)
    boundary[N_eff] = K  # at expiry, exercise iff payoff > 0

    for n in range(N_eff - 1, -1, -1):
        j_n = np.arange(n + 1)
        stock_prices_n = S * (params.u**j_n) * (params.d ** (n - j_n))
        continuation = discount * (
            params.p * V[1 : n + 2] + (1 - params.p) * V[0 : n + 1]
        )
        exercise = payoff(stock_prices_n)
        is_exercise = exercise >= continuation
        if option_type == "put" and np.any(is_exercise):
            # Largest stock price at which exercise is optimal
            boundary[n] = stock_prices_n[is_exercise].max()
        elif option_type == "call" and np.any(is_exercise & (exercise > 0)):
            # Smallest stock price at which exercise is optimal
            mask = is_exercise & (exercise > 0)
            boundary[n] = stock_prices_n[mask].min()
        V = np.maximum(continuation, exercise)

    return boundary

def richardson_extrapolation(
    price_N: float,
    price_2N: float,
    rho: float = 2.0,
) -> float:
    """Richardson-extrapolate two binomial prices to a higher-order estimate.

    Given prices ``C(N)`` and ``C(2N)`` from a tree scheme whose pricing
    error e(N) = C(N) - C_true satisfies a known limit of the error
    ratio rho = e(N) / e(2N), the Richardson-extrapolated estimate

        C_hat(2N) = (rho * C(2N) - C(N)) / (rho - 1)

    has lower-order error than either input. This is Tian (1999),
    eq. (17). For schemes with smooth (monotone) convergence at rate
    O(1/N), the error ratio rho -> 2; this is the default.

    The technique only improves accuracy when convergence is smooth.
    For oscillatory schemes (such as standard CRR for vanilla
    options), the error ratio fluctuates and extrapolation can amplify
    rather than reduce error.

    Args:
        price_N: Price computed with N time steps.
        price_2N: Price computed with 2N time steps.
        rho: Limit of the error ratio. Default 2.0, appropriate for
            O(1/N) schemes including the strike-aligned Tian (1999)
            flexible binomial.

    Returns:
        The Richardson-extrapolated price estimate.

    Raises:
        ValueError: If rho == 1 (the formula is singular).

    References:
        Tian, Y. (1999). A Flexible Binomial Option Pricing Model.
        Journal of Futures Markets, 19(7), 817-843. See eqs. (16),
        (17), (18).
    """
    if rho == 1.0:
        raise ValueError("Richardson extrapolation requires rho != 1")
    return (rho * price_2N - price_N) / (rho - 1.0)