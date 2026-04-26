"""Peizer-Pratt approximation to the normal CDF.

The Peizer-Pratt approximation provides a closed-form rational
approximation to the standard normal cumulative distribution function
that, when used to define a binomial probability, makes the resulting
binomial CDF match the normal CDF to O(1/N^2) accuracy. This is the
inversion at the heart of the Leisen-Reimer (1996) binomial pricing
scheme.

References:
    Peizer, D. B., and Pratt, J. W. (1968). A Normal Approximation for
    Binomial, F, Beta, and Other Common, Related Tail Probabilities, I.
    Journal of the American Statistical Association, 63(324), 1416-1456.

    Leisen, D. P. J., and Reimer, M. (1996). Binomial Models for Option
    Valuation -- Examining and Improving Convergence. Applied
    Mathematical Finance, 3(4), 319-346.
"""

from __future__ import annotations

import numpy as np

__all__ = ["peizer_pratt_inversion"]


def peizer_pratt_inversion(z: float, N: int) -> float:
    """Compute the Peizer-Pratt inversion h(z, N).

    Returns h(z, N) such that the resulting binomial probability,
    when used in a binomial tree with N steps, produces a binomial
    CDF that agrees with N(z) (the standard normal CDF) at the
    relevant quantile to O(1/N^2) accuracy.

    The formula is:

        h(z, N) = 1/2 + sgn(z)/2 * sqrt(1 - exp(-(z / (N + 1/3 + 0.1/(N+1)))^2 * (N + 1/6)))

    Args:
        z: Standard normal quantile (i.e., d1 or d2 from Black-Scholes).
        N: Number of binomial steps. Must be a positive odd integer.

    Returns:
        The Peizer-Pratt approximation h(z, N) to N(z), in [0, 1].

    Raises:
        ValueError: If N is not a positive odd integer.

    Notes:
        For N even, the Leisen-Reimer scheme typically uses N+1 (odd)
        internally, so this function assumes its caller has already
        adjusted to odd N.
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if N % 2 == 0:
        raise ValueError(f"N must be odd for Peizer-Pratt; got even N={N}")

    sign_z = np.sign(z) if z != 0 else 1.0  # convention: at z=0, sign is +1

    denom = N + 1.0 / 3.0 + 0.1 / (N + 1)
    exponent_arg = -((z / denom) ** 2) * (N + 1.0 / 6.0)
    inner = 1.0 - np.exp(exponent_arg)

    # Numerical guard: inner should be in [0, 1] but floating-point can
    # produce tiny negative values for z very close to 0
    inner = max(0.0, min(1.0, inner))

    return 0.5 + sign_z * 0.5 * np.sqrt(inner)