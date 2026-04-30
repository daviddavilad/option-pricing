"""Fit empirical European convergence rates for Table 7.1.

Computes |C_0^(N) - C^BS| for CRR, Tian (1999), and Leisen--Reimer
across a grid of N values, then fits

    log|epsilon_N| = alpha - beta * log N

by ordinary least squares.

Test case: standard ATM, S=K=100, T=1, r=0.05, sigma=0.20.
N range: [101, 1001], odd values only (LR requires odd N).

Note: at K=S=100 with even N, Tian (1999) collapses to CRR
because the strike-aligning tilt parameter lambda vanishes. We use
odd N throughout to keep all three schemes apples-to-apples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from option_pricing.parameterizations import (
    crr_parameters,
    leisen_reimer_parameters,
    tian_1999_parameters,
)
from option_pricing.pricers import binomial_price


# Test case
S = 100.0
K = 100.0
T = 1.0
r = 0.05
SIGMA = 0.20


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def fit_rate(N_arr: np.ndarray, err_arr: np.ndarray) -> tuple[float, float]:
    """Fit log|err| = alpha - beta * log(N). Returns (beta, intercept)."""
    log_N = np.log(N_arr)
    log_err = np.log(err_arr)
    slope, intercept = np.polyfit(log_N, log_err, 1)
    return -slope, intercept  # rate beta is negative slope


def main() -> None:
    bs_price = black_scholes_call(S, K, T, r, SIGMA)
    print(f"Black-Scholes reference: {bs_price:.6f}")

    # Odd N from 101 to 1001
    N_values = np.arange(101, 1002, 2)
    print(f"Computing errors at {len(N_values)} odd N from {N_values[0]} to {N_values[-1]}...")

    errors = {"crr": [], "tian": [], "lr": []}
    for N in N_values:
        N = int(N)

        p_crr = crr_parameters(T=T, N=N, r=r, sigma=SIGMA)
        price_crr = binomial_price(S=S, K=K, T=T, r=r, N=N, params=p_crr,
                                    option_type="call", exercise_style="european")
        errors["crr"].append(abs(price_crr - bs_price))

        p_tian = tian_1999_parameters(S=S, K=K, T=T, N=N, r=r, sigma=SIGMA)
        price_tian = binomial_price(S=S, K=K, T=T, r=r, N=N, params=p_tian,
                                     option_type="call", exercise_style="european")
        errors["tian"].append(abs(price_tian - bs_price))

        p_lr = leisen_reimer_parameters(S=S, K=K, T=T, N=N, r=r, sigma=SIGMA)
        price_lr = binomial_price(S=S, K=K, T=T, r=r, N=N, params=p_lr,
                                   option_type="call", exercise_style="european")
        errors["lr"].append(abs(price_lr - bs_price))

    errors = {k: np.array(v) for k, v in errors.items()}

    print()
    print("Fitted convergence rates (European call, S=K=100, T=1, r=0.05, sigma=0.20):")
    print()
    print(f"  {'Scheme':<20} {'beta':>8}   Theoretical")
    print(f"  {'-'*20} {'-'*8}   {'-'*11}")
    beta_crr, _ = fit_rate(N_values, errors["crr"])
    beta_tian, _ = fit_rate(N_values, errors["tian"])
    beta_lr, _ = fit_rate(N_values, errors["lr"])
    print(f"  {'CRR':<20} {beta_crr:>8.3f}   1")
    print(f"  {'Tian (1999)':<20} {beta_tian:>8.3f}   1")
    print(f"  {'Leisen--Reimer':<20} {beta_lr:>8.3f}   2")
    print()

    # Also report a sanity check via successive doublings for LR
    # since we expect rate ~2 and want to confirm not noise
    print("LR successive-doublings sanity check (rate near 2 expected):")
    for N in [101, 201, 401, 801]:
        p_lr_n = leisen_reimer_parameters(S=S, K=K, T=T, N=N, r=r, sigma=SIGMA)
        err_n = abs(binomial_price(S=S, K=K, T=T, r=r, N=N, params=p_lr_n,
                                    option_type="call", exercise_style="european") - bs_price)
        p_lr_2n = leisen_reimer_parameters(S=S, K=K, T=T, N=2*N+1, r=r, sigma=SIGMA)
        err_2n = abs(binomial_price(S=S, K=K, T=T, r=r, N=2*N+1, params=p_lr_2n,
                                     option_type="call", exercise_style="european") - bs_price)
        ratio = err_n / err_2n
        rate = np.log2(ratio)
        print(f"  N={N:4d} -> {2*N+1:4d}:  err ratio = {ratio:6.2f}, rate = {rate:.2f}")


if __name__ == "__main__":
    main()