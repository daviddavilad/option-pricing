"""Benchmark wall-clock runtime for the three binomial schemes.

Measures runtime for a single European call price evaluation across
CRR, Tian (1999), and Leisen--Reimer at four values of N. Used to
populate Table~\\ref{tab:runtime} in the LaTeX note.

Methodology:
  - Best-of-N timing via timeit (mitigates OS scheduling noise).
  - Inner repeat count chosen so each measurement takes 0.5-2 seconds.
  - Reports best-of-5 mean of inner-repeat batches, in milliseconds.
  - Hardware/Python metadata printed at top for reproducibility.

Note: all schemes share the same O(N^2) backward-induction inner
loop. Differences between schemes are dominated by parameter-
calibration cost (constant in N), which is negligible for N >= 100.
We expect runtimes to be very close across schemes at fixed N, with
the main variation coming from N itself.
"""

from __future__ import annotations

import platform
import sys
import timeit
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from option_pricing.parameterizations import (
    crr_parameters,
    leisen_reimer_parameters,
    tian_1999_parameters,
)
from option_pricing.pricers import binomial_price


# Test case parameters
S = 100.0
K = 100.0
T = 1.0
r = 0.05
SIGMA = 0.20

# N values to benchmark
N_VALUES = [101, 501, 1001, 5001]


def _bench_single(callable_factory, n_inner: int, n_outer: int = 5) -> float:
    """Best-of-n_outer timing of n_inner repeats, returns ms per call."""
    timer = timeit.Timer(callable_factory)
    times = timer.repeat(repeat=n_outer, number=n_inner)
    best = min(times)
    return (best / n_inner) * 1000.0


def _pick_inner_count(callable_factory, target_seconds: float = 0.5) -> int:
    """Pick n_inner so a single timing batch takes ~target_seconds."""
    timer = timeit.Timer(callable_factory)
    one_call_time = timer.timeit(number=1)
    if one_call_time <= 0:
        return 1000
    n = max(1, int(target_seconds / one_call_time))
    return n


def benchmark_scheme(scheme_name: str, N: int) -> float:
    """Return the runtime in milliseconds for one price evaluation."""
    if scheme_name == "crr":
        params = crr_parameters(T=T, N=N, r=r, sigma=SIGMA)
    elif scheme_name == "tian":
        params = tian_1999_parameters(S=S, K=K, T=T, N=N, r=r, sigma=SIGMA)
    elif scheme_name == "lr":
        params = leisen_reimer_parameters(S=S, K=K, T=T, N=N, r=r, sigma=SIGMA)
    else:
        raise ValueError(f"Unknown scheme: {scheme_name}")

    def call() -> None:
        binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params,
            option_type="call", exercise_style="european",
        )

    # Warmup
    call()
    call()

    n_inner = _pick_inner_count(call, target_seconds=0.5)
    return _bench_single(call, n_inner=n_inner, n_outer=5)


def print_metadata() -> None:
    print("Benchmark environment:")
    print(f"  Platform:    {platform.platform()}")
    print(f"  Processor:   {platform.processor() or platform.machine()}")
    print(f"  Python:      {sys.version.split()[0]}")
    print(f"  NumPy:       {np.__version__}")
    print()


def main() -> None:
    print_metadata()
    print(f"Test case: S={S}, K={K}, T={T}, r={r}, sigma={SIGMA}")
    print()
    print("Best-of-5 mean runtime per call, in milliseconds:")
    print()
    print(f"  {'N':>6}    {'CRR':>10}   {'Tian':>10}   {'LR':>10}")
    print(f"  {'-'*6}    {'-'*10}   {'-'*10}   {'-'*10}")

    rows = []
    for N in N_VALUES:
        ms_crr = benchmark_scheme("crr", N)
        ms_tian = benchmark_scheme("tian", N)
        ms_lr = benchmark_scheme("lr", N)
        rows.append((N, ms_crr, ms_tian, ms_lr))
        print(f"  {N:>6}    {ms_crr:>9.3f}    {ms_tian:>9.3f}    {ms_lr:>9.3f}")

    print()
    print("LaTeX table rows (paste into Table tab:runtime):")
    print()
    for N, ms_crr, ms_tian, ms_lr in rows:
        if N >= 1000:
            n_str = f"{N // 1000}{{,}}{N % 1000:03d}"
        else:
            n_str = str(N)
        print(f"        {n_str} & {ms_crr:.3g} & {ms_tian:.3g} & {ms_lr:.3g} \\\\")


if __name__ == "__main__":
    main()