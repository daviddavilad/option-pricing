"""Upper-envelope decay rates for CRR and strict-aligned Tian.

A standard view of CRR convergence (see Diener and Diener 2004, Walsh
2003) is to look not at every individual error but at the *upper
envelope* of the oscillating scatter. The upper envelope of CRR decays
considerably more slowly than $1/N$ in regimes where the strike is
poorly aligned with terminal nodes, even though many individual N
values reach near-$1/N$ accuracy by luck. Strict strike-aligned Tian
trades that lottery for a more controlled envelope: every N is forced
into the same alignment regime, so the resulting scatter is much
tighter and its envelope is closer to a clean $1/N$ line.

This figure plots the rolling 90th percentile (in log-error space) for
each scheme as the upper envelope. Light scatter sits behind for
context. The point of comparison is the *crossing* of the two upper
envelopes: that crossing tells us where Tian's controlled discreteness
becomes a more reliable choice than CRR's lottery.

Output: figures/tian_envelope_crossings.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _style import COLORS, FIGSIZE_WIDE, set_style

from option_pricing.black_scholes import black_scholes_call
from option_pricing.parameterizations import (
    crr_parameters,
    tian_parameters,
)
from option_pricing.pricers import binomial_price


# Same regime as the other Tian-vs-CRR figures, for direct comparison.
S = 100.0
K = 120.0
T = 0.25
r = 0.05
SIGMA = 0.40

N_VALUES = np.concatenate([
    np.arange(10, 100, 1),
    np.arange(100, 500, 5),
    np.arange(500, 2001, 25),
])

# Window for the rolling envelope. Wider than the win-rate window
# (which uses 21) because we are estimating an upper quantile rather
# than a mean, and quantile estimates are noisier at small samples.
WINDOW = 41
UPPER_QUANTILE = 0.90


def compute_errors() -> dict[str, np.ndarray]:
    """Compute |C^N - C^BS| for CRR and strict Tian across N."""
    bs_price = black_scholes_call(S=S, K=K, T=T, r=r, sigma=SIGMA)
    errors = {"crr": [], "tian": []}

    for N in N_VALUES:
        N = int(N)
        params_crr = crr_parameters(T=T, N=N, r=r, sigma=SIGMA)
        price_crr = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params_crr,
            option_type="call", exercise_style="european",
        )
        errors["crr"].append(abs(price_crr - bs_price))

        params_tian = tian_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=SIGMA
        )
        price_tian = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params_tian,
            option_type="call", exercise_style="european",
        )
        errors["tian"].append(abs(price_tian - bs_price))

    return {k: np.array(v) for k, v in errors.items()}


def rolling_quantile_log(values: np.ndarray, window: int,
                         q: float) -> np.ndarray:
    """Centered rolling q-quantile of values, computed in log space.

    Computing the quantile of log-values and exponentiating gives a
    geometrically symmetric envelope, which is the natural choice for
    multiplicative error data on a log scale. Returns NaN at the edges
    where the window does not fit.
    """
    log_vals = np.log(values)
    smoothed = np.full_like(log_vals, np.nan)
    half = window // 2
    for i in range(half, len(log_vals) - half):
        smoothed[i] = np.quantile(log_vals[i - half : i + half + 1], q)
    return np.exp(smoothed)


def find_crossings(x: np.ndarray, y_a: np.ndarray,
                   y_b: np.ndarray) -> np.ndarray:
    """Indices i where the sign of (y_a - y_b) flips between i and i+1.

    NaNs in either curve are skipped. Used as a stdout diagnostic to
    show where the two envelopes cross.
    """
    diff = y_a - y_b
    valid = ~np.isnan(diff)
    crossings = []
    for i in range(len(diff) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        if diff[i] == 0 or diff[i] * diff[i + 1] < 0:
            crossings.append(i)
    return np.array(crossings, dtype=int)


def main() -> None:
    set_style()

    print(f"Computing errors for {len(N_VALUES)} values of N...")
    print(f"Test case: S={S}, K={K}, T={T}, sigma={SIGMA}")
    errors = compute_errors()
    print("Done.")

    env_crr = rolling_quantile_log(errors["crr"], window=WINDOW,
                                   q=UPPER_QUANTILE)
    env_tian = rolling_quantile_log(errors["tian"], window=WINDOW,
                                    q=UPPER_QUANTILE)

    # Diagnostic only: print the crossings to stdout. We deliberately
    # do not draw vertical markers on the figure -- with two noisy
    # rolling envelopes the "crossings" form clusters rather than a
    # single clean transition, and adding markers tends to overstate
    # how cleanly the regimes separate. The shaded regions on
    # tian_regime_analysis.pdf already convey that information.
    crossings = find_crossings(N_VALUES.astype(float), env_tian, env_crr)
    if len(crossings):
        crossing_Ns = N_VALUES[crossings]
        print(f"\nUpper-envelope crossings at N = {crossing_Ns.tolist()}")
    else:
        print("\nNo upper-envelope crossings found in this range.")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Light scatter behind everything.
    ax.loglog(
        N_VALUES, errors["crr"],
        marker="o", linestyle="none", color=COLORS["crr"],
        markersize=2.0, alpha=0.30, zorder=2,
        label="CRR (1979), individual $N$",
    )
    ax.loglog(
        N_VALUES, errors["tian"],
        marker="o", linestyle="none", color=COLORS["tian"],
        markersize=2.0, alpha=0.30, zorder=3,
        label="Tian (1999), individual $N$",
    )

    # 1/N reference, anchored to the CRR envelope around N=100 so the
    # slope-comparison is visually fair.
    n_ref = np.array([10, 2200])
    median_crr_at_100 = np.median(
        errors["crr"][(N_VALUES >= 80) & (N_VALUES <= 120)]
    )
    ref_constant = median_crr_at_100 * 100
    ax.loglog(
        n_ref, ref_constant / n_ref,
        color=COLORS["envelope"], linestyle="--", linewidth=1.0, alpha=0.7,
        label=r"$\mathcal{O}(1/N)$ reference",
        zorder=1,
    )

    # The upper envelopes themselves.
    ax.loglog(
        N_VALUES, env_crr,
        color=COLORS["crr"], linewidth=2.0, zorder=5,
        label=f"CRR upper envelope ({int(UPPER_QUANTILE*100)}th pct)",
    )
    ax.loglog(
        N_VALUES, env_tian,
        color=COLORS["tian"], linewidth=2.0, zorder=6,
        label=f"Tian upper envelope ({int(UPPER_QUANTILE*100)}th pct)",
    )

    ax.set_xlabel(r"Number of binomial steps, $N$")
    ax.set_ylabel(r"Absolute pricing error, $|C^{(N)}_0 - C^{\mathrm{BS}}|$")
    ax.set_xlim(8, 2500)
    ax.legend(loc="lower left", ncol=2, columnspacing=1.2)

    plt.tight_layout()

    output_path = (
        Path(__file__).parent.parent
        / "figures" / "tian_envelope_crossings.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\nSaved {output_path}")
    plt.close()


if __name__ == "__main__":
    main()