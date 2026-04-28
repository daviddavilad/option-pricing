"""Two-panel analysis of CRR vs strict-aligned Tian as N varies.

Top panel: absolute pricing error vs N for both schemes (log-log).
Bottom panel: ratio of errors (Tian/CRR) on log scale, with the y=1
line marked. Below 1: Tian wins; above 1: CRR wins.

The figure reveals the regime structure: Tian dominates at low and
high N, while a transition zone near N~100-200 temporarily favors
CRR's lucky alignments. The advantage of Tian grows at large N where
its smoother convergence outpaces CRR's oscillating envelope.

Output: figures/tian_regime_analysis.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _style import COLORS, set_style

from option_pricing.black_scholes import black_scholes_call
from option_pricing.parameterizations import (
    crr_parameters,
    tian_parameters,
)
from option_pricing.pricers import binomial_price


# Same test case as tian_alignment_wins for consistency
S = 100.0
K = 120.0
T = 0.25
r = 0.05
SIGMA = 0.40

# Wide N range to see the full regime structure
N_VALUES = np.concatenate([
    np.arange(10, 100, 1),      # dense at low N
    np.arange(100, 500, 5),     # medium density
    np.arange(500, 2001, 25),   # extend to high N to show Tian's dominance
])


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


def rolling_geometric_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth a positive-valued series with a rolling geometric mean.

    Geometric mean is appropriate for ratio data on a log scale.
    Returns NaN at edges where window doesn't fit.
    """
    log_vals = np.log(values)
    smoothed = np.full_like(log_vals, np.nan)
    half = window // 2
    for i in range(half, len(log_vals) - half):
        smoothed[i] = np.exp(np.mean(log_vals[i - half : i + half + 1]))
    return smoothed


def main() -> None:
    set_style()

    print(f"Computing errors for {len(N_VALUES)} values of N...")
    print(f"Test case: S={S}, K={K}, T={T}, sigma={SIGMA}")
    errors = compute_errors()
    print("Done.")

    # Compute ratio of errors. Tian/CRR < 1 means Tian wins.
    ratio = errors["tian"] / errors["crr"]
    smoothed_ratio = rolling_geometric_mean(ratio, window=15)

    # Two-panel figure with shared x-axis
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.5], "hspace": 0.08},
    )

    # ---- Top panel: absolute errors ----
    n_ref = np.array([10, 2200])
    median_crr_at_100 = np.median(
        errors["crr"][(N_VALUES >= 80) & (N_VALUES <= 120)]
    )
    ref_constant = median_crr_at_100 * 100
    ax_top.loglog(
        n_ref, ref_constant / n_ref,
        color=COLORS["envelope"], linestyle="--", linewidth=1.0, alpha=0.7,
        label=r"$\mathcal{O}(1/N)$ reference",
        zorder=1,
    )
    ax_top.loglog(
        N_VALUES, errors["crr"],
        marker="o", linestyle="none", color=COLORS["crr"],
        markersize=2.0, alpha=0.7,
        label="CRR (1979)",
        zorder=3,
    )
    ax_top.loglog(
        N_VALUES, errors["tian"],
        marker="o", linestyle="none", color=COLORS["tian"],
        markersize=2.0, alpha=0.7,
        label="Tian (1999), strict strike-aligned",
        zorder=2,
    )
    ax_top.set_ylabel(r"Absolute error, $|C^{(N)}_0 - C^{\mathrm{BS}}|$")
    ax_top.legend(loc="lower left")
    ax_top.set_xlim(8, 2500)

    # ---- Bottom panel: ratio of errors ----
    # Light scatter of raw ratios, then smoothed line on top
    ax_bot.semilogx(
        N_VALUES, ratio,
        marker="o", linestyle="none", color="#888888",
        markersize=1.5, alpha=0.4, zorder=2,
    )
    ax_bot.semilogx(
        N_VALUES, smoothed_ratio,
        color="#222222", linewidth=1.6, zorder=4,
    )
    # Reference line at ratio=1 (parity between schemes)
    ax_bot.axhline(1.0, color=COLORS["envelope"], linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=1)
    # Shade regions where Tian wins (smoothed ratio < 1)
    ax_bot.fill_between(
        N_VALUES, 0.01, 100,
        where=(smoothed_ratio < 1) & ~np.isnan(smoothed_ratio),
        color=COLORS["tian"], alpha=0.10, zorder=0,
    )
    ax_bot.fill_between(
        N_VALUES, 0.01, 100,
        where=(smoothed_ratio > 1) & ~np.isnan(smoothed_ratio),
        color=COLORS["crr"], alpha=0.10, zorder=0,
    )
    ax_bot.set_yscale("log")
    ax_bot.set_ylim(0.05, 20)
    ax_bot.set_xlabel(r"Number of binomial steps, $N$")
    ax_bot.set_ylabel(r"$\varepsilon_{\mathrm{Tian}} / \varepsilon_{\mathrm{CRR}}$")

    # Annotations on the ratio panel showing winner regions
    # Compute typical N values for each region for label placement
    ax_bot.text(
        25, 0.10, "Tian wins", color=COLORS["tian"],
        fontsize=9, ha="center", va="center",
        fontweight="bold", alpha=0.8,
    )
    ax_bot.text(
        150, 8, "CRR wins\n(transition)", color=COLORS["crr"],
        fontsize=9, ha="center", va="center",
        fontweight="bold", alpha=0.8,
    )
    ax_bot.text(
        1200, 0.08, "Tian wins", color=COLORS["tian"],
        fontsize=9, ha="center", va="center",
        fontweight="bold", alpha=0.8,
    )

    plt.tight_layout()

    output_path = (
        Path(__file__).parent.parent / "figures" / "tian_regime_analysis.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()


if __name__ == "__main__":
    main()