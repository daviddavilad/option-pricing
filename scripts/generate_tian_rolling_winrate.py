"""Rolling win-rate of strict-aligned Tian against CRR as N varies.

For each N, we compute the fraction of nearby N values (sliding window)
where Tian's absolute error is strictly less than CRR's. The result is a
single curve in [0, 1]: above 0.5 means Tian wins more often than not,
below 0.5 means CRR does. This complements the ratio-of-errors plot in
tian_regime_analysis.pdf: the ratio measures the *magnitude* of the
advantage in either direction; the win rate measures its *frequency*.
The two can diverge in interesting ways. For example, in a regime where
CRR has lucky integer alignments at half the N values and is heavily
penalised at the other half, a 50% win rate can sit alongside a strongly
sub-1 geometric mean ratio.

Output: figures/tian_rolling_winrate.pdf
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


# Same regime as tian_regime_analysis.pdf for direct comparison.
S = 100.0
K = 120.0
T = 0.25
r = 0.05
SIGMA = 0.40

# Wide N range so the regime structure is visible. We deliberately use
# the same grid as the regime-analysis figure so the curves line up.
N_VALUES = np.concatenate([
    np.arange(10, 100, 1),
    np.arange(100, 500, 5),
    np.arange(500, 2001, 25),
])

# Window for the rolling fraction. With ~270 grid points, a window of 21
# spans roughly 8% of the range and produces a curve that is smooth
# enough to read but not so smoothed that it hides the transition.
WINDOW = 21


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


def rolling_mean_centered(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean. Returns NaN at the edges where the window
    does not fit, matching the convention used elsewhere in this project.
    """
    smoothed = np.full_like(values, np.nan, dtype=float)
    half = window // 2
    for i in range(half, len(values) - half):
        smoothed[i] = np.mean(values[i - half : i + half + 1])
    return smoothed


def main() -> None:
    set_style()

    print(f"Computing errors for {len(N_VALUES)} values of N...")
    print(f"Test case: S={S}, K={K}, T={T}, sigma={SIGMA}")
    errors = compute_errors()
    print("Done.")

    # Boolean: 1 where Tian wins (smaller error), 0 where CRR wins.
    # Strict inequality; ties (effectively measure zero) treated as no win.
    tian_wins = (errors["tian"] < errors["crr"]).astype(float)
    win_rate = rolling_mean_centered(tian_wins, window=WINDOW)

    # Print regime summary to stdout for cross-checking against the
    # regime-analysis figure's shaded regions.
    valid = ~np.isnan(win_rate)
    print(f"\nOverall Tian win rate: {tian_wins.mean():.2%}")
    print("Win rate by N decade:")
    for lo, hi in [(10, 50), (50, 100), (100, 200),
                   (200, 500), (500, 1000), (1000, 2000)]:
        mask = (N_VALUES >= lo) & (N_VALUES < hi)
        if mask.any():
            print(f"  N in [{lo:>4d}, {hi:>4d}): "
                  f"{tian_wins[mask].mean():.2%} ({mask.sum()} points)")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Light scatter of the underlying 0/1 indicator with jitter for context.
    rng = np.random.default_rng(seed=0)
    jitter = rng.uniform(-0.015, 0.015, size=tian_wins.shape)
    ax.semilogx(
        N_VALUES, tian_wins + jitter,
        marker="o", linestyle="none", color="#888888",
        markersize=1.5, alpha=0.35, zorder=2,
    )

    # Shade above/below the parity line.
    ax.fill_between(
        N_VALUES, 0.5, 1.05,
        where=(win_rate > 0.5) & valid,
        color=COLORS["tian"], alpha=0.10, zorder=0,
    )
    ax.fill_between(
        N_VALUES, -0.05, 0.5,
        where=(win_rate < 0.5) & valid,
        color=COLORS["crr"], alpha=0.10, zorder=0,
    )

    # Parity reference at 0.5.
    ax.axhline(0.5, color=COLORS["envelope"], linestyle="--",
               linewidth=1.0, alpha=0.7, zorder=1)

    # The win-rate curve itself.
    ax.semilogx(
        N_VALUES, win_rate,
        color="#222222", linewidth=1.6, zorder=4,
        label=f"rolling fraction (window={WINDOW})",
    )

    ax.set_xlabel(r"Number of binomial steps, $N$")
    ax.set_ylabel(
        r"Fraction with $\varepsilon_{\mathrm{Tian}} < \varepsilon_{\mathrm{CRR}}$"
    )
    ax.set_xlim(8, 2500)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # Region labels positioned in the middle of each major regime.
    ax.text(
        25, 0.92, "Tian wins more often",
        color=COLORS["tian"], fontsize=9,
        ha="center", va="center", fontweight="bold", alpha=0.85,
    )
    ax.text(
        1200, 0.92, "Tian wins more often",
        color=COLORS["tian"], fontsize=9,
        ha="center", va="center", fontweight="bold", alpha=0.85,
    )
    ax.text(
        150, 0.08, "CRR wins more often",
        color=COLORS["crr"], fontsize=9,
        ha="center", va="center", fontweight="bold", alpha=0.85,
    )

    ax.legend(loc="lower left")

    plt.tight_layout()

    output_path = (
        Path(__file__).parent.parent / "figures" / "tian_rolling_winrate.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\nSaved {output_path}")
    plt.close()


if __name__ == "__main__":
    main()