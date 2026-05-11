"""Generate a social preview image for the GitHub repository.

Produces a 1280x640 PNG (2:1 aspect ratio, GitHub's recommended dimensions)
showing the convergence comparison at K = 95 (ITM), where the three-scheme
distinction is most visually striking.

Output: figures/social_preview.png
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
    leisen_reimer_parameters,
    tian_1999_parameters,
)
from option_pricing.pricers import binomial_price


# Use the ITM case where Tian's smooth curve is visible against CRR scatter
S = 100.0
K = 95.0
T = 1.0
r = 0.05
SIGMA = 0.20

N_VALUES = np.concatenate([
    np.arange(10, 100, 1),
    np.arange(100, 500, 5),
    np.arange(500, 1001, 25),
])


def compute_errors() -> dict[str, np.ndarray]:
    bs_price = black_scholes_call(S=S, K=K, T=T, r=r, sigma=SIGMA)
    errors = {"crr": [], "tian": [], "lr": []}

    for N in N_VALUES:
        N = int(N)

        p_crr = crr_parameters(T=T, N=N, r=r, sigma=SIGMA)
        errors["crr"].append(abs(binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=p_crr,
            option_type="call", exercise_style="european",
        ) - bs_price))

        p_tian = tian_1999_parameters(S=S, K=K, T=T, N=N, r=r, sigma=SIGMA)
        errors["tian"].append(abs(binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=p_tian,
            option_type="call", exercise_style="european",
        ) - bs_price))

        p_lr = leisen_reimer_parameters(S=S, K=K, T=T, N=N, r=r, sigma=SIGMA)
        errors["lr"].append(abs(binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=p_lr,
            option_type="call", exercise_style="european",
        ) - bs_price))

    return {k: np.array(v) for k, v in errors.items()}


def main() -> None:
    set_style()

    print("Computing errors for social preview...")
    errors = compute_errors()
    print("Done.")

    # 2:1 aspect ratio for GitHub social preview (1280x640 at dpi=160).
    # Slightly taller than strict 2:1 to give labels room before crop.
    fig, ax = plt.subplots(figsize=(12.8, 6.4))

    n_ref = np.array([10, 1100])

    median_crr_at_100 = np.median(
        errors["crr"][(N_VALUES >= 80) & (N_VALUES <= 120)]
    )
    ref_constant_1 = median_crr_at_100 * 100
    ax.loglog(
        n_ref, ref_constant_1 / n_ref,
        color=COLORS["envelope"], linestyle="--", linewidth=1.2, alpha=0.7,
        label=r"$\mathcal{O}(1/N)$ reference",
        zorder=1,
    )

    median_lr_at_100 = np.median(
        errors["lr"][(N_VALUES >= 80) & (N_VALUES <= 120)]
    )
    ref_constant_2 = median_lr_at_100 * 100**2
    ax.loglog(
        n_ref, ref_constant_2 / n_ref**2,
        color=COLORS["envelope"], linestyle=":", linewidth=1.2, alpha=0.7,
        label=r"$\mathcal{O}(1/N^2)$ reference",
        zorder=1,
    )

    ax.loglog(
        N_VALUES, errors["crr"],
        marker="o", linestyle="none", color=COLORS["crr"],
        markersize=4.0, alpha=0.7,
        label="CRR (1979)",
        zorder=3,
    )

    ax.loglog(
        N_VALUES, errors["tian"],
        color=COLORS["tian"], linewidth=2.2, linestyle="-",
        label="Tian (1999)",
        zorder=4,
    )

    ax.loglog(
        N_VALUES, errors["lr"],
        color=COLORS["lr"], linewidth=2.2,
        label="Leisen--Reimer (1996)",
        zorder=5,
    )

    # Larger fonts so they survive thumbnail compression
    ax.set_xlabel(r"Number of binomial steps, $N$", fontsize=14)
    ax.set_ylabel(r"Absolute pricing error", fontsize=14)
    ax.set_title(
        "Binomial Option Pricing: Convergence to Black--Scholes",
        fontsize=16, pad=12,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(8, 1200)
    ax.legend(loc="lower left", fontsize=12)

    plt.tight_layout()

    output_path = (
        Path(__file__).parent.parent / "figures" / "social_preview.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100)  # 12.8*100 x 6.4*100 = 1280x640
    print(f"Saved {output_path}")
    plt.close()


if __name__ == "__main__":
    main()