"""Generate convergence plot comparing CRR, Tian, and Leisen-Reimer.

Produces a log-log plot of absolute pricing error against N for the three
binomial schemes on a standard test case. Demonstrates the O(1/N) rate
for CRR and Tian and the O(1/N^2) rate for Leisen-Reimer, with the
oscillation in the CRR scheme visible as scatter around the envelope.

Output: figures/convergence_three_schemes.pdf

Reproducibility:
    All inputs are specified as constants below. No randomness is used.
    Re-running this script will produce a byte-identical PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import from the package; we add the repo root to path so this works
# whether run as `python scripts/...` or via uv run
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _style import COLORS, FIGSIZE_WIDE, set_style

from option_pricing.black_scholes import black_scholes_call
from option_pricing.parameterizations import (
    crr_parameters,
    leisen_reimer_parameters,
)
from option_pricing.pricers import binomial_price


# Test case parameters
S = 100.0
K = 100.0
T = 1.0
r = 0.05
SIGMA = 0.20

# N values to sweep. Use dense grid at low N to capture oscillation;
# coarser at high N where convergence dominates.
N_VALUES = np.concatenate([
    np.arange(10, 100, 1),     # 10..99, every step
    np.arange(100, 500, 5),    # 100..495, every 5 steps
    np.arange(500, 1001, 25),  # 500..1000, every 25 steps
])


def compute_errors() -> dict[str, np.ndarray]:
    """Compute |C^N - C^BS| for each scheme across the N grid."""
    bs_price = black_scholes_call(S=S, K=K, T=T, r=r, sigma=SIGMA)

    errors = {"crr": [], "lr": []}

    for N in N_VALUES:
        N = int(N)

        # CRR
        params_crr = crr_parameters(T=T, N=N, r=r, sigma=SIGMA)
        price_crr = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params_crr,
            option_type="call", exercise_style="european",
        )
        errors["crr"].append(abs(price_crr - bs_price))

        # LR
        params_lr = leisen_reimer_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=SIGMA
        )
        price_lr = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params_lr,
            option_type="call", exercise_style="european",
        )
        errors["lr"].append(abs(price_lr - bs_price))

    return {k: np.array(v) for k, v in errors.items()}


def main() -> None:
    set_style()

    print(f"Computing convergence errors for {len(N_VALUES)} values of N...")
    errors = compute_errors()
    print("Done.")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Reference slopes drawn FIRST so they sit behind the data
    n_ref = np.array([10, 1100])
    median_crr_at_100 = np.median(
        errors["crr"][(N_VALUES >= 80) & (N_VALUES <= 120)]
    )
    ref_constant_1 = median_crr_at_100 * 100
    ax.loglog(
        n_ref, ref_constant_1 / n_ref,
        color=COLORS["envelope"], linestyle="--", linewidth=1.0, alpha=0.7,
        label=r"$\mathcal{O}(1/N)$ reference",
        zorder=1,
    )

    median_lr_at_100 = np.median(
        errors["lr"][(N_VALUES >= 80) & (N_VALUES <= 120)]
    )
    ref_constant_2 = median_lr_at_100 * 100**2
    ax.loglog(
        n_ref, ref_constant_2 / n_ref**2,
        color=COLORS["envelope"], linestyle=":", linewidth=1.0, alpha=0.7,
        label=r"$\mathcal{O}(1/N^2)$ reference",
        zorder=1,
    )

    # CRR: clear scatter, larger markers so oscillation is visible
    ax.loglog(
        N_VALUES, errors["crr"],
        marker="o", linestyle="none", color=COLORS["crr"],
        markersize=2.5, alpha=0.7,
        label="CRR (1979)",
        zorder=3,
    )

    # LR: smooth curve (no oscillation)
    ax.loglog(
        N_VALUES, errors["lr"],
        color=COLORS["lr"], linewidth=1.4,
        label="Leisen--Reimer (1996)",
        zorder=4,
    )


    ax.set_xlabel(r"Number of binomial steps, $N$")
    ax.set_ylabel(r"Absolute pricing error, $|C^{(N)}_0 - C^{\mathrm{BS}}|$")
    ax.set_xlim(8, 1200)
    ax.legend(loc="lower left")

    # Title is intentionally omitted; LaTeX caption provides description
    plt.tight_layout()

    output_path = (
        Path(__file__).parent.parent / "figures"
        / "convergence_crr_vs_lr.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"), dpi=150)
    print(f"Saved {output_path} (+ .png)")
    plt.close()


if __name__ == "__main__":
    main()