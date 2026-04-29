"""Generate convergence plot for American put pricing.

Compares CRR, Tian (1999), and Leisen--Reimer convergence for an
American put option. No closed-form benchmark exists for American
options, so we use a Leisen--Reimer reference price computed at very
high N (25001) as a proxy for the true value.

The figure illustrates two points:

    1. Convergence patterns for American puts retain the per-scheme
       characteristics seen in the European case: CRR oscillates,
       Tian (1999) is comparatively smooth, LR dominates both. The
       early-exercise constraint does not eliminate the alignment-
       induced oscillation.

    2. The LR rate appears to degrade slightly from O(1/N^2) in the
       European case toward something between O(1/N) and O(1/N^2)
       for American puts, because the Peizer--Pratt inversion
       underlying LR is calibrated to the European Black--Scholes
       tails, not to the American value process which has no
       closed-form representation in terms of normal CDFs.

Setup: at-the-money, T=1, r=5%, sigma=20%.

Output: figures/american_convergence.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _style import COLORS, FIGSIZE_WIDE, set_style

from option_pricing.parameterizations import (
    crr_parameters,
    leisen_reimer_parameters,
    tian_1999_parameters,
)
from option_pricing.pricers import binomial_price


# Test case parameters: American put, ATM, standard test case to match
# the European convergence figure.
S = 100.0
K = 100.0
T = 1.0
r = 0.05
SIGMA = 0.20

# High-N LR reference price. Verified stable to ~5 decimals at this N.
N_REFERENCE = 25001

# N grid for the convergence comparison. Use odd N throughout because
# Leisen--Reimer requires odd N; this keeps the comparison apples-to-apples.
N_VALUES = np.concatenate([
    np.arange(11, 100, 2),     # odd N from 11 to 99
    np.arange(101, 501, 10),   # odd N from 101 to 491
    np.arange(501, 1002, 50),  # odd N up to 1001
])


def _price_american_put(scheme_params, N: int) -> float:
    return binomial_price(
        S=S, K=K, T=T, r=r, N=N, params=scheme_params,
        option_type="put", exercise_style="american",
    )


def compute_reference_price() -> float:
    """High-N LR reference for the American put."""
    print(f"Computing LR reference at N = {N_REFERENCE}...")
    params = leisen_reimer_parameters(S=S, K=K, T=T, N=N_REFERENCE,
                                       r=r, sigma=SIGMA)
    return _price_american_put(params, N_REFERENCE)


def compute_errors(p_ref: float) -> dict[str, np.ndarray]:
    errors = {"crr": [], "tian": [], "lr": []}

    for N in N_VALUES:
        N = int(N)

        params_crr = crr_parameters(T=T, N=N, r=r, sigma=SIGMA)
        errors["crr"].append(abs(_price_american_put(params_crr, N) - p_ref))

        params_tian = tian_1999_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=SIGMA,
        )
        errors["tian"].append(
            abs(_price_american_put(params_tian, N) - p_ref)
        )

        params_lr = leisen_reimer_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=SIGMA,
        )
        errors["lr"].append(abs(_price_american_put(params_lr, N) - p_ref))

    return {k: np.array(v) for k, v in errors.items()}


def main() -> None:
    set_style()

    p_ref = compute_reference_price()
    print(f"Reference price: {p_ref:.6f}")

    print(f"Computing errors for {len(N_VALUES)} values of N...")
    errors = compute_errors(p_ref)
    print("Done.")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

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

    # CRR scatter (oscillation visible)
    ax.loglog(
        N_VALUES, errors["crr"],
        marker="o", linestyle="none", color=COLORS["crr"],
        markersize=2.5, alpha=0.7,
        label="CRR (1979)",
        zorder=3,
    )

    # Tian (1999): smooth-ish but not monotone for American
    ax.loglog(
        N_VALUES, errors["tian"],
        marker="s", linestyle="-", color=COLORS["tian"],
        markersize=2.5, alpha=0.85, linewidth=0.9,
        label="Tian (1999), strike-aligned",
        zorder=4,
    )

    # LR
    ax.loglog(
        N_VALUES, errors["lr"],
        color=COLORS["lr"], linewidth=1.4,
        label="Leisen--Reimer (1996)",
        zorder=5,
    )

    ax.set_xlabel(r"Number of binomial steps, $N$")
    ax.set_ylabel(
        r"Absolute pricing error, "
        r"$|P^{\mathrm{AM}, (N)}_0 - P^{\mathrm{AM}, \mathrm{ref}}|$"
    )
    ax.set_xlim(8, 1200)
    ax.legend(loc="lower left")

    plt.tight_layout()

    output_path = (
        Path(__file__).parent.parent / "figures" / "american_convergence.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()


if __name__ == "__main__":
    main()