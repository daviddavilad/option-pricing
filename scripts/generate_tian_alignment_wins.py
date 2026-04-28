"""Demonstrate Tian's strict strike-alignment benefit (favorable regime).

In the far-OTM short-maturity high-volatility regime, the strike sits in
the sparse-node tail of the binomial tree, where the integer-cutoff
oscillation in CRR is most damaging to the option price. Strict
strike-alignment in Tian (1999) eliminates this oscillation and produces
visibly lower errors than CRR at the same N. See generate_tian_alignment_tradeoff.py
for the complementary regime where the trade-off goes against Tian.

Output: figures/tian_alignment_wins.pdf
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


# Test case: 20% OTM short-dated high-vol call.
# In this regime, the strike sits in the sparse-node tail of the binomial
# tree, where CRR's integer-cutoff oscillation is most damaging. Strict
# strike-alignment (Tian) eliminates this error and visibly outperforms
# CRR at the cost of slightly degraded distributional matching.
S = 100.0
K = 120.0
T = 0.25
r = 0.05
SIGMA = 0.40

N_VALUES = np.concatenate([
    np.arange(10, 100, 1),
    np.arange(100, 500, 5),
    np.arange(500, 1001, 25),
])


def compute_errors() -> dict[str, np.ndarray]:
    """Compute |C^N - C^BS| for CRR and strict-aligned Tian."""
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


def main() -> None:
    set_style()

    print(f"Computing Tian-vs-CRR errors for {len(N_VALUES)} values of N")
    print(f"OTM call: S={S}, K={K} (K/S = {K/S:.2f})")
    errors = compute_errors()
    print("Done.")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Reference 1/N slope, anchored to CRR median around N=100
    n_ref = np.array([10, 1100])
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

    # CRR scatter showing oscillation
    ax.loglog(
        N_VALUES, errors["crr"],
        marker="o", linestyle="none", color=COLORS["crr"],
        markersize=2.5, alpha=0.7,
        label="CRR (1979)",
        zorder=3,
    )

    # Tian as smooth scatter (no oscillation)
    ax.loglog(
        N_VALUES, errors["tian"],
        marker="o", linestyle="none", color=COLORS["tian"],
        markersize=2.5, alpha=0.7,
        label="Tian (1999), strict strike-aligned",
        zorder=2,
    )

    ax.set_xlabel(r"Number of binomial steps, $N$")
    ax.set_ylabel(r"Absolute pricing error, $|C^{(N)}_0 - C^{\mathrm{BS}}|$")
    ax.set_xlim(8, 1200)
    ax.legend(loc="lower left")

    plt.tight_layout()

    output_path = Path(__file__).parent.parent / "figures" / "tian_alignment_wins.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()


if __name__ == "__main__":
    main()