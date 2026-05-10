"""Generate convergence plot showing Tian (1999) smoothness vs CRR oscillation.

Reproduces the qualitative content of Tian (1999) Figure 2 (page 825):
the strike-aligned flexible binomial converges essentially monotonically
to the Black-Scholes price, while the CRR scheme oscillates above and
below it. This is the entire point of strike-alignment -- the smoothness
of convergence, NOT a lower per-N error.

Setup matches Tian (1999) Figure 2: in-the-money call,
    S = 100, K = 95, T = 0.5, r = 0.06, sigma = 0.20.

Output: figures/tian_smoothness.pdf
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
    tian_1999_parameters,
)
from option_pricing.pricers import binomial_price


# Tian (1999) Figure 2 setup
S = 100.0
K = 95.0
T = 0.5
r = 0.06
SIGMA = 0.20

# Match Tian's range: N from 10 to 100, every step
N_VALUES = np.arange(10, 101, 1)


def compute_errors() -> dict[str, np.ndarray]:
    """Compute pricing error (signed, so we can show oscillation)."""
    bs = black_scholes_call(S=S, K=K, T=T, r=r, sigma=SIGMA)

    crr_errors = []
    tian_errors = []

    for N in N_VALUES:
        N = int(N)

        params_crr = crr_parameters(T=T, N=N, r=r, sigma=SIGMA)
        price_crr = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params_crr,
            option_type="call", exercise_style="european",
        )
        crr_errors.append(price_crr - bs)

        params_tian = tian_1999_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=SIGMA,
        )
        price_tian = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params_tian,
            option_type="call", exercise_style="european",
        )
        tian_errors.append(price_tian - bs)

    return {
        "crr": np.array(crr_errors),
        "tian": np.array(tian_errors),
    }


def main() -> None:
    set_style()

    print(f"Computing errors for N in [{N_VALUES.min()}, {N_VALUES.max()}]...")
    errors = compute_errors()
    print("Done.")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Zero reference line
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5, zorder=1)

    # CRR with oscillation visible -- markers + thin connecting line
    ax.plot(
        N_VALUES, errors["crr"],
        marker="o", linestyle="-",
        color=COLORS["crr"], markersize=2.5, linewidth=0.8, alpha=0.85,
        label="CRR (1979)",
        zorder=3,
    )

    # Tian (1999) strike-aligned: smooth curve
    ax.plot(
        N_VALUES, errors["tian"],
        color=COLORS["tian"], linewidth=1.6,
        label="Tian (1999), strike-aligned",
        zorder=4,
    )

    ax.set_xlabel(r"Number of binomial steps, $N$")
    ax.set_ylabel(r"Pricing error, $C^{(N)}_0 - C^{\mathrm{BS}}$")
    ax.set_xlim(N_VALUES.min() - 2, N_VALUES.max() + 2)
    ax.legend(loc="upper right")

    plt.tight_layout()

    output_path = (
        Path(__file__).parent.parent / "figures"
        / "tian_smoothness.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"), dpi=150)
    print(f"Saved {output_path} (+ .png)")
    plt.close()


if __name__ == "__main__":
    main()