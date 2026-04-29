"""Generate convergence plot showing Richardson extrapolation gains for Tian (1999).

Reproduces the qualitative content of Tian (1999) Figure 3 (page 832):
applying Richardson extrapolation (eq. 17) to the smoothly-convergent
strike-aligned flexible binomial collapses the pricing error far below
either the un-extrapolated CRR or un-extrapolated Tian price. This is
the practical payoff of strike-alignment: the smoothness is what enables
the extrapolation, and the extrapolation is what produces dramatic
accuracy gains. CRR cannot use this technique because its oscillation
breaks the constant-error-ratio assumption.

Setup matches Tian (1999) Figure 3: in-the-money call,
    S = 100, K = 95, T = 0.5, r = 0.06, sigma = 0.20.

Output: figures/tian_extrapolation.pdf
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
from option_pricing.pricers import binomial_price, richardson_extrapolation


# Tian (1999) Figure 3 setup
S = 100.0
K = 95.0
T = 0.5
r = 0.06
SIGMA = 0.20

# Use even N so Richardson extrapolation always has a paired N/2 available.
N_VALUES = np.arange(10, 101, 2)


def _price(scheme_params, N: int) -> float:
    return binomial_price(
        S=S, K=K, T=T, r=r, N=N, params=scheme_params,
        option_type="call", exercise_style="european",
    )


def compute_errors() -> dict[str, np.ndarray]:
    """Compute |C^N - C^BS| for CRR, Tian (1999), and Richardson-extrapolated Tian."""
    bs = black_scholes_call(S=S, K=K, T=T, r=r, sigma=SIGMA)

    errors = {"crr": [], "tian": [], "tian_extrap": []}

    for N in N_VALUES:
        N = int(N)

        params_crr = crr_parameters(T=T, N=N, r=r, sigma=SIGMA)
        price_crr = _price(params_crr, N)
        errors["crr"].append(abs(price_crr - bs))

        params_tian_N = tian_1999_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=SIGMA,
        )
        price_tian_N = _price(params_tian_N, N)
        errors["tian"].append(abs(price_tian_N - bs))

        # Richardson extrapolation needs paired N/2 and N. We use the
        # standard rho = 2 from Tian (1999) eq. (17).
        N_half = N // 2
        params_tian_half = tian_1999_parameters(
            S=S, K=K, T=T, N=N_half, r=r, sigma=SIGMA,
        )
        price_tian_half = _price(params_tian_half, N_half)
        price_extrap = richardson_extrapolation(
            price_N=price_tian_half, price_2N=price_tian_N, rho=2.0,
        )
        errors["tian_extrap"].append(abs(price_extrap - bs))

    return {k: np.array(v) for k, v in errors.items()}


def main() -> None:
    set_style()

    print(f"Computing errors for {len(N_VALUES)} values of N...")
    errors = compute_errors()
    print("Done.")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # CRR: scattered points (oscillatory)
    ax.semilogy(
        N_VALUES, errors["crr"],
        marker="o", linestyle="-",
        color=COLORS["crr"], markersize=2.5, linewidth=0.8, alpha=0.85,
        label="CRR (1979)",
        zorder=3,
    )

    # Tian (1999) un-extrapolated: smooth monotone curve
    ax.semilogy(
        N_VALUES, errors["tian"],
        color=COLORS["tian"], linewidth=1.6,
        label="Tian (1999), strike-aligned",
        zorder=4,
    )

    # Tian (1999) with Richardson extrapolation: this is the real win
    ax.semilogy(
        N_VALUES, errors["tian_extrap"],
        color=COLORS["lr"], linewidth=1.6, linestyle="--",
        label="Tian (1999) with Richardson extrapolation",
        zorder=5,
    )

    ax.set_xlabel(r"Number of binomial steps, $N$")
    ax.set_ylabel(r"Absolute pricing error, $|C^{(N)}_0 - C^{\mathrm{BS}}|$")
    ax.set_xlim(N_VALUES.min() - 2, N_VALUES.max() + 2)
    ax.legend(loc="upper right")

    plt.tight_layout()

    output_path = (
        Path(__file__).parent.parent / "figures" / "tian_extrapolation.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()


if __name__ == "__main__":
    main()