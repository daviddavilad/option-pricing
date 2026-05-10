"""Generate a two-panel convergence plot comparing CRR, Tian (1999), and LR.

Two panels at different strikes:
    - ATM (K = S = 100): Tian (1999) collapses to CRR at even N because
      the strike-aligning tilt parameter lambda vanishes. The orange and
      blue curves sit on top of each other, illustrating that strict
      strike-alignment buys nothing here.
    - In-the-money (K = 95): Tian's tilt is non-zero and the strike-aligned
      scheme appears as a smooth monotone curve at the upper envelope of
      the CRR scatter. The smoothness is what enables Richardson
      extrapolation; see Figure tian_extrapolation.pdf.

The two panels together tell the honest pedagogical point: Tian's
strike-alignment trades per-N scatter for monotonicity. It does not
improve on CRR's lucky-alignment N values.

Output: figures/convergence_three_schemes.pdf
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


# Common parameters
S = 100.0
T = 1.0
r = 0.05
SIGMA = 0.20

# Two strike regimes for the two panels
K_ATM = 100.0
K_ITM = 95.0

# N grid
N_VALUES = np.concatenate([
    np.arange(10, 100, 1),
    np.arange(100, 500, 5),
    np.arange(500, 1001, 25),
])


def compute_errors(K: float) -> dict[str, np.ndarray]:
    """Compute |C^N - C^BS| for each scheme across the N grid at strike K."""
    bs_price = black_scholes_call(S=S, K=K, T=T, r=r, sigma=SIGMA)

    errors = {"crr": [], "tian": [], "lr": []}

    for N in N_VALUES:
        N = int(N)

        params_crr = crr_parameters(T=T, N=N, r=r, sigma=SIGMA)
        price_crr = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params_crr,
            option_type="call", exercise_style="european",
        )
        errors["crr"].append(abs(price_crr - bs_price))

        params_tian = tian_1999_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=SIGMA,
        )
        price_tian = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params_tian,
            option_type="call", exercise_style="european",
        )
        errors["tian"].append(abs(price_tian - bs_price))

        params_lr = leisen_reimer_parameters(
            S=S, K=K, T=T, N=N, r=r, sigma=SIGMA,
        )
        price_lr = binomial_price(
            S=S, K=K, T=T, r=r, N=N, params=params_lr,
            option_type="call", exercise_style="european",
        )
        errors["lr"].append(abs(price_lr - bs_price))

    return {k: np.array(v) for k, v in errors.items()}


def plot_panel(ax, errors: dict[str, np.ndarray], title: str,
               show_ylabel: bool = True, show_legend: bool = False) -> None:
    """Render one panel."""
    n_ref = np.array([10, 1100])

    # Reference slopes anchored to median CRR/LR error around N=100
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

    ax.loglog(
        N_VALUES, errors["crr"],
        marker="o", linestyle="none", color=COLORS["crr"],
        markersize=2.5, alpha=0.7,
        label="CRR (1979)",
        zorder=3,
    )

    ax.loglog(
        N_VALUES, errors["tian"],
        color=COLORS["tian"], linewidth=1.4, linestyle="-",
        label="Tian (1999), strike-aligned",
        zorder=4,
    )

    ax.loglog(
        N_VALUES, errors["lr"],
        color=COLORS["lr"], linewidth=1.4,
        label="Leisen--Reimer (1996)",
        zorder=5,
    )

    ax.set_xlabel(r"Number of binomial steps, $N$")
    if show_ylabel:
        ax.set_ylabel(r"Absolute pricing error, $|C^{(N)}_0 - C^{\mathrm{BS}}|$")
    ax.set_xlim(8, 1200)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="lower left", fontsize=8)


def main() -> None:
    set_style()

    print(f"Computing errors at K = {K_ATM} (ATM)...")
    errors_atm = compute_errors(K_ATM)
    print(f"Computing errors at K = {K_ITM} (ITM)...")
    errors_itm = compute_errors(K_ITM)
    print("Done.")

    # Wider figure to accommodate two panels side by side.
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    plot_panel(
        ax_left, errors_atm,
        title=fr"ATM: $S = K = {K_ATM:.0f}$",
        show_ylabel=True, show_legend=True,
    )
    plot_panel(
        ax_right, errors_itm,
        title=fr"ITM: $S = {S:.0f}$, $K = {K_ITM:.0f}$",
        show_ylabel=False, show_legend=False,
    )

    # Match y-limits across panels for fair visual comparison.
    y_lo = min(ax_left.get_ylim()[0], ax_right.get_ylim()[0])
    y_hi = max(ax_left.get_ylim()[1], ax_right.get_ylim()[1])
    ax_left.set_ylim(y_lo, y_hi)
    ax_right.set_ylim(y_lo, y_hi)

    plt.tight_layout()

    output_path = (
        Path(__file__).parent.parent / "figures"
        / "convergence_three_schemes.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"), dpi=150)
    print(f"Saved {output_path} (+ .png)")
    plt.close()

if __name__ == "__main__":
    main()