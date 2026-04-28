"""Multi-regime CRR oscillation: signed pricing error vs N.

A 2x2 panel showing $\\varepsilon_N = C_0^{(N)} - C^{\\mathrm{BS}}$
on linear axes for four parameter regimes. The point of using signed
error rather than absolute error (and linear rather than log axes) is
that the period-2 alternation is visible only on this view: log-log
abs-error plots collapse the sign and obscure the alternation that
makes CRR convergence non-monotone.

Layout convention:
  Rows correspond to (volatility, maturity) regime; columns to strike
  position. Reading across a row isolates the strike effect; reading
  down a column isolates the vol/maturity effect.

The figure visualises the mechanism argued in section 6.2 of the
note: oscillation amplitude is governed by the strike's position
relative to the terminal binomial nodes. The ATM cells show clean
period-2 alternation; the OTM cells show period-2 alternation
modulated by slower beats whose period depends on the strike's
geometric placement among the terminal nodes.

Output: figures/crr_oscillation_regimes.pdf
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
from option_pricing.parameterizations import crr_parameters
from option_pricing.pricers import binomial_price


# Four regimes. Ordering is (row, col) in the 2x2 panel.
# Row 0: long-dated, low-vol (T=1, sigma=0.20). Row 1: short-dated,
# high-vol (T=0.25, sigma=0.40). Col 0: ATM. Col 1: OTM.
REGIMES = [
    {"S": 100, "K": 100, "T": 1.0,  "r": 0.05, "sigma": 0.20,
     "label": r"ATM ($K=100$), $T=1$, $\sigma=0.20$"},
    {"S": 100, "K": 110, "T": 1.0,  "r": 0.05, "sigma": 0.20,
     "label": r"10% OTM ($K=110$), $T=1$, $\sigma=0.20$"},
    {"S": 100, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.40,
     "label": r"ATM ($K=100$), $T=0.25$, $\sigma=0.40$"},
    {"S": 100, "K": 120, "T": 0.25, "r": 0.05, "sigma": 0.40,
     "label": r"20% OTM ($K=120$), $T=0.25$, $\sigma=0.40$"},
]

# Linear-axes plot benefits from a moderate N range. Going beyond N~150
# compresses the period-2 alternation against the y-axis at each
# individual N; going below ~10 includes pre-asymptotic noise.
N_VALUES = np.arange(10, 151)


def signed_errors(S: float, K: float, T: float,
                  r: float, sigma: float) -> np.ndarray:
    """Signed CRR pricing error C_0^(N) - C^BS across N_VALUES."""
    bs = black_scholes_call(S=S, K=K, T=T, r=r, sigma=sigma)
    errs = np.empty_like(N_VALUES, dtype=float)
    for i, N in enumerate(N_VALUES):
        params = crr_parameters(T=T, N=int(N), r=r, sigma=sigma)
        price = binomial_price(
            S=S, K=K, T=T, r=r, N=int(N), params=params,
            option_type="call", exercise_style="european",
        )
        errs[i] = price - bs
    return errs


def main() -> None:
    set_style()

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5),
                             sharex=True)

    for i, regime in enumerate(REGIMES):
        ax = axes[i // 2, i % 2]
        errs = signed_errors(regime["S"], regime["K"], regime["T"],
                             regime["r"], regime["sigma"])

        # Diagnostic: amplitude and sign-change count, useful to
        # cross-check that ATM cells have perfect period-2 (sign
        # changes == N-1) while OTM cells have fewer (beat structure).
        amp = np.max(np.abs(errs))
        signs = np.sign(errs)
        flips = int(np.sum(signs[:-1] * signs[1:] < 0))
        print(f"{regime['label']}")
        print(f"  max |eps_N| = {amp:.4e}, "
              f"sign changes = {flips} of {len(errs)-1}")

        ax.axhline(0, color=COLORS["envelope"], linewidth=0.8,
                   linestyle="--", alpha=0.7, zorder=0)
        ax.plot(N_VALUES, errs, color=COLORS["crr"],
                linewidth=0.9, marker="o", markersize=2.5,
                alpha=0.85, zorder=2)

        ax.set_title(regime["label"], fontsize=10)
        ax.set_xlim(8, 152)

        # Symmetric y-limits about zero so the alternation is visually
        # centered. Pad by 10% so markers don't kiss the axis edge.
        y_pad = 1.10 * amp
        ax.set_ylim(-y_pad, y_pad)

        if i // 2 == 1:
            ax.set_xlabel(r"Number of binomial steps, $N$")
        if i % 2 == 0:
            ax.set_ylabel(
                r"$\varepsilon_N = C_0^{(N)} - C^{\mathrm{BS}}$"
            )

    plt.tight_layout()

    output_path = (
        Path(__file__).parent.parent
        / "figures" / "crr_oscillation_regimes.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\nSaved {output_path}")
    plt.close()


if __name__ == "__main__":
    main()