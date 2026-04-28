"""Shared matplotlib styling for figure-generation scripts.

Imports of this module configure matplotlib's rcParams to produce
publication-quality figures with serif fonts, restrained colors, and
tight layouts suitable for embedding in the LaTeX note.

Usage:
    from _style import set_style, COLORS, FIGSIZE_SINGLE, FIGSIZE_WIDE
    set_style()
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Color palette: muted, distinguishable, colorblind-safe
# Chosen from Tableau 10 with deliberate selection for distinguishability
COLORS = {
    "crr": "#1f77b4",       # blue
    "tian": "#ff7f0e",      # orange
    "lr": "#2ca02c",        # green
    "bs": "#d62728",        # red (reference line)
    "envelope": "#7f7f7f",  # gray (envelope curves)
    "neutral": "#000000",   # black (data points)
}

# Figure sizes (in inches); LaTeX page is ~6.5 inches wide for our geometry
FIGSIZE_SINGLE = (5.5, 3.8)   # standard single figure
FIGSIZE_WIDE = (6.5, 3.5)     # wider, shorter for log-log convergence
FIGSIZE_TALL = (5.5, 5.0)     # taller for stacked subplots


def set_style() -> None:
    """Configure matplotlib rcParams for publication-quality output."""
    mpl.rcParams.update({
        # Fonts
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # LaTeX-style math
        "mathtext.fontset": "cm",

        # Lines and markers
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Grid: light, behind data
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "grid.color": "#888888",
        "axes.axisbelow": True,

        # No background fill in figure or axes
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # Legend
        "legend.frameon": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.6,

        # Saving
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
    })