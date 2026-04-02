"""
beta_density_plots.py
---------------------
Plot 5 Beta(alpha, beta) density curves to explore how different parameter
sets shape the segment-size distribution.

Output: beta_density_plots.png  (saved next to this script)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Allow importing BetaDensity from the server package without installing it.
SERVER_ROOT = Path(__file__).resolve().parents[1] / "arya_fullstack_app" / "server"
sys.path.insert(0, str(SERVER_ROOT))

from app.beta_density import BetaDensity  # noqa: E402

# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------

CONFIGS: list[tuple[float, float, str, str]] = [
    # (alpha, beta, label, interpretation)
    (0.5,  0.5,  "α=0.5, β=0.5",  "U-shaped — mass at extremes (polarised segments)"),
    (1.0,  1.0,  "α=1.0, β=1.0",  "Uniform — all positions equally likely"),
    (2.0,  5.0,  "α=2.0, β=5.0",  "Left-skewed — most segments lean low-preference"),
    (5.0,  2.0,  "α=5.0, β=2.0",  "Right-skewed — most segments lean high-preference"),
    (3.0,  3.0,  "α=3.0, β=3.0",  "Symmetric bell — mass concentrated in the middle"),
]

COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#6a4c93"]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(CONFIGS),
    ncols=1,
    figsize=(9, 3.2 * len(CONFIGS)),
    sharex=True,
)
fig.suptitle(
    "Beta Distribution — Segment Density Shapes",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)

x = np.linspace(0.0, 1.0, 500)

for ax, (alpha, beta, label, interp), color in zip(axes, CONFIGS, COLORS):
    bd = BetaDensity(alpha=alpha, beta=beta)
    _, y = bd.sample_grid(n_points=500)

    ax.plot(x, y, color=color, linewidth=2.2, label=label)
    ax.fill_between(x, y, alpha=0.15, color=color)

    ax.set_title(f"{label}  —  {interp}", fontsize=10.5, pad=6)
    ax.set_ylabel("density", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

axes[-1].set_xlabel("segment position  [0, 1]", fontsize=10)

fig.tight_layout()

out_path = Path(__file__).with_name("beta_density_plots.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
