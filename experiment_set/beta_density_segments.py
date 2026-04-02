"""
beta_density_segments.py
------------------------
Discretise 5 Beta density curves into N customer segments.

Steps
-----
1. Load the User sheet from the Excel workbook.
2. Sort users by w_cost (cost sensitivity) ascending.
3. Assign each user a position = (rank + 0.5) / N on [0, 1].
   This is the midpoint of their equal-width partition cell.
4. For each Beta(alpha, beta) config, evaluate the PDF at every position.
5. Normalise → each bar height becomes the segment's fraction of total demand.
6. Plot: continuous curve in the background, one bar per user on top.

Output: beta_density_segments.png  (saved next to this script)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parent
ARYA_ROOT      = EXPERIMENT_DIR.parent
SERVER_ROOT    = ARYA_ROOT / "arya_fullstack_app" / "server"
XLSX_PATH      = ARYA_ROOT / "Arya_Phones_Supplier_Selection.xlsx"

sys.path.insert(0, str(SERVER_ROOT))

from app.beta_density import BetaDensity                          # noqa: E402
from app.optimization_controller import load_supplier_user_tables  # noqa: E402

# ---------------------------------------------------------------------------
# Load & sort users
# ---------------------------------------------------------------------------

_, users_df = load_supplier_user_tables(XLSX_PATH)
users_sorted = (
    users_df[["user_id", "w_cost"]]
    .sort_values("w_cost")
    .reset_index(drop=True)
)

N = len(users_sorted)
bar_width = 1.0 / N

# Midpoint position of each user's equal-width cell on [0, 1].
positions = np.array([(i + 0.5) / N for i in range(N)])
user_labels = users_sorted["user_id"].astype(str).tolist()

# ---------------------------------------------------------------------------
# Beta configs  (alpha, beta, color)
# ---------------------------------------------------------------------------

CONFIGS: list[tuple[float, float, str, str]] = [
    (0.5, 0.5,  "#e63946", "U-shaped  (mass at extremes)"),
    (1.0, 1.0,  "#457b9d", "Uniform"),
    (2.0, 5.0,  "#2a9d8f", "Left-skewed  (low cost-sens. dominant)"),
    (5.0, 2.0,  "#e9c46a", "Right-skewed  (high cost-sens. dominant)"),
    (3.0, 3.0,  "#6a4c93", "Symmetric bell  (middle dominant)"),
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(CONFIGS),
    ncols=1,
    figsize=(14, 3.8 * len(CONFIGS)),
    sharex=True,
)
fig.suptitle(
    f"Beta Density  ->  {N} Customer Segments  (sorted by w_cost)",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

x_curve = np.linspace(0.0, 1.0, 500)

for ax, (alpha, beta_param, color, interp) in zip(axes, CONFIGS):
    bd = BetaDensity(alpha=alpha, beta=beta_param)

    # Continuous curve
    y_curve = bd.pdf(x_curve)
    ax.plot(x_curve, y_curve, color=color, linewidth=1.8, alpha=0.5, zorder=1)

    # Bar heights = raw density at each midpoint
    raw_heights = np.array([bd.density_at(p) for p in positions])

    # Normalised fractions: weight_i = pdf(pos_i) * bar_width / sum(pdf * bar_width)
    # Since bar_width is constant it cancels; just normalise the heights.
    fractions = raw_heights / raw_heights.sum()

    # Draw bars
    bars = ax.bar(
        positions,
        raw_heights,
        width=bar_width * 0.88,
        color=color,
        alpha=0.75,
        zorder=2,
        edgecolor="white",
        linewidth=0.5,
    )

    # Fraction label on each bar — clip to axis so infinite-PDF edges don't warn
    y_cap = np.nanpercentile(raw_heights[np.isfinite(raw_heights)], 95) if np.any(np.isfinite(raw_heights)) else 1.0
    for bar, frac, h in zip(bars, fractions, raw_heights):
        if frac >= 0.01 and np.isfinite(h) and h <= y_cap * 1.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(h, y_cap) + y_cap * 0.04,
                f"{frac:.1%}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#333333",
            )

    ax.set_title(
        f"Beta(alpha={alpha}, beta={beta_param})  --  {interp}",
        fontsize=10,
        pad=5,
    )
    ax.set_ylabel("density", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

# X-axis: user IDs
axes[-1].set_xticks(positions)
axes[-1].set_xticklabels(
    [f"u{uid}" for uid in user_labels],
    rotation=45,
    ha="right",
    fontsize=7.5,
)
axes[-1].set_xlabel("users sorted by w_cost  (low -> high)", fontsize=9)

fig.tight_layout()

out_path = EXPERIMENT_DIR / "beta_density_segments.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Print fraction table for the 5 configs
# ---------------------------------------------------------------------------

print(f"\n{'user_id':>8}  {'w_cost':>7}", end="")
for alpha, beta_param, _, _ in CONFIGS:
    print(f"  B({alpha},{beta_param})".ljust(14), end="")
print()

for i, row in users_sorted.iterrows():
    print(f"{str(row['user_id']):>8}  {row['w_cost']:>7.4f}", end="")
    for alpha, beta_param, _, _ in CONFIGS:
        bd = BetaDensity(alpha=alpha, beta=beta_param)
        raw = np.array([bd.density_at(p) for p in positions])
        frac = raw / raw.sum()
        print(f"  {frac[i]:>11.2%}   ", end="")
    print()
