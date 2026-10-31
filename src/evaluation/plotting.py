"""Headline and ablation plots (matplotlib), following the dataviz guidance.

One axis, 2 px lines, hairline grid, text in ink tokens (never the series colour), a legend
for two or more series, a direct label on the line the story is about, and a secondary
encoding (dash pattern) so lines survive greyscale. Categorical colours are slots 1 to 5 of
the reference palette, validated in both modes; colour follows the system, not its rank.
Two themes: "light" and "dark" (the dark one matches the charcoal slide theme).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", grid="#e1e0d9", axis="#c3c2b7", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781",
                  colours=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "dark": dict(surface="#1a1a19", grid="#2c2c2a", axis="#383835", ink="#f0efec", ink2="#c3c2b7",
                 muted="#898781",
                 colours=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]),
}

# system -> (categorical slot, dash pattern). Slot order is fixed; the dash is the secondary encoding.
SERIES = {
    "ORBIT-PRISM":         (0, "-"),
    "Capture order":       (1, (0, (6, 3))),
    "Discard cloudy":      (2, (0, (2, 2))),
    "Optical-only triage": (3, (0, (8, 3, 2, 3))),
    "Colour-index triage": (4, (0, (1, 2))),
    "No radar (ablation)": (1, (0, (6, 3))),
    "No dial (ablation)":  (2, (0, (2, 2))),
}


def recall_plot(curves: Dict[str, pd.DataFrame], out_path: Path, title: str, subtitle: str,
                scaled_budget_mb: Optional[float] = None, synthetic: bool = False,
                direct_label: Optional[str] = "ORBIT-PRISM", theme: str = "light") -> None:
    """Critical-tile recall against cumulative megabytes sent, one line per system."""
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.2, 5.9), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.grid(True, color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], labelsize=10, length=0)
    ax.set_xlabel("Cumulative data sent (MB)", color=t["ink2"], fontsize=11)
    ax.set_ylabel("Critical-tile recall (full tiles delivered)", color=t["ink2"], fontsize=11)
    ax.set_ylim(0, 1.09)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    xmax = max((c["cum_bytes"].max() for c in curves.values() if len(c)), default=1.0) / 1e6
    for name, curve in curves.items():
        slot, dash = SERIES.get(name, (0, "-"))
        colour = t["colours"][slot]
        x, y = curve["cum_bytes"] / 1e6, curve["recall"]
        ax.plot(x, y, color=colour, linewidth=2, linestyle=dash, drawstyle="steps-post",
                solid_capstyle="round", label=name)
        if name == direct_label and len(curve):
            ax.plot([x.iloc[-1]], [y.iloc[-1]], marker="o", markersize=8, color=colour,
                    markeredgecolor=t["surface"], markeredgewidth=2, linestyle="none")
            ax.annotate(f"{name}  {y.iloc[-1]:.0%}", (x.iloc[-1], y.iloc[-1]), xytext=(-4, 9),
                        textcoords="offset points", color=t["ink"], fontsize=10, fontweight="bold",
                        ha="right", va="bottom")
    if scaled_budget_mb:
        ax.axvline(scaled_budget_mb, color=t["muted"], linewidth=1, linestyle="-")
        ax.annotate(f"scaled pass\n{scaled_budget_mb:.2f} MB", (scaled_budget_mb, 0.04), xytext=(5, 0),
                    textcoords="offset points", color=t["ink2"], fontsize=9, va="bottom")
    ax.set_xlim(0, xmax * 1.16)  # room for the direct label
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=t["ink2"], handlelength=3.2)

    fig.text(0.075, 0.965, title, color=t["ink"], fontsize=14, fontweight="bold", ha="left")
    y = 0.925
    if synthetic:
        fig.text(0.075, y, "SYNTHETIC PLACEHOLDER SCORES: NOT RESULTS", color=t["ink"], fontsize=10,
                 fontweight="bold", ha="left")
        y -= 0.036
    fig.text(0.075, y, subtitle, color=t["ink2"], fontsize=9, ha="left", va="top", linespacing=1.4)
    fig.subplots_adjust(top=0.80 if synthetic else 0.83, left=0.075, right=0.97, bottom=0.11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=t["surface"])
    plt.close(fig)
