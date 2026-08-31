# figure for the bar-scale robustness sweep: OOS r-squared and slope against
# median bar duration, threshold recalibrated in-sample at each scale. also a
# quarter-by-scale r-squared heatmap as an out-of-paper diagnostic.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
figs = root / "figs"
figs.mkdir(parents=True, exist_ok=True)

ink      = "#0f1d2e"
off      = "#5a6878"
acc1     = "#1f6feb"
acc2     = "#c6502c"
grid_col = "#dfe3e8"
bg       = "#fbfbfd"

mpl.rcParams.update({
    "figure.facecolor":  bg,
    "axes.facecolor":    bg,
    "axes.edgecolor":    ink,
    "axes.linewidth":    0.8,
    "axes.labelcolor":   ink,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.titleweight":  "semibold",
    "axes.titlecolor":   ink,
    "axes.grid":         True,
    "grid.color":        grid_col,
    "grid.linewidth":    0.6,
    "grid.alpha":        0.9,
    "xtick.color":       ink,
    "ytick.color":       ink,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.frameon":    False,
    "legend.fontsize":   9,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       160,
    "savefig.bbox":      "tight",
    "savefig.facecolor": bg,
})

DUR_TICKS = [2, 5, 10, 30, 60, 120, 240]


def scale_curve():
    d = pd.read_csv(root / "results" / "bar_scale.csv").sort_values("median_dur_min")
    base = d[d["bars_per_day"] == 10].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"wspace": 0.28})

    ax = axes[0]
    ax.plot(d["median_dur_min"], d["r2"], color=ink, linewidth=1.4,
            marker="o", markersize=4.5)
    ax.plot(base["median_dur_min"], base["r2"], marker="o", markersize=9,
            markerfacecolor="none", markeredgecolor=acc1, markeredgewidth=1.6,
            linestyle="none")
    ax.annotate("paper calibration\n(10 bars/day)",
                (base["median_dur_min"], base["r2"]),
                textcoords="offset points", xytext=(10, -28),
                fontsize=9, color=acc1)
    ax.set_xscale("log")
    ax.set_xticks(DUR_TICKS)
    ax.set_xticklabels([str(t) for t in DUR_TICKS])
    ax.set_ylim(0, 0.40)
    ax.set_xlabel("Median OOS bar duration (minutes)")
    ax.set_ylabel("OOS $R^2$")
    ax.set_title("Explanatory power across bar scales", loc="left")

    ax = axes[1]
    ax.plot(d["median_dur_min"], d["beta0"], color=ink, linewidth=1.4,
            marker="o", markersize=4.5)
    ax.plot(base["median_dur_min"], base["beta0"], marker="o", markersize=9,
            markerfacecolor="none", markeredgecolor=acc1, markeredgewidth=1.6,
            linestyle="none")
    ax.set_xscale("log")
    ax.set_xticks(DUR_TICKS)
    ax.set_xticklabels([str(t) for t in DUR_TICKS])
    ax.set_ylim(0, 0.30)
    ax.set_xlabel("Median OOS bar duration (minutes)")
    ax.set_ylabel(r"$\hat\beta_0$ (mid-quote $\Delta$ per contract)")
    ax.set_title("Contemporaneous slope across bar scales", loc="left")

    out = figs / "fig6_barscale.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(root)}")


def quarterly_heatmap():
    q = pd.read_csv(root / "results" / "bar_scale_quarterly.csv")
    piv = q.pivot(index="bars_per_day", columns="quarter", values="r2").sort_index()

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.grid(False)
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis",
                   vmin=0, vmax=piv.values.max())
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{b:g}" for b in piv.index])
    ax.set_ylabel("Bars per day")
    ax.set_title("OOS $R^2$ by calendar quarter and bar scale", loc="left")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.6 * piv.values.max() else ink)
    fig.colorbar(im, ax=ax, shrink=0.9, label="$R^2$")

    out = figs / "fig_barscale_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(root)}")


if __name__ == "__main__":
    scale_curve()
    quarterly_heatmap()
