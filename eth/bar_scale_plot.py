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

BPD_TICKS = [1, 3, 10, 30, 100, 300, 1000]
FLOOR_BPD = 200   # beyond this most bars complete within a single minute


def load_sweep():
    cols = ["bars_per_day", "beta0", "r2", "median_dur_min"]
    d = pd.concat([pd.read_csv(root / "results" / "bar_scale.csv")[cols],
                   pd.read_csv(root / "results" / "bar_scale_ext.csv")[cols]])
    return d.sort_values("bars_per_day")


def scale_curve():
    d = load_sweep()
    base = d[d["bars_per_day"] == 10].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"wspace": 0.28})

    for ax, col, ylim, ylab, title in [
            (axes[0], "r2", 0.40, "OOS $R^2$",
             "Explanatory power across bar scales"),
            (axes[1], "beta0", 0.30,
             r"$\hat\beta_0$ (mid-quote $\Delta$ per contract)",
             "Contemporaneous slope across bar scales")]:
        ax.axvspan(FLOOR_BPD, 1300, color=grid_col, alpha=0.55, zorder=0)
        ax.text(450, ylim * 0.06, "one-minute\ngrid binds", fontsize=8,
                color=off, ha="center")
        ax.plot(d["bars_per_day"], d[col], color=ink, linewidth=1.4,
                marker="o", markersize=4.5)
        ax.plot(base["bars_per_day"], base[col], marker="o", markersize=9,
                markerfacecolor="none", markeredgecolor=acc1, markeredgewidth=1.6,
                linestyle="none")
        ax.set_xscale("log")
        ax.set_xticks(BPD_TICKS)
        ax.set_xticklabels([str(t) for t in BPD_TICKS])
        ax.set_xlim(0.8, 1300)
        ax.set_ylim(0, ylim)
        ax.set_xlabel("Target bars per day")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")

    axes[0].annotate("paper calibration\n(10 bars/day)",
                     (base["bars_per_day"], base["r2"]),
                     textcoords="offset points", xytext=(10, -28),
                     fontsize=9, color=acc1)

    out = figs / "fig6_barscale.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(root)}")


def quarterly_heatmap():
    q = pd.concat([pd.read_csv(root / "results" / "bar_scale_quarterly.csv"),
                   pd.read_csv(root / "results" / "bar_scale_ext_quarterly.csv")])
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
