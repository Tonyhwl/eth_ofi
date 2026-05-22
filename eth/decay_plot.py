# plot signal decay profile (0.5 min to 24 h).
# placebo column is loaded for the robustness check but not plotted here.

from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

ink, off = "#0f1d2e", "#5a6878"
acc1, acc2, acc3 = "#1f6feb", "#c6502c", "#0e8a6f"
grid_col, bg = "#dfe3e8", "#fbfbfd"
mpl.rcParams.update({
    "figure.facecolor": bg, "axes.facecolor": bg, "axes.edgecolor": ink,
    "axes.linewidth": 0.8, "axes.labelcolor": ink, "axes.labelsize": 10,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": ink,
    "axes.grid": True, "grid.color": grid_col, "grid.linewidth": 0.6,
    "grid.alpha": 0.9, "xtick.color": ink, "ytick.color": ink,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
    "legend.fontsize": 9, "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": bg,
})


def main():
    df = pd.read_csv(ROOT / "results" / "decay_profile.csv")
    horizons = df["horizon_min"].values
    ofi = df["ofi_mean_bp"].values
    std_error = df["ofi_se_bp"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(horizons, ofi - 1.96 * std_error, ofi + 1.96 * std_error,
                    color=acc1, alpha=0.15, linewidth=0, label="95% CI")
    ax.plot(horizons, ofi, color=ink, linewidth=1.6, marker="o", markersize=5,
            markerfacecolor=acc1, markeredgecolor=ink, markeredgewidth=0.6,
            label="Mean signed cumulative return")
    ax.axhline(0, color=ink, linewidth=0.6)
    ax.axvline(10, color=acc2, linewidth=1.0, linestyle=(0, (4, 3)),
               label="10-min clock cap (risk management)")
    ax.axvline(47, color=acc3, linewidth=0.9, linestyle=(0, (2, 3)),
               label="$\\approx$ bar duration (47 min)")

    ax.set_xlabel("Minutes since signal entry")
    ax.set_ylabel("Cumulative mid-quote change, signed by direction (bp)")
    ax.set_title("OOS signal decay from OFI entry (n $\\approx$ 4,500 trades)",
                 loc="left")
    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 5, 10, 30, 60, 240, 720, 1440]
    labels = ["0.5", "1", "2", "5", "10", "30", "60", "4h", "12h", "24h"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left")

    out = ROOT / "figs" / "fig4_decay.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
