# figure 1: binned OOS mid-quote change against contemporaneous OFI

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from signal_impact import panel_path, oos_start

figs = root / "figs"
figs.mkdir(parents=True, exist_ok=True)

ink      = "#0f1d2e"
off      = "#5a6878"
acc1     = "#1f6feb"
acc2     = "#c6502c"
acc3     = "#0e8a6f"
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


def fig5_impact():
    df = pd.read_parquet(panel_path).sort_index()
    if "d_mid" not in df.columns:
        df["d_mid"] = df["mid_close"].diff()
    oos = df.loc[df.index >= oos_start]
    x = oos["ofi"].values.astype(float)
    y = oos["d_mid"].values.astype(float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    n_bins = 25
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    bin_index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    bin_x, bin_y, bin_err = [], [], []
    for b in range(len(edges) - 1):
        sel = bin_index == b
        if sel.sum() < 5:
            continue
        bin_x.append(float(x[sel].mean()))
        bin_y.append(float(y[sel].mean()))
        bin_err.append(float(y[sel].std(ddof=1) / math.sqrt(sel.sum())))
    bin_x, bin_y, bin_err = np.array(bin_x), np.array(bin_y), np.array(bin_err)
    slope, intercept = np.polyfit(x, y, 1)
    # power fit at the profiled exponent (impact_shape.py)
    gamma = 0.58
    xg = np.sign(x) * np.abs(x) ** gamma
    b_pow, a_pow = np.polyfit(xg, y, 1)
    xpad = (bin_x.max() - bin_x.min()) * 0.06
    xs = np.linspace(bin_x.min() - xpad, bin_x.max() + xpad, 100)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axhline(0, color=off, linewidth=0.7)
    ax.axvline(0, color=off, linewidth=0.7)
    ax.plot(xs, slope * xs + intercept, color=acc2, linewidth=1.7, zorder=3,
            label="OLS fit")
    ax.plot(xs, b_pow * np.sign(xs) * np.abs(xs) ** gamma + a_pow, color=acc3,
            linewidth=1.6, linestyle=(0, (5, 3)), zorder=3,
            label=r"power fit, $\gamma = 0.58$")
    ax.errorbar(bin_x, bin_y, yerr=bin_err, fmt="o", color=acc1, markersize=5.5,
                elinewidth=1.0, capsize=2.5, zorder=4,
                label="binned mean, 25 quantile bins")
    ax.set_xlabel("Bar OFI")
    ax.set_ylabel("Bar mid-quote change")
    ax.set_title("OOS price impact: bar mid-quote change against contemporaneous OFI",
                 loc="left")
    ax.legend(loc="upper left")
    ax.set_xlim(bin_x.min() - xpad, bin_x.max() + xpad)
    out = figs / "fig5_impact.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(root)}")


if __name__ == "__main__":
    fig5_impact()
