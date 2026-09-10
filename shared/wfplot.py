# walk-forward figures: OOS equity + drawdown, per-fold IS vs OOS sharpe scatter

import math

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

_INK  = "#0f1d2e"
_OFF  = "#5a6878"
_ACC1 = "#1f6feb"
_ACC2 = "#c6502c"
_GRID = "#dfe3e8"
_BG   = "#fbfbfd"


def _apply_style():
    mpl.rcParams.update({
        "figure.facecolor": _BG, "axes.facecolor": _BG, "axes.edgecolor": _INK,
        "axes.linewidth": 0.8, "axes.labelcolor": _INK, "axes.labelsize": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": _INK,
        "axes.grid": True, "grid.color": _GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "xtick.color": _INK, "ytick.color": _INK, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": _BG,
    })


def _metrics(pnl, capital):
    years = (pnl.index.max() - pnl.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(pnl) / years
    sharpe = float(pnl.mean() / pnl.std() * math.sqrt(bars_per_year))
    total = float(pnl.sum() / capital * 100)
    cagr = ((1 + total / 100) ** (1 / years) - 1) * 100
    cum = pnl.cumsum()
    max_dd = float(((cum - cum.cummax()) / capital * 100).min())
    return sharpe, cagr, total, max_dd


def save_equity_figure(label, wf_pnl, locked_pnl, capital, out_path):
    """OOS equity and drawdown, walk-forward vs locked config"""
    _apply_style()
    equity_wf     = (capital + wf_pnl.cumsum()) / 1e6
    equity_locked = (capital + locked_pnl.cumsum()) / 1e6
    cum = wf_pnl.cumsum()
    dd_pct = (cum - cum.cummax()) / capital * 100
    sharpe, cagr, total, max_dd = _metrics(wf_pnl, capital)

    fig = plt.figure(figsize=(11, 6.2))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(equity_locked.index, equity_locked.values, color=_OFF, linewidth=1.0,
             linestyle=(0, (4, 3)), label="locked config")
    ax1.plot(equity_wf.index, equity_wf.values, color=_INK, linewidth=1.5,
             label="walk-forward")
    ax1.fill_between(equity_wf.index, capital / 1e6, equity_wf.values,
                     where=equity_wf.values >= capital / 1e6, color=_ACC1, alpha=0.06,
                     interpolate=True)
    ax1.axhline(capital / 1e6, color=_OFF, linewidth=0.7, linestyle=(0, (4, 3)))
    y_low, y_high = ax1.get_ylim()
    ax1.set_ylim(y_low, y_high + (y_high - y_low) * 0.12)   # headroom for the stats box
    ax1.set_ylabel("Equity (\\$M)")
    ax1.set_title(f"{label} walk-forward OOS equity", loc="left")
    ax1.text(0.012, 0.97,
             f"WF Sharpe {sharpe:+.2f}    CAGR {cagr:+.1f}%    "
             f"Total {total:+.1f}%    MaxDD {max_dd:+.1f}%",
             transform=ax1.transAxes, va="top", fontsize=10, color=_INK, weight="semibold",
             bbox=dict(facecolor="white", edgecolor=_GRID, boxstyle="round,pad=0.4"))
    ax1.legend(loc="lower right")

    ax2.fill_between(dd_pct.index, dd_pct.values, 0, color=_ACC2, alpha=0.40, linewidth=0)
    ax2.plot(dd_pct.index, dd_pct.values, color=_ACC2, linewidth=0.9)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(min(-1.0, dd_pct.min() * 1.05), 0.5)

    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.savefig(out_path)
    plt.close(fig)


def save_folds_figure(label, fold_is, fold_oos, out_path):
    """per-fold IS vs OOS sharpe against y=x"""
    _apply_style()
    is_sharpe = np.array(fold_is, dtype=float)
    oos_sharpe = np.array(fold_oos, dtype=float)
    finite = np.isfinite(is_sharpe) & np.isfinite(oos_sharpe)
    is_sharpe, oos_sharpe = is_sharpe[finite], oos_sharpe[finite]
    axis_lo = min(is_sharpe.min(), oos_sharpe.min(), 0.0) - 0.5
    axis_hi = max(is_sharpe.max(), oos_sharpe.max()) + 0.5

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot([axis_lo, axis_hi], [axis_lo, axis_hi], color=_INK, linewidth=0.8,
            linestyle="--", alpha=0.7, label="OOS = IS")
    ax.axhline(0, color=_INK, linewidth=0.6, alpha=0.4)
    ax.scatter(is_sharpe, oos_sharpe, s=46, color=_ACC1, edgecolor=_INK, linewidth=0.6, zorder=3)
    ax.set_xlim(axis_lo, axis_hi)
    ax.set_ylim(axis_lo, axis_hi)
    ax.set_xlabel("in-sample Sharpe (per fold)")
    ax.set_ylabel("out-of-sample Sharpe (per fold)")
    ax.set_title(f"{label}: per-fold IS vs OOS Sharpe", loc="left")
    ax.legend(loc="upper left")

    fig.savefig(out_path)
    plt.close(fig)
