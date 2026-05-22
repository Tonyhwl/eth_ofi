# ETH OFI paper figures. locked config: K=5, H=3, |z|>=1, zlb=200,
# filters off, 10-min cap. outputs fig1_equity, fig2_yearly, fig3_trades,
# fig5_impact, fig6_cap under figs/.

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

root = Path(__file__).resolve().parent
figs = root / "figs"
figs.mkdir(parents=True, exist_ok=True)

ink      = "#0f1d2e"
off      = "#5a6878"
acc1     = "#1f6feb"
acc2     = "#c6502c"
acc3     = "#0e8a6f"
acc4     = "#a5572b"
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


def _oos_trades_and_pnl():
    sys.path.insert(0, str(root))
    from strategy import (
        simulate, vol_target_contracts, pnl_from_trades,
        build_event_calendar, oos_start, panel_path, contract_mult,
    )
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year_panel = len(df) / years
    n_contracts = vol_target_contracts(df, bars_per_year_panel)
    events = build_event_calendar()
    trades = simulate(df, n_contracts, use_event_filter=False, use_weekend_filter=False,
                      cap_hours=10/60, events=events)
    pnl_full = pnl_from_trades(trades, df, convention="realistic", cap_hours=10/60)
    pnl_oos = pnl_full.loc[df.index >= oos_start]
    df_oos = df.loc[df.index >= oos_start]
    years_oos = (df_oos.index.max() - df_oos.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year_oos = len(df_oos) / years_oos

    # per-trade pnl for the trade histogram, using the same realistic convention
    oos_trades = [t for t in trades if t.entry_time >= oos_start]
    trade_pnl = []
    for t in oos_trades:
        sub = pnl_full.loc[(pnl_full.index >= t.entry_time) & (pnl_full.index <= t.exit_time)]
        trade_pnl.append(float(sub.sum()))
    return pnl_oos, bars_per_year_oos, np.array(trade_pnl)


def fig1_equity(pnl, bars_per_year):
    cap = 1_000_000
    equity = cap + pnl.cumsum()
    cumulative = pnl.cumsum()
    dd_dollar = cumulative - cumulative.cummax()
    dd_pct = dd_dollar / cap * 100
    sharpe = float(pnl.mean() / pnl.std() * math.sqrt(bars_per_year))
    total_ret = float(pnl.sum() / cap * 100)
    years = (pnl.index.max() - pnl.index.min()).total_seconds() / (365.25 * 86400)
    cagr = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    max_dd = dd_pct.min()

    fig = plt.figure(figsize=(11, 6.2))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(equity.index, equity.values / 1e6, color=ink, linewidth=1.4)
    ax1.fill_between(equity.index, cap / 1e6, equity.values / 1e6,
                     where=equity.values >= cap, color=acc1, alpha=0.06, interpolate=True)
    ax1.axhline(cap / 1e6, color=off, linewidth=0.7, linestyle=(0, (4, 3)))
    ax1.set_ylabel("Equity ($M)")
    ax1.set_title("ETH OFI locked strategy. OOS equity, vol-targeted, $1M starting capital",
                  loc="left")
    ax1.text(0.012, 0.94,
             f"Sharpe {sharpe:+.2f}    CAGR {cagr:+.1f}%    "
             f"Total {total_ret:+.1f}%    MaxDD {max_dd:+.1f}%",
             transform=ax1.transAxes, fontsize=10, color=ink, weight="semibold",
             bbox=dict(facecolor="white", edgecolor=grid_col, boxstyle="round,pad=0.4"))

    ax2.fill_between(dd_pct.index, dd_pct.values, 0, color=acc2, alpha=0.40, linewidth=0)
    ax2.plot(dd_pct.index, dd_pct.values, color=acc2, linewidth=0.9)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(min(-1.0, dd_pct.min() * 1.05), 0.5)

    plt.setp(ax1.get_xticklabels(), visible=False)
    out = figs / "fig1_equity.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(root)}")


def fig2_yearly(pnl):
    cap = 1_000_000
    rows = []
    for yr, sub in pnl.groupby(pnl.index.year):
        if len(sub) < 50:
            continue
        calendar_years = (sub.index.max() - sub.index.min()).total_seconds() / (365.25 * 86400)
        bars_per_year_local = len(sub) / max(calendar_years, 1e-9)
        sharpe = float(sub.mean() / sub.std() * math.sqrt(bars_per_year_local))
        rows.append({"year": yr, "ret_pct": float(sub.sum() / cap * 100), "sharpe": sharpe})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    colors = [acc3 if v >= 0 else acc4 for v in df["ret_pct"]]
    bars = ax.bar(df["year"].astype(str), df["ret_pct"], color=colors,
                  edgecolor=ink, linewidth=0.6, width=0.55)
    ax.axhline(0, color=ink, linewidth=0.6)
    for b, v in zip(bars, df["ret_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.6 if v >= 0 else -1.4),
                f"{v:+.1f}%", ha="center", fontsize=9, color=ink)
    ax.set_ylabel("Annual return (%)")
    ax.set_title("OOS yearly returns", loc="left")
    pad = max(abs(df["ret_pct"]).max(), 5)
    ax.set_ylim(-pad - 4, pad + 6)

    ax = axes[1]
    bars = ax.bar(df["year"].astype(str), df["sharpe"],
                  color=[acc1 if v >= 0 else acc2 for v in df["sharpe"]],
                  edgecolor=ink, linewidth=0.6, width=0.55)
    ax.axhline(0, color=ink, linewidth=0.6)
    for b, v in zip(bars, df["sharpe"]):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.05 if v >= 0 else -0.15),
                f"{v:+.2f}", ha="center", fontsize=9, color=ink)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("OOS yearly Sharpe", loc="left")

    fig.suptitle("Per-year OOS performance, $1M starting capital",
                 fontsize=11, color=ink, weight="semibold", x=0.07, ha="left", y=0.99)
    out = figs / "fig2_yearly.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(root)}")


def fig3_trades(trade_pnl):
    pnl = trade_pnl
    n_trades = len(pnl)
    n_win = int((pnl > 0).sum())
    n_los = int((pnl < 0).sum())
    n_flat = n_trades - n_win - n_los
    win_rate = n_win / n_trades * 100 if n_trades else 0
    mean = float(pnl.mean())
    median = float(np.median(pnl))
    p01  = float(np.percentile(pnl, 1))
    p99  = float(np.percentile(pnl, 99))
    worst = float(pnl.min())
    best  = float(pnl.max())
    total = float(pnl.sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                             gridspec_kw={"wspace": 0.28, "width_ratios": [1.4, 1.0]})

    ax = axes[0]
    lo, hi = np.percentile(pnl, [0.5, 99.5])
    bins = np.linspace(lo, hi, 71)
    clipped = np.clip(pnl, lo, hi)
    ax.hist(clipped[clipped < 0], bins=bins, color=acc2, alpha=0.75,
            edgecolor=ink, linewidth=0.3, label=f"loss  (n = {n_los:,})")
    ax.hist(clipped[clipped > 0], bins=bins, color=acc3, alpha=0.75,
            edgecolor=ink, linewidth=0.3, label=f"win   (n = {n_win:,})")
    ax.axvline(0, color=ink, linewidth=0.7)
    ax.axvline(mean, color=acc1, linewidth=1.4, linestyle=(0, (4, 3)),
               label=f"mean = ${mean:,.0f}")
    ax.set_xlabel("Trade PnL ($), clipped at 0.5/99.5 pct")
    ax.set_ylabel("Trade count")
    ax.set_title(f"OOS trade-level PnL  (n = {n_trades:,}, win rate {win_rate:.1f}%)",
                 loc="left")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.axis("off")
    summary = [
        ("Trades",          f"{n_trades:,}"),
        ("Win rate",        f"{win_rate:.1f}%"),
        ("Mean / trade",    f"${mean:,.0f}"),
        ("Median / trade",  f"${median:,.0f}"),
        ("1st percentile",  f"${p01:,.0f}"),
        ("99th percentile", f"${p99:,.0f}"),
        ("Worst trade",     f"${worst:,.0f}"),
        ("Best trade",      f"${best:,.0f}"),
        ("Total OOS PnL",   f"${total:,.0f}"),
    ]
    y = 0.95
    for k, v in summary:
        ax.text(0.02, y, k, fontsize=10, color=off, transform=ax.transAxes)
        ax.text(0.55, y, v, fontsize=10, color=ink, weight="semibold",
                transform=ax.transAxes)
        y -= 0.10

    out = figs / "fig3_trades.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(root)}")


def fig5_impact():
    """Binned OOS scatter of bar mid-quote change against contemporaneous OFI."""
    sys.path.insert(0, str(root))
    from strategy import panel_path, oos_start
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
    xpad = (bin_x.max() - bin_x.min()) * 0.06
    xs = np.linspace(bin_x.min() - xpad, bin_x.max() + xpad, 100)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axhline(0, color=off, linewidth=0.7)
    ax.axvline(0, color=off, linewidth=0.7)
    ax.plot(xs, slope * xs + intercept, color=acc2, linewidth=1.7, zorder=3,
            label="OLS fit")
    ax.errorbar(bin_x, bin_y, yerr=bin_err, fmt="o", color=acc1, markersize=5.5,
                elinewidth=1.0, capsize=2.5, zorder=4,
                label="binned mean, 25 quantile bins")
    ax.set_xlabel("Bar OFI")
    ax.set_ylabel("Bar mid-quote change")
    ax.set_title("OOS signal impact: bar mid-quote change against contemporaneous OFI",
                 loc="left")
    ax.legend(loc="upper left")
    ax.set_xlim(bin_x.min() - xpad, bin_x.max() + xpad)
    out = figs / "fig5_impact.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(root)}")


def fig6_cap():
    """In-sample and OOS Sharpe across a fine clock-cap sweep, other parameters locked."""
    sweep = pd.read_csv(root / "results" / "cap_sweep.csv").sort_values("cap_min")
    cap_min = sweep["cap_min"].values.astype(float)
    sharpe_is = sweep["sr_is"].values
    sharpe_oos = sweep["sr_oos"].values
    top = max(float(sharpe_oos.max()), float(sharpe_is.max()))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axvline(10, color=acc2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.plot(cap_min, sharpe_is, "o-", color=off, linewidth=1.4, markersize=4,
            zorder=3, label="in-sample")
    ax.plot(cap_min, sharpe_oos, "o-", color=acc1, linewidth=1.7, markersize=4,
            zorder=4, label="out-of-sample")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([1, 3, 10, 30, 60, 180, 720, 1440])
    ax.set_xticklabels(["1", "3", "10", "30", "60", "180", "720", "1440"])
    ax.set_xlabel("Clock cap (minutes, log scale)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_ylim(0, top * 1.15)
    ax.text(10, top * 1.06, "locked", color=acc2, fontsize=9, ha="center")
    ax.set_title("Sharpe against clock-cap length, remaining parameters at the locked values",
                 loc="left")
    ax.legend(loc="upper right")
    out = figs / "fig6_cap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(root)}")


def main():
    print("Loading OOS PnL and trade list ...")
    pnl, bars_per_year, trade_pnl = _oos_trades_and_pnl()
    print(f"  {len(pnl):,} OOS bars, {len(trade_pnl):,} OOS trades")
    print("Generating figures under figs/ ...")
    fig1_equity(pnl, bars_per_year)
    fig2_yearly(pnl)
    fig3_trades(trade_pnl)
    fig5_impact()
    fig6_cap()
    print("Done.")


if __name__ == "__main__":
    main()
