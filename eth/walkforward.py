# walk-forward for the ETH OFI strategy: re-select parameters on a rolling in-sample
# window, test on the next window, roll forward. the stitched OOS curve exposes overfit.

import sys
import os
import time
import itertools
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import strategy as strat
from strategy import (simulate, vol_target_contracts, pnl_from_trades, panel_path,
                      oos_start, capital)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "shared"))
from metrics import sharpe as ann_sharpe

# parameter grid searched on each train window (zscore lookback fixed at the locked value)
SIGNAL_BARS_GRID = [3, 5, 10]
ENTRY_GRID       = [1.0, 1.5, 2.0]
MAX_HOLD_GRID    = [1, 3, 5]
CAP_GRID         = [5 / 60, 10 / 60, 15 / 60]
ZLB              = 200
LOCKED           = (5, 1.0, 3, 10 / 60)   # README locked config, for the equity overlay

# test windows tile contiguously (step == test); train window is held fixed
WINDOW_SCHEMES = [(1, 1, "monthly"), (3, 3, "quarterly")]

# worker-process globals, set once per worker so the bar panel is not re-pickled per task
_DF = None
_NC = None


def _init_worker(df, n_contracts):
    global _DF, _NC
    _DF = df
    _NC = n_contracts


def _run_combo(params):
    """Full-panel per-bar pnl for one parameter combo (runs in a worker process)."""
    sb, et, mh, cap = params
    strat.signal_bars     = sb
    strat.entry_threshold = et
    strat.max_hold_bars   = mh
    strat.zscore_lookback = ZLB
    trades = simulate(_DF, _NC, use_event_filter=False, use_weekend_filter=False,
                      cap_hours=cap, events=None)
    return params, pnl_from_trades(trades, _DF, "realistic", cap_hours=cap)


def build_combo_pnl(df, n_contracts, combos, workers):
    """Per-combo full-panel pnl, in parallel across cores (or serially if workers <= 1)."""
    if workers <= 1:
        _init_worker(df, n_contracts)
        return {p: _run_combo(p)[1] for p in combos}
    out = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(df, n_contracts)) as ex:
        for params, pnl in ex.map(_run_combo, combos):
            out[params] = pnl
    return out


def make_folds(index, train, test, step):
    """Rolling (train_start, train_end, test_end) windows with full test windows only."""
    folds = []
    train_start = index.min()
    last = index.max()
    while True:
        train_end = train_start + pd.DateOffset(months=train)
        test_end  = train_end + pd.DateOffset(months=test)
        if test_end > last:
            break
        folds.append((train_start, train_end, test_end))
        train_start = train_start + pd.DateOffset(months=step)
    return folds


def window_sharpe(pnl, start, end):
    return ann_sharpe(pnl[(pnl.index >= start) & (pnl.index < end)])


def evaluate(combo_pnl, index, oos_start_ts, train, test, step, label):
    """Run one walk-forward scheme and print a one-line summary."""
    folds = make_folds(index, train, test, step)
    rows = []
    for train_start, train_end, test_end in folds:
        best, best_sr = None, -np.inf
        for combo, pnl in combo_pnl.items():
            sr = window_sharpe(pnl, train_start, train_end)
            if np.isfinite(sr) and sr > best_sr:
                best_sr, best = sr, combo
        if best is None:
            continue
        chosen_pnl = combo_pnl[best]
        test_pnl = chosen_pnl[(chosen_pnl.index >= train_end) & (chosen_pnl.index < test_end)]
        rows.append({"config": best, "is_sr": best_sr,
                     "oos_sr": ann_sharpe(test_pnl), "pnl": test_pnl})
    stitched = pd.concat([r["pnl"] for r in rows]).sort_index()
    sharpe_full = ann_sharpe(stitched)
    sharpe_oos  = ann_sharpe(stitched[stitched.index >= oos_start_ts])
    mean_is  = np.nanmean([r["is_sr"] for r in rows])
    mean_oos = np.nanmean([r["oos_sr"] for r in rows])
    n_configs = len(set(r["config"] for r in rows))
    print(f"  [{label:<9}] train {train}m/test {test}m  folds={len(rows):>3}  "
          f"WF(full)={sharpe_full:+.2f}  WF(2024+)={sharpe_oos:+.2f}  "
          f"IS->OOS={mean_is:+.2f}->{mean_oos:+.2f}  configs={n_configs}")
    return stitched, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--train", type=int, default=18, help="train window in months")
    args = ap.parse_args()

    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    n_contracts = vol_target_contracts(df)
    combos = list(itertools.product(SIGNAL_BARS_GRID, ENTRY_GRID, MAX_HOLD_GRID, CAP_GRID))

    print(f"ETH walk-forward: {len(combos)} combos, {args.workers} workers, "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    t0 = time.time()
    combo_pnl = build_combo_pnl(df, n_contracts, combos, args.workers)
    print(f"  built per-combo pnl in {time.time() - t0:.1f}s")

    # single-split baseline (scheme-independent): best combo on pre-OOS, applied to OOS
    best, best_sr = None, -np.inf
    for combo, pnl in combo_pnl.items():
        sr = window_sharpe(pnl, df.index.min(), oos_start)
        if np.isfinite(sr) and sr > best_sr:
            best_sr, best = sr, combo
    split = ann_sharpe(combo_pnl[best][combo_pnl[best].index >= oos_start])
    print(f"  single-split OOS Sharpe (headline-style): {split:+.3f}  [config {best}]\n")

    results = {}
    for test, step, label in WINDOW_SCHEMES:
        results[label] = evaluate(combo_pnl, df.index, oos_start, args.train, test, step, label)

    from wfplot import save_equity_figure, save_folds_figure
    quarterly_pnl, quarterly_folds = results["quarterly"]
    wf_oos = quarterly_pnl[quarterly_pnl.index >= oos_start]
    locked_oos = combo_pnl[LOCKED][combo_pnl[LOCKED].index >= oos_start]
    locked_oos = locked_oos[locked_oos.index <= wf_oos.index.max()]   # match the walk-forward window
    figs = ROOT / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    save_equity_figure("ETH", wf_oos, locked_oos, capital, figs / "fig_walkforward.png")
    save_folds_figure("ETH", [r["is_sr"] for r in quarterly_folds],
                      [r["oos_sr"] for r in quarterly_folds], figs / "fig_walkforward_folds.png")
    print(f"\n  wrote {figs / 'fig_walkforward.png'} (+ fig_walkforward_folds.png)")


if __name__ == "__main__":
    main()
