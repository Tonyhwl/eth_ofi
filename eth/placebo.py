# direction-shuffled placebo for the locked strategy. holds the locked
# trade list fixed and permutes the long and short labels across 200
# seeds, so each seed keeps every entry time, exit time and size but
# reassigns the trade direction. this checks that the edge comes from
# the ofi direction call and not from the trade timing.

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from strategy import (
    simulate, vol_target_contracts, build_event_calendar, pnl_from_trades,
    compute_metrics, oos_start, panel_path, capital,
)

n_seeds   = 200
cap_hours = 10 / 60


def main():
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(df) / years

    n_contracts = vol_target_contracts(df, bars_per_year)
    events = build_event_calendar()
    trades = simulate(df, n_contracts, use_event_filter=False,
                      use_weekend_filter=False, cap_hours=cap_hours, events=events)

    oos_trades = [t for t in trades if t.entry_time >= oos_start]
    oos_index  = df.index[df.index >= oos_start]
    directions = np.array([t.direction for t in oos_trades])
    n_long  = int((directions > 0).sum())
    n_short = int((directions < 0).sum())

    reference = compute_metrics(
        pnl_from_trades(oos_trades, df, "realistic", cap_hours=cap_hours).loc[oos_index])
    ref_sharpe = reference["sharpe"]
    print(f"locked OOS trades: {len(oos_trades)}  (long {n_long}, short {n_short})")
    print(f"reference Sharpe, true directions: {ref_sharpe:+.3f}")

    rows = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        permuted = directions[rng.permutation(len(oos_trades))]
        shuffled = [replace(t, direction=int(permuted[i]))
                    for i, t in enumerate(oos_trades)]
        pnl_oos = pnl_from_trades(shuffled, df, "realistic",
                                  cap_hours=cap_hours).loc[oos_index]
        metrics_result = compute_metrics(pnl_oos)
        if metrics_result is None:
            continue
        rows.append({
            "seed":        seed,
            "sharpe":      metrics_result["sharpe"],
            "cagr_pct":    metrics_result["cagr_pct"],
            "mdd_pct":     metrics_result["max_dd_pct"],
            "ann_vol_pct": metrics_result["ann_vol_pct"],
            "total_pct":   round(float(pnl_oos.sum()) / capital * 100, 3),
        })

    results = pd.DataFrame(rows)
    out_path = root / "results" / "random_baseline_shuffled.csv"
    results.to_csv(out_path, index=False)

    sharpe = results["sharpe"]
    print(f"\ndirection-shuffled placebo over {len(results)} seeds")
    print(f"  mean Sharpe   {sharpe.mean():+.3f}")
    print(f"  p05 / p95     {sharpe.quantile(0.05):+.3f} / {sharpe.quantile(0.95):+.3f}")
    print(f"  max Sharpe    {sharpe.max():+.3f}")
    print(f"  mean CAGR     {results['cagr_pct'].mean():+.2f}%")
    print(f"  mean max DD   {results['mdd_pct'].mean():+.2f}%")
    print(f"  mean total    {results['total_pct'].mean():+.2f}%")
    print(f"  seeds reaching the locked Sharpe: {int((sharpe >= ref_sharpe).sum())} / {len(results)}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
