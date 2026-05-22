# clock-cap sensitivity sweep for the locked eth ofi strategy. descriptive
# sweep over clock-cap lengths, not a parameter search, so it does not enter
# the dsr.

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from strategy import (
    simulate, vol_target_contracts, build_event_calendar, pnl_from_trades,
    oos_start, panel_path,
)

cap_grid_min = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120, 180,
                240, 360, 480, 720, 1440]


def bars_per_year(index):
    """Bars per year for a datetime index."""
    years = (index.max() - index.min()).total_seconds() / (365.25 * 86400)
    return len(index) / years


def annualized_sharpe(pnl, bars_per_year_value):
    """Annualised Sharpe of a per-bar pnl array."""
    returns = pnl[np.isfinite(pnl)]
    if len(returns) < 10 or returns.std(ddof=1) == 0:
        return float("nan")
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(bars_per_year_value))


def main():
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    bars_per_year_panel = bars_per_year(df.index)
    n_contracts = vol_target_contracts(df, bars_per_year_panel)
    events = build_event_calendar()

    is_mask = df.index < oos_start
    oos_mask = df.index >= oos_start
    bars_per_year_is = bars_per_year(df.index[is_mask])
    bars_per_year_oos = bars_per_year(df.index[oos_mask])
    is_index = np.where(is_mask)[0]
    oos_index = np.where(oos_mask)[0]

    print(f"panel {len(df):,} bars   IS {int(is_mask.sum()):,}   "
          f"OOS {int(oos_mask.sum()):,}")
    rows = []
    for cap_min in cap_grid_min:
        cap_h = cap_min / 60.0
        trades = simulate(df, n_contracts, use_event_filter=False,
                          use_weekend_filter=False, cap_hours=cap_h, events=events)
        pnl = pnl_from_trades(trades, df, convention="realistic",
                              cap_hours=cap_h).values
        sharpe_is = annualized_sharpe(pnl[is_index], bars_per_year_is)
        sharpe_oos = annualized_sharpe(pnl[oos_index], bars_per_year_oos)
        n_trades_oos = sum(1 for t in trades if t.entry_time >= oos_start)
        rows.append({"cap_min": cap_min, "sr_is": round(sharpe_is, 4),
                     "sr_oos": round(sharpe_oos, 4), "n_trades_oos": n_trades_oos})
        print(f"  cap {cap_min:>5} min   IS {sharpe_is:+.3f}   OOS {sharpe_oos:+.3f}   "
              f"OOS trades {n_trades_oos}")

    out = root / "results" / "cap_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
