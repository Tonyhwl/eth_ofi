# extend the bar-scale sweep past 100 bars/day, toward the panel's one-minute floor
# one_min_share tracks degeneration into plain minute bars

import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from signal_impact import run_regression, oos_start
from bar_scale import build_vbars, median_duration_min, quarterly_r2

panel_min = root / "results" / "eth_ofi_minute.parquet"
out_path = root / "results" / "bar_scale_ext.csv"
quarterly_path = root / "results" / "bar_scale_ext_quarterly.csv"

TARGETS = [150, 200, 300, 500, 1000]


def one_min_share(vbars):
    """share of one-minute OOS bars, rolls excluded"""
    oos = vbars[vbars.index >= oos_start]
    gaps = oos.index.to_series().diff()[~oos["roll"]]
    return float((gaps <= pd.Timedelta(minutes=1)).mean())


def main():
    df = pd.read_parquet(panel_min).sort_index()
    df["dv"] = (df["buy_dollar"] + df["sell_dollar"]).fillna(0)
    daily_dv_is = df.loc[df.index < oos_start, "dv"].resample("1D").sum()
    is_median = float(daily_dv_is.median())

    rows, q_rows = [], []
    for target in TARGETS:
        thr = is_median / target
        print(f"\n=== {target} bars/day  (threshold ${thr:,.0f}) ===", flush=True)
        vbars = build_vbars(df, thr)
        r = run_regression(vbars[vbars.index >= oos_start], f"OOS {target} bars/day")
        rows.append({"bars_per_day": target, "threshold": round(thr, 2),
                     "n_oos": r["n"], "beta0": r["beta_ofi"], "t0": r["t_ofi"],
                     "beta1": r["beta_ofi_lag"], "t1": r["t_ofi_lag"],
                     "r2": r["r2"],
                     "median_dur_min": round(median_duration_min(vbars), 1),
                     "one_min_share": round(one_min_share(vbars), 3)})
        q_rows.extend(quarterly_r2(vbars, target))

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    pd.DataFrame(q_rows).to_csv(quarterly_path, index=False)
    print(f"\nwrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
