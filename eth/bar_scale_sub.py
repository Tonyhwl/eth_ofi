# bar-scale sweep on the ten-second bucket panel: dollar-volume bars whose
# boundaries resolve at ten seconds instead of one minute, pushing the sweep
# past the minute floor. the in-sample threshold rule carries over from the
# minute panel because daily dollar volume does not depend on bucket size.
# also runs the plain ten-second interval regression (the cont-kukanov-stoikov
# sampling grid), on consecutive trade-active buckets.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from signal_impact import run_regression, oos_start
from bar_scale import build_vbars

panel_min = root / "results" / "eth_ofi_minute.parquet"
panel_10s = root / "results" / "eth_ofi_10s_oos.parquet"
out_path = root / "results" / "bar_scale_sub.csv"

TARGETS = [10, 50, 100, 200, 500, 1000, 2000, 5000]


def bucket_stats(vbars):
    """Median bar duration in seconds and share of single-bucket bars, OOS,
    roll boundaries excluded."""
    gaps = vbars.index.to_series().diff()[~vbars["roll"]]
    return (gaps.median().total_seconds(),
            float((gaps <= pd.Timedelta(seconds=10)).mean()))


def main():
    mp = pd.read_parquet(panel_min).sort_index()
    mp["dv"] = (mp["buy_dollar"] + mp["sell_dollar"]).fillna(0)
    daily_dv_is = mp.loc[mp.index < oos_start, "dv"].resample("1D").sum()
    is_median = float(daily_dv_is.median())
    print(f"IS median daily dollar volume (from minute panel): ${is_median:,.0f}")

    df = pd.read_parquet(panel_10s).sort_index()
    df["dv"] = (df["buy_dollar"] + df["sell_dollar"]).fillna(0)
    print(f"10s panel: {len(df):,} buckets, {df.index.min()} -> {df.index.max()}")

    rows = []
    for target in TARGETS:
        thr = is_median / target
        print(f"\n=== {target} bars/day  (threshold ${thr:,.0f}) ===", flush=True)
        vbars = build_vbars(df, thr)
        r = run_regression(vbars, f"OOS {target} bars/day (10s grid)")
        med_s, one_share = bucket_stats(vbars)
        rows.append({"bars_per_day": target, "threshold": round(thr, 2),
                     "n_oos": r["n"], "beta0": r["beta_ofi"], "t0": r["t_ofi"],
                     "beta1": r["beta_ofi_lag"], "t1": r["t_ofi_lag"],
                     "r2": r["r2"], "median_dur_sec": round(med_s, 1),
                     "one_bucket_share": round(one_share, 3)})

    # plain ten-second intervals: every trade-active bucket is an observation
    print("\n=== plain 10-second buckets ===", flush=True)
    tb = df[["ofi", "mid_close", "front_sym"]].copy()
    tb["roll"] = (tb["front_sym"] != tb["front_sym"].shift(1)).fillna(True)
    tb["d_mid"] = tb["mid_close"].diff()
    tb.loc[tb["roll"], "d_mid"] = np.nan
    r = run_regression(tb, "OOS plain 10s buckets")
    gaps = tb.index.to_series().diff()[~tb["roll"]]
    rows.append({"bars_per_day": np.nan, "threshold": np.nan,
                 "n_oos": r["n"], "beta0": r["beta_ofi"], "t0": r["t_ofi"],
                 "beta1": r["beta_ofi_lag"], "t1": r["t_ofi_lag"],
                 "r2": r["r2"], "median_dur_sec": round(gaps.median().total_seconds(), 1),
                 "one_bucket_share": 1.0})

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
