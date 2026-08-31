# bar-scale robustness for the price-impact regression: rebuild the dollar-volume
# bars at several target bar counts per day (threshold always calibrated on the
# in-sample window only) and re-run the identical regression at each scale. the
# ten-bars-per-day row must reproduce the paper's table exactly.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from signal_impact import run_regression, oos_start

panel_min = root / "results" / "eth_ofi_minute.parquet"
out_path = root / "results" / "bar_scale.csv"
quarterly_path = root / "results" / "bar_scale_quarterly.csv"

TARGETS = [1, 1.5, 2, 3, 5, 7, 10, 14, 20, 30, 50, 70, 100]


def build_vbars(df, bar_threshold):
    """Dollar-volume bars from the minute panel, same construction as robust.py
    but returned in memory so the live panel on disk is never touched."""
    bars = []
    for sym, sub in df.groupby("front_sym", sort=False):
        sub = sub.sort_index()
        cumulative_dv = sub["dv"].cumsum()
        bar_id = (cumulative_dv // bar_threshold).astype(int)
        agg = sub.assign(bar_id=bar_id).groupby("bar_id").agg(
            ts=("mid_close", lambda s: s.index[-1]),
            ofi=("ofi", "sum"),
            mid_close=("mid_close", "last"),
        )
        agg["front_sym"] = sym
        bars.append(agg)
    vbars = pd.concat(bars).reset_index(drop=True).set_index("ts").sort_index()
    vbars["roll"] = (vbars["front_sym"] != vbars["front_sym"].shift(1)).fillna(True)
    vbars["d_mid"] = vbars["mid_close"].diff()
    vbars.loc[vbars["roll"], "d_mid"] = np.nan
    return vbars


def median_duration_min(vbars):
    """Median gap between consecutive OOS bar closes, roll boundaries excluded."""
    oos = vbars[vbars.index >= oos_start]
    gaps = oos.index.to_series().diff()[~oos["roll"]]
    return gaps.median().total_seconds() / 60


def quarterly_r2(vbars, target):
    """Plain OLS R-squared of the two-regressor fit, per OOS calendar quarter."""
    d = vbars[vbars.index >= oos_start].copy()
    d["ofi_lag1"] = d["ofi"].shift(1)
    d = d.dropna(subset=["ofi", "d_mid", "ofi_lag1"])
    rows = []
    for q, sub in d.groupby(d.index.to_period("Q")):
        y = sub["d_mid"].values
        X = np.column_stack([np.ones(len(y)), sub["ofi"].values,
                             sub["ofi_lag1"].values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append({"bars_per_day": target, "quarter": str(q),
                     "n": len(y), "r2": r2})
    return rows


def main():
    df = pd.read_parquet(panel_min).sort_index()
    df["dv"] = (df["buy_dollar"] + df["sell_dollar"]).fillna(0)
    daily_dv_is = df.loc[df.index < oos_start, "dv"].resample("1D").sum()
    is_median = float(daily_dv_is.median())
    print(f"IS median daily dollar volume: ${is_median:,.0f}")

    rows, q_rows = [], []
    for target in TARGETS:
        thr = is_median / target
        print(f"\n=== {target} bars/day  (threshold ${thr:,.0f}) ===")
        vbars = build_vbars(df, thr)
        r = run_regression(vbars[vbars.index >= oos_start], f"OOS {target} bars/day")
        rows.append({"bars_per_day": target, "threshold": round(thr, 2),
                     "n_oos": r["n"], "beta0": r["beta_ofi"], "t0": r["t_ofi"],
                     "beta1": r["beta_ofi_lag"], "t1": r["t_ofi_lag"],
                     "r2": r["r2"],
                     "median_dur_min": round(median_duration_min(vbars), 1)})
        q_rows.extend(quarterly_r2(vbars, target))

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    pd.DataFrame(q_rows).to_csv(quarterly_path, index=False)
    print(f"\nwrote {out_path}")
    print(out.to_string(index=False))

    base = out[out["bars_per_day"] == 10].iloc[0]
    assert base["n_oos"] == 28711, f"baseline n mismatch: {base['n_oos']}"
    assert abs(base["beta0"] - 0.215) < 5e-4, f"baseline beta0 mismatch: {base['beta0']}"
    assert abs(base["r2"] - 0.30) < 5e-3, f"baseline r2 mismatch: {base['r2']}"
    print("\nbaseline check passed: 10 bars/day reproduces the paper's OOS row")


if __name__ == "__main__":
    main()
