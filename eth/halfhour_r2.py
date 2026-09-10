# CKS-matched statistic: per half-hour OLS of d_mid on ofi at 10s intervals
# mean r2 over windows, not a pooled fit

from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
panel_10s = root / "results" / "eth_ofi_10s_oos.parquet"
out_path = root / "results" / "halfhour_r2.csv"

MIN_OBS = [10, 30, 90]


def main():
    df = pd.read_parquet(panel_10s).sort_index()
    tb = df[["ofi", "mid_close", "front_sym"]].copy()
    tb["roll"] = (tb["front_sym"] != tb["front_sym"].shift(1)).fillna(True)
    tb["d_mid"] = tb["mid_close"].diff()
    tb.loc[tb["roll"], "d_mid"] = np.nan
    tb = tb.dropna(subset=["ofi", "d_mid"])
    tb = tb[np.isfinite(tb["ofi"]) & np.isfinite(tb["d_mid"])]

    window = tb.index.floor("30min")
    r2s = {}
    for (w, sub) in tb.groupby(window):
        n = len(sub)
        if n < min(MIN_OBS):
            continue
        y = sub["d_mid"].values
        X = np.column_stack([np.ones(n), sub["ofi"].values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot <= 0:
            continue
        r2s[w] = (n, 1 - np.sum(resid ** 2) / ss_tot)

    rows = []
    for m in MIN_OBS:
        vals = np.array([r2 for n, r2 in r2s.values() if n >= m])
        ns = np.array([n for n, r2 in r2s.values() if n >= m])
        rows.append({"min_obs": m, "n_windows": len(vals),
                     "mean_r2": vals.mean(),
                     "weighted_r2": float(np.average(vals, weights=ns)),
                     "median_r2": float(np.median(vals))})
        print(f"min {m:>3} intervals/window: {len(vals):,} windows  "
              f"mean R2 {vals.mean():.4f}  n-weighted {np.average(vals, weights=ns):.4f}")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
