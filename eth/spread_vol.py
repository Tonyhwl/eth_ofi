# is the spread effect on impact a depth story or volatility in disguise?
# wide spreads and large moves are both symptoms of volatile periods, so the
# OFI x spread interaction is re-estimated with a predetermined realized-vol
# interaction alongside, with the spread lagged one bar, and within volatility
# terciles. all in basis-point returns, NW-5 errors.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from impact_shape import load_oos, nw_cov, minute_path

out_path = root / "results" / "spread_vol.csv"
VOL_BARS = 20


def interaction(df, cols, label):
    """d_mid_bp on [1, ofi, interactions..., ofi_lag1]; returns coef/t per interaction."""
    y = df["d_mid_bp"].values
    X = np.column_stack([np.ones(len(y)), df["ofi"].values]
                        + [df["ofi"].values * df[c].values for c in cols]
                        + [df["ofi_lag1"].values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    cov = nw_cov(X, y - X @ beta)
    out = {}
    parts = []
    for k, c in enumerate(cols):
        t = beta[2 + k] / np.sqrt(cov[2 + k, 2 + k])
        out[c] = (beta[2 + k], t)
        parts.append(f"{c}: {beta[2 + k]:+.4f} (t {t:+.2f})")
    print(f"[{label}]  " + "   ".join(parts) + f"   n {len(y):,}")
    return out


def main():
    df = load_oos()
    mp = pd.read_parquet(minute_path)[["bid_close", "ask_close"]]
    sp = ((mp["ask_close"] - mp["bid_close"])
          / ((mp["ask_close"] + mp["bid_close"]) / 2) * 1e4)
    df = df.join(sp.rename("spread_bp"), how="inner")
    df = df[np.isfinite(df["spread_bp"]) & (df["spread_bp"] > 0)]

    # predetermined state: previous bar's closing spread, trailing realized vol
    df["s_lag"] = df["spread_bp"].shift(1)
    df["vol"] = df["d_mid_bp"].rolling(VOL_BARS).std().shift(1)
    df = df.dropna(subset=["s_lag", "vol"])
    for c in ["spread_bp", "s_lag", "vol"]:
        df[c + "_c"] = df[c] - df[c].median()

    print(f"bars: {len(df):,}   corr(spread, trailing vol) = "
          f"{df['spread_bp'].corr(df['vol']):.3f}   "
          f"corr(lagged spread, trailing vol) = {df['s_lag'].corr(df['vol']):.3f}\n")

    rows = []
    for cols, label in [
            (["spread_bp_c"], "baseline: contemporaneous spread"),
            (["s_lag_c"], "spread lagged one bar"),
            (["spread_bp_c", "vol_c"], "spread + volatility control"),
            (["s_lag_c", "vol_c"], "lagged spread + volatility control")]:
        res = interaction(df, cols, label)
        for c, (b, t) in res.items():
            rows.append({"spec": label, "term": c, "coef": b, "t": t, "n": len(df)})

    print()
    df["vol_tercile"] = pd.qcut(df["vol"], 3, labels=["low", "mid", "high"])
    for name, sub in df.groupby("vol_tercile", observed=True):
        res = interaction(sub, ["s_lag_c"], f"vol tercile {name}")
        b, t = res["s_lag_c"]
        rows.append({"spec": f"vol_tercile_{name}", "term": "s_lag_c",
                     "coef": b, "t": t, "n": len(sub)})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
