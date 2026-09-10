# minute-resolution signal-impact regression: per-minute mid change on
# contemporaneous and lag-1 per-minute OFI. the bar regression's finer counterpart.

from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
panel_path = root / "results" / "eth_ofi_minute.parquet"
oos_start = pd.Timestamp("2024-01-01", tz="US/Eastern")
hac_lags = 30          # minutes; wider than the bar regression's 5 given minute autocorrelation


def newey_west_se(X, resid, lags):
    """vectorised NW HAC SEs, same estimator as signal_impact.py"""
    n = len(resid)
    XX_inv = np.linalg.inv(X.T @ X)
    moment = X * resid[:, None]                       # row t is residual_t * x_t
    S = (moment.T @ moment) / n                        # lag-0 term
    for j in range(1, lags + 1):
        weight = 1.0 - j / (lags + 1)
        gamma = (moment[j:].T @ moment[:n - j]) / n
        S += weight * (gamma + gamma.T)
    cov_beta = n * XX_inv @ S @ XX_inv
    return np.sqrt(np.diag(cov_beta))


def run_regression(df, label):
    df = df.dropna(subset=["ofi", "d_mid", "ofi_lag1"])
    y = df["d_mid"].values
    ofi = df["ofi"].values
    ofi_lag1 = df["ofi_lag1"].values
    X = np.column_stack([np.ones(len(y)), ofi, ofi_lag1])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = newey_west_se(X, resid, hac_lags)
    t_stat = beta / se
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"\n[{label}]  n = {len(y):,} minutes")
    print(f"  OFI_t        {beta[1]:+.4e}  HAC SE {se[1]:.4e}  t = {t_stat[1]:+.2f}")
    print(f"  OFI_(t-1)    {beta[2]:+.4e}  HAC SE {se[2]:.4e}  t = {t_stat[2]:+.2f}")
    print(f"  R-squared    {r2:.4f}")
    return {"label": label, "n": int(len(y)), "beta_ofi": float(beta[1]),
            "t_ofi": float(t_stat[1]), "beta_lag": float(beta[2]),
            "t_lag": float(t_stat[2]), "r2": float(r2)}


def main():
    df = pd.read_parquet(panel_path).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")

    # d_mid and lag-1 OFI dropped across rolls and non-consecutive minutes
    roll = df["front_sym"] != df["front_sym"].shift(1)
    gap_min = df.index.to_series().diff().dt.total_seconds() / 60.0
    boundary = roll | (gap_min > 1.5)
    df["d_mid"] = df["mid_close"].diff()
    df["ofi_lag1"] = df["ofi"].shift(1)
    df.loc[boundary, ["d_mid", "ofi_lag1"]] = np.nan

    print("=" * 72)
    print("ETH OFI signal-impact regression at minute resolution")
    print("=" * 72)
    print(f"Panel: {len(df):,} minutes, {df.index.min().date()} to {df.index.max().date()}")
    print(f"usable consecutive-minute changes: {int(df['d_mid'].notna().sum()):,}  (HAC lags = {hac_lags})")

    rows = []
    rows.append(run_regression(df, "Full sample"))
    rows.append(run_regression(df[df.index < oos_start], "In-sample 2021-02 to 2023-12"))
    rows.append(run_regression(df[df.index >= oos_start], "Out-of-sample 2024-01 onward"))
    out = root / "results" / "ofi_signal_impact_minute.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
