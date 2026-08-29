# OLS of bar mid-quote change on contemporaneous + lag-1 OFI, with Newey-West
# HAC standard errors.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps


root       = Path(__file__).resolve().parent
panel_path = root / "results" / "eth_ofi_vbars_isfix.parquet"

oos_start = pd.Timestamp("2024-01-01", tz="US/Eastern")


def newey_west_se(X, resid, lags):
    n, k = X.shape
    XX_inv = np.linalg.inv(X.T @ X)
    S = np.zeros((k, k))
    for j in range(0, lags + 1):
        weight = 1.0 - j / (lags + 1)
        gamma = np.zeros((k, k))
        for t in range(j, n):
            u_t  = resid[t]
            u_tj = resid[t - j]
            gamma += u_t * u_tj * np.outer(X[t], X[t - j])
        gamma /= n
        if j == 0:
            S += gamma
        else:
            S += weight * (gamma + gamma.T)
    cov_beta = n * XX_inv @ S @ XX_inv
    return np.sqrt(np.diag(cov_beta))


def run_regression(df, label, lags=5):
    # lag built before dropping rows, so post-roll bars get the true previous bar's ofi
    df = df.copy()
    df["ofi_lag1"] = df["ofi"].shift(1)
    df = df.dropna(subset=["ofi", "d_mid", "ofi_lag1"])
    df = df[np.isfinite(df["ofi"]) & np.isfinite(df["d_mid"]) & np.isfinite(df["ofi_lag1"])]
    y = df["d_mid"].values
    ofi = df["ofi"].values
    ofi_lag1 = df["ofi_lag1"].values
    X = np.column_stack([np.ones(len(y)), ofi, ofi_lag1])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = newey_west_se(X, resid, lags=lags)
    t_stat = beta / se
    p_value = 2 * (1 - sps.norm.cdf(np.abs(t_stat)))
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n[{label}]  n = {len(y):,} bars")
    print(f"  intercept    {beta[0]:+.4e}  HAC SE {se[0]:.4e}  t = {t_stat[0]:+.2f}  p = {p_value[0]:.4f}")
    print(f"  OFI_t        {beta[1]:+.4e}  HAC SE {se[1]:.4e}  t = {t_stat[1]:+.2f}  p = {p_value[1]:.6f}")
    print(f"  OFI_{{t-1}}    {beta[2]:+.4e}  HAC SE {se[2]:.4e}  t = {t_stat[2]:+.2f}  p = {p_value[2]:.6f}")
    print(f"  R-squared    {r2:.4f}")
    return {"label": label, "n": len(y), "beta_ofi": beta[1], "t_ofi": t_stat[1],
            "p_ofi": p_value[1], "beta_ofi_lag": beta[2], "t_ofi_lag": t_stat[2],
            "p_ofi_lag": p_value[2], "r2": r2}


def main():
    df = pd.read_parquet(panel_path)
    if df.index.name != "ts":
        if "ts" in df.columns:
            df = df.set_index("ts")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    elif str(df.index.tz) != "US/Eastern":
        df.index = df.index.tz_convert("US/Eastern")

    print("=" * 72)
    print("ETH OFI signal-impact regression  (Cont-Kukanov-Stoikov 2014 style)")
    print("=" * 72)
    print(f"Panel: {len(df):,} dollar-volume bars from {df.index.min()} to {df.index.max()}")

    rows = []
    rows.append(run_regression(df, "Full sample"))
    # labels derive from the data so they cannot go stale when the panel grows
    is_df, oos_df = df[df.index < oos_start], df[df.index >= oos_start]
    span = lambda d: f"{d.index.min():%Y-%m} to {d.index.max():%Y-%m}"
    rows.append(run_regression(is_df,  f"In-sample {span(is_df)}"))
    rows.append(run_regression(oos_df, f"Out-of-sample {span(oos_df)}"))

    pd.DataFrame(rows).to_csv(root / "results" / "ofi_signal_impact.csv", index=False)
    print(f"\nwrote {root / 'results' / 'ofi_signal_impact.csv'}")


if __name__ == "__main__":
    main()
