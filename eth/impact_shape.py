# shape and state dependence of the impact relation, on the OOS bar panel.
# (a) concavity: fit d_mid = a + b sign(O)|O|^gamma, gamma profiled by OLS on a
#     grid, day-block bootstrap CI. gamma = 1 is the linear model; 0.5 is the
#     square-root impact law.
# (b) sign asymmetry: separate slopes for positive and negative OFI, NW errors,
#     Wald test of equality using the full NW covariance.
# (c) spread conditioning: beta0 by spread tercile (spread at the bar boundary,
#     from the minute panel), in basis-point returns so the price level drops out.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from signal_impact import panel_path, oos_start

minute_path = root / "results" / "eth_ofi_minute.parquet"
out_path = root / "results" / "impact_shape.csv"

GAMMA_GRID = np.round(np.arange(0.30, 1.31, 0.02), 2)
N_BOOT = 500


def load_oos():
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    df["ofi_lag1"] = df["ofi"].shift(1)
    df["mid_prev"] = df["mid_close"].shift(1)
    df = df[df.index >= oos_start]
    df = df.dropna(subset=["ofi", "d_mid", "ofi_lag1", "mid_prev"])
    df = df[np.isfinite(df["ofi"]) & np.isfinite(df["d_mid"])]
    df["d_mid_bp"] = df["d_mid"] / df["mid_prev"] * 1e4
    return df


def nw_cov(X, resid, lags=5):
    n, k = X.shape
    XX_inv = np.linalg.inv(X.T @ X)
    S = np.zeros((k, k))
    for j in range(0, lags + 1):
        w = 1.0 - j / (lags + 1)
        u = resid[:, None] * X
        gamma = (u[j:].T @ u[:n - j]) / n
        S += gamma if j == 0 else w * (gamma + gamma.T)
    return n * XX_inv @ S @ XX_inv


def profile_gamma(o, y):
    """OLS SSE profiled over the gamma grid; returns (gamma_hat, r2_at_hat)."""
    best = (None, np.inf, None)
    ss_tot = np.sum((y - y.mean()) ** 2)
    for g in GAMMA_GRID:
        x = np.sign(o) * np.abs(o) ** g
        X = np.column_stack([np.ones(len(y)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = np.sum((y - X @ beta) ** 2)
        if sse < best[1]:
            best = (g, sse, 1 - sse / ss_tot)
    return best[0], best[2]


def concavity(df):
    o, y = df["ofi"].values, df["d_mid"].values
    g_hat, r2_pow = profile_gamma(o, y)
    # linear univariate benchmark (the fitted line of figure 1)
    X = np.column_stack([np.ones(len(y)), o])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2_lin = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)

    days = np.array([ts.date() for ts in df.index])
    uniq = np.unique(days)
    day_idx = {d: np.where(days == d)[0] for d in uniq}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([day_idx[d] for d in pick])
        gb, _ = profile_gamma(o[idx], y[idx])
        boots.append(gb)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"[concavity] gamma_hat = {g_hat:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  "
          f"R2 power {r2_pow:.4f} vs linear {r2_lin:.4f}  (n = {len(y):,})")
    return {"analysis": "concavity", "gamma": g_hat, "ci_lo": lo, "ci_hi": hi,
            "r2_power": r2_pow, "r2_linear": r2_lin, "n": len(y)}


def asymmetry(df):
    y = df["d_mid"].values
    o = df["ofi"].values
    X = np.column_stack([np.ones(len(y)), np.maximum(o, 0), np.minimum(o, 0),
                         df["ofi_lag1"].values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    cov = nw_cov(X, y - X @ beta)
    se = np.sqrt(np.diag(cov))
    diff = beta[1] - beta[2]
    se_diff = np.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
    print(f"[asymmetry] beta_pos = {beta[1]:+.4f} (t {beta[1]/se[1]:+.1f})  "
          f"beta_neg = {beta[2]:+.4f} (t {beta[2]/se[2]:+.1f})  "
          f"diff = {diff:+.4f}  t(diff) = {diff/se_diff:+.2f}")
    return {"analysis": "asymmetry", "beta_pos": beta[1], "t_pos": beta[1] / se[1],
            "beta_neg": beta[2], "t_neg": beta[2] / se[2],
            "diff": diff, "t_diff": diff / se_diff, "n": len(y)}


def spread_terciles(df):
    mp = pd.read_parquet(minute_path)[["bid_close", "ask_close"]]
    sp = ((mp["ask_close"] - mp["bid_close"])
          / ((mp["ask_close"] + mp["bid_close"]) / 2) * 1e4)
    df = df.join(sp.rename("spread_bp"), how="inner")
    df = df[np.isfinite(df["spread_bp"]) & (df["spread_bp"] > 0)]
    df["tercile"] = pd.qcut(df["spread_bp"], 3, labels=["narrow", "middle", "wide"])

    rows = []
    for name, sub in df.groupby("tercile", observed=True):
        y = sub["d_mid_bp"].values
        X = np.column_stack([np.ones(len(y)), sub["ofi"].values,
                             sub["ofi_lag1"].values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        cov = nw_cov(X, y - X @ beta)
        r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"[spread {name:>6}] median spread {sub['spread_bp'].median():.2f} bp  "
              f"beta0 = {beta[1]:+.5f} bp/contract (t {beta[1]/np.sqrt(cov[1,1]):+.1f})  "
              f"R2 {r2:.3f}  n {len(y):,}")
        rows.append({"analysis": f"spread_{name}", "beta_pos": np.nan,
                     "median_spread_bp": sub["spread_bp"].median(),
                     "beta0_bp": beta[1], "t_beta0": beta[1] / np.sqrt(cov[1, 1]),
                     "r2": r2, "n": len(y)})

    # continuous check: interaction of OFI with de-medianed spread, bp returns
    y = df["d_mid_bp"].values
    s_c = df["spread_bp"].values - df["spread_bp"].median()
    X = np.column_stack([np.ones(len(y)), df["ofi"].values,
                         df["ofi"].values * s_c, df["ofi_lag1"].values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    cov = nw_cov(X, y - X @ beta)
    print(f"[spread interaction] delta = {beta[2]:+.5f} per bp of spread  "
          f"t = {beta[2]/np.sqrt(cov[2,2]):+.2f}")
    rows.append({"analysis": "spread_interaction", "beta0_bp": beta[2],
                 "t_beta0": beta[2] / np.sqrt(cov[2, 2]), "n": len(y)})
    return rows


def main():
    df = load_oos()
    print(f"OOS bars: {len(df):,}\n")
    rows = [concavity(df), asymmetry(df)]
    rows.extend(spread_terciles(df))
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
