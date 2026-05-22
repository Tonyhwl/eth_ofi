# eth ofi robustness suite: is-only bar calibration, dsr, hansen spa, block bootstrap

import sys, math, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

panel_min  = root / "results" / "eth_ofi_minute.parquet"
vbars_fix  = root / "results" / "eth_ofi_vbars_isfix.parquet"
out_path   = root / "results" / "eth_ofi_robust.json"
oos_start  = pd.Timestamp("2024-01-01", tz="US/Eastern")

signal_bars      = 5
entry_threshold  = 1.0
max_hold_bars    = 3
direction_sign   = +1
zscore_lookback  = 200
contract_mult    = 50.0
tick_size        = 0.05
capital          = 1_000_000.0
vol_target       = 0.15
vol_lookback     = 200


# 1. rebuild dollar-volume bars with is-only threshold

def rebuild_vbars_is_only(target_bars_per_day=10):
    df = pd.read_parquet(panel_min).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    df["dv"] = (df["buy_dollar"] + df["sell_dollar"]).fillna(0)

    is_mask = df.index < oos_start
    daily_dv_is = df.loc[is_mask, "dv"].resample("1D").sum()
    bar_threshold = float(daily_dv_is.median() / target_bars_per_day)
    print(f"  IS-only bar threshold: ${bar_threshold:,.0f} / bar  "
          f"(IS median dv = {daily_dv_is.median():,.0f})")

    bars = []
    for sym, sub in df.groupby("front_sym", sort=False):
        sub = sub.sort_index()
        cumulative_dv = sub["dv"].cumsum()
        bar_id = (cumulative_dv // bar_threshold).astype(int)
        agg = sub.assign(bar_id=bar_id).groupby("bar_id").agg(
            ts=("mid_close", lambda s: s.index[-1]),
            ofi=("ofi", "sum"),
            mid_open=("mid_open", "first"),
            mid_close=("mid_close", "last"),
            buy_dollar=("buy_dollar", "sum"),
            sell_dollar=("sell_dollar", "sum"),
            n_min=("ofi", "count"),
        )
        agg["front_sym"] = sym
        bars.append(agg)
    vbars = pd.concat(bars).reset_index(drop=True).set_index("ts").sort_index()
    vbars["roll"] = (vbars["front_sym"] != vbars["front_sym"].shift(1)).fillna(True)
    vbars["d_mid"] = vbars["mid_close"].diff()
    vbars.loc[vbars["roll"], "d_mid"] = np.nan
    vbars["ret"] = vbars["mid_close"].pct_change()
    vbars.loc[vbars["roll"], "ret"] = np.nan
    vbars.to_parquet(vbars_fix)
    print(f"  rebuilt vbars: {len(vbars):,} bars  -> {vbars_fix.name}")
    return vbars


# 2a. deflated sharpe ratio (bailey and lopez de prado 2014)

euler_mascheroni = 0.5772156649


def expected_max_sharpe(n_trials, var_sr_trials):
    if n_trials <= 1 or var_sr_trials <= 0:
        return 0.0
    gamma = euler_mascheroni
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr_trials) * ((1 - gamma) * z1 + gamma * z2)


def deflated_sharpe(returns, sr_trials_annual, bars_per_year):
    returns_clean = np.asarray(returns, dtype=float)
    returns_clean = returns_clean[np.isfinite(returns_clean)]
    n_obs = len(returns_clean)
    if n_obs < 50 or returns_clean.std(ddof=1) == 0:
        return None
    sharpe_obs_per_bar = returns_clean.mean() / returns_clean.std(ddof=1)
    returns_active = returns_clean[returns_clean != 0]
    skew = float(stats.skew(returns_active, bias=False))
    kurtosis = float(stats.kurtosis(returns_active, fisher=False, bias=False))

    sharpe_trials_per_bar = np.asarray(sr_trials_annual) / math.sqrt(bars_per_year)
    var_sharpe_per_bar = float(np.var(sharpe_trials_per_bar, ddof=1))
    sharpe0_per_bar = expected_max_sharpe(len(sharpe_trials_per_bar), var_sharpe_per_bar)

    denominator = math.sqrt(max(1.0 - skew * sharpe_obs_per_bar + ((kurtosis - 1.0) / 4.0) * sharpe_obs_per_bar ** 2, 1e-12))
    z_dsr = (sharpe_obs_per_bar - sharpe0_per_bar) * math.sqrt(n_obs - 1) / denominator
    dsr_prob = float(stats.norm.cdf(z_dsr))
    return {
        "sr_obs_per_bar":   round(sharpe_obs_per_bar, 5),
        "sr0_per_bar":      round(sharpe0_per_bar, 5),
        "sr_obs_annual":    round(sharpe_obs_per_bar * math.sqrt(bars_per_year), 4),
        "sr0_annual":       round(sharpe0_per_bar    * math.sqrt(bars_per_year), 4),
        "skew":             round(skew, 3),
        "kurt_non_excess":  round(kurtosis, 3),
        "n_trials":         int(len(sharpe_trials_per_bar)),
        "var_sr_trials_pb": round(var_sharpe_per_bar, 6),
        "T":                int(n_obs),
        "z_dsr":            round(z_dsr, 3),
        "dsr_prob":         round(dsr_prob, 4),
    }


# 2b. hansen (2005) spa-consistent test

def _stationary_bootstrap_indices(n, mean_block, rng):
    prob = 1.0 / mean_block
    index = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        length = max(1, int(rng.geometric(prob)))
        end = min(i + length, n)
        for j in range(end - i):
            index[i + j] = (start + j) % n
        i = end
    return index


def hansen_spa_consistent(pnl_matrix, n_boot=2000, mean_block=5.0, seed=7):
    pnl = np.asarray(pnl_matrix, dtype=float)
    n_obs, n_strategies = pnl.shape
    rng = np.random.default_rng(seed)

    pnl_mean = pnl.mean(axis=0)
    standard_error = np.zeros(n_strategies)
    for _ in range(200):
        index = _stationary_bootstrap_indices(n_obs, mean_block, rng)
        standard_error += (pnl[index].mean(axis=0) - pnl_mean) ** 2
    standard_error = np.sqrt(standard_error / 200)
    standard_error = np.where(standard_error > 1e-15, standard_error, 1e-15)

    t_per_strategy = pnl_mean / standard_error
    t_stat = float(np.maximum(t_per_strategy, 0).max())

    a_n = math.sqrt(2.0 * math.log(math.log(n_obs))) * standard_error
    g_consistent = np.where(pnl_mean >= -a_n, pnl_mean, 0.0)

    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        index = _stationary_bootstrap_indices(n_obs, mean_block, rng)
        pnl_boot_mean = pnl[index].mean(axis=0)
        z = float(np.maximum((pnl_boot_mean - g_consistent) / standard_error, 0).max())
        boot_stats[b] = z

    p_spa = float((boot_stats >= t_stat).mean())
    p_spa = max(p_spa, 1.0 / n_boot)
    best_index = int(np.argmax(t_per_strategy))
    return {
        "n_strategies":      int(n_strategies),
        "n_obs":             int(n_obs),
        "t_stat_consistent": round(t_stat, 4),
        "p_spa_consistent":  round(p_spa, 5),
        "best_strategy_idx": best_index,
        "best_strategy_t":   round(float(t_per_strategy[best_index]), 3),
        "n_boot":            int(n_boot),
        "mean_block":        float(mean_block),
        "boot_stats":        boot_stats.tolist(),
    }


# 3. circular block bootstrap on locked oos pnl

def circular_block_bootstrap(pnl, bars_per_year, block_size, n_boot=5000, seed=11):
    returns = np.asarray(pnl, dtype=float)
    returns = returns[np.isfinite(returns)]
    n_obs = len(returns)
    if n_obs < 100 or returns.std(ddof=1) == 0:
        return None
    rng = np.random.default_rng(seed)
    block_length = max(1, int(block_size))
    n_starts = max(1, math.ceil(n_obs / block_length))
    sharpe_obs = float(returns.mean() / returns.std(ddof=1) * math.sqrt(bars_per_year))
    null = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n_obs, size=n_starts)
        index = (starts[:, None] + np.arange(block_length)[None, :]) % n_obs
        sample = returns[index.ravel()][:n_obs]
        signs = rng.choice([-1.0, 1.0], size=n_starts).repeat(block_length)[:n_obs]
        sample = sample * signs
        sample_std = sample.std(ddof=1)
        null[b] = 0.0 if sample_std == 0 else sample.mean() / sample_std * math.sqrt(bars_per_year)
    z = float((sharpe_obs - null.mean()) / null.std(ddof=1))
    p_value = max(float((null >= sharpe_obs).mean()), 1.0 / n_boot)
    return {
        "block_size":   int(block_length),
        "n_boot":       int(n_boot),
        "sr_obs":       round(sharpe_obs, 4),
        "null_mean":    round(float(null.mean()), 4),
        "null_std":     round(float(null.std(ddof=1)), 4),
        "z_block":      round(z, 3),
        "p_block":      round(p_value, 5),
        "null_sharpes": null.tolist(),
    }


# strategy machinery

def _signal_position_vec(df, n_signal, entry_z, max_hold, sig_dir, z_lb, n_contracts):
    cum_ofi = df["ofi"].rolling(n_signal, min_periods=n_signal).sum()
    rolling_mean = cum_ofi.rolling(z_lb, min_periods=z_lb).mean().shift(1)
    rolling_std  = cum_ofi.rolling(z_lb, min_periods=z_lb).std(ddof=1).shift(1)
    z_scores = ((cum_ofi - rolling_mean) / rolling_std).values
    contract_sym = df["front_sym"].values
    n_bars = len(df)
    positions = np.zeros(n_bars, dtype=float)
    pos_size = 0.0; bars_held = 0; pos_dir = 0; prev_contract = None
    for i in range(n_bars):
        if contract_sym[i] != prev_contract:
            pos_size = 0.0; bars_held = 0; pos_dir = 0
        z = z_scores[i]
        if not np.isfinite(z):
            positions[i] = pos_size; prev_contract = contract_sym[i]; continue
        if pos_dir != 0 and bars_held >= max_hold:
            pos_size = 0.0; bars_held = 0; pos_dir = 0
        if pos_dir == +1 and z <= 0:
            pos_size = 0.0; bars_held = 0; pos_dir = 0
        elif pos_dir == -1 and z >= 0:
            pos_size = 0.0; bars_held = 0; pos_dir = 0
        if pos_dir == 0:
            if z >= entry_z:
                direction = +1 * sig_dir
                contract_count = max(1, int(round(n_contracts[i])))
                pos_dir = direction; pos_size = direction * contract_count; bars_held = 1
            elif z <= -entry_z:
                direction = -1 * sig_dir
                contract_count = max(1, int(round(n_contracts[i])))
                pos_dir = direction; pos_size = direction * contract_count; bars_held = 1
        elif pos_dir != 0:
            bars_held += 1
        positions[i] = pos_size
        prev_contract = contract_sym[i]
    return positions


def _trade_pnl(df, position, spread_ticks=1.0):
    prev_position = np.concatenate([[0.0], position[:-1]])
    mid_change = df["d_mid"].fillna(0).values
    gross_pnl = prev_position * contract_mult * mid_change
    cost = np.abs(position - prev_position) * spread_ticks * tick_size * contract_mult
    return gross_pnl - cost


def _vol_target_contracts(df, bars_per_year):
    returns = df["mid_close"].pct_change().mask(df["roll"]).fillna(0)
    rolling_std = returns.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1)
    annualized_vol = rolling_std * math.sqrt(bars_per_year)
    notional = contract_mult * df["mid_close"]
    n_target = (vol_target * capital) / (notional * annualized_vol.replace(0, np.nan))
    return n_target.clip(lower=1).fillna(1).values


def _bars_per_year(df):
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    return len(df) / years


def _annualized_sharpe(pnl, bars_per_year):
    returns = pnl[np.isfinite(pnl)]
    if len(returns) < 10 or returns.std(ddof=1) == 0:
        return float("nan")
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(bars_per_year))


def run_locked(df, spread_ticks=1.0):
    bars_per_year = _bars_per_year(df)
    n_contracts = _vol_target_contracts(df, bars_per_year)
    positions = _signal_position_vec(df, signal_bars, entry_threshold, max_hold_bars,
                                direction_sign, zscore_lookback, n_contracts)
    pnl = _trade_pnl(df, positions, spread_ticks)
    return positions, pnl, bars_per_year


def run_grid(df, bars_per_year):
    grid = []
    for n_signal in [3, 5, 10, 20]:
        for ez in [1.0, 1.5, 2.0]:
            for n_hold in [1, 3, 5, 10]:
                for z_lb in [200, 500]:
                    grid.append((n_signal, ez, n_hold, z_lb))
    contracts_fixed = np.ones(len(df))
    n_configs = len(grid)
    is_mask  = df.index <  oos_start
    oos_mask = df.index >= oos_start
    is_index   = np.where(is_mask)[0]
    oos_index  = np.where(oos_mask)[0]
    n_obs_is  = len(is_index)
    n_obs_oos = len(oos_index)
    cap_norm = contract_mult * 2500.0

    pnl_is_mat  = np.zeros((n_obs_is, n_configs))
    pnl_oos_mat = np.zeros((n_obs_oos, n_configs))
    sharpe_is  = np.zeros(n_configs)
    sharpe_oos = np.zeros(n_configs)
    configs = []
    for k_i, (n_signal, ez, n_hold, z_lb) in enumerate(grid):
        positions = _signal_position_vec(df, n_signal, ez, n_hold, direction_sign,
                                    z_lb, contracts_fixed)
        pnl = _trade_pnl(df, positions, spread_ticks=1.0) / cap_norm
        pnl_is_mat[:,  k_i] = pnl[is_index]
        pnl_oos_mat[:, k_i] = pnl[oos_index]
        sharpe_is[k_i]  = _annualized_sharpe(pnl[is_index],  bars_per_year)
        sharpe_oos[k_i] = _annualized_sharpe(pnl[oos_index], bars_per_year)
        configs.append({"signal_bars": n_signal, "entry_threshold": ez,
                        "max_hold_bars": n_hold, "direction_sign": direction_sign,
                        "zscore_lookback": z_lb,
                        "sharpe_is":  round(float(sharpe_is[k_i]),  3),
                        "sharpe_oos": round(float(sharpe_oos[k_i]), 3)})
    return configs, pnl_oos_mat, pnl_is_mat, sharpe_is, sharpe_oos


def main():
    print("=" * 78)
    print("ETH OFI - robustness suite  (DSR + SPA + block-bootstrap + IS-fix)")
    print("=" * 78)

    print("\n[1] Rebuilding dollar-volume bars with IS-only threshold...")
    if vbars_fix.exists():
        vbars = pd.read_parquet(vbars_fix)
        print(f"  loaded cached panel: {len(vbars):,} bars")
    else:
        vbars = rebuild_vbars_is_only()

    df = vbars.sort_index()
    df_is  = df.loc[df.index <  oos_start]
    df_oos = df.loc[df.index >= oos_start]
    bars_per_year = _bars_per_year(df)
    print(f"  bars/yr: {bars_per_year:.0f}   IS: {len(df_is):,} bars   OOS: {len(df_oos):,} bars")

    positions, pnl, _ = run_locked(df, spread_ticks=1.0)
    pnl_oos = pnl[df.index >= oos_start]
    sharpe_oos_isfix = _annualized_sharpe(pnl_oos, _bars_per_year(df_oos))
    print(f"\n  Locked OOS Sharpe (IS-fix panel): {sharpe_oos_isfix:+.3f}")

    print("\n[2] Running 96-config grid on IS-fix panel...")
    configs, pnl_oos_mat, pnl_is_mat, sharpe_is_arr, sharpe_oos_arr = run_grid(df, bars_per_year)
    finite = np.isfinite(sharpe_is_arr)
    print(f"  finite IS Sharpes: {finite.sum()}/{len(sharpe_is_arr)}  "
          f"max IS SR = {sharpe_is_arr[finite].max():.2f}")

    order = np.argsort(-sharpe_is_arr)
    top3_index = [int(i) for i in order[:3]]
    print("\n  Top-3 by IS Sharpe:")
    for i in top3_index:
        config = configs[i]
        print(f"    signal_bars={config['signal_bars']:>2}  entry_z={config['entry_threshold']}  "
              f"max_hold={config['max_hold_bars']:>2}  zlb={config['zscore_lookback']}   "
              f"IS_SR={config['sharpe_is']:+.2f}  OOS_SR={config['sharpe_oos']:+.2f}")

    locked_index = next(i for i, config in enumerate(configs)
                      if (config["signal_bars"], config["entry_threshold"],
                          config["max_hold_bars"], config["zscore_lookback"])
                      == (signal_bars, entry_threshold, max_hold_bars, zscore_lookback))
    print(f"\n[2a] Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014)")
    dsr = deflated_sharpe(pnl_is_mat[:, locked_index], sharpe_is_arr[finite], bars_per_year)
    if dsr is not None:
        print(f"     SR_obs annual = {dsr['sr_obs_annual']:+.3f}    "
              f"SR_0 expected-max = {dsr['sr0_annual']:+.3f}")
        print(f"     skew = {dsr['skew']:+.2f}   kurtosis = {dsr['kurt_non_excess']:.2f}   "
              f"T = {dsr['T']:,}   N_trials = {dsr['n_trials']}")
        print(f"     z_DSR = {dsr['z_dsr']:+.3f}    "
              f"P(true SR > 0 | data) = {dsr['dsr_prob']:.4f}")

    print(f"\n[2b] Hansen (2005) SPA-Consistent test ({len(configs)} configs)")
    spa = hansen_spa_consistent(pnl_oos_mat, n_boot=2000, mean_block=5.0)
    print(f"     studentized max t = {spa['t_stat_consistent']:.3f}    "
          f"p_SPA = {spa['p_spa_consistent']:.4f}")

    print(f"\n[3] Circular block bootstrap on locked OOS PnL (block = max_hold = {max_hold_bars})")
    block_boot = circular_block_bootstrap(pnl_oos / capital, _bars_per_year(df_oos),
                                   block_size=max_hold_bars, n_boot=5000)
    if block_boot is not None:
        print(f"     observed Sharpe: {block_boot['sr_obs']:+.3f}")
        print(f"     null mean: {block_boot['null_mean']:+.3f} +/- {block_boot['null_std']:.3f}")
        print(f"     z_block = {block_boot['z_block']:+.3f}    p_block = {block_boot['p_block']:.4f}")

    out = {
        "is_fix_panel": {
            "bar_threshold_basis": "IS-only daily-DV median / 10",
            "n_bars": int(len(df)),
            "sr_oos_isfix": round(float(sharpe_oos_isfix), 4),
        },
        "grid": configs,
        "top3_idx_by_is_sharpe": top3_index,
        "deflated_sharpe": dsr,
        "spa_consistent": {k2: v for k2, v in spa.items() if k2 != "boot_stats"},
        "spa_t_observed": spa["t_stat_consistent"],
        "block_bootstrap": {k2: v for k2, v in (block_boot or {}).items() if k2 != "null_sharpes"},
        "config": {
            "signal_bars": signal_bars, "entry_threshold": entry_threshold,
            "max_hold_bars": max_hold_bars, "direction_sign": direction_sign,
            "zscore_lookback": zscore_lookback, "capital": capital, "vol_target": vol_target,
        },
    }
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nWrote {out_path.name}")

    np.savez(root / "results" / "eth_ofi_robust_arrays.npz",
             sr_is=sharpe_is_arr, sr_oos=sharpe_oos_arr,
             pnl_oos_locked=pnl_oos,
             null_block=np.array(block_boot["null_sharpes"]) if block_boot else np.zeros(0),
             spa_boot=np.array(spa["boot_stats"]),
             top3_idx=np.array(top3_index, dtype=int),
             locked_idx=np.array([locked_index], dtype=int))
    print("Wrote eth_ofi_robust_arrays.npz")


if __name__ == "__main__":
    main()
