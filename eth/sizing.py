# ETH OFI sizing comparison: fixed 1, fixed 4, vol-target on $1M
# locked config: signal_bars=5, entry_threshold=1.0, max_hold_bars=3, zlb=200

import math, json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent

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
oos_start = pd.Timestamp("2024-01-01", tz="US/Eastern")

panel_path = root / "results" / "eth_ofi_vbars_isfix.parquet"


def load_panel():
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    if "d_mid" not in df.columns:
        df["d_mid"] = df["mid_close"].diff()
        df.loc[df["roll"], "d_mid"] = np.nan
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(df) / years
    return df, bars_per_year


def signal_position(df, n_contracts_series):
    rolling_sum = df["ofi"].rolling(signal_bars, min_periods=signal_bars).sum()
    rolling_mean = rolling_sum.rolling(zscore_lookback, min_periods=zscore_lookback).mean().shift(1)
    rolling_std  = rolling_sum.rolling(zscore_lookback, min_periods=zscore_lookback).std(ddof=1).shift(1)
    z = (rolling_sum - rolling_mean) / rolling_std

    sym           = df["front_sym"].values
    z_arr         = z.values
    contracts_arr = n_contracts_series.fillna(0).values
    pos = np.zeros(len(df), dtype=float)
    pos_size = 0.0; bars_held = 0; pos_dir = 0; prev_contract = None
    for i in range(len(df)):
        if sym[i] != prev_contract:
            pos_size = 0.0; bars_held = 0; pos_dir = 0
        z_value = z_arr[i]
        if not np.isfinite(z_value):
            pos[i] = pos_size; prev_contract = sym[i]; continue
        if pos_dir != 0 and bars_held >= max_hold_bars:
            pos_size = 0.0; bars_held = 0; pos_dir = 0
        if pos_dir == +1 and z_value <= 0:
            pos_size = 0.0; bars_held = 0; pos_dir = 0
        elif pos_dir == -1 and z_value >= 0:
            pos_size = 0.0; bars_held = 0; pos_dir = 0
        if pos_dir == 0:
            if z_value >= entry_threshold:
                direction = +1 * direction_sign
                n_contracts = max(1, int(round(contracts_arr[i])))
                pos_dir = direction; pos_size = direction * n_contracts; bars_held = 1
            elif z_value <= -entry_threshold:
                direction = -1 * direction_sign
                n_contracts = max(1, int(round(contracts_arr[i])))
                pos_dir = direction; pos_size = direction * n_contracts; bars_held = 1
        elif pos_dir != 0:
            bars_held += 1
        pos[i] = pos_size
        prev_contract = sym[i]
    return pd.Series(pos, index=df.index, dtype=float)


def vol_target_contracts(df, bars_per_year):
    returns = df["mid_close"].pct_change().mask(df["roll"]).fillna(0)
    rolling_std = returns.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1)
    ann_vol = rolling_std * math.sqrt(bars_per_year)
    notional_per_contract = contract_mult * df["mid_close"]
    n_target = (vol_target * capital) / (notional_per_contract * ann_vol.replace(0, np.nan))
    return n_target.clip(lower=1).fillna(1)


def trade_pnl(df, position, spread_ticks):
    prev = position.shift(1).fillna(0)
    d_mid = df["d_mid"].fillna(0)
    gross_pnl = prev * contract_mult * d_mid
    cost = (position - prev).abs() * spread_ticks * tick_size * contract_mult
    return (gross_pnl - cost).fillna(0)


def full_metrics(pnl, bars_per_year, cap=capital):
    returns = (pnl / cap).dropna()
    if len(returns) < 50 or returns.std() == 0:
        return None
    calendar_years = (returns.index.max() - returns.index.min()).total_seconds() / (365.25 * 86400)
    effective_years = calendar_years if calendar_years > 0 else len(returns) / bars_per_year
    bars_per_year_local = len(returns) / calendar_years if calendar_years > 0 else bars_per_year
    sharpe = float(returns.mean() / returns.std() * math.sqrt(bars_per_year_local))
    ann_arith = float(returns.mean() * bars_per_year_local)
    ann_vol = float(returns.std() * math.sqrt(bars_per_year_local))
    total_ret = float(pnl.sum() / cap)
    cagr = (1 + total_ret) ** (1 / effective_years) - 1 if effective_years > 0 else 0
    cumulative = pnl.cumsum(); drawdown = cumulative - cumulative.cummax()
    max_dd_dollar = float(drawdown.min())
    max_dd_pct = max_dd_dollar / cap * 100
    downside = returns[returns < 0]
    down_vol = float(downside.std() * math.sqrt(bars_per_year_local)) if len(downside) > 0 else 0
    sortino = ann_arith / down_vol if down_vol > 0 else float("nan")
    calmar = cagr / abs(max_dd_pct / 100) if max_dd_pct != 0 else 0
    pct_underwater = float((drawdown < 0).mean()) * 100
    return {
        "sharpe": round(sharpe, 3),
        "cagr_pct": round(cagr * 100, 2),
        "ann_arith_pct": round(ann_arith * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "total_net": round(float(pnl.sum()), 0),
        "total_return_pct": round(total_ret * 100, 2),
        "max_dd_dollar": round(max_dd_dollar, 0),
        "max_dd_pct": round(max_dd_pct, 2),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "pct_underwater": round(pct_underwater, 1),
        "n_bars": len(returns),
        "eff_years": round(effective_years, 2),
    }


def sign_flip(pnl, bars_per_year, n=5000, seed=43, cap=capital):
    returns_series = (pnl / cap).dropna()
    if len(returns_series) < 100 or returns_series.std() == 0:
        return None
    calendar_years = (returns_series.index.max() - returns_series.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year_local = len(returns_series) / calendar_years if calendar_years > 0 else bars_per_year
    returns = returns_series.values
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n):
        flipped = returns * rng.choice([-1, 1], size=len(returns))
        if flipped.std() == 0:
            continue
        null.append(flipped.mean() / flipped.std() * math.sqrt(bars_per_year_local))
    null_array = np.array(null)
    observed = returns.mean() / returns.std() * math.sqrt(bars_per_year_local)
    z = (observed - null_array.mean()) / null_array.std() if null_array.std() > 0 else 0
    p = float((null_array >= observed).mean())
    return {"z": round(float(z), 2), "p": round(float(max(p, 1 / n)), 4)}


def yearly_breakdown(pnl, bars_per_year, cap=capital):
    rows = []
    for yr, sub in pnl.groupby(pnl.index.year):
        returns = (sub / cap).dropna()
        if len(returns) < 50 or returns.std() == 0:
            continue
        calendar_years = (returns.index.max() - returns.index.min()).total_seconds() / (365.25 * 86400)
        bars_per_year_local = len(returns) / calendar_years if calendar_years > 0 else bars_per_year
        sharpe = float(returns.mean() / returns.std() * math.sqrt(bars_per_year_local))
        cumulative = sub.cumsum(); drawdown = cumulative - cumulative.cummax()
        rows.append({
            "year": int(yr),
            "n_bars": len(returns),
            "net": round(float(sub.sum()), 0),
            "ret_pct": round(float(sub.sum() / cap) * 100, 2),
            "ann_vol_pct": round(float(returns.std() * math.sqrt(bars_per_year_local) * 100), 2),
            "sharpe": round(sharpe, 2),
            "max_dd_pct": round(float(drawdown.min()) / cap * 100, 2),
        })
    return rows


def run_mode(df, df_oos, bars_per_year, mode, spread_ticks):
    if mode == "fixed_1":
        n_contracts = pd.Series(1.0, index=df.index)
    elif mode == "fixed_4":
        n_contracts = pd.Series(4.0, index=df.index)
    elif mode == "vol_target":
        n_contracts = vol_target_contracts(df, bars_per_year)
    else:
        raise ValueError(mode)
    pos = signal_position(df, n_contracts)
    pnl = trade_pnl(df, pos, spread_ticks)
    pnl_oos = pnl.loc[df_oos.index]
    pos_oos = pos.loc[df_oos.index]
    return pos_oos, pnl_oos, n_contracts.loc[df_oos.index]


def main():
    df, bars_per_year = load_panel()
    df_oos = df.loc[df.index >= oos_start]
    print(f"Volume bars: {len(df):,}  bars/yr: {bars_per_year:.0f}")
    print(f"OOS slice  : {len(df_oos):,} bars   {df_oos.index.min().date()} -> {df_oos.index.max().date()}")
    print(f"Capital    : ${capital:,.0f}   vol target: {vol_target*100:.0f}%")
    print(f"Locked cfg : signal_bars={signal_bars} entry={entry_threshold} "
          f"max_hold={max_hold_bars} direction={direction_sign:+d} zlb={zscore_lookback}\n")

    modes = ["fixed_1", "fixed_4", "vol_target"]
    cost_modes = [(1, "1-tick"), (2, "2-tick stress")]

    summary = {}
    for sp_ticks, sp_label in cost_modes:
        print(f"\n{'='*108}")
        print(f"  COST: {sp_label} half-spread (${sp_ticks*tick_size*contract_mult:.2f}/contract per cross)")
        print(f"{'='*108}")
        print(f"  {'mode':<14}{'Sharpe':>8}{'CAGR%':>9}{'Vol%':>8}{'Net $':>15}{'Ret%':>8}"
              f"{'MaxDD$':>14}{'DD%':>8}{'Sortino':>9}{'Calmar':>9}{'%UW':>7}")
        print("-" * 108)
        for mode in modes:
            pos, pnl, n_contracts = run_mode(df, df_oos, bars_per_year, mode, sp_ticks)
            metrics = full_metrics(pnl, bars_per_year)
            sign_flip_result = sign_flip(pnl, bars_per_year, n=3000)
            yearly = yearly_breakdown(pnl, bars_per_year=bars_per_year)
            avg_contracts = float(n_contracts.mean()); med_contracts = float(n_contracts.median())
            n_trades = int((pos.diff().fillna(0) != 0).sum())
            if metrics is None:
                print(f"  {mode:<14} insufficient"); continue
            print(f"  {mode:<14}{metrics['sharpe']:>+8.2f}{metrics['cagr_pct']:>+8.1f}%"
                  f"{metrics['ann_vol_pct']:>+7.1f}%{metrics['total_net']:>+15,.0f}"
                  f"{metrics['total_return_pct']:>+7.1f}%{metrics['max_dd_dollar']:>+14,.0f}"
                  f"{metrics['max_dd_pct']:>+7.1f}%{metrics['sortino']:>+9.2f}"
                  f"{metrics['calmar']:>9.2f}{metrics['pct_underwater']:>6.0f}%")
            summary[f"{mode}_{sp_label.replace(' ', '_')}"] = {
                "config": {
                    "signal_bars": signal_bars, "entry_threshold": entry_threshold,
                    "max_hold_bars": max_hold_bars, "direction_sign": direction_sign,
                    "zscore_lookback": zscore_lookback, "capital": capital,
                    "vol_target": vol_target, "vol_lookback": vol_lookback,
                    "spread_ticks": sp_ticks,
                },
                "metrics": metrics, "sign_flip": sign_flip_result, "yearly": yearly,
                "n_trades": n_trades,
                "avg_contracts": round(avg_contracts, 2),
                "median_contracts": round(med_contracts, 2),
            }
            print(f"    sign-flip z={sign_flip_result['z'] if sign_flip_result else 0:+.2f} (p={sign_flip_result['p'] if sign_flip_result else 0:.4g})  "
                  f"trades={n_trades}  avg n_contracts={avg_contracts:.2f} (med {med_contracts:.1f})")

    headline_key = "vol_target_1-tick"
    if headline_key in summary:
        headline = summary[headline_key]
        print(f"\n{'='*72}")
        print(f"  headline: vol_target sizing, 1-tick spread, $1M capital")
        print(f"{'='*72}")
        print(f"  {'Year':<6}{'Bars':>7}{'Net $':>14}{'Ret%':>9}{'Vol%':>8}{'Sharpe':>9}{'MaxDD%':>10}")
        for row in headline["yearly"]:
            print(f"  {row['year']:<6}{row['n_bars']:>7,}{row['net']:>+14,.0f}"
                  f"{row['ret_pct']:>+8.1f}%{row['ann_vol_pct']:>+7.1f}%"
                  f"{row['sharpe']:>+9.2f}{row['max_dd_pct']:>+9.1f}%")
        metrics = headline["metrics"]
        print(f"\n  Total OOS: {metrics['total_return_pct']:+.1f}% over {metrics['eff_years']:.2f} yrs  "
              f"(CAGR {metrics['cagr_pct']:+.1f}%, Sharpe {metrics['sharpe']:+.2f}, "
              f"MaxDD {metrics['max_dd_pct']:+.1f}%, Calmar {metrics['calmar']:.2f})")

    out_path = root / "results" / "eth_ofi_sizing.json"
    with out_path.open("w") as file_handle:
        json.dump(summary, file_handle, indent=2, default=str)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
