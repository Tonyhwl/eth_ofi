# eth ofi capacity sweep using the almgren-chriss (2000) square-root market
# impact model.
# impact ($) = gamma * sigma_bar * sqrt(n / v_per_bar) * n * contract_mult * mid
# adv sweep: 5k, 10k, 20k contracts/day; capital sweep: $1m to $50m

import math
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategy import (
    signal_bars, entry_threshold, max_hold_bars, direction_sign, zscore_lookback,
    contract_mult, tick_size, spread_ticks, vol_target, vol_lookback,
    build_event_calendar, simulate, pnl_from_trades, panel_path,
)

oos_start        = pd.Timestamp("2024-01-01", tz="US/Eastern")
cap_hours_default = 10/60
gamma = 0.1
adv_cases = {
    "pessimistic (5k)":  5_000,
    "base (10k)":       10_000,
    "optimistic (20k)": 20_000,
}
capitals = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]


def load_panel():
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    if "d_mid" not in df.columns:
        df["d_mid"] = df["mid_close"].diff()
        df.loc[df["roll"], "d_mid"] = np.nan
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(df) / years
    return df, bars_per_year


def vol_target_contracts(df, bars_per_year, cap):
    ret = df["mid_close"].pct_change().mask(df["roll"]).fillna(0)
    rolling_std = ret.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1)
    ann_vol = rolling_std * math.sqrt(bars_per_year)
    notional = contract_mult * df["mid_close"]
    n_target = (vol_target * cap) / (notional * ann_vol.replace(0, np.nan))
    return n_target.clip(lower=1).fillna(1)


def bar_vol_series(df, bars_per_year):
    ret = df["mid_close"].pct_change().mask(df["roll"]).fillna(0)
    return ret.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1).fillna(0)


def pnl_with_impact(trades, df, sigma_bar, bars_per_year, adv_daily):
    bars_per_day = bars_per_year / 252
    v_per_bar = adv_daily / bars_per_day

    mid   = df["mid_close"].values
    sigma = sigma_bar.values

    pnl = pnl_from_trades(trades, df, convention="realistic", cap_hours=cap_hours_default)

    for t in trades:
        entry_idx = t.entry_idx
        exit_idx = t.exit_idx
        n  = t.size
        sigma_entry = max(sigma[entry_idx], 1e-8)
        sigma_exit = max(sigma[exit_idx], 1e-8)
        mid_entry = mid[entry_idx]
        mid_exit = mid[exit_idx]
        impact_entry = gamma * sigma_entry * math.sqrt(n / v_per_bar) * n * contract_mult * mid_entry
        impact_exit = gamma * sigma_exit * math.sqrt(n / v_per_bar) * n * contract_mult * mid_exit
        pnl.iloc[entry_idx] -= impact_entry
        pnl.iloc[exit_idx] -= impact_exit

    return pnl


def run_metrics(pnl_oos, cap):
    ret = (pnl_oos / cap).dropna()
    if len(ret) < 50 or ret.std() == 0:
        return None
    cal_years = (ret.index.max() - ret.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(ret) / cal_years
    sharpe = float(ret.mean() / ret.std() * math.sqrt(bars_per_year))
    cumulative = pnl_oos.cumsum()
    drawdown = cumulative - cumulative.cummax()
    max_drawdown = float(drawdown.min()) / cap * 100
    cagr = (1 + float(pnl_oos.sum() / cap)) ** (1 / cal_years) - 1
    return {
        "sharpe": round(sharpe, 2),
        "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(max_drawdown, 1),
    }


def main():
    df, bars_per_year = load_panel()
    df_oos  = df.loc[df.index >= oos_start]
    events  = build_event_calendar()
    sigma_bar = bar_vol_series(df, bars_per_year)
    bars_per_day = bars_per_year / 252

    print(f"Panel: {len(df):,} bars  bpy={bars_per_year:.0f}  OOS: {len(df_oos):,} bars")
    print(f"Impact model: gamma={gamma}  (Almgren-Chriss sqrt)\n")

    for adv_label, adv_daily in adv_cases.items():
        v_per_bar = adv_daily / bars_per_day
        print(f"{'='*70}")
        print(f"ADV = {adv_label}  ->  V_per_bar = {v_per_bar:.0f} contracts")
        print(f"{'='*70}")
        print(f"  {'Capital':>10}  {'Avg n':>7}  {'Sharpe':>8}  {'CAGR':>7}  "
              f"{'MaxDD':>7}  {'Impact/Spread':>14}")
        print(f"  {'-'*65}")

        for cap in capitals:
            n_contracts = vol_target_contracts(df, bars_per_year, cap)
            trades = simulate(df, n_contracts, use_event_filter=False, use_weekend_filter=False,
                              cap_hours=cap_hours_default, events=events)
            trades_oos = [t for t in trades if t.entry_time >= oos_start]

            pnl = pnl_with_impact(trades, df, sigma_bar, bars_per_year, adv_daily)
            pnl_oos = pnl.loc[df_oos.index]
            metrics = run_metrics(pnl_oos, cap)
            if metrics is None:
                continue

            avg_n = np.mean([t.size for t in trades_oos]) if trades_oos else 0

            spread_total = sum(t.size * spread_ticks * tick_size * contract_mult * 2
                               for t in trades_oos)
            impact_total = 0.0
            for t in trades_oos:
                n   = t.size
                sigma_entry = max(float(sigma_bar.iloc[t.entry_idx]), 1e-8)
                mid_entry = float(df["mid_close"].iloc[t.entry_idx])
                mid_exit = float(df["mid_close"].iloc[t.exit_idx])
                sigma_exit = max(float(sigma_bar.iloc[t.exit_idx]), 1e-8)
                impact_total += (gamma * sigma_entry * math.sqrt(n / v_per_bar) * n * contract_mult * mid_entry +
                                 gamma * sigma_exit  * math.sqrt(n / v_per_bar) * n * contract_mult * mid_exit)

            ratio = impact_total / spread_total if spread_total > 0 else 0

            print(f"  ${cap/1e6:>8.0f}M  {avg_n:>7.1f}  "
                  f"{metrics['sharpe']:>+8.2f}  {metrics['cagr_pct']:>+6.1f}%  "
                  f"{metrics['max_dd_pct']:>+6.1f}%  {ratio:>13.1%}")
        print()


if __name__ == "__main__":
    main()
