# OOS Sharpe vs per-side adverse-selection cost (locked strategy).

import math
from pathlib import Path

import pandas as pd

import strategy as strat
from strategy import (
    vol_target_contracts, simulate, pnl_from_trades,
    build_event_calendar, oos_start, capital, panel_path,
)


def load_panel():
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(df) / years
    return df, bars_per_year


COST_LEVELS_TICKS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]


def metrics(pnl_oos):
    returns = (pnl_oos / capital).dropna()
    if len(returns) < 50 or returns.std() == 0:
        return None
    calendar_years = (returns.index.max() - returns.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(returns) / calendar_years
    sharpe = float(returns.mean() / returns.std() * math.sqrt(bars_per_year))
    ann_arith = float(returns.mean() * bars_per_year)
    ann_vol = float(returns.std() * math.sqrt(bars_per_year))
    cumulative = pnl_oos.cumsum()
    drawdown = cumulative - cumulative.cummax()
    max_dd_dollar = float(drawdown.min())
    effective_years = calendar_years if calendar_years > 0 else len(returns) / bars_per_year
    total_return = float(pnl_oos.sum() / capital)
    cagr = (1 + total_return) ** (1 / effective_years) - 1 if effective_years > 0 else 0
    return {
        "sharpe": sharpe,
        "cagr_pct": cagr * 100,
        "ann_vol_pct": ann_vol * 100,
        "max_dd_pct": max_dd_dollar / capital * 100,
    }


def main():
    df, bars_per_year = load_panel()
    df_oos = df.loc[df.index >= oos_start]
    print("=" * 78)
    print("ETH OFI - cost sensitivity on the locked strategy")
    print("=" * 78)
    print(f"OOS: {df_oos.index.min().date()} -> {df_oos.index.max().date()}  "
          f"({len(df_oos):,} bars)")
    print("Locked config: filters off, 10-minute clock cap")
    print()

    events      = build_event_calendar()
    n_contracts = vol_target_contracts(df, bars_per_year)
    trades      = simulate(df, n_contracts, use_event_filter=False,
                           use_weekend_filter=False, cap_hours=10/60, events=events)
    print(f"Simulated {len(trades):,} trades.")

    # use a typical front-month eth price to translate ticks to bp
    mean_price            = float(df_oos["mid_close"].mean())
    notional_per_contract = strat.contract_mult * mean_price
    bp_per_tick           = (strat.tick_size * strat.contract_mult) / notional_per_contract * 1e4

    print()
    print(f"  {'ticks/side':>10}  {'~bp/side':>9}  {'Sharpe':>7}  "
          f"{'CAGR':>7}  {'Vol':>6}  {'MaxDD':>8}")
    print("  " + "-" * 60)

    rows = []
    original_ticks = strat.spread_ticks
    try:
        for ticks in COST_LEVELS_TICKS:
            strat.spread_ticks = ticks
            pnl = pnl_from_trades(trades, df, convention="realistic", cap_hours=10/60)
            pnl_oos = pnl.loc[df_oos.index]
            metrics_result = metrics(pnl_oos)
            if metrics_result is None:
                print(f"  {ticks:>10.2f}  insufficient")
                continue
            bp_per_side = ticks * bp_per_tick
            print(f"  {ticks:>10.2f}  {bp_per_side:>+9.2f}  {metrics_result['sharpe']:>+7.2f}  "
                  f"{metrics_result['cagr_pct']:>+6.1f}%  {metrics_result['ann_vol_pct']:>5.1f}%  "
                  f"{metrics_result['max_dd_pct']:>+7.1f}%")
            rows.append({"ticks_per_side": ticks, "bp_per_side": bp_per_side, **metrics_result})
    finally:
        strat.spread_ticks = original_ticks

    out = Path(__file__).resolve().parent / "results" / "eth_ofi_cost_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
