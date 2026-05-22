# ETH OFI strategy. signal = rolling z-score of cumulative OFI over 5
# dollar-volume bars; entry |z| >= 1; exit on z-cross, 3-bar hold, contract
# roll, or 10-min clock cap. realistic PnL convention on cap-fire bars.

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps


root       = Path(__file__).resolve().parent
panel_path = root / "results" / "eth_ofi_vbars_isfix.parquet"

# strategy parameters (locked after IS selection)
signal_bars      = 5
entry_threshold  = 1.0
max_hold_bars    = 3
direction_sign   = +1   # +1 = follow ofi
zscore_lookback  = 200

# instrument
contract_mult = 50.0
tick_size     = 0.05
spread_ticks  = 1.0

# sizing
capital      = 1_000_000.0
vol_target   = 0.15
vol_lookback = 200

oos_start = pd.Timestamp("2024-01-01", tz="US/Eastern")

# event filter (minutes around each macro release)
event_pre_min  = 10
event_post_min = 30

# friday filter (minutes before 5pm ET CME weekly close)
friday_no_entry_min   = 30
friday_force_flat_min = 15
friday_close_hour     = 17

fomc_dates = [
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29",
]


@dataclass
class Trade:
    entry_time:  pd.Timestamp
    entry_idx:   int
    exit_time:   pd.Timestamp
    exit_idx:    int
    direction:   int    # +1 long, -1 short
    size:        int    # contracts
    exit_reason: str    # 'hold', 'zcross', 'event', 'weekend', 'roll', 'cap', 'end'

    def duration_hours(self):
        return (self.exit_time - self.entry_time).total_seconds() / 3600


def _first_friday(year, month):
    date = pd.Timestamp(year=year, month=month, day=1)
    return date + pd.Timedelta(days=(4 - date.weekday()) % 7)


def _cpi_date(year, month):
    date = pd.Timestamp(year=year, month=month, day=12)
    if date.weekday() == 5:
        date += pd.Timedelta(days=2)
    elif date.weekday() == 6:
        date += pd.Timedelta(days=1)
    return date


def build_event_calendar():
    events = []
    for date_str in fomc_dates:
        events.append(pd.Timestamp(date_str + " 14:00", tz="US/Eastern"))
    for year in range(2021, 2027):
        for month in range(1, 13):
            cpi_date = _cpi_date(year, month)
            events.append(pd.Timestamp(cpi_date.strftime("%Y-%m-%d") + " 08:30", tz="US/Eastern"))
            nfp_date = _first_friday(year, month)
            events.append(pd.Timestamp(nfp_date.strftime("%Y-%m-%d") + " 08:30", tz="US/Eastern"))
    return sorted(events)


def in_event_window(ts, events):
    for event in events:
        delta = (ts - event).total_seconds() / 60
        if -event_pre_min <= delta <= event_post_min:
            return True
    return False


def in_friday_window(ts, mins_before):
    if ts.weekday() != 4:
        return False
    close = ts.replace(hour=friday_close_hour, minute=0, second=0, microsecond=0)
    delta = (close - ts).total_seconds() / 60
    return 0 <= delta <= mins_before


def vol_target_contracts(df, bars_per_year):
    returns     = df["mid_close"].pct_change().mask(df["roll"]).fillna(0)
    rolling_std = returns.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1)
    ann_vol     = rolling_std * math.sqrt(bars_per_year)
    notional    = contract_mult * df["mid_close"]
    n_contracts = (vol_target * capital) / (notional * ann_vol.replace(0, np.nan))
    return n_contracts.clip(lower=1).fillna(1)


def simulate(df, n_contracts_series, use_event_filter, use_weekend_filter,
             cap_hours=None, events=None):
    """Run the state machine and return a list of Trade objects."""
    # exit priority: clock cap > event flat > friday flat > roll > hold > z-cross

    if use_event_filter and events is None:
        events = build_event_calendar()

    cumulative_ofi = df["ofi"].rolling(signal_bars, min_periods=signal_bars).sum()
    rolling_mean   = cumulative_ofi.rolling(zscore_lookback, min_periods=zscore_lookback).mean().shift(1)
    rolling_std    = cumulative_ofi.rolling(zscore_lookback, min_periods=zscore_lookback).std(ddof=1).shift(1)
    z_series       = ((cumulative_ofi - rolling_mean) / rolling_std).values

    contracts_arr = n_contracts_series.fillna(0).values
    contract_sym  = df["front_sym"].values
    timestamps    = df.index
    n_bars        = len(df)
    cap_seconds   = cap_hours * 3600 if cap_hours is not None else None

    trades        = []
    pos_dir       = 0
    pos_size      = 0
    bars_held     = 0
    entry_time    = None
    entry_idx     = None
    prev_contract = None

    for i in range(n_bars):
        ts      = timestamps[i]
        z_score = z_series[i]

        cap_fired   = (pos_dir != 0 and cap_seconds is not None
                       and entry_time is not None
                       and (ts - entry_time).total_seconds() >= cap_seconds)
        block_event = use_event_filter   and in_event_window(ts, events)
        force_flat  = use_weekend_filter and in_friday_window(ts, friday_force_flat_min)
        block_entry = block_event or (use_weekend_filter and in_friday_window(ts, friday_no_entry_min))

        if cap_fired:
            trades.append(Trade(entry_time, entry_idx, ts, i, pos_dir, pos_size, "cap"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_idx = None
            prev_contract = contract_sym[i]; continue

        if pos_dir != 0 and block_event:
            trades.append(Trade(entry_time, entry_idx, ts, i, pos_dir, pos_size, "event"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_idx = None
            prev_contract = contract_sym[i]; continue

        if pos_dir != 0 and force_flat:
            trades.append(Trade(entry_time, entry_idx, ts, i, pos_dir, pos_size, "weekend"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_idx = None
            prev_contract = contract_sym[i]; continue

        if contract_sym[i] != prev_contract and prev_contract is not None and pos_dir != 0:
            trades.append(Trade(entry_time, entry_idx, ts, i, pos_dir, pos_size, "roll"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_idx = None

        if not np.isfinite(z_score):
            prev_contract = contract_sym[i]; continue

        if pos_dir != 0 and bars_held >= max_hold_bars:
            trades.append(Trade(entry_time, entry_idx, ts, i, pos_dir, pos_size, "hold"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_idx = None

        if pos_dir == +1 and z_score <= 0:
            trades.append(Trade(entry_time, entry_idx, ts, i, pos_dir, pos_size, "zcross"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_idx = None
        elif pos_dir == -1 and z_score >= 0:
            trades.append(Trade(entry_time, entry_idx, ts, i, pos_dir, pos_size, "zcross"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_idx = None

        if pos_dir == 0 and not block_entry:
            if z_score >= entry_threshold:
                pos_dir = +1 * direction_sign
                pos_size = max(1, int(round(contracts_arr[i])))
                bars_held = 1; entry_time = ts; entry_idx = i
            elif z_score <= -entry_threshold:
                pos_dir = -1 * direction_sign
                pos_size = max(1, int(round(contracts_arr[i])))
                bars_held = 1; entry_time = ts; entry_idx = i
        elif pos_dir != 0:
            bars_held += 1

        prev_contract = contract_sym[i]

    if pos_dir != 0:
        trades.append(Trade(entry_time, entry_idx, timestamps[n_bars-1], n_bars-1,
                            pos_dir, pos_size, "end"))
    return trades


def pnl_from_trades(trades, df, convention="realistic", cap_hours=None):
    """Bar-level PnL. convention: 'realistic' | 'optimistic' | 'conservative'."""
    n_bars  = len(df)
    d_mid   = df["d_mid"].fillna(0).values
    times   = df.index
    position       = np.zeros(n_bars)
    skip_bars      = set()
    fractional_bars = {}

    for t in trades:
        signed = t.direction * t.size
        for j in range(t.entry_idx, t.exit_idx):
            position[j] = signed

        if t.exit_reason in ("event", "weekend", "cap"):
            if convention == "optimistic":
                skip_bars.add(t.exit_idx)
            elif convention == "realistic":
                if t.exit_idx == 0:
                    continue
                prev_time    = times[t.exit_idx - 1]
                curr_time    = times[t.exit_idx]
                bar_duration = (curr_time - prev_time).total_seconds()
                if bar_duration <= 0:
                    continue
                if t.exit_reason == "cap" and cap_hours is not None:
                    fire_time = t.entry_time + pd.Timedelta(hours=cap_hours)
                elif t.exit_reason == "weekend":
                    fire_time = curr_time.replace(hour=16, minute=45, second=0, microsecond=0)
                else:
                    fire_time = prev_time + pd.Timedelta(seconds=bar_duration * 0.5)
                fraction = max(0.0, min(1.0, (fire_time - prev_time).total_seconds() / bar_duration))
                fractional_bars[t.exit_idx] = fraction

    prev_pos = np.concatenate([[0.0], position[:-1]])
    pnl = np.zeros(n_bars)
    for i in range(n_bars):
        if i in skip_bars:
            pnl[i] = 0.0
        elif i in fractional_bars:
            pnl[i] = prev_pos[i] * contract_mult * d_mid[i] * fractional_bars[i]
        else:
            pnl[i] = prev_pos[i] * contract_mult * d_mid[i]

    for t in trades:
        cost = t.size * spread_ticks * tick_size * contract_mult
        pnl[t.entry_idx] -= cost
        pnl[t.exit_idx]  -= cost

    return pd.Series(pnl, index=df.index)


def compute_metrics(pnl):
    returns = (pnl / capital).dropna()
    if len(returns) < 50 or returns.std() == 0:
        return None
    calendar_years = (returns.index.max() - returns.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year  = len(returns) / calendar_years
    sharpe         = float(returns.mean() / returns.std() * math.sqrt(bars_per_year))
    cumulative     = pnl.cumsum()
    drawdown       = cumulative - cumulative.cummax()
    max_drawdown   = float(drawdown.min()) / capital * 100
    cagr           = (1 + float(pnl.sum() / capital)) ** (1 / calendar_years) - 1
    active         = returns[returns != 0]
    return {
        "sharpe":      round(sharpe, 3),
        "cagr_pct":    round(cagr * 100, 2),
        "ann_vol_pct": round(float(returns.std() * math.sqrt(bars_per_year)) * 100, 2),
        "max_dd_pct":  round(max_drawdown, 2),
        "calmar":      round(cagr / abs(max_drawdown / 100), 3) if max_drawdown != 0 else 0,
        "skew":        round(float(sps.skew(active)), 3) if len(active) > 5 else 0,
        "kurt":        round(float(sps.kurtosis(active, fisher=False)), 3) if len(active) > 5 else 0,
        "ann_pnl":     round(float(pnl.sum()) / calendar_years, 0),
    }


def main():
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(df) / years
    n_contracts = vol_target_contracts(df, bars_per_year)
    df_oos  = df.loc[df.index >= oos_start]
    events  = build_event_calendar()

    print(f"Panel: {len(df):,} bars  ({years:.2f} yrs)")
    print(f"OOS:   {len(df_oos):,} bars  "
          f"{df_oos.index.min().date()} -> {df_oos.index.max().date()}")

    for label, use_event, use_weekend, cap_hours in [
        ("bare locked",          False, False, None),
        ("+ event & weekend",    True,  True,  None),
        ("+ 3h cap",             True,  True,  3),
        ("locked (10-min cap)",  False, False, 10/60),
    ]:
        trades     = simulate(df, n_contracts, use_event, use_weekend, cap_hours=cap_hours, events=events)
        trades_oos = [t for t in trades if t.entry_time >= oos_start]
        pnl_real = pnl_from_trades(trades, df, "realistic",    cap_hours=cap_hours)
        pnl_opt  = pnl_from_trades(trades, df, "optimistic",   cap_hours=cap_hours)
        pnl_con  = pnl_from_trades(trades, df, "conservative", cap_hours=cap_hours)
        metrics = compute_metrics(pnl_real.loc[df_oos.index])
        print(f"\n--- {label}  ({len(trades_oos)} trades) ---")
        if metrics:
            print(f"  Sharpe={metrics['sharpe']:+.3f}  CAGR={metrics['cagr_pct']:+.1f}%  "
                  f"DD={metrics['max_dd_pct']:+.1f}%  kurt={metrics['kurt']:.1f}")
            metrics_opt = compute_metrics(pnl_opt.loc[df_oos.index])
            metrics_con = compute_metrics(pnl_con.loc[df_oos.index])
            if metrics_opt and metrics_con:
                print(f"  pnl convention range: "
                      f"[{metrics_con['sharpe']:+.3f} conservative, "
                      f"{metrics_opt['sharpe']:+.3f} optimistic]")

    print("\n--- yearly (final config) ---")
    trades_final = simulate(df, n_contracts, False, False, cap_hours=10/60, events=events)
    pnl_oos = pnl_from_trades(trades_final, df, "realistic", cap_hours=10/60).loc[df_oos.index]
    print(f"  {'year':<6}{'bars':>8}{'PnL $':>14}{'ret%':>9}{'Sharpe':>9}{'DD%':>9}")
    for yr, year_pnl in pnl_oos.groupby(pnl_oos.index.year):
        returns = (year_pnl / capital).dropna()
        if len(returns) < 50:
            continue
        calendar_years = (returns.index.max() - returns.index.min()).total_seconds() / (365.25 * 86400)
        year_bars_per_year = len(returns) / max(calendar_years, 1e-9)
        sharpe     = float(returns.mean() / returns.std() * math.sqrt(year_bars_per_year))
        cumulative = year_pnl.cumsum(); drawdown = cumulative - cumulative.cummax()
        print(f"  {yr:<6}{len(year_pnl):>8}{float(year_pnl.sum()):>+13,.0f}"
              f"{float(year_pnl.sum()/capital*100):>+8.1f}%{sharpe:>+9.2f}"
              f"{float(drawdown.min())/capital*100:>+8.1f}%")


if __name__ == "__main__":
    main()
