# market-order tick-fill sim using raw TBBO. worst-case retail execution
# benchmark; not used for the paper's colocated 1-tick headline.

import sys
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import databento as db

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategy import (
    simulate, vol_target_contracts, pnl_from_trades, build_event_calendar,
    oos_start, panel_path, contract_mult, tick_size, capital,
)

ROOT     = Path(__file__).resolve().parent
TBBO_DIR = ROOT.parent / "data" / "eth_tbbo"


def load_tbbo_day(date_obj):
    """Load one day of TBBO data as a frame of front-month bid/ask/sizes."""
    filename = f"glbx-mdp3-{date_obj.strftime('%Y%m%d')}.tbbo.dbn.zst"
    file_path = TBBO_DIR / filename
    if not file_path.exists():
        return None
    try:
        store = db.DBNStore.from_file(str(file_path))
        df = store.to_df()
    except Exception:
        return None
    if df.empty or "symbol" not in df.columns:
        return None
    df.index = pd.DatetimeIndex(df.index).tz_convert("US/Eastern")
    outrights = df[~df["symbol"].astype(str).str.contains("-", na=False)].copy()
    if outrights.empty:
        return None
    # find front month (smallest non-negative days to expiry)
    month_codes = "FGHJKMNQUVXZ"
    best_symbol, best_days_to_expiry = None, 1e9
    for symbol in outrights["symbol"].unique():
        match = re.match(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d+)$", str(symbol))
        if not match:
            continue
        _, month_code, year = match.groups()
        month = month_codes.index(month_code) + 1
        year = 2020 + int(year) if len(year) == 1 else 2000 + int(year)
        expiry = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        while expiry.weekday() >= 5:
            expiry -= pd.Timedelta(days=1)
        days_to_expiry = (expiry.date() - date_obj).days
        if 0 <= days_to_expiry < best_days_to_expiry:
            best_symbol, best_days_to_expiry = symbol, days_to_expiry
    if best_symbol is None:
        return None
    sub = outrights[outrights["symbol"] == best_symbol]
    bid = sub["bid_px_00"].astype(float).values
    ask = sub["ask_px_00"].astype(float).values
    bid_size = sub["bid_sz_00"].astype(float).values
    ask_size = sub["ask_sz_00"].astype(float).values
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    return pd.DataFrame({"bid": bid[valid], "ask": ask[valid],
                         "bid_sz": bid_size[valid], "ask_sz": ask_size[valid]},
                        index=sub.index[valid]).sort_index()


def bbo_at(tbbo_df, ts):
    """Last TBBO update at or before ts. Returns dict of bid/ask/sizes or None."""
    if tbbo_df is None or len(tbbo_df) == 0:
        return None
    pos = tbbo_df.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    row = tbbo_df.iloc[pos]
    return {"bid": float(row["bid"]), "ask": float(row["ask"]),
            "bid_sz": float(row["bid_sz"]), "ask_sz": float(row["ask_sz"])}


def fill_price(bbo, direction, size, side, tick_size):
    """Market order fill price, at the touch if it fits or one tick deeper if it does not."""
    # direction is +1 for a long position, -1 for a short; side is 'entry' or 'exit'
    is_buy = (direction == +1 and side == "entry") or (direction == -1 and side == "exit")
    if is_buy:
        if size <= bbo["ask_sz"]:
            return bbo["ask"], False
        return bbo["ask"] + tick_size, True
    else:
        if size <= bbo["bid_sz"]:
            return bbo["bid"], False
        return bbo["bid"] - tick_size, True


def main():
    print("=" * 78)
    print("ETH OFI - tick-level execution simulation")
    print("=" * 78)
    print("loading panel and running strategy ...")
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(df) / years
    n_contracts = vol_target_contracts(df, bars_per_year)
    events  = build_event_calendar()
    trades  = simulate(df, n_contracts, use_event_filter=False, use_weekend_filter=False,
                       cap_hours=10/60, events=events)
    oos_trades = [t for t in trades if t.entry_time >= oos_start]
    print(f"  {len(trades):,} total trades, {len(oos_trades):,} OOS trades")

    # bar-level pnl with parametric 1-tick/side cost (paper headline) for reference
    pnl_param = pnl_from_trades(trades, df, convention="realistic", cap_hours=10/60)
    pnl_param_oos = pnl_param.loc[df.index >= oos_start]
    sharpe_param = float(pnl_param_oos.mean() / pnl_param_oos.std()
                     * math.sqrt(len(pnl_param_oos)
                                 / ((pnl_param_oos.index.max() - pnl_param_oos.index.min())
                                    .total_seconds() / (365.25 * 86400))))
    print(f"  paper headline (1 tick/side parametric): OOS Sharpe = {sharpe_param:.3f}")
    print()

    # walk OOS trades day by day; load each day's TBBO once
    trades_by_day = {}
    for t in oos_trades:
        for tag, ts in [("entry", t.entry_time), ("exit", t.exit_time)]:
            trade_date = ts.date()
            trades_by_day.setdefault(trade_date, []).append((t, tag, ts))

    realised_half_spreads = []   # in ticks, per side
    swept = []                   # boolean per side
    fill_records = []            # for reconstructing PnL

    print("scanning TBBO ...")
    for i, (day, day_legs) in enumerate(sorted(trades_by_day.items())):
        tbbo = load_tbbo_day(day)
        if tbbo is None:
            for trade, tag, ts in day_legs:
                # fall back to parametric 1-tick assumption
                fill_records.append({"trade_id": id(trade), "tag": tag, "ts": ts,
                                     "half_spread_ticks": 1.0, "swept": False,
                                     "fill_px": np.nan, "mid_at_ts": np.nan,
                                     "size": trade.size, "direction": trade.direction})
            continue
        for trade, tag, ts in day_legs:
            bbo = bbo_at(tbbo, ts)
            if bbo is None:
                fill_records.append({"trade_id": id(trade), "tag": tag, "ts": ts,
                                     "half_spread_ticks": 1.0, "swept": False,
                                     "fill_px": np.nan, "mid_at_ts": np.nan,
                                     "size": trade.size, "direction": trade.direction})
                continue
            mid = (bbo["bid"] + bbo["ask"]) / 2
            fill_px, did_sweep = fill_price(bbo, trade.direction, trade.size, tag, tick_size)
            half_ticks = abs(fill_px - mid) / tick_size
            realised_half_spreads.append(half_ticks)
            swept.append(did_sweep)
            fill_records.append({"trade_id": id(trade), "tag": tag, "ts": ts,
                                 "half_spread_ticks": half_ticks, "swept": did_sweep,
                                 "fill_px": fill_px, "mid_at_ts": mid,
                                 "size": trade.size, "direction": trade.direction})
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(trades_by_day)} days "
                  f"({len(realised_half_spreads):,} legs so far)")

    print(f"\ndone scanning {len(trades_by_day)} days, "
          f"{len(realised_half_spreads):,} legs with TBBO coverage")

    half_spreads = np.array(realised_half_spreads)
    sweeps = np.array(swept)
    print()
    print(f"  Realised half-spread distribution (ticks per side):")
    print(f"    median            = {np.median(half_spreads):.3f}")
    print(f"    mean              = {half_spreads.mean():.3f}")
    print(f"    p90               = {np.percentile(half_spreads, 90):.3f}")
    print(f"    p95               = {np.percentile(half_spreads, 95):.3f}")
    print(f"    p99               = {np.percentile(half_spreads, 99):.3f}")
    print(f"    max               = {half_spreads.max():.3f}")
    print(f"  Sweep rate          = {sweeps.mean()*100:.1f}% of fills exceeded top-of-book size")

    # rebuild pnl replacing the 1-tick parametric cost with realised half-spreads
    fill_df = pd.DataFrame(fill_records)
    fill_df["realised_cost_per_contract"] = fill_df["half_spread_ticks"] * tick_size
    fill_df["realised_cost_total"] = fill_df["realised_cost_per_contract"] * fill_df["size"] * contract_mult

    # pnl adjustment: paper assumed cost = 1 tick * size * mult per side,
    # new cost = realised half-spread (in $) * size * mult per side
    paper_cost_per_leg = 1.0 * tick_size * fill_df["size"] * contract_mult
    cost_delta = fill_df["realised_cost_total"] - paper_cost_per_leg  # positive = worse than paper

    # pnl is bar-indexed, so push the cost_delta back onto bar timestamps
    pnl_tick = pnl_param.copy()
    for _, row in fill_df.iterrows():
        ts_idx = pnl_tick.index.get_indexer([row["ts"]], method="nearest")[0]
        pnl_tick.iloc[ts_idx] -= cost_delta.loc[row.name]
    pnl_tick_oos = pnl_tick.loc[df.index >= oos_start]

    years_oos = (pnl_tick_oos.index.max() - pnl_tick_oos.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year_oos = len(pnl_tick_oos) / years_oos
    sharpe_tick = float(pnl_tick_oos.mean() / pnl_tick_oos.std() * math.sqrt(bars_per_year_oos))
    cumulative = pnl_tick_oos.cumsum()
    drawdown   = (cumulative - cumulative.cummax()) / capital * 100
    mdd     = float(drawdown.min())
    total   = float(pnl_tick_oos.sum() / capital)
    cagr    = (1 + total) ** (1 / years_oos) - 1

    print()
    print("OOS performance under tick-level fills:")
    print(f"  Sharpe              = {sharpe_tick:+.3f}   (paper headline {sharpe_param:+.3f})")
    print(f"  CAGR                = {cagr*100:+.2f}%")
    print(f"  Max drawdown        = {mdd:+.2f}%")
    print(f"  Sharpe change       = {sharpe_tick - sharpe_param:+.3f}")

    out_csv = ROOT / "results" / "tick_fill_sim.csv"
    summary = pd.DataFrame([{
        "n_legs_scanned":   len(half_spreads),
        "median_half_tk":   float(np.median(half_spreads)),
        "mean_half_tk":     float(half_spreads.mean()),
        "p90_half_tk":      float(np.percentile(half_spreads, 90)),
        "p95_half_tk":      float(np.percentile(half_spreads, 95)),
        "p99_half_tk":      float(np.percentile(half_spreads, 99)),
        "sweep_rate":       float(sweeps.mean()),
        "sharpe_paper":     sharpe_param,
        "sharpe_tick_sim":  sharpe_tick,
        "cagr_tick_sim":    cagr * 100,
        "mdd_tick_sim":     mdd,
    }])
    summary.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
