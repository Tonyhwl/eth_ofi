# empirical fill simulation from raw TBBO under three execution variants:
# aggressive (walk the book), passive (rest at touch, 60s queue), staggered (5 child orders).

import sys, math, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import databento as db

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from strategy import (
    simulate, vol_target_contracts, pnl_from_trades, build_event_calendar,
    oos_start, panel_path, contract_mult, tick_size, capital,
)

TBBO_DIR = ROOT.parent / "data" / "eth_tbbo"

CAP = 10 / 60          # locked 10-minute clock cap

PASSIVE_WINDOW_SECS  = 60
STAGGER_N_CHILDREN   = 5
STAGGER_INTERVAL_SECS = 60

month_codes = "FGHJKMNQUVXZ"


def load_tbbo_day(date_obj):
    filename = f"glbx-mdp3-{date_obj.strftime('%Y%m%d')}.tbbo.dbn.zst"
    file_path = TBBO_DIR / filename
    if not file_path.exists():
        return None
    try:
        df = db.DBNStore.from_file(str(file_path)).to_df()
    except Exception:
        return None
    if df.empty or "symbol" not in df.columns:
        return None
    df.index = pd.DatetimeIndex(df.index).tz_convert("US/Eastern")
    outrights = df[~df["symbol"].astype(str).str.contains("-", na=False)].copy()
    if outrights.empty:
        return None
    best_symbol, best_days_to_expiry = None, 1e9
    for symbol in outrights["symbol"].unique():
        match = re.match(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d+)$", str(symbol))
        if not match: continue
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
    is_trade = (sub.get("action", pd.Series(["N"]*len(sub))).astype(str).values == "T")
    side     = sub.get("side", pd.Series(["N"]*len(sub))).astype(str).values
    price    = sub.get("price", pd.Series(np.full(len(sub), np.nan))).astype(float).values
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    return pd.DataFrame({
        "bid": bid[valid], "ask": ask[valid], "bid_sz": bid_size[valid], "ask_sz": ask_size[valid],
        "is_trade": is_trade[valid], "side": side[valid], "trade_px": price[valid],
    }, index=sub.index[valid]).sort_index()


def decision_pos(tbbo, ts, on_bar=True):
    """Index position of the book state a decision at ts could actually act on."""
    if not on_bar:                      # clock-cap exit: fires mid-bar at ts
        return tbbo.index.searchsorted(ts, side="right") - 1
    end = ts + pd.Timedelta(seconds=60)
    pos = tbbo.index.searchsorted(end, side="left") - 1
    if pos < 0 or tbbo.index[pos] < ts:
        return -1
    return pos


def leg_time(t, tag, cap_hours):
    """Timestamp a leg is actually executable: cap exits fire on the clock."""
    if tag == "exit" and t.exit_reason == "cap" and cap_hours is not None:
        return t.entry_time + pd.Timedelta(hours=cap_hours), False
    return (t.entry_time if tag == "entry" else t.exit_time), True


def aggressive_fill(tbbo, ts, direction, side, size, on_bar=True):
    """Market order that walks the book, using top-of-book size as a depth proxy."""
    pos = decision_pos(tbbo, ts, on_bar)
    if pos < 0:
        return None
    row = tbbo.iloc[pos]
    bid, ask = float(row["bid"]), float(row["ask"])
    mid = (bid + ask) / 2
    is_buy = (direction == +1 and side == "entry") or (direction == -1 and side == "exit")
    if is_buy:
        top_price, top_size = ask, max(1.0, float(row["ask_sz"]))
    else:
        top_price, top_size = bid, max(1.0, float(row["bid_sz"]))
    if size <= top_size:
        return float(top_price), float(mid)
    # walk the book one level deeper at +1 tick per level
    n_levels = int(math.ceil(size / top_size))
    walk_cost = 0.0
    remaining = size
    for level in range(n_levels):
        take = min(top_size, remaining)
        level_px = top_price + (level * tick_size if is_buy else -level * tick_size)
        walk_cost += take * level_px
        remaining -= take
    vwap = walk_cost / size
    return float(vwap), float(mid)


def passive_fill(tbbo, ts, direction, side, size, on_bar=True, window_secs=PASSIVE_WINDOW_SECS):
    """Post at the touch and fill only if a trade prints at our price within the window."""
    pos = decision_pos(tbbo, ts, on_bar)
    if pos < 0:
        return None
    row = tbbo.iloc[pos]
    bid, ask = float(row["bid"]), float(row["ask"])
    mid = (bid + ask) / 2
    is_buy = (direction == +1 and side == "entry") or (direction == -1 and side == "exit")
    post_price = bid if is_buy else ask
    decided = tbbo.index[pos]
    end_ts = decided + pd.Timedelta(seconds=window_secs)
    start_pos = pos + 1
    end_pos = tbbo.index.searchsorted(end_ts, side="right")
    window = tbbo.iloc[start_pos:end_pos]
    trades = window[window["is_trade"]]
    if len(trades) == 0:
        return None
    if is_buy:
        hit = trades[trades["trade_px"] <= post_price]
    else:
        hit = trades[trades["trade_px"] >= post_price]
    if len(hit) == 0:
        return None
    return float(post_price), float(mid)


def staggered_fill(tbbo, ts, direction, side, size, on_bar=True,
                    n_children=STAGGER_N_CHILDREN, interval_secs=STAGGER_INTERVAL_SECS):
    """Split into n_children equal-size aggressive fills one minute apart."""
    child_size = max(1, int(round(size / n_children)))
    if child_size * n_children < size:
        sizes = [child_size] * (n_children - 1) + [size - child_size * (n_children - 1)]
    else:
        sizes = [child_size] * n_children
    vwap_sum, mid_first = 0.0, None
    for k, sz in enumerate(sizes):
        child_ts = ts + pd.Timedelta(seconds=k * interval_secs)
        fill = aggressive_fill(tbbo, child_ts, direction, side, sz, on_bar and k == 0)
        if fill is None:
            return None
        if mid_first is None:
            mid_first = fill[1]
        vwap_sum += sz * fill[0]
    return float(vwap_sum / size), float(mid_first)


def trade_pnl(entry_px, exit_px, direction, size):
    return direction * size * contract_mult * (exit_px - entry_px)


def main():
    print("ETH OFI fill simulation: aggressive / passive / staggered\n")
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(df) / years
    n_contracts = vol_target_contracts(df)
    events = build_event_calendar()
    trades = simulate(df, n_contracts, use_event_filter=False, use_weekend_filter=False,
                       cap_hours=CAP, events=events)
    oos_trades = [t for t in trades if t.entry_time >= oos_start]
    print(f"  {len(oos_trades):,} OOS trades")

    df_oos = df.loc[df.index >= oos_start]
    years_oos = (df_oos.index.max() - df_oos.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year_oos = len(df_oos) / years_oos

    # group OOS trades by date so each day's TBBO loads once
    by_day = defaultdict(list)
    for t in oos_trades:
        by_day[leg_time(t, "entry", CAP)[0].tz_convert("UTC").date()].append(("entry", t))
        by_day[leg_time(t, "exit", CAP)[0].tz_convert("UTC").date()].append(("exit", t))

    fills = {"aggressive": {}, "passive": {}, "staggered": {}}
    n_passive_missed = 0
    n_days = 0
    for day, day_legs in sorted(by_day.items()):
        n_days += 1
        tbbo = load_tbbo_day(day)
        if tbbo is None:
            continue
        for tag, t in day_legs:
            ts, on_bar = leg_time(t, tag, CAP)
            fill_aggressive = aggressive_fill(tbbo, ts, t.direction, tag, t.size, on_bar)
            fill_passive = passive_fill(tbbo,    ts, t.direction, tag, t.size, on_bar)
            fill_staggered = staggered_fill(tbbo,  ts, t.direction, tag, t.size, on_bar)
            if fill_aggressive is not None: fills["aggressive"].setdefault(id(t), {})[tag] = fill_aggressive
            if fill_passive is not None: fills["passive"].setdefault(id(t), {})[tag] = fill_passive
            else:                n_passive_missed += 1
            if fill_staggered is not None: fills["staggered"].setdefault(id(t), {})[tag] = fill_staggered
        if n_days % 50 == 0:
            print(f"  {n_days}/{len(by_day)} days processed", flush=True)

    print(f"\n  passive missed legs: {n_passive_missed:,}")

    def metrics(label, fill_dict):
        pnl = np.zeros(len(df))
        n_used = 0
        for t in oos_trades:
            trade_fills = fill_dict.get(id(t), {})
            if "entry" not in trade_fills or "exit" not in trade_fills:
                continue
            entry_px, _ = trade_fills["entry"]
            exit_px, _ = trade_fills["exit"]
            pnl[t.exit_idx] += trade_pnl(entry_px, exit_px, t.direction, t.size)
            n_used += 1
        pnl_oos = pd.Series(pnl, index=df.index).loc[df_oos.index]
        returns = (pnl_oos / capital)
        if returns.std() == 0:
            return
        sharpe = float(returns.mean() / returns.std() * math.sqrt(bars_per_year_oos))
        cumulative = pnl_oos.cumsum()
        mdd = float(((cumulative - cumulative.cummax()) / capital * 100).min())
        total = float(pnl_oos.sum() / capital * 100)
        cagr = (1 + total/100) ** (1/years_oos) - 1
        print(f"  {label:<11}  trades={n_used:>5,d}  Sharpe={sharpe:+.2f}  "
              f"CAGR={cagr*100:+.1f}%  MaxDD={mdd:+.1f}%  Total={total:+.1f}%")

    print("\nOOS performance under each fill model:")
    metrics("aggressive", fills["aggressive"])
    metrics("passive",    fills["passive"])
    metrics("staggered",  fills["staggered"])


if __name__ == "__main__":
    main()
