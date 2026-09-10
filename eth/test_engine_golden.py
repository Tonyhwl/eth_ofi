# golden replay: OFIEngine must match strategy.simulate on the pinned window
# guards history against drift when data is appended

import math

import pandas as pd

from strategy import (
    simulate, pnl_from_trades, compute_metrics, vol_target_contracts,
    build_event_calendar, panel_path, oos_start,
)
from engine import simulate_streaming

CAP = 10 / 60   # locked 10-minute clock cap

# last complete pre-append bar, final partial dollar bar excluded
GOLDEN_ASOF = pd.Timestamp("2026-04-27 19:43:00", tz="US/Eastern")

# frozen expectations on that window (sizing uses strategy.bars_per_year_sizing)
GOLDEN_BARS       = 37_953
GOLDEN_TRADES     = 6_858
GOLDEN_OOS_TRADES = 4_531
GOLDEN_OOS_SHARPE = 5.212


def trade_key(t):
    return (t.entry_idx, t.exit_idx, t.direction, t.size, t.exit_reason)


def main():
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    df = df.loc[df.index <= GOLDEN_ASOF]
    n_contracts = vol_target_contracts(df)
    events = build_event_calendar()

    batch = simulate(df, n_contracts, False, False, cap_hours=CAP, events=events)
    stream = simulate_streaming(df, n_contracts, False, False, cap_hours=CAP, events=events)

    print(f"pinned window: {len(df):,} bars through {GOLDEN_ASOF}")
    print(f"batch trades:  {len(batch):,}")
    print(f"stream trades: {len(stream):,}")

    assert len(df) == GOLDEN_BARS, f"panel prefix {len(df)} != {GOLDEN_BARS} (history changed)"
    assert len(batch) == len(stream), "trade COUNT differs between batch and streaming"
    mismatches = [(a, b) for a, b in zip(batch, stream) if trade_key(a) != trade_key(b)]
    assert not mismatches, f"{len(mismatches)} trade(s) differ; first: {trade_key(mismatches[0][0])}"
    assert len(batch) == GOLDEN_TRADES, f"total trades {len(batch)} != {GOLDEN_TRADES}"

    df_oos = df.loc[df.index >= oos_start]
    n_oos = len([t for t in stream if t.entry_time >= oos_start])
    sharpe_batch = compute_metrics(pnl_from_trades(batch, df, "realistic", cap_hours=CAP).loc[df_oos.index])["sharpe"]
    sharpe_stream = compute_metrics(pnl_from_trades(stream, df, "realistic", cap_hours=CAP).loc[df_oos.index])["sharpe"]

    print(f"OOS trades:    {n_oos:,}")
    print(f"OOS Sharpe:    batch {sharpe_batch:+.4f}  /  stream {sharpe_stream:+.4f}")

    assert n_oos == GOLDEN_OOS_TRADES, f"OOS trades {n_oos} != {GOLDEN_OOS_TRADES}"
    assert abs(sharpe_stream - sharpe_batch) < 1e-9, "batch and streaming Sharpe differ"
    assert abs(sharpe_stream - GOLDEN_OOS_SHARPE) < 5e-4, \
        f"OOS Sharpe {sharpe_stream:.4f} != {GOLDEN_OOS_SHARPE} (history moved)"

    print("\nGOLDEN REPLAY PASSED: engine == strategy.simulate, pinned history reproduced.")


if __name__ == "__main__":
    main()
