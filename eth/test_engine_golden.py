# golden replay: the streaming OFIEngine must reproduce strategy.simulate exactly
# on the locked config (filters off, 10-minute cap). this binds the live engine to
# the published backtest, so any forward divergence is the market, not the code.

import math

import pandas as pd

from strategy import (
    simulate, pnl_from_trades, compute_metrics, vol_target_contracts,
    build_event_calendar, panel_path, oos_start,
)
from engine import simulate_streaming

CAP = 10 / 60   # locked 10-minute clock cap


def trade_key(t):
    return (t.entry_idx, t.exit_idx, t.direction, t.size, t.exit_reason)


def main():
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    n_contracts = vol_target_contracts(df, len(df) / years)
    events = build_event_calendar()

    batch = simulate(df, n_contracts, False, False, cap_hours=CAP, events=events)
    stream = simulate_streaming(df, n_contracts, False, False, cap_hours=CAP, events=events)

    print(f"batch trades:  {len(batch):,}")
    print(f"stream trades: {len(stream):,}")

    assert len(batch) == len(stream), "trade COUNT differs"
    mismatches = [(a, b) for a, b in zip(batch, stream) if trade_key(a) != trade_key(b)]
    assert not mismatches, f"{len(mismatches)} trade(s) differ; first: {trade_key(mismatches[0][0])} vs {trade_key(mismatches[0][1])}"

    # OOS headline must reproduce: 4,531 trades, +5.11 Sharpe
    df_oos = df.loc[df.index >= oos_start]
    n_oos = len([t for t in stream if t.entry_time >= oos_start])
    pnl_batch = pnl_from_trades(batch, df, "realistic", cap_hours=CAP).loc[df_oos.index]
    pnl_stream = pnl_from_trades(stream, df, "realistic", cap_hours=CAP).loc[df_oos.index]
    sharpe_batch = compute_metrics(pnl_batch)["sharpe"]
    sharpe_stream = compute_metrics(pnl_stream)["sharpe"]

    print(f"OOS trades:    {n_oos:,}")
    print(f"OOS Sharpe:    batch {sharpe_batch:+.3f}  /  stream {sharpe_stream:+.3f}")

    assert n_oos == 4531, f"OOS trade count {n_oos} != 4531"
    assert abs(sharpe_stream - sharpe_batch) < 1e-9, "OOS Sharpe differs"
    assert abs(sharpe_stream - 5.11) < 0.02, f"OOS Sharpe {sharpe_stream} not ~+5.11"

    print("\nGOLDEN REPLAY PASSED: streaming engine == strategy.simulate, headline reproduced.")


if __name__ == "__main__":
    main()
