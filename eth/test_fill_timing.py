# guards against the fill-timing error: every reconstructed fill must be priced
# at a market event the strategy could actually have acted on. a bar's mid_close
# comes from the last event inside its final minute, and that is only knowable
# once the minute has elapsed, so a bar-boundary fill must occur at or after
# label + 60s. clock-cap exits fire on the wall clock and are exempt.

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategy import (
    simulate, vol_target_contracts, build_event_calendar, oos_start, panel_path,
)
from fill_sim import load_tbbo_day, decision_pos, leg_time, CAP

SAMPLE_EVERY = 7      # sample days; the invariant is structural, not statistical


def main():
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    trades = simulate(df, vol_target_contracts(df), False, False,
                      cap_hours=CAP, events=build_event_calendar())
    oos = [t for t in trades if t.entry_time >= oos_start]

    by_day = defaultdict(list)
    for t in oos:
        for tag in ("entry", "exit"):
            ts, on_bar = leg_time(t, tag, CAP)
            by_day[ts.tz_convert("UTC").date()].append((ts, on_bar))

    checked = early = 0
    worst = pd.Timedelta(0)
    for day in sorted(by_day)[::SAMPLE_EVERY]:
        tbbo = load_tbbo_day(day)
        if tbbo is None or len(tbbo) == 0:
            continue
        for ts, on_bar in by_day[day]:
            if not on_bar:                      # clock-cap exit: exempt
                continue
            pos = decision_pos(tbbo, ts, True)
            if pos < 0:
                continue
            checked += 1
            knowable = ts + pd.Timedelta(seconds=60)
            if tbbo.index[pos] < knowable:
                early += 1
                worst = max(worst, knowable - tbbo.index[pos])

    print(f"bar-boundary fills checked: {checked:,}")
    print(f"priced before the bar was knowable: {early:,}")
    if early:
        print(f"worst case: {worst.total_seconds():.1f}s early")
    assert early == 0, f"{early} of {checked} fills priced before the decision was possible"
    print("\nFILL TIMING PASSED: no fill is priced earlier than the strategy could act.")


if __name__ == "__main__":
    main()
