# sub-bar OFI signal decay. for each OOS entry, walk forward in TBBO and
# measure mean cumulative mid-quote change at fixed horizons out to 24h,
# signed by trade direction. also runs a direction-shuffled placebo at
# the same entry times to check the result is not just market drift.

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategy import (
    simulate, vol_target_contracts, build_event_calendar,
    oos_start, panel_path,
)
from tick_fill_sim import load_tbbo_day

ROOT = Path(__file__).resolve().parent
HORIZONS_MIN = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120,
                180, 240, 360, 480, 720, 1080, 1440]  # out to 24h
MAX_STALE_SEC = 120     # reject if matched BBO is more than 2 min stale
N_SHUFFLE_SEEDS = 50    # placebo seeds for direction-shuffled baseline


def bbo_at_fresh(tbbo_df, ts):
    """BBO at or before ts, with matched timestamp."""
    if tbbo_df is None or len(tbbo_df) == 0:
        return None
    pos = tbbo_df.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    row = tbbo_df.iloc[pos]
    return {"bid": float(row["bid"]), "ask": float(row["ask"]),
            "ts": tbbo_df.index[pos]}


def load_window(d0, n_days):
    """Concatenate n_days of TBBO starting at d0, for long forward windows."""
    frames = []
    for k in range(n_days):
        d = d0 + pd.Timedelta(days=k)
        tbbo = load_tbbo_day(d.date() if hasattr(d, "date") else d)
        if tbbo is not None and len(tbbo) > 0:
            frames.append(tbbo)
    if not frames:
        return None
    return pd.concat(frames).sort_index()


def main():
    print("=" * 78)
    print("ETH OFI - signal decay profile, sub-bar to 24h, with placebo")
    print("=" * 78)
    print("loading panel and running strategy ...")
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(df) / years
    n_contracts = vol_target_contracts(df, bars_per_year)
    events = build_event_calendar()
    trades = simulate(df, n_contracts, use_event_filter=False, use_weekend_filter=False,
                      cap_hours=10/60, events=events)
    oos = [t for t in trades if t.entry_time >= oos_start]
    print(f"  {len(oos):,} OOS trades")

    # pre-extract entry data so we can shuffle directions without re-running
    entries = [(t.entry_time, t.direction) for t in oos]
    directions = np.array([d for _, d in entries], dtype=int)

    # group by entry day for cache; load 2 days of TBBO to cover 24h forward
    by_day = {}
    for index, (ts, _) in enumerate(entries):
        by_day.setdefault(ts.date(), []).append(index)

    n_horizons = len(HORIZONS_MIN)
    n_trades = len(entries)
    # per-trade signed bp return matrix (rows = trades, cols = horizons),
    # nan where measurement is missing (no fresh BBO at that horizon)
    returns = np.full((n_trades, n_horizons), np.nan)

    print("scanning TBBO (2 days per entry day to cover 24h forward) ...")
    for i, (day, day_indices) in enumerate(sorted(by_day.items())):
        tbbo = load_window(pd.Timestamp(day), n_days=2)
        if tbbo is None:
            continue
        for index in day_indices:
            ts_entry, _ = entries[index]
            bbo0 = bbo_at_fresh(tbbo, ts_entry)
            if bbo0 is None:
                continue
            mid0 = (bbo0["bid"] + bbo0["ask"]) / 2
            for j, h in enumerate(HORIZONS_MIN):
                ts = ts_entry + pd.Timedelta(seconds=int(h * 60))
                bbo_h = bbo_at_fresh(tbbo, ts)
                if bbo_h is None:
                    continue
                if (ts - bbo_h["ts"]).total_seconds() > MAX_STALE_SEC:
                    continue
                mid_h = (bbo_h["bid"] + bbo_h["ask"]) / 2
                # unsigned return; sign applied later for ofi vs placebo
                returns[index, j] = (mid_h - mid0) / mid0 * 1e4
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(by_day)} days "
                  f"(coverage at 24h horizon: "
                  f"{np.isfinite(returns[:, -1]).sum():,}/{n_trades:,})")

    print(f"\ndone scanning {len(by_day)} days\n")

    # ofi-signed forward return
    signed_ofi = returns * directions[:, None].astype(float)

    rng = np.random.default_rng(0)
    placebo_means = np.zeros((N_SHUFFLE_SEEDS, n_horizons))
    for s in range(N_SHUFFLE_SEEDS):
        flips = rng.choice([-1, 1], size=n_trades)
        signed_p = returns * flips[:, None].astype(float)
        with np.errstate(invalid="ignore"):
            placebo_means[s] = np.nanmean(signed_p, axis=0)
    placebo_mean = placebo_means.mean(axis=0)
    placebo_sd = placebo_means.std(axis=0, ddof=1)

    rows = []
    for j, h in enumerate(HORIZONS_MIN):
        col = signed_ofi[:, j]
        finite_returns = col[np.isfinite(col)]
        if len(finite_returns) == 0:
            continue
        mean = float(finite_returns.mean())
        se = float(finite_returns.std(ddof=1) / math.sqrt(len(finite_returns)))
        rows.append({
            "horizon_min":  h,
            "ofi_mean_bp":  round(mean, 3),
            "ofi_se_bp":    round(se, 3),
            "ofi_t":        round(mean / se, 2) if se > 0 else 0,
            "placebo_bp":   round(float(placebo_mean[j]), 3),
            "placebo_sd":   round(float(placebo_sd[j]), 3),
            "n":            int(len(finite_returns)),
        })

    out = pd.DataFrame(rows)
    print("Forward signed return: OFI direction vs direction-shuffled placebo")
    print(out.to_string(index=False))

    out_csv = ROOT / "results" / "decay_profile.csv"
    out_csv.parent.mkdir(exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
