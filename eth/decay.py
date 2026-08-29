# sub-bar OFI signal decay: for each OOS entry, walk forward in TBBO and measure
# mean signed cumulative mid-quote change at fixed horizons out to 24h.

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
from fill_sim import load_tbbo_day

ROOT = Path(__file__).resolve().parent
HORIZONS_MIN = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120,
                180, 240, 360, 480, 720, 1080, 1440]  # out to 24h
MAX_STALE_SEC = 120     # reject if matched BBO is more than 2 min stale
N_SHUFFLE_SEEDS = 50    # seeds for the sign-randomised placebo baseline
N_BOOT = 2000           # day-block bootstrap resamples for honest decay SEs


def decision_bbo(tbbo_df, ts):
    """BBO at the last event inside the bar's final minute, i.e. the decision instant."""
    if tbbo_df is None or len(tbbo_df) == 0:
        return None
    pos = tbbo_df.index.searchsorted(ts + pd.Timedelta(seconds=60), side="left") - 1
    if pos < 0 or tbbo_df.index[pos] < ts:
        return None
    row = tbbo_df.iloc[pos]
    return {"bid": float(row["bid"]), "ask": float(row["ask"]),
            "ts": tbbo_df.index[pos]}


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


def non_overlap_stats(signed_col, entry_ts, horizon_min):
    """Mean and t on a non-overlapping subset: keep entries at least horizon_min apart."""
    gap = pd.Timedelta(minutes=horizon_min)
    last = None
    vals = []
    for k in np.argsort(entry_ts.values):
        if not np.isfinite(signed_col[k]):
            continue
        ts = entry_ts[k]
        if last is None or (ts - last) >= gap:
            vals.append(float(signed_col[k]))
            last = ts
    vals = np.array(vals)
    if len(vals) < 10:
        return float("nan"), float("nan"), len(vals)
    se = vals.std(ddof=1) / math.sqrt(len(vals))
    return float(vals.mean()), (float(vals.mean() / se) if se > 0 else 0.0), len(vals)


def main():
    print("=" * 78)
    print("ETH OFI - signal decay profile, sub-bar to 24h, with placebo")
    print("=" * 78)
    print("loading panel and running strategy ...")
    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    bars_per_year = len(df) / years
    n_contracts = vol_target_contracts(df)
    events = build_event_calendar()
    trades = simulate(df, n_contracts, use_event_filter=False, use_weekend_filter=False,
                      cap_hours=10/60, events=events)
    oos = [t for t in trades if t.entry_time >= oos_start]
    print(f"  {len(oos):,} OOS trades")

    # pre-extract entry data so we can shuffle directions without re-running
    entries = [(t.entry_time, t.direction) for t in oos]
    directions = np.array([d for _, d in entries], dtype=int)
    entry_ts = pd.DatetimeIndex([ts for ts, _ in entries])
    entry_days = np.array([ts.date() for ts, _ in entries])

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
            bbo0 = decision_bbo(tbbo, ts_entry)
            if bbo0 is None:
                continue
            mid0 = (bbo0["bid"] + bbo0["ask"]) / 2
            t_zero = bbo0["ts"]
            for j, h in enumerate(HORIZONS_MIN):
                ts = t_zero + pd.Timedelta(seconds=int(h * 60))
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

    # block bootstrap by trading day so overlapping forward windows stay bundled
    unique_days = sorted(set(entry_days.tolist()))
    day_rows = [np.where(entry_days == day)[0] for day in unique_days]
    rng_boot = np.random.default_rng(7)
    boot_means = np.full((N_BOOT, n_horizons), np.nan)
    for b in range(N_BOOT):
        sampled_days = rng_boot.integers(0, len(unique_days), size=len(unique_days))
        sampled_rows = np.concatenate([day_rows[day_i] for day_i in sampled_days])
        with np.errstate(invalid="ignore"):
            boot_means[b] = np.nanmean(signed_ofi[sampled_rows], axis=0)

    rows = []
    for j, h in enumerate(HORIZONS_MIN):
        col = signed_ofi[:, j]
        finite_returns = col[np.isfinite(col)]
        if len(finite_returns) == 0:
            continue
        mean = float(finite_returns.mean())
        se_crosssec = float(finite_returns.std(ddof=1) / math.sqrt(len(finite_returns)))
        boot_at_h = boot_means[:, j][np.isfinite(boot_means[:, j])]
        se_boot = float(boot_at_h.std(ddof=1)) if len(boot_at_h) > 1 else float("nan")
        ci_lo = float(np.percentile(boot_at_h, 2.5)) if len(boot_at_h) else float("nan")
        ci_hi = float(np.percentile(boot_at_h, 97.5)) if len(boot_at_h) else float("nan")
        _, t_nonoverlap, n_nonoverlap = non_overlap_stats(col, entry_ts, h)
        rows.append({
            "horizon_min":  h,
            "ofi_mean_bp":  round(mean, 3),
            "ofi_se_bp":    round(se_boot, 3),
            "t_crosssec":   round(mean / se_crosssec, 2) if se_crosssec > 0 else 0,
            "t_blockboot":  round(mean / se_boot, 2) if se_boot > 0 else 0,
            "t_nonoverlap": round(t_nonoverlap, 2),
            "ci_lo_bp":     round(ci_lo, 3),
            "ci_hi_bp":     round(ci_hi, 3),
            "n":            int(len(finite_returns)),
            "n_noov":       n_nonoverlap,
            "placebo_bp":   round(float(placebo_mean[j]), 3),
            "placebo_sd":   round(float(placebo_sd[j]), 3),
        })

    out = pd.DataFrame(rows)
    print("Forward signed return: OFI direction vs sign-randomised placebo")
    print(out.to_string(index=False))

    out_csv = ROOT / "results" / "decay_profile.csv"
    out_csv.parent.mkdir(exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
