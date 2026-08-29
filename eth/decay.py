# forward price-impact decay after OFI signal events. an event is the first
# upward crossing of |z| >= 1 on the dollar-volume bars (a timestamp and a
# direction only - no trading state machine is involved). t=0 is anchored at
# the exact instant the event bar becomes knowable: the trade at which the
# contract's cumulative dollar volume crosses the bar threshold.

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import databento as db

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategy import (panel_path, oos_start, signal_bars, entry_threshold,
                      zscore_lookback)

ROOT = Path(__file__).resolve().parent
TBBO_DIR = ROOT.parent / "data" / "eth_tbbo"
MINUTE_PATH = ROOT / "results" / "eth_ofi_minute.parquet"

HORIZONS_MIN = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120,
                180, 240, 360, 480, 720, 1080, 1440]   # out to 24h
MAX_STALE_SEC = 120     # reject if matched BBO is more than 2 min stale
N_SHUFFLE_SEEDS = 50    # seeds for the sign-randomised placebo baseline
N_BOOT = 2000           # day-block bootstrap resamples
TARGET_BARS_PER_DAY = 10


def load_day_for_symbol(utc_date, symbol):
    """One UTC day of TBBO for one contract: quotes, trades, sizes."""
    path = TBBO_DIR / f"glbx-mdp3-{utc_date.strftime('%Y%m%d')}.tbbo.dbn.zst"
    if not path.exists():
        return None
    try:
        df = db.DBNStore.from_file(str(path)).to_df()
    except Exception:
        return None
    if df.empty or "symbol" not in df.columns:
        return None
    df.index = pd.DatetimeIndex(df.index).tz_convert("US/Eastern")
    sub = df[df["symbol"].astype(str) == symbol]
    if sub.empty:
        return None
    bid = sub["bid_px_00"].astype(float).values
    ask = sub["ask_px_00"].astype(float).values
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    is_trade = (sub.get("action", pd.Series(["N"] * len(sub))).astype(str).values == "T")
    side = sub.get("side", pd.Series(["N"] * len(sub))).astype(str).values
    px = sub.get("price", pd.Series(np.full(len(sub), np.nan))).astype(float).values
    sz = sub.get("size", pd.Series(np.zeros(len(sub)))).astype(float).values
    out = pd.DataFrame({"bid": bid, "ask": ask, "is_trade": is_trade,
                        "side": side, "px": px, "sz": sz}, index=sub.index)
    return out[valid].sort_index()


def load_window(utc_date, symbol, n_days=2):
    frames = []
    for k in range(n_days):
        f = load_day_for_symbol(utc_date + pd.Timedelta(days=k), symbol)
        if f is not None and len(f):
            frames.append(f)
    if not frames:
        return None
    return pd.concat(frames).sort_index()


def first_crossing_events(bars):
    """First upward crossings of |z| >= entry_threshold; timestamp + direction."""
    cum = bars["ofi"].rolling(signal_bars, min_periods=signal_bars).sum()
    mean = cum.rolling(zscore_lookback, min_periods=zscore_lookback).mean().shift(1)
    std = cum.rolling(zscore_lookback, min_periods=zscore_lookback).std(ddof=1).shift(1)
    z = (cum - mean) / std
    above = z.abs() >= entry_threshold
    prev_above = above.shift(1).fillna(False)
    cross = above & ~prev_above & z.notna()
    ev = bars.loc[cross, ["front_sym"]].copy()
    ev["direction"] = np.sign(z.loc[cross]).astype(int)
    return ev


def main():
    print("=" * 78)
    print("ETH OFI - forward impact decay from first-crossing signal events")
    print("=" * 78)

    bars = pd.read_parquet(panel_path).sort_index()
    bars["roll"] = (bars["front_sym"] != bars["front_sym"].shift(1)).fillna(True)
    events = first_crossing_events(bars)
    events = events.loc[events.index >= oos_start]
    print(f"OOS first-crossing events: {len(events):,}")

    # threshold and per-contract cumulative dollar volume, exactly as robust.py
    mp = pd.read_parquet(MINUTE_PATH).sort_index()
    mp["dv"] = (mp["buy_dollar"] + mp["sell_dollar"]).fillna(0)
    thr = float(mp.loc[mp.index < oos_start, "dv"].resample("1D").sum().median()
                / TARGET_BARS_PER_DAY)
    print(f"bar threshold: ${thr:,.2f}")
    cum_by_sym = {s: g["dv"].cumsum() for s, g in mp.groupby("front_sym", sort=False)}
    last_minute = {s: g.index.max() for s, g in mp.groupby("front_sym", sort=False)}

    n_h = len(HORIZONS_MIN)
    n_ev = len(events)
    returns = np.full((n_ev, n_h), np.nan)     # unsigned bp, signed later
    spreads_bp = np.full(n_ev, np.nan)
    stale_secs = []
    dropped = {"no_data": 0, "no_crossing": 0}

    # group events by (utc day, contract) so each file loads once
    by_key = {}
    ev_ts = events.index
    for i in range(n_ev):
        key = (ev_ts[i].tz_convert("UTC").date(), events["front_sym"].iloc[i])
        by_key.setdefault(key, []).append(i)

    for j, ((day, sym), idxs) in enumerate(sorted(by_key.items())):
        tb = load_window(pd.Timestamp(day), sym)
        if tb is None:
            dropped["no_data"] += len(idxs)
            continue
        idx = tb.index
        tr_mask = tb["is_trade"].values & np.isin(tb["side"].values, ["B", "A"])
        tr_dollars = np.where(tr_mask, tb["px"].values * tb["sz"].values, 0.0)
        for i in idxs:
            label = ev_ts[i]
            cum = cum_by_sym[sym]
            if label not in cum.index:
                dropped["no_data"] += 1
                continue
            k = math.floor(cum.loc[label] / thr)
            needed = (k + 1) * thr - float(cum.loc[label])
            # accumulate trade dollars from the start of the NEXT minute until
            # the bar threshold is crossed; that trade is the anchor instant
            start = idx.searchsorted(label + pd.Timedelta(seconds=60), side="left")
            stop_ts = last_minute[sym] + pd.Timedelta(seconds=60)
            acc = 0.0
            pos_cross = -1
            for p in range(start, len(idx)):
                if idx[p] >= stop_ts:
                    break
                acc += tr_dollars[p]
                if acc >= needed - 1e-6:
                    pos_cross = p
                    break
            if pos_cross < 0:
                dropped["no_crossing"] += 1
                continue
            row0 = tb.iloc[pos_cross]
            mid0 = (row0["bid"] + row0["ask"]) / 2
            t0 = idx[pos_cross]
            spreads_bp[i] = (row0["ask"] - row0["bid"]) / mid0 * 1e4
            for h_i, h in enumerate(HORIZONS_MIN):
                target = t0 + pd.Timedelta(seconds=int(h * 60))
                q = idx.searchsorted(target, side="right") - 1
                if q < 0:
                    continue
                stale = (target - idx[q]).total_seconds()
                if stale > MAX_STALE_SEC:
                    continue
                rq = tb.iloc[q]
                mid_h = (rq["bid"] + rq["ask"]) / 2
                returns[i, h_i] = (mid_h - mid0) / mid0 * 1e4
                stale_secs.append(stale)
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(by_key)} day-contract groups", flush=True)

    print(f"\ndropped: {dropped}   anchored events: "
          f"{int(np.isfinite(spreads_bp).sum()):,} of {n_ev:,}")

    directions = events["direction"].values.astype(float)
    signed = returns * directions[:, None]
    ev_days = np.array([ts.date() for ts in ev_ts])

    # sign-randomised placebo
    rng = np.random.default_rng(0)
    plc = np.zeros((N_SHUFFLE_SEEDS, n_h))
    for s in range(N_SHUFFLE_SEEDS):
        flips = rng.choice([-1.0, 1.0], size=n_ev)
        with np.errstate(invalid="ignore"):
            plc[s] = np.nanmean(returns * flips[:, None], axis=0)
    placebo_mean, placebo_sd = plc.mean(axis=0), plc.std(axis=0, ddof=1)

    # by-day block bootstrap, pooled and common-sample
    common = np.all(np.isfinite(signed), axis=1)
    print(f"common-sample events (measurable at all {n_h} horizons): {common.sum():,}")
    unique_days = sorted(set(ev_days.tolist()))
    rows_by_day = [np.where(ev_days == d)[0] for d in unique_days]
    rng_b = np.random.default_rng(7)
    boot = np.full((N_BOOT, n_h), np.nan)
    boot_cs = np.full((N_BOOT, n_h), np.nan)
    for b in range(N_BOOT):
        sample_days = rng_b.integers(0, len(unique_days), size=len(unique_days))
        rows = np.concatenate([rows_by_day[d] for d in sample_days])
        with np.errstate(invalid="ignore"):
            boot[b] = np.nanmean(signed[rows], axis=0)
            cs_rows = rows[common[rows]]
            if len(cs_rows):
                boot_cs[b] = np.nanmean(signed[cs_rows], axis=0)

    def non_overlap_t(col, horizon_min):
        gap = pd.Timedelta(minutes=horizon_min)
        last, vals = None, []
        order = np.argsort(ev_ts.values)
        for k in order:
            if not np.isfinite(col[k]):
                continue
            if last is None or (ev_ts[k] - last) >= gap:
                vals.append(col[k]); last = ev_ts[k]
        vals = np.array(vals)
        if len(vals) < 10:
            return float("nan"), len(vals)
        se = vals.std(ddof=1) / math.sqrt(len(vals))
        return (float(vals.mean() / se) if se > 0 else 0.0), len(vals)

    out_rows = []
    for h_i, h in enumerate(HORIZONS_MIN):
        col = signed[:, h_i]
        fin = col[np.isfinite(col)]
        if not len(fin):
            continue
        mean = float(fin.mean())
        bcol = boot[:, h_i][np.isfinite(boot[:, h_i])]
        se_b = float(bcol.std(ddof=1)) if len(bcol) > 1 else float("nan")
        cs_col = signed[common, h_i]
        bcs = boot_cs[:, h_i][np.isfinite(boot_cs[:, h_i])]
        cs_se = float(bcs.std(ddof=1)) if len(bcs) > 1 else float("nan")
        t_no, n_no = non_overlap_t(col, h)
        out_rows.append({
            "horizon_min": h,
            "ofi_mean_bp": round(mean, 3),
            "ofi_se_bp": round(se_b, 3),
            "t_blockboot": round(mean / se_b, 2) if se_b and se_b > 0 else 0,
            "ci_lo_bp": round(float(np.percentile(bcol, 2.5)), 3) if len(bcol) else np.nan,
            "ci_hi_bp": round(float(np.percentile(bcol, 97.5)), 3) if len(bcol) else np.nan,
            "n": int(len(fin)),
            "cs_mean_bp": round(float(cs_col.mean()), 3) if common.sum() else np.nan,
            "cs_t": round(float(cs_col.mean() / cs_se), 2) if cs_se and cs_se > 0 else np.nan,
            "cs_n": int(common.sum()),
            "t_nonoverlap": round(t_no, 2),
            "n_noov": n_no,
            "placebo_bp": round(float(placebo_mean[h_i]), 3),
            "placebo_sd": round(float(placebo_sd[h_i]), 3),
        })

    out = pd.DataFrame(out_rows)
    print("\npooled vs common-sample profile:")
    print(out[["horizon_min", "ofi_mean_bp", "t_blockboot", "n",
               "cs_mean_bp", "cs_t", "cs_n"]].to_string(index=False))

    sp = spreads_bp[np.isfinite(spreads_bp)]
    st = np.array(stale_secs)
    print(f"\nquoted spread at the event instant (bp, full): "
          f"median {np.median(sp):.2f}  IQR [{np.percentile(sp,25):.2f}, {np.percentile(sp,75):.2f}]")
    print(f"matched-quote staleness at horizons (s): median {np.median(st):.1f}  "
          f"p90 {np.percentile(st,90):.1f}")

    out_csv = ROOT / "results" / "decay_profile.csv"
    out.to_csv(out_csv, index=False)
    meta = {"events_oos": int(n_ev), "anchored": int(np.isfinite(spreads_bp).sum()),
            "common_sample": int(common.sum()), "threshold": thr,
            "spread_bp_median": float(np.median(sp)),
            "spread_bp_iqr_lo": float(np.percentile(sp, 25)),
            "spread_bp_iqr_hi": float(np.percentile(sp, 75)),
            "staleness_median_s": float(np.median(st))}
    pd.Series(meta).to_json(ROOT / "results" / "decay_meta.json")
    print(f"\nwrote {out_csv} and decay_meta.json")


if __name__ == "__main__":
    main()
