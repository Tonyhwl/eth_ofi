# robustness of the decay profile to the event-definition parameters: repeat
# the first-crossing measurement for alternative K and |z| thresholds, with
# forward mids taken from the minute panel (minute resolution is sufficient
# for a relative comparison across configurations).

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategy import panel_path, oos_start, zscore_lookback

ROOT = Path(__file__).resolve().parent
MINUTE_PATH = ROOT / "results" / "eth_ofi_minute.parquet"
CONFIGS = [(3, 1.0), (5, 1.0), (10, 1.0), (5, 1.5), (5, 2.0)]
HORIZONS = [1, 10, 30, 60, 240, 480]          # minutes
STALE_MIN = 2


def events_for(bars, k_bars, z_thr):
    cum = bars["ofi"].rolling(k_bars, min_periods=k_bars).sum()
    mean = cum.rolling(zscore_lookback, min_periods=zscore_lookback).mean().shift(1)
    std = cum.rolling(zscore_lookback, min_periods=zscore_lookback).std(ddof=1).shift(1)
    z = (cum - mean) / std
    above = z.abs() >= z_thr
    cross = above & ~above.shift(1).fillna(False) & z.notna()
    ev = bars.loc[cross]
    return ev.index, np.sign(z.loc[cross]).values


def main():
    bars = pd.read_parquet(panel_path).sort_index()
    mp = pd.read_parquet(MINUTE_PATH).sort_index()
    mids = mp["mid_close"]
    midx = mids.index

    print(f"{'K':>3} {'|z|':>4} {'events':>7} | " +
          " ".join(f"{h:>6}m" for h in HORIZONS) + "   (mean signed bp)")
    for k_bars, z_thr in CONFIGS:
        ts, dirs = events_for(bars, k_bars, z_thr)
        keep = ts >= oos_start
        ts, dirs = ts[keep], dirs[keep]
        sums = {h: [] for h in HORIZONS}
        for t, d in zip(ts, dirs):
            p0 = midx.searchsorted(t + pd.Timedelta(minutes=1), side="left")
            if p0 >= len(midx) or (midx[p0] - t).total_seconds() > 180:
                continue
            m0 = mids.iloc[p0]
            t0 = midx[p0]
            for h in HORIZONS:
                q = midx.searchsorted(t0 + pd.Timedelta(minutes=h), side="right") - 1
                if q < 0:
                    continue
                if (t0 + pd.Timedelta(minutes=h) - midx[q]).total_seconds() > STALE_MIN * 60:
                    continue
                sums[h].append(d * (mids.iloc[q] - m0) / m0 * 1e4)
        line = " ".join(f"{np.mean(sums[h]):>+7.2f}" if sums[h] else "     --"
                        for h in HORIZONS)
        print(f"{k_bars:>3} {z_thr:>4.1f} {len(ts):>7,} | {line}")


if __name__ == "__main__":
    main()
