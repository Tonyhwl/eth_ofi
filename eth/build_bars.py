# dollar-volume bars from the minute panel, threshold from IS median daily dv

from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
panel_min = root / "results" / "eth_ofi_minute.parquet"
vbars_path = root / "results" / "eth_ofi_vbars_isfix.parquet"
oos_start = pd.Timestamp("2024-01-01", tz="US/Eastern")


def rebuild_vbars_is_only(target_bars_per_day=10):
    df = pd.read_parquet(panel_min).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    df["dv"] = (df["buy_dollar"] + df["sell_dollar"]).fillna(0)

    is_mask = df.index < oos_start
    daily_dv_is = df.loc[is_mask, "dv"].resample("1D").sum()
    bar_threshold = float(daily_dv_is.median() / target_bars_per_day)
    print(f"  IS-only bar threshold: ${bar_threshold:,.0f} / bar  "
          f"(IS median dv = {daily_dv_is.median():,.0f})")

    bars = []
    for sym, sub in df.groupby("front_sym", sort=False):
        sub = sub.sort_index()
        cumulative_dv = sub["dv"].cumsum()
        bar_id = (cumulative_dv // bar_threshold).astype(int)
        agg = sub.assign(bar_id=bar_id).groupby("bar_id").agg(
            ts=("mid_close", lambda s: s.index[-1]),
            ofi=("ofi", "sum"),
            mid_open=("mid_open", "first"),
            mid_close=("mid_close", "last"),
            buy_dollar=("buy_dollar", "sum"),
            sell_dollar=("sell_dollar", "sum"),
            n_min=("ofi", "count"),
        )
        agg["front_sym"] = sym
        bars.append(agg)
    vbars = pd.concat(bars).reset_index(drop=True).set_index("ts").sort_index()
    vbars["roll"] = (vbars["front_sym"] != vbars["front_sym"].shift(1)).fillna(True)
    vbars["d_mid"] = vbars["mid_close"].diff()
    vbars.loc[vbars["roll"], "d_mid"] = np.nan
    vbars["ret"] = vbars["mid_close"].pct_change()
    vbars.loc[vbars["roll"], "ret"] = np.nan
    vbars.to_parquet(vbars_path)
    print(f"  rebuilt vbars: {len(vbars):,} bars  -> {vbars_path.name}")
    return vbars


if __name__ == "__main__":
    rebuild_vbars_is_only()
