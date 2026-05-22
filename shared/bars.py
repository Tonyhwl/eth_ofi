# dollar-volume bar construction from a minute panel.

import numpy as np
import pandas as pd


def build_volume_bars(panel, target, is_start, is_end):
    """Aggregate a minute panel into dollar-volume bars at about target per day."""

    df = panel.copy()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    df["dollar_volume"] = (df["buy_dollar"] + df["sell_dollar"]).fillna(0)

    # calibrate the bar size on in-sample days only, so out-of-sample volume cannot leak in
    daily_dollar_volume = df["dollar_volume"].resample("1D").sum()
    is_days = daily_dollar_volume[(daily_dollar_volume.index >= is_start)
                                  & (daily_dollar_volume.index <= is_end)
                                  & (daily_dollar_volume > 0)]
    bar_threshold = float(is_days.median() / target)

    bars = []
    for sym, contract_minutes in df.groupby("front_sym", sort=False):
        contract_minutes = contract_minutes.sort_index()
        cumulative_volume = contract_minutes["dollar_volume"].cumsum()
        bar_id = (cumulative_volume // bar_threshold).astype(int)

        agg = contract_minutes.assign(bar_id=bar_id).groupby("bar_id").agg(
            timestamp   = ("mid_close",   lambda s: s.index[-1]),
            ofi         = ("ofi",         "sum"),
            mid_open    = ("mid_open",    "first"),
            mid_close   = ("mid_close",   "last"),
            buy_dollar  = ("buy_dollar",  "sum"),
            sell_dollar = ("sell_dollar", "sum"),
            n_minutes   = ("ofi",         "count"),
        )
        agg["front_sym"] = sym
        bars.append(agg)

    volume_bars = pd.concat(bars).reset_index(drop=True)
    volume_bars = volume_bars.set_index("timestamp").sort_index()

    # a roll bar joins two contracts, so its price change is not a real return
    volume_bars["roll"] = (volume_bars["front_sym"] != volume_bars["front_sym"].shift(1)).fillna(True)
    volume_bars["mid_difference"] = volume_bars["mid_close"].diff()
    volume_bars.loc[volume_bars["roll"], "mid_difference"] = np.nan
    volume_bars["ret"] = volume_bars["mid_close"].pct_change()
    volume_bars.loc[volume_bars["roll"], "ret"] = np.nan

    return volume_bars
