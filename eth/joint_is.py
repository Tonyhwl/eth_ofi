# joint IS sweep over signal_bars x entry_z x max_hold x zlb x filters x cap.
# picks the best IS Sharpe and runs SPA and DSR over the full 1920 trial set.
# precomputes event and friday masks and reuses z-series.

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from strategy import (
    vol_target_contracts, pnl_from_trades, build_event_calendar,
    oos_start, panel_path, contract_mult, Trade,
    event_pre_min, event_post_min,
    friday_no_entry_min, friday_force_flat_min, friday_close_hour,
    direction_sign,
)
from robust import (
    _bars_per_year, _annualized_sharpe, hansen_spa_consistent, deflated_sharpe,
)

OUT_JSON = ROOT / "results" / "eth_ofi_joint_is.json"
OUT_CSV  = ROOT / "results" / "eth_ofi_joint_is.csv"

SIGNAL_BARS_GRID = [3, 5, 10, 20]
ENTRY_Z_GRID     = [1.0, 1.5, 2.0]
MAX_HOLD_GRID    = [1, 3, 5, 10]
ZLB_GRID         = [200, 500]
FILTER_GRID      = [(False, False), (True, False), (False, True), (True, True)]
CAP_GRID         = [1/60, 10/60, 30/60, 1.0, 3.0]

CAP_NORM = contract_mult * 2500.0


def build_event_mask(timestamps, events):
    """Boolean array: True on bars within [-event_pre_min, +event_post_min] of any event."""
    n_bars = len(timestamps)
    mask = np.zeros(n_bars, dtype=bool)
    if not events:
        return mask
    ts_secs = np.asarray([t.value for t in timestamps], dtype=np.int64)  # ns
    ev_secs = np.asarray([e.value for e in events],     dtype=np.int64)
    pre_ns  = int(event_pre_min  * 60 * 1_000_000_000)
    post_ns = int(event_post_min * 60 * 1_000_000_000)
    for e in ev_secs:
        mask |= (ts_secs >= e - pre_ns) & (ts_secs <= e + post_ns)
    return mask


def build_friday_mask(timestamps, mins_before):
    n_bars = len(timestamps)
    mask = np.zeros(n_bars, dtype=bool)
    for i, t in enumerate(timestamps):
        if t.weekday() != 4:
            continue
        close = t.replace(hour=friday_close_hour, minute=0, second=0, microsecond=0)
        delta_min = (close - t).total_seconds() / 60.0
        if 0 <= delta_min <= mins_before:
            mask[i] = True
    return mask


def precompute_z(df, signal_bars, zlb):
    cum_ofi  = df["ofi"].rolling(signal_bars, min_periods=signal_bars).sum()
    rolling_mean = cum_ofi.rolling(zlb, min_periods=zlb).mean().shift(1)
    rolling_std  = cum_ofi.rolling(zlb, min_periods=zlb).std(ddof=1).shift(1)
    return ((cum_ofi - rolling_mean) / rolling_std).values


def fast_simulate(timestamps, z_series, contracts_arr, contract_sym, n_bars,
                  entry_z, max_hold, cap_secs,
                  event_mask, friday_flat_mask, friday_no_entry_mask,
                  use_event, use_weekend):
    trades = []
    pos_dir = 0; pos_size = 0; bars_held = 0
    entry_time = None; entry_index = None
    prev_contract = None
    no_event   = not use_event
    no_weekend = not use_weekend

    for i in range(n_bars):
        ts = timestamps[i]
        z  = z_series[i]

        cap_fired   = (pos_dir != 0 and cap_secs is not None
                       and entry_time is not None
                       and (ts - entry_time).total_seconds() >= cap_secs)
        block_event = False if no_event   else event_mask[i]
        force_flat  = False if no_weekend else friday_flat_mask[i]
        block_entry = block_event or (False if no_weekend else friday_no_entry_mask[i])

        if cap_fired:
            trades.append(Trade(entry_time, entry_index, ts, i, pos_dir, pos_size, "cap"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_index = None
            prev_contract = contract_sym[i]; continue

        if pos_dir != 0 and block_event:
            trades.append(Trade(entry_time, entry_index, ts, i, pos_dir, pos_size, "event"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_index = None
            prev_contract = contract_sym[i]; continue

        if pos_dir != 0 and force_flat:
            trades.append(Trade(entry_time, entry_index, ts, i, pos_dir, pos_size, "weekend"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_index = None
            prev_contract = contract_sym[i]; continue

        if contract_sym[i] != prev_contract and prev_contract is not None and pos_dir != 0:
            trades.append(Trade(entry_time, entry_index, ts, i, pos_dir, pos_size, "roll"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_index = None

        if z != z:  # nan check
            prev_contract = contract_sym[i]; continue

        if pos_dir != 0 and bars_held >= max_hold:
            trades.append(Trade(entry_time, entry_index, ts, i, pos_dir, pos_size, "hold"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_index = None

        if pos_dir == +1 and z <= 0:
            trades.append(Trade(entry_time, entry_index, ts, i, pos_dir, pos_size, "zcross"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_index = None
        elif pos_dir == -1 and z >= 0:
            trades.append(Trade(entry_time, entry_index, ts, i, pos_dir, pos_size, "zcross"))
            pos_dir = 0; pos_size = 0; bars_held = 0; entry_time = None; entry_index = None

        if pos_dir == 0 and not block_entry:
            if z >= entry_z:
                pos_dir = +1 * direction_sign
                pos_size = max(1, int(round(contracts_arr[i])))
                bars_held = 1; entry_time = ts; entry_index = i
            elif z <= -entry_z:
                pos_dir = -1 * direction_sign
                pos_size = max(1, int(round(contracts_arr[i])))
                bars_held = 1; entry_time = ts; entry_index = i
        elif pos_dir != 0:
            bars_held += 1

        prev_contract = contract_sym[i]

    if pos_dir != 0:
        trades.append(Trade(entry_time, entry_index, timestamps[n_bars-1], n_bars-1,
                            pos_dir, pos_size, "end"))
    return trades


def main():
    print("ETH OFI joint IS sweep")
    n_total = (len(SIGNAL_BARS_GRID) * len(ENTRY_Z_GRID) * len(MAX_HOLD_GRID)
               * len(ZLB_GRID) * len(FILTER_GRID) * len(CAP_GRID))
    print(f"  {n_total} configurations", flush=True)

    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    df_is  = df.loc[df.index <  oos_start]
    df_oos = df.loc[df.index >= oos_start]
    bars_per_year_panel = _bars_per_year(df)
    bars_per_year_is    = _bars_per_year(df_is)
    bars_per_year_oos   = _bars_per_year(df_oos)
    events    = build_event_calendar()
    n_contracts = vol_target_contracts(df, bars_per_year_panel)
    is_index    = np.where(df.index <  oos_start)[0]
    oos_index   = np.where(df.index >= oos_start)[0]

    timestamps   = df.index
    contract_sym = df["front_sym"].values
    n_bars       = len(df)
    contracts_arr = n_contracts.fillna(0).values

    print(f"  panel: {n_bars:,} bars, IS {len(df_is):,}, OOS {len(df_oos):,}", flush=True)
    print(f"  bars/yr IS={bars_per_year_is:.0f}, OOS={bars_per_year_oos:.0f}", flush=True)

    print("  precomputing event / friday masks...", flush=True)
    t_pre = time.time()
    event_mask  = build_event_mask(timestamps, events)
    friday_flat_mask     = build_friday_mask(timestamps, friday_force_flat_min)
    friday_no_entry_mask = build_friday_mask(timestamps, friday_no_entry_min)
    print(f"    event mask: {event_mask.sum():,} bars blocked", flush=True)
    print(f"    friday flat mask: {friday_flat_mask.sum():,} bars", flush=True)
    print(f"    friday no-entry mask: {friday_no_entry_mask.sum():,} bars", flush=True)
    print(f"    precompute time: {time.time()-t_pre:.1f}s", flush=True)

    print("  precomputing z-score series for all (signal_bars, zlb) combos...", flush=True)
    t_pre = time.time()
    z_cache = {}
    for n_signal in SIGNAL_BARS_GRID:
        for zlb in ZLB_GRID:
            z_cache[(n_signal, zlb)] = precompute_z(df, n_signal, zlb)
    print(f"    {len(z_cache)} z-series cached in {time.time()-t_pre:.1f}s", flush=True)

    rows = []
    pnl_oos_cols = []
    sharpe_is_arr  = []
    sharpe_oos_arr = []
    t0 = time.time()
    i = 0

    for n_signal in SIGNAL_BARS_GRID:
        for zlb in ZLB_GRID:
            z_series = z_cache[(n_signal, zlb)]
            for ez in ENTRY_Z_GRID:
                for n_hold in MAX_HOLD_GRID:
                    for ev, wk in FILTER_GRID:
                        for cap_h in CAP_GRID:
                            i += 1
                            cap_secs = cap_h * 3600
                            trades = fast_simulate(
                                timestamps, z_series, contracts_arr, contract_sym, n_bars,
                                entry_z=ez, max_hold=n_hold, cap_secs=cap_secs,
                                event_mask=event_mask,
                                friday_flat_mask=friday_flat_mask,
                                friday_no_entry_mask=friday_no_entry_mask,
                                use_event=ev, use_weekend=wk,
                            )
                            pnl = pnl_from_trades(trades, df, convention="realistic",
                                                  cap_hours=cap_h).values
                            pnl_norm = pnl / CAP_NORM
                            sharpe_is  = _annualized_sharpe(pnl_norm[is_index],  bars_per_year_is)
                            sharpe_oos = _annualized_sharpe(pnl_norm[oos_index], bars_per_year_oos)
                            rows.append({
                                "signal_bars": n_signal, "entry_z": ez,
                                "max_hold": n_hold, "zlb": zlb,
                                "filter_event": ev, "filter_weekend": wk,
                                "cap_h": cap_h,
                                "n_trades": len(trades),
                                "sr_is":  sharpe_is,
                                "sr_oos": sharpe_oos,
                            })
                            sharpe_is_arr.append(sharpe_is)
                            sharpe_oos_arr.append(sharpe_oos)
                            pnl_oos_cols.append(pnl_norm[oos_index])
                            if i % 50 == 0:
                                elapsed = time.time() - t0
                                rate = i / elapsed
                                eta  = (n_total - i) / rate
                                print(f"  [{i:>5}/{n_total}]  {rate:.2f} cfg/s  "
                                      f"ETA {eta/60:.1f} min", flush=True)

    print(f"\n  finished {n_total} configs in {(time.time()-t0)/60:.1f} min", flush=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"  wrote {OUT_CSV}", flush=True)

    sharpe_is_arr  = np.asarray(sharpe_is_arr)
    sharpe_oos_arr = np.asarray(sharpe_oos_arr)
    pnl_oos_mat = np.asarray(pnl_oos_cols).T

    is_order = np.argsort(-np.where(np.isfinite(sharpe_is_arr), sharpe_is_arr, -np.inf))
    print("\nTop 10 by IS Sharpe:", flush=True)
    print(f"  {'idx':>5} {'sig':>4} {'ez':>5} {'hold':>5} {'zlb':>5} "
          f"{'ev':>3} {'wk':>3} {'cap_h':>7}  {'IS':>7} {'OOS':>7}", flush=True)
    for k in is_order[:10]:
        row = df_out.iloc[k]
        print(f"  {k:>5} {int(row.signal_bars):>4} {row.entry_z:>5.1f} "
              f"{int(row.max_hold):>5} {int(row.zlb):>5} "
              f"{int(row.filter_event):>3} {int(row.filter_weekend):>3} "
              f"{row.cap_h:>7.4f}  {row.sr_is:>+7.3f} {row.sr_oos:>+7.3f}", flush=True)

    best_index = int(is_order[0])
    best     = df_out.iloc[best_index].to_dict()
    print(f"\nBest IS joint config: idx {best_index}", flush=True)
    print(f"  signal_bars = {int(best['signal_bars'])}, entry_z = {best['entry_z']}, "
          f"max_hold = {int(best['max_hold'])}, zlb = {int(best['zlb'])}", flush=True)
    print(f"  filter_event = {bool(best['filter_event'])}, "
          f"filter_weekend = {bool(best['filter_weekend'])}, "
          f"cap_h = {best['cap_h']:.4f}", flush=True)
    print(f"  IS Sharpe = {best['sr_is']:+.3f}, OOS Sharpe = {best['sr_oos']:+.3f}", flush=True)

    print(f"\nHansen SPA on {n_total}-strategy set ...", flush=True)
    spa = hansen_spa_consistent(pnl_oos_mat, n_boot=2000, mean_block=5.0)
    print(f"  studentized max t = {spa['t_stat_consistent']:.3f}", flush=True)
    print(f"  p_SPA             = {spa['p_spa_consistent']:.5f}", flush=True)

    # rebuild is pnl for the focal config
    row = df_out.iloc[best_index]
    z_focal = z_cache[(int(row.signal_bars), int(row.zlb))]
    trades_focal = fast_simulate(
        timestamps, z_focal, contracts_arr, contract_sym, n_bars,
        entry_z=float(row.entry_z), max_hold=int(row.max_hold),
        cap_secs=float(row.cap_h) * 3600,
        event_mask=event_mask,
        friday_flat_mask=friday_flat_mask,
        friday_no_entry_mask=friday_no_entry_mask,
        use_event=bool(row.filter_event), use_weekend=bool(row.filter_weekend),
    )
    pnl_focal = pnl_from_trades(trades_focal, df, convention="realistic",
                                 cap_hours=float(row.cap_h)).values / CAP_NORM
    finite = np.isfinite(sharpe_is_arr)
    dsr = deflated_sharpe(pnl_focal[is_index], sharpe_is_arr[finite], bars_per_year_is)
    print(f"\nDeflated Sharpe on {n_total}-trial set (focal = IS-best) ...", flush=True)
    print(f"  focal IS Sharpe   = {sharpe_is_arr[best_index]:+.3f}", flush=True)
    print(f"  N_trials          = {dsr['n_trials']}", flush=True)
    print(f"  SR_0 expected-max = {dsr['sr0_annual']:+.3f}", flush=True)
    print(f"  z_DSR             = {dsr['z_dsr']:+.3f}", flush=True)
    print(f"  DSR p-value       = {dsr['dsr_prob']:.4f}", flush=True)

    out = {
        "n_strategies":   n_total,
        "best_idx":       best_index,
        "best_config":    {k: (bool(v) if isinstance(v, np.bool_) else
                                int(v)  if isinstance(v, np.integer) else
                                float(v) if isinstance(v, np.floating) else v)
                           for k, v in best.items()},
        "spa_t":          spa["t_stat_consistent"],
        "spa_p":          spa["p_spa_consistent"],
        "dsr_n_trials":   dsr["n_trials"],
        "dsr_sr0_annual": dsr["sr0_annual"],
        "dsr_z":          dsr["z_dsr"],
        "dsr_p":          dsr["dsr_prob"],
    }
    with OUT_JSON.open("w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
