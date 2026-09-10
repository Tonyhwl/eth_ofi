# one-bar-at-a-time OFI engine, mirrors strategy.simulate
# equivalence pinned by test_engine_golden.py

from collections import deque

import numpy as np

from strategy import (
    signal_bars, entry_threshold, max_hold_bars, direction_sign, zscore_lookback,
    friday_force_flat_min, friday_no_entry_min,
    Trade, Position, in_event_window, in_friday_window, build_event_calendar,
)


class OFIEngine:
    """incremental runner for the locked strategy"""

    def __init__(self, use_event_filter=False, use_weekend_filter=False,
                 cap_hours=None, events=None,
                 signal_bars=signal_bars, entry_threshold=entry_threshold,
                 max_hold_bars=max_hold_bars, direction_sign=direction_sign,
                 zscore_lookback=zscore_lookback):
        self.signal_bars = signal_bars
        self.entry_threshold = entry_threshold
        self.max_hold_bars = max_hold_bars
        self.direction_sign = direction_sign
        self.zscore_lookback = zscore_lookback
        self.use_event_filter = use_event_filter
        self.use_weekend_filter = use_weekend_filter
        self.cap_seconds = cap_hours * 3600 if cap_hours is not None else None
        if events is None and use_event_filter:
            events = build_event_calendar()
        self.events = events

        self.ofi_buffer = deque(maxlen=signal_bars)        # last K bar OFIs
        self.cofi_hist = deque(maxlen=zscore_lookback)     # prior cumulative-OFI values
        self.pos = Position()
        self.prev_contract = None
        self.trades = []
        self._i = -1

    def _zscore(self, ofi):
        """rolling z of cumulative OFI, shifted window"""
        self.ofi_buffer.append(ofi)
        if len(self.ofi_buffer) == self.signal_bars:
            cofi = float(np.sum(self.ofi_buffer))
        else:
            cofi = np.nan
        # window excludes current bar (shift by one)
        if len(self.cofi_hist) == self.zscore_lookback:
            window = np.fromiter(self.cofi_hist, dtype=float, count=self.zscore_lookback)
            if np.all(np.isfinite(window)) and np.isfinite(cofi):
                std = window.std(ddof=1)
                z = (cofi - window.mean()) / std if std > 0 else np.nan
            else:
                z = np.nan
        else:
            z = np.nan
        self.cofi_hist.append(cofi)
        return z

    def _open(self, direction, n_contracts, ts, i):
        self.pos.direction = direction
        self.pos.size = max(1, int(round(n_contracts)))
        self.pos.bars_held = 1
        self.pos.entry_time = ts
        self.pos.entry_idx = i

    def _close(self, ts, i, reason):
        self.trades.append(Trade(self.pos.entry_time, self.pos.entry_idx, ts, i,
                                 self.pos.direction, self.pos.size, reason))
        self.pos.reset()

    def step(self, ts, ofi, contract_sym, n_contracts):
        """one bar, returns actions"""
        self._i += 1
        i = self._i
        z = self._zscore(ofi)
        actions = []

        cap_fired = (self.pos.is_open() and self.cap_seconds is not None
                     and self.pos.entry_time is not None
                     and (ts - self.pos.entry_time).total_seconds() >= self.cap_seconds)
        block_event = self.use_event_filter and in_event_window(ts, self.events)
        force_flat = self.use_weekend_filter and in_friday_window(ts, friday_force_flat_min)
        block_entry = block_event or (self.use_weekend_filter
                                      and in_friday_window(ts, friday_no_entry_min))

        # exit priority: cap > event > weekend > roll > hold > z-cross
        if cap_fired:
            self._close(ts, i, "cap"); self.prev_contract = contract_sym
            return ["exit:cap"]
        if self.pos.is_open() and block_event:
            self._close(ts, i, "event"); self.prev_contract = contract_sym
            return ["exit:event"]
        if self.pos.is_open() and force_flat:
            self._close(ts, i, "weekend"); self.prev_contract = contract_sym
            return ["exit:weekend"]
        if (contract_sym != self.prev_contract and self.prev_contract is not None
                and self.pos.is_open()):
            self._close(ts, i, "roll"); actions.append("exit:roll")

        if not np.isfinite(z):
            self.prev_contract = contract_sym
            return actions

        if self.pos.is_open() and self.pos.bars_held >= self.max_hold_bars:
            self._close(ts, i, "hold"); actions.append("exit:hold")
        if self.pos.direction == +1 and z <= 0:
            self._close(ts, i, "zcross"); actions.append("exit:zcross")
        elif self.pos.direction == -1 and z >= 0:
            self._close(ts, i, "zcross"); actions.append("exit:zcross")

        if not self.pos.is_open() and not block_entry:
            if z >= self.entry_threshold:
                self._open(+1 * self.direction_sign, n_contracts, ts, i)
                actions.append(f"enter:{self.pos.direction:+d}")
            elif z <= -self.entry_threshold:
                self._open(-1 * self.direction_sign, n_contracts, ts, i)
                actions.append(f"enter:{self.pos.direction:+d}")
        elif self.pos.is_open():
            self.pos.bars_held += 1

        self.prev_contract = contract_sym
        return actions

    def finalize(self, ts, i):
        """end-of-data flush, exit reason 'end'"""
        if self.pos.is_open():
            self._close(ts, i, "end")


def simulate_streaming(df, n_contracts_series, use_event_filter, use_weekend_filter,
                       cap_hours=None, events=None):
    """OFIEngine over a panel, same output as strategy.simulate"""
    eng = OFIEngine(use_event_filter, use_weekend_filter, cap_hours, events)
    contracts = n_contracts_series.fillna(0).values
    syms = df["front_sym"].values
    ofi = df["ofi"].values
    ts_index = df.index
    for i in range(len(df)):
        eng.step(ts_index[i], ofi[i], syms[i], contracts[i])
    eng.finalize(ts_index[len(df) - 1], len(df) - 1)
    return eng.trades
