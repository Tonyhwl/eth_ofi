# front-contract selection and order-flow imbalance from a tbbo event stream.

import numpy as np
import pandas as pd

month_codes = "FGHJKMNQUVXZ"


def parse_outright(sym):
    """Parse a contract symbol such as 'BTCK6' into ('BTC', 2026, 5)."""

    # spread symbols such as 'BTCK6 - BTCM6' are not outright contracts
    if "-" in sym:
        return None

    # walk back over the trailing year digits to separate root, month, year
    i = len(sym)
    while i > 0 and sym[i - 1].isdigit():
        i -= 1
    root = sym[:i - 1]
    month_letter = sym[i - 1]
    year_digits = sym[i:]

    month = month_codes.index(month_letter) + 1

    # single-digit years are decade-ambiguous, so resolve them for the 2017 to 2026 range
    if int(year_digits) <= 6:
        year = 2020 + int(year_digits)
    else:
        year = 2010 + int(year_digits)

    return (root, year, month)


def expiry_date(year, month):
    """Last weekday of the given contract month."""

    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    while end.weekday() >= 5:
        end -= pd.Timedelta(days=1)
    return end


def find_front(symbols, date):
    """Return the nearest unexpired contract and its days to expiry."""

    best = None
    best_dte = 1e9
    for sym in symbols:
        parsed = parse_outright(sym)
        if parsed is None:
            continue
        _, year, month = parsed
        dte = (expiry_date(year, month) - date).days
        if 0 <= dte < best_dte:
            best_dte = dte
            best = sym
    return best, best_dte


def compute_event_ofi(bid_price, ask_price, bid_size, ask_size):
    """Order-flow imbalance contribution for each consecutive event."""

    n = len(bid_price)
    e = np.zeros(n)
    for i in range(1, n):
        # bid side: a higher bid adds size, a lower bid removes the previous size
        if bid_price[i] > bid_price[i - 1]:
            bid_points = bid_size[i]
        elif bid_price[i] < bid_price[i - 1]:
            bid_points = -bid_size[i - 1]
        else:
            bid_points = bid_size[i] - bid_size[i - 1]

        # ask side is symmetric with the opposite sign
        if ask_price[i] < ask_price[i - 1]:
            ask_points = -ask_size[i]
        elif ask_price[i] > ask_price[i - 1]:
            ask_points = ask_size[i - 1]
        else:
            ask_points = ask_size[i - 1] - ask_size[i]

        e[i] = bid_points + ask_points

    return e
