# front-contract selection and OFI from a TBBO event stream.

import numpy as np
import pandas as pd

month_codes = "FGHJKMNQUVXZ"


def parse_outright(sym):
    """'BTCK6' -> ('BTC', 2026, 5)"""

    # skip spread symbols
    if "-" in sym:
        return None

    # walk back over trailing year digits
    i = len(sym)
    while i > 0 and sym[i - 1].isdigit():
        i -= 1
    root = sym[:i - 1]
    month_letter = sym[i - 1]
    year_digits = sym[i:]

    month = month_codes.index(month_letter) + 1

    # decade split for the 2017-2026 range
    if int(year_digits) <= 6:
        year = 2020 + int(year_digits)
    else:
        year = 2010 + int(year_digits)

    return (root, year, month)


def expiry_date(year, month):
    """last weekday of the contract month"""

    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    while end.weekday() >= 5:
        end -= pd.Timedelta(days=1)
    return end


def find_front(symbols, date):
    """nearest unexpired contract and days to expiry"""

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
    """per-event OFI contribution"""

    n = len(bid_price)
    e = np.zeros(n)
    for i in range(1, n):
        # bid up adds size, bid down removes prior size
        if bid_price[i] > bid_price[i - 1]:
            bid_points = bid_size[i]
        elif bid_price[i] < bid_price[i - 1]:
            bid_points = -bid_size[i - 1]
        else:
            bid_points = bid_size[i] - bid_size[i - 1]

        # ask side symmetric, opposite sign
        if ask_price[i] < ask_price[i - 1]:
            ask_points = -ask_size[i]
        elif ask_price[i] > ask_price[i - 1]:
            ask_points = ask_size[i - 1]
        else:
            ask_points = ask_size[i - 1] - ask_size[i]

        e[i] = bid_points + ask_points

    return e
