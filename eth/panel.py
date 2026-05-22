# build per-minute ETH front-month OFI panel from a TBBO event stream.
# cont-stoikov (2014) ofi: bid up -> +bid_sz, bid flat -> +delta_sz, bid dn -> -prev_sz.

import sys, glob, re, argparse, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import databento as db
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "shared"))
from ofi import find_front, compute_event_ofi

tbbo_dir = root.parent / "data" / "eth_tbbo"
out_path = root / "results" / "eth_ofi_minute.parquet"


def process_day(file_path, the_date):
    store = db.DBNStore.from_file(str(file_path))
    df = store.to_df()
    if df.empty or "symbol" not in df.columns:
        return None
    df.index = df.index.tz_convert("US/Eastern")

    out_right = df[~df["symbol"].astype(str).str.contains("-", na=False)].copy()
    if out_right.empty:
        return None

    front_sym, days_to_expiry = find_front(out_right["symbol"].unique(), the_date)
    if front_sym is None:
        return None
    front = out_right[out_right["symbol"] == front_sym].copy()
    if front.empty:
        return None

    bid_px = front["bid_px_00"].astype(float).values
    ask_px = front["ask_px_00"].astype(float).values
    bid_sz = front["bid_sz_00"].astype(float).values
    ask_sz = front["ask_sz_00"].astype(float).values
    valid = np.isfinite(bid_px) & np.isfinite(ask_px) & (bid_px > 0) & (ask_px > bid_px)
    front = front.loc[valid]
    if len(front) < 10:
        return None
    bid_px = front["bid_px_00"].astype(float).values
    ask_px = front["ask_px_00"].astype(float).values
    bid_sz = front["bid_sz_00"].astype(float).values
    ask_sz = front["ask_sz_00"].astype(float).values
    mid = (bid_px + ask_px) / 2

    e_ofi = compute_event_ofi(bid_px, ask_px, bid_sz, ask_sz)

    is_trade = (front["action"].astype(str).values == "T") if "action" in front.columns else np.zeros(len(front), dtype=bool)
    side = front["side"].astype(str).values if "side" in front.columns else np.array(["N"] * len(front))
    size = front["size"].astype(float).values if "size" in front.columns else np.zeros(len(front))
    price = front["price"].astype(float).values if "price" in front.columns else mid

    buy_size  = np.where(is_trade & (side == "B"), size, 0.0)
    sell_size = np.where(is_trade & (side == "A"), size, 0.0)
    buy_dollar  = buy_size  * price
    sell_dollar = sell_size * price

    large_thresh = 5
    is_large = size >= large_thresh
    buy_large_dollar  = np.where(is_trade & (side == "B") & is_large,  size * price, 0.0)
    sell_large_dollar = np.where(is_trade & (side == "A") & is_large,  size * price, 0.0)
    buy_small_dollar  = np.where(is_trade & (side == "B") & ~is_large, size * price, 0.0)
    sell_small_dollar = np.where(is_trade & (side == "A") & ~is_large, size * price, 0.0)

    minute = front.index.floor("1min")
    event_df = pd.DataFrame({
        "minute": minute,
        "mid": mid,
        "bid": bid_px, "ask": ask_px, "bid_sz": bid_sz, "ask_sz": ask_sz,
        "ofi": e_ofi,
        "buy_size": buy_size, "sell_size": sell_size,
        "buy_dollar": buy_dollar, "sell_dollar": sell_dollar,
        "buy_large_dollar": buy_large_dollar, "sell_large_dollar": sell_large_dollar,
        "buy_small_dollar": buy_small_dollar, "sell_small_dollar": sell_small_dollar,
        "is_trade": is_trade.astype(int),
    })
    agg = event_df.groupby("minute").agg(
        mid_open=("mid", "first"),
        mid_close=("mid", "last"),
        bid_close=("bid", "last"),
        ask_close=("ask", "last"),
        bid_sz_close=("bid_sz", "last"),
        ask_sz_close=("ask_sz", "last"),
        ofi=("ofi", "sum"),
        buy_size=("buy_size", "sum"),
        sell_size=("sell_size", "sum"),
        buy_dollar=("buy_dollar", "sum"),
        sell_dollar=("sell_dollar", "sum"),
        buy_large_dollar=("buy_large_dollar", "sum"),
        sell_large_dollar=("sell_large_dollar", "sum"),
        buy_small_dollar=("buy_small_dollar", "sum"),
        sell_small_dollar=("sell_small_dollar", "sum"),
        n_trades=("is_trade", "sum"),
        n_events=("mid", "count"),
    )
    agg["front_sym"] = front_sym
    agg["dte"] = days_to_expiry
    return agg


def _worker(fp_str):
    fp = Path(fp_str)
    date_match = re.search(r"(\d{8})", fp.name)
    if not date_match:
        return (fp.name, None, "no date in name")
    day = pd.Timestamp(date_match.group(1))
    try:
        chunk = process_day(fp, day)
        return (fp.name, chunk, None)
    except Exception as e:
        return (fp.name, None, f"{type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-i", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(out_path))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()

    files = sorted(glob.glob(str(tbbo_dir / "*.tbbo.dbn.zst")))
    if args.limit is not None:
        files = files[args.start_i: args.start_i + args.limit]
    else:
        files = files[args.start_i:]
    print(f"Processing {len(files)} TBBO files (start_i={args.start_i}, workers={args.workers})")

    all_chunks = []
    start_time = time.time()
    n_kept = 0
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, fp): fp for fp in files}
        for fut in as_completed(futures):
            name, chunk, err = fut.result()
            n_done += 1
            if err is not None:
                print(f"  err {name}: {err}")
            elif chunk is not None and not chunk.empty:
                all_chunks.append(chunk)
                n_kept += len(chunk)
            if n_done % 100 == 0:
                elapsed = time.time() - start_time
                rate = n_done / elapsed
                eta_min = (len(files) - n_done) / rate / 60
                print(f"  {n_done}/{len(files)}  rows so far {n_kept:,}  "
                      f"rate {rate:.1f} files/s  eta {eta_min:.1f} min")

    if not all_chunks:
        print("No data produced.")
        return

    df = pd.concat(all_chunks).sort_index()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nSaved: {out}")
    print(f"  rows: {len(df):,}")
    print(f"  range: {df.index.min()} -> {df.index.max()}")
    print(f"  unique fronts: {df['front_sym'].nunique()}")
    print(f"  median ofi/min: {df['ofi'].median():.2f}, std: {df['ofi'].std():.2f}")
    print(f"  median $ vol/min (buy+sell): ${(df['buy_dollar']+df['sell_dollar']).median():,.0f}")
    print(f"  median trades/min: {df['n_trades'].median():.0f}")


if __name__ == "__main__":
    main()
