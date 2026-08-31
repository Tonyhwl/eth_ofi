# build a sub-minute ETH front-month OFI panel from the TBBO event stream.
# same per-event construction as panel.py, floored to a finer bucket (default
# ten seconds, the cont-kukanov-stoikov sampling interval). built for the OOS
# window only: the bar threshold depends on daily dollar volume, which is
# bucket-invariant, so the in-sample calibration carries over from the minute
# panel unchanged.

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

FREQ = "10s"
# first UTC file that can contain ET timestamps on or after 2024-01-01
START_FILE_DATE = "20231230"


def process_day(file_path, the_date, freq):
    store = db.DBNStore.from_file(str(file_path))
    df = store.to_df()
    if df.empty or "symbol" not in df.columns:
        return None
    df.index = df.index.tz_convert("US/Eastern")

    out_right = df[~df["symbol"].astype(str).str.contains("-", na=False)].copy()
    if out_right.empty:
        return None

    front_sym, _ = find_front(out_right["symbol"].unique(), the_date)
    if front_sym is None:
        return None
    front = out_right[out_right["symbol"] == front_sym].copy()
    if front.empty:
        return None

    bid_px = front["bid_px_00"].astype(float).values
    ask_px = front["ask_px_00"].astype(float).values
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

    buy_dollar = np.where(is_trade & (side == "B"), size * price, 0.0)
    sell_dollar = np.where(is_trade & (side == "A"), size * price, 0.0)

    bucket = front.index.floor(freq)
    event_df = pd.DataFrame({
        "bucket": bucket, "mid": mid, "ofi": e_ofi,
        "buy_dollar": buy_dollar, "sell_dollar": sell_dollar,
    })
    agg = event_df.groupby("bucket").agg(
        mid_close=("mid", "last"),
        ofi=("ofi", "sum"),
        buy_dollar=("buy_dollar", "sum"),
        sell_dollar=("sell_dollar", "sum"),
        n_events=("mid", "count"),
    )
    agg["front_sym"] = front_sym
    return agg


def _worker(args):
    fp_str, freq = args
    fp = Path(fp_str)
    date_match = re.search(r"(\d{8})", fp.name)
    if not date_match:
        return (fp.name, None, "no date in name")
    day = pd.Timestamp(date_match.group(1))
    try:
        chunk = process_day(fp, day, freq)
        return (fp.name, chunk, None)
    except Exception as e:
        return (fp.name, None, f"{type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=str, default=FREQ)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()

    out = Path(args.out) if args.out else root / "results" / f"eth_ofi_{args.freq}_oos.parquet"
    files = [f for f in sorted(glob.glob(str(tbbo_dir / "*.tbbo.dbn.zst")))
             if re.search(r"(\d{8})", Path(f).name).group(1) >= START_FILE_DATE]
    if args.limit is not None:
        files = files[:args.limit]
    print(f"Processing {len(files)} TBBO files at {args.freq} buckets (workers={args.workers})")

    all_chunks = []
    start_time = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, (fp, args.freq)): fp for fp in files}
        for fut in as_completed(futures):
            name, chunk, err = fut.result()
            n_done += 1
            if err is not None:
                print(f"  err {name}: {err}")
            elif chunk is not None and not chunk.empty:
                all_chunks.append(chunk)
            if n_done % 100 == 0:
                elapsed = time.time() - start_time
                eta_min = (len(files) - n_done) / (n_done / elapsed) / 60
                print(f"  {n_done}/{len(files)}  eta {eta_min:.1f} min", flush=True)

    df = pd.concat(all_chunks).sort_index()
    df = df[df.index >= pd.Timestamp("2024-01-01", tz="US/Eastern")]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nSaved: {out}")
    print(f"  rows: {len(df):,}   range: {df.index.min()} -> {df.index.max()}")
    print(f"  median events/bucket: {df['n_events'].median():.0f}   "
          f"single-event buckets: {(df['n_events'] == 1).mean():.1%}")


if __name__ == "__main__":
    main()
