# grid-design sensitivity: re-run the quarterly walk-forward under perturbed
# parameter grids to test whether the result depends on the grid values searched.

import os
import time
import argparse
import itertools

import pandas as pd

from strategy import panel_path, oos_start, vol_target_contracts
from walkforward import build_combo_pnl, evaluate, LOCKED

# baseline = the walkforward.py grid; shifted avoids the locked values entirely;
# widened spans a much larger range around them
GRID_VARIANTS = {
    "baseline": {"sb": [3, 5, 10], "z": [1.0, 1.5, 2.0],    "hold": [1, 3, 5], "cap_min": [5, 10, 15]},
    "shifted":  {"sb": [4, 6, 8],  "z": [0.75, 1.25, 1.75], "hold": [2, 4, 6], "cap_min": [7.5, 12.5, 20]},
    "widened":  {"sb": [2, 5, 15], "z": [0.5, 1.0, 2.5],    "hold": [1, 3, 8], "cap_min": [2, 10, 30]},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    df = pd.read_parquet(panel_path).sort_index()
    df["roll"] = (df["front_sym"] != df["front_sym"].shift(1)).fillna(True)
    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    n_contracts = vol_target_contracts(df)

    print(f"ETH grid-design sensitivity: quarterly walk-forward under {len(GRID_VARIANTS)} grids")
    print(f"  locked config {LOCKED} is only present in the baseline grid\n")

    for name, grid in GRID_VARIANTS.items():
        combos = list(itertools.product(grid["sb"], grid["z"], grid["hold"],
                                        [m / 60 for m in grid["cap_min"]]))
        contains_locked = LOCKED in combos
        start = time.time()
        combo_pnl = build_combo_pnl(df, n_contracts, combos, args.workers)
        print(f"  [{name:<9}] {len(combos)} combos in {time.time() - start:.0f}s  "
              f"(contains locked config: {contains_locked})")
        evaluate(combo_pnl, df.index, oos_start, 18, 3, 3, name)
        print()


if __name__ == "__main__":
    main()
