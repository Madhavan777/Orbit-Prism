"""Per-tile flood statistics from the real Sen1Floods11 hand labels.

For every LabelHand chip (-1 no data, 0 not water, 1 water): remove permanent
water using the JRC layer, cut the 512x512 chip into 16 tiles of 128x128 and
compute water_frac per tile (ignoring -1 pixels).

Output: results/real_water_stats.csv (event, chip, row, col, split, water_frac)

Run from the repo root:  python -I scripts/real_stats.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LABELS = RAW / "data" / "flood_events" / "HandLabeled" / "LabelHand"
JRC = RAW / "data" / "flood_events" / "HandLabeled" / "JRCWaterHand"
SPLITS = RAW / "splits" / "flood_handlabeled"
OUT = ROOT / "results" / "real_water_stats.csv"

TILE = 128
GRID = 4
CRITICAL_THRESHOLD = 0.30  # same constant as scripts/make_fake_data.py

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "holdout": "flood_bolivia_data.csv",  # Bolivia, the held-out event
}


def load_splits():
    out = {}
    for split, name in SPLIT_FILES.items():
        with open(SPLITS / name, newline="") as f:
            for row in csv.reader(f):
                if row:
                    out[row[1].replace("_LabelHand.tif", "")] = split
    return out


def tile_fracs(label, jrc):
    valid = label != -1
    flood = (label == 1) & (jrc != 1)  # non-permanent water only
    rows = []
    for r in range(GRID):
        for c in range(GRID):
            sl = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))
            n = int(valid[sl].sum())
            rows.append((r, c, np.nan if n == 0 else float(flood[sl].sum()) / n))
    return rows


def main():
    splits = load_splits()
    records, perm_removed = [], 0
    for lab_path in sorted(LABELS.glob("*_LabelHand.tif")):
        chip = lab_path.name.replace("_LabelHand.tif", "")
        event = chip.split("_", 1)[0]
        label = tifffile.imread(lab_path)
        jrc = tifffile.imread(JRC / f"{chip}_JRCWaterHand.tif")
        perm_removed += int(((label == 1) & (jrc == 1)).sum())
        for r, c, wf in tile_fracs(label, jrc):
            records.append((event, chip, r, c, splits[chip], wf))

    n_all = len(records)
    records = [x for x in records if not np.isnan(x[5])]
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "chip", "row", "col", "split", "water_frac"])
        for e, ch, r, c, s, wf in records:
            w.writerow([e, ch, r, c, s, f"{wf:.6f}"])

    wf = np.array([x[5] for x in records])
    print(f"chips: {n_all // 16}, tiles kept: {len(records)} "
          f"(dropped {n_all - len(records)} with no valid pixels)")
    print(f"permanent-water pixels removed: {perm_removed}")
    print(f"\n{'event':<10} {'tiles':>6} {'share >= ' + str(CRITICAL_THRESHOLD):>12}")
    for ev in sorted({x[0] for x in records}):
        v = np.array([x[5] for x in records if x[0] == ev])
        print(f"{ev:<10} {len(v):>6} {np.mean(v >= CRITICAL_THRESHOLD):>12.3f}")
    print(f"\nshare of tiles with water_frac >= {CRITICAL_THRESHOLD}: "
          f"{np.mean(wf >= CRITICAL_THRESHOLD):.3f}")
    print(f"share with water_frac >= 0.05: {np.mean(wf >= 0.05):.3f}")
    print(f"share with water_frac == 0: {np.mean(wf == 0):.3f}")
    q = np.percentile(wf, [0, 25, 50, 75, 100])
    print("quartiles (min, Q1, median, Q3, max): " + ", ".join(f"{x:.4f}" for x in q))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
