"""Per-tile burned-area statistics from the Satellite Burned Area Dataset (Politecnico di Torino).

For every area (folder) whose *_mask.tiff is found in data/raw/fire_torino/zips/*.zip, cut
the mask into 128x128 tiles (the bottom/right remainder that does not fill a tile is
dropped) and compute burn_frac per tile.

ASSUMPTION: burned = mask > 0. The masks are Copernicus EMS damage grades stored as
0 plus 2 to 4 non-zero codes that differ between areas (seen: 32, 64, 128, 192, 255),
so the codes are not comparable across areas and are collapsed to burned / not burned.
Unlike Sen1Floods11 there is no "no data" value, so every pixel counts.

Output: results/fire/real_burn_stats.csv (event, chip, row, col, fold, burn_frac)
  event = EMS activation code (EMSR226), chip = area folder name, fold = the dataset's
  own cross-validation fold (a colour). Folds keep every activation together.
  A train/valid/holdout mapping is a team decision, not made here.

The script prints a PARTIAL warning unless all 73 areas are present.

Run from the repo root:  python -I scripts/fire_stats.py
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.constants import CRITICAL_THRESHOLD, TILE  # noqa: E402

ZIPS = ROOT / "data" / "raw" / "fire_torino" / "zips"
META = ROOT / "data" / "fire" / "satellite_data.csv"
OUT = ROOT / "results" / "fire" / "real_burn_stats.csv"
N_AREAS_FULL = 73


def load_folds():
    with open(META, newline="") as f:
        return {r["folder"]: r["fold"] for r in csv.DictReader(f)}


def tile_fracs(burned):
    h, w = burned.shape
    nr, nc = h // TILE, w // TILE
    t = burned[: nr * TILE, : nc * TILE].reshape(nr, TILE, nc, TILE)
    fr = t.mean(axis=(1, 3))
    return [(r, c, float(fr[r, c])) for r in range(nr) for c in range(nc)]


def main():
    zips = sorted(ZIPS.glob("*.zip"))
    if not zips:
        sys.exit(f"No zips in {ZIPS}\nDownload the Zenodo files first (see data/README.md).")
    folds = load_folds()
    records, codes, skipped = [], {}, []
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for name in sorted(z.namelist()):
                if not name.endswith("_mask.tiff"):
                    continue
                area = name.split("/")[1]
                if area not in folds:
                    skipped.append(area)
                    continue
                mask = tifffile.imread(io.BytesIO(z.read(name)))
                codes[area] = sorted(np.unique(mask).tolist())
                for r, c, bf in tile_fracs(mask > 0):
                    records.append((area.split("_", 1)[0], area, r, c, folds[area], bf))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["event", "chip", "row", "col", "fold", "burn_frac"])
        for e, a, r, c, fo, bf in records:
            w.writerow([e, a, r, c, fo, f"{bf:.6f}"])

    n_areas = len(codes)
    bf = np.array([x[5] for x in records])
    if n_areas < N_AREAS_FULL:
        print(f"*** PARTIAL: {n_areas} of {N_AREAS_FULL} areas found. Do not use these numbers "
              f"as dataset statistics. ***\n")
    if skipped:
        print(f"skipped {len(skipped)} areas missing from {META.name}: {skipped}")
    print(f"areas: {n_areas}, tiles: {len(records)}, mask codes seen: "
          f"{sorted({v for c in codes.values() for v in c})}")
    print(f"\n{'event':<8} {'areas':>5} {'tiles':>6} {'share >= ' + str(CRITICAL_THRESHOLD):>12}")
    for ev in sorted({x[0] for x in records}):
        v = np.array([x[5] for x in records if x[0] == ev])
        n_a = len({x[1] for x in records if x[0] == ev})
        print(f"{ev:<8} {n_a:>5} {len(v):>6} {np.mean(v >= CRITICAL_THRESHOLD):>12.3f}")
    print(f"\nshare of tiles with burn_frac >= {CRITICAL_THRESHOLD}: "
          f"{np.mean(bf >= CRITICAL_THRESHOLD):.3f}")
    print(f"share with burn_frac >= 0.05: {np.mean(bf >= 0.05):.3f}")
    print(f"share with burn_frac == 0: {np.mean(bf == 0):.3f}")
    q = np.percentile(bf, [0, 25, 50, 75, 100])
    print("quartiles (min, Q1, median, Q3, max): " + ", ".join(f"{x:.4f}" for x in q))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
