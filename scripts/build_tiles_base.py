"""Build data/tiles_base.csv: the fixed per-tile inputs, with the cloud_frac the pixels really get.

Real (from results/real_water_stats.csv): event, chip, row, col, split, water_frac, critical.
Simulated clouds (SYNTHETIC; Sen1Floods11 has no cloud mask): cloud_frac and sensor_used come
from src.data.cloud_simulation, the same function the pixel pipeline (src/data/dataset.py)
uses to blend clouds into the optical tiles. The mask depends only on the chip id, so this
script reproduces the pipeline's per-tile cloud_frac exactly without any imagery.

Output (9 columns): event, chip, row, col, split, water_frac, critical, cloud_frac, sensor_used

Run from the repo root:  python -I scripts/build_tiles_base.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.constants import CLOUD_SWITCH, CRITICAL_THRESHOLD, GRID, TILE  # noqa: E402
from src.data.cloud_simulation import simulate_chip_cloud_mask  # noqa: E402

SRC = ROOT / "results" / "real_water_stats.csv"
OUT = ROOT / "data" / "tiles_base.csv"
COLUMNS = ["event", "chip", "row", "col", "split", "water_frac", "critical",
           "cloud_frac", "sensor_used"]


def main():
    with open(SRC, newline="") as f:
        rows = list(csv.DictReader(f))

    cloud = {}
    for chip in sorted({r["chip"] for r in rows}):
        alpha = simulate_chip_cloud_mask(chip)  # SYNTHETIC clouds, seeded by chip id
        for r in range(GRID):
            for c in range(GRID):
                tile = alpha[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
                cloud[(chip, r, c)] = float((tile >= 0.5).mean())

    out = []
    for r in rows:
        wf = float(r["water_frac"])
        cf = cloud[(r["chip"], int(r["row"]), int(r["col"]))]
        out.append([r["event"], r["chip"], int(r["row"]), int(r["col"]), r["split"],
                    round(wf, 6), int(wf >= CRITICAL_THRESHOLD), round(cf, 4),
                    "radar" if cf >= CLOUD_SWITCH else "optical"])

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        w.writerows(out)
    n_radar = sum(x[8] == "radar" for x in out)
    print(f"wrote {OUT}: {len(out)} tiles, {sum(x[6] for x in out)} critical, "
          f"{n_radar} radar ({n_radar / len(out):.2f})")


if __name__ == "__main__":
    main()
