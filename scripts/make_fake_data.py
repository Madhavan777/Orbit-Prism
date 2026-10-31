# water_frac and critical come from real Sen1Floods11 labels; cloud_frac, p_flood, extent are synthetic placeholders.
"""Build data/fake_tiles.csv from results/real_water_stats.csv.

Real (copied from results/real_water_stats.csv, one row per real tile, so the
event mix, chip counts, 4x4 grid and official split are exactly the real ones):
    event, chip, row, col, split, water_frac, critical

Synthetic (columns follow CONTEXT.md):
    cloud_frac   - SIMULATED. Sen1Floods11 has no cloud mask, so each chip gets
                   a random cloud level and each tile varies around it.
    sensor_used  - radar if cloud_frac >= 0.5, else optical
    p_flood      - placeholder model score; optical noise grows with cloud_frac,
                   radar noise does not depend on cloud
    extent       - noisy placeholder for flood extent

Note: the official train/valid/test split is by chip across events; only
Bolivia is a held-out event ("holdout").

Run from the repo root:  python -I scripts/make_fake_data.py
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "real_water_stats.csv"
OUT = ROOT / "data" / "fake_tiles.csv"

CRITICAL_THRESHOLD = 0.30
CLOUD_SWITCH = 0.5
SEED = 7

# p_flood = sigmoid(SLOPE * (water_frac - CRITICAL_THRESHOLD) + noise)
SLOPE = 10.0
OPTICAL_NOISE_CLEAR, OPTICAL_NOISE_CLOUD = 0.5, 4.0  # sd at cloud_frac 0 / 1
RADAR_NOISE = 1.2                                     # independent of cloud
EXTENT_NOISE = 0.05


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    rng = np.random.default_rng(SEED)
    with open(SRC, newline="") as f:
        rows = list(csv.DictReader(f))

    chip_cloud = {c: rng.beta(0.7, 1.0) for c in sorted({r["chip"] for r in rows})}

    out = []
    for r in rows:
        wf = float(r["water_frac"])
        cloud = float(np.clip(chip_cloud[r["chip"]] + rng.normal(0, 0.15), 0, 1))
        sensor = "radar" if cloud >= CLOUD_SWITCH else "optical"
        sd = RADAR_NOISE if sensor == "radar" else (
            OPTICAL_NOISE_CLEAR + (OPTICAL_NOISE_CLOUD - OPTICAL_NOISE_CLEAR) * cloud)
        p = float(sigmoid(SLOPE * (wf - CRITICAL_THRESHOLD) + rng.normal(0, sd)))
        extent = float(np.clip(wf + rng.normal(0, EXTENT_NOISE), 0, 1))
        out.append([r["event"], r["chip"], int(r["row"]), int(r["col"]), r["split"],
                    round(wf, 4), int(wf >= CRITICAL_THRESHOLD), round(cloud, 4),
                    sensor, round(p, 4), round(extent, 4)])

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "chip", "row", "col", "split", "water_frac", "critical",
                    "cloud_frac", "sensor_used", "p_flood", "extent"])
        w.writerows(out)
    n_crit = sum(x[6] for x in out)
    print(f"wrote {OUT}: {len(out)} tiles, {n_crit} critical "
          f"({n_crit / len(out):.3f}), {sum(x[8] == 'radar' for x in out)} radar")


if __name__ == "__main__":
    main()
