# water_frac and critical come from real Sen1Floods11 labels; cloud_frac is simulated (src/data/cloud_simulation.py); p_flood and extent are synthetic placeholders.
"""Build data/fake_tiles.csv: tiles_base.csv plus PLACEHOLDER model scores.

From data/tiles_base.csv (run scripts/build_tiles_base.py first; nothing is invented here):
    event, chip, row, col, split, water_frac, critical, cloud_frac, sensor_used
    water_frac and critical are real labels; cloud_frac and sensor_used come from the same
    simulated clouds the pixel pipeline uses, so this file agrees with the pixels.

Synthetic, invented here (columns follow CONTEXT.md):
    p_flood      - placeholder model score; optical noise grows with cloud_frac,
                   radar noise does not depend on cloud
    extent       - noisy placeholder for flood extent

Note: the official train/valid/test split is by chip across events; only
Bolivia is a held-out event ("holdout").

Run from the repo root:  python -I scripts/make_fake_data.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "tiles_base.csv"
OUT = ROOT / "data" / "fake_tiles.csv"

SEED = 7

# p_flood = sigmoid(SLOPE * (water_frac - 0.30) + noise)
SLOPE = 5.0
CENTRE = 0.30
OPTICAL_NOISE_CLEAR, OPTICAL_NOISE_CLOUD = 1.0, 5.0  # sd at cloud_frac 0 / 1; optical tiles only
                                                      # exist below CLOUD_SWITCH, so sd tops out at 3.0
RADAR_NOISE = 1.8                                     # independent of cloud
RADAR_FALSE_ALARM = (0.06, 2.0)  # (probability, logit boost): smooth tarmac/sand looks like water
EVENT_SHIFT_SD = 0.5             # per-event logit offset, so held-out Bolivia is not in-distribution
EXTENT_NOISE = 0.12


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    if not SRC.exists():
        sys.exit(f"Missing {SRC}. Run: python scripts/build_tiles_base.py")
    rng = np.random.default_rng(SEED)
    with open(SRC, newline="") as f:
        rows = list(csv.DictReader(f))

    event_shift = {e: rng.normal(0, EVENT_SHIFT_SD) for e in sorted({r["event"] for r in rows})}

    out = []
    for r in rows:
        wf = float(r["water_frac"])
        cloud = float(r["cloud_frac"])
        sensor = r["sensor_used"]
        sd = RADAR_NOISE if sensor == "radar" else (
            OPTICAL_NOISE_CLEAR + (OPTICAL_NOISE_CLOUD - OPTICAL_NOISE_CLEAR) * cloud)
        logit = SLOPE * (wf - CENTRE) + event_shift[r["event"]] + rng.normal(0, sd)
        if sensor == "radar" and rng.random() < RADAR_FALSE_ALARM[0]:
            logit += RADAR_FALSE_ALARM[1]
        p = float(sigmoid(logit))
        extent = float(np.clip(wf + rng.normal(0, EXTENT_NOISE), 0, 1))
        out.append([r["event"], r["chip"], int(r["row"]), int(r["col"]), r["split"],
                    r["water_frac"], int(r["critical"]), r["cloud_frac"],
                    sensor, round(p, 4), round(extent, 4)])

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["event", "chip", "row", "col", "split", "water_frac", "critical",
                    "cloud_frac", "sensor_used", "p_flood", "extent"])
        w.writerows(out)
    n_crit = sum(x[6] for x in out)
    print(f"wrote {OUT}: {len(out)} tiles, {n_crit} critical "
          f"({n_crit / len(out):.3f}), {sum(x[8] == 'radar' for x in out)} radar")


if __name__ == "__main__":
    main()
