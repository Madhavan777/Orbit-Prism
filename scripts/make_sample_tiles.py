"""Extract sample single-tile NPZ files for UI testing and triage demonstration.

Creates ready-to-upload .npz tile files in sample_tiles/:
1. sample_optical_clear.npz: Clear flood tile (water_frac=0.37, cloud_frac=0.08).
2. sample_radar_cloudy.npz: Dense-cloud flood tile (water_frac=0.39, cloud_frac=0.86).
3. sample_non_flood.npz: Dry/non-critical tile (water_frac=0.00).

Each file contains:
- 'optical': (128, 128, 4) uint8 [B2, B3, B4, B8]
- 'radar':   (128, 128, 2) uint8 [VV, VH]
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "sample_tiles"
OUT_DIR.mkdir(exist_ok=True)

TARGET_TILES = [
    {
        "filename": "sample_optical_clear.npz",
        "event": "Bolivia",
        "tile_id": "Bolivia_103757_3_2",
        "description": "Clear flood tile (water_frac=0.37, cloud_frac=0.08). Tested with optical model.",
    },
    {
        "filename": "sample_radar_cloudy.npz",
        "event": "Bolivia",
        "tile_id": "Bolivia_242570_0_1",
        "description": "Dense-cloud flood tile (water_frac=0.39, cloud_frac=0.86). Tested with radar model.",
    },
    {
        "filename": "sample_non_flood.npz",
        "event": "Bolivia",
        "tile_id": "Bolivia_195474_0_0",
        "description": "Dry/non-flood tile (water_frac=0.00).",
    },
]


def extract_sample_tiles():
    print("=" * 70)
    print("ORBIT-PRISM: Generating Sample Tile NPZ Files for UI Triage")
    print("=" * 70)

    for item in TARGET_TILES:
        event = item["event"]
        tile_id = item["tile_id"]
        npz_path = ROOT / "data" / "processed" / f"{event}.npz"
        if not npz_path.exists():
            sys.exit(f"Missing {npz_path}. Run python scripts/build_processed.py first.")

        with np.load(npz_path) as z:
            ids = list(z["tile_id"])
            if tile_id not in ids:
                sys.exit(f"Tile {tile_id} not found in {npz_path}")
            idx = ids.index(tile_id)
            optical = z["optical"][idx]  # (128, 128, 4) uint8
            radar = z["radar"][idx]      # (128, 128, 2) uint8
            wf = float(z["water_frac"][idx])
            cf = float(z["cloud_frac"][idx])

        dest = OUT_DIR / item["filename"]
        np.savez(dest, optical=optical, radar=radar)
        size_kb = dest.stat().st_size / 1024.0
        print(f"Created {dest.name:30s} | {tile_id} | WF: {wf:.2f} | CF: {cf:.2f} | Size: {size_kb:.1f} KB")

    print("\nSample tiles ready in sample_tiles/")


if __name__ == "__main__":
    extract_sample_tiles()
