"""NDWI colour-index baseline: a tile score from the green and near-infrared bands, no neural network.

NDWI = (green - NIR) / (green + NIR) per pixel, with green = B3 and NIR = B8 of the 4-band optical
tile [B2, B3, B4, B8]. A pixel counts as water when NDWI > 0 (the standard threshold, not tuned).
The tile score is the share of water pixels, written as both p_flood and extent. Higher = more water.

The bands are read from data/processed/<event>.npz, which already carry the simulated clouds, so
this baseline sees exactly the pixels the optical model sees: a cloud (pure white, green = NIR)
gives NDWI = 0 and is not water. This is the "Colour-index tile triage" system: it ranks every tile
by this score and pays the optical tile size for every tile, with no radar and no dial.

Events whose processed file is not on disk are skipped and reported; the held-out event (Bolivia)
must be present. Output: results/scores_ndwi.csv  (tile_id,p_flood,extent)

Run from the repo root:  python scripts/ndwi_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "scores_ndwi.csv"
HELD_OUT_EVENT = "Bolivia"


def ndwi_water_share(optical: np.ndarray) -> np.ndarray:
    """optical: (N, 128, 128, 4) uint8 [B2, B3, B4, B8] -> share of pixels with NDWI > 0, shape (N,)."""
    green = optical[..., 1].astype(np.float32)
    nir = optical[..., 3].astype(np.float32)
    denom = green + nir
    ndwi = np.divide(green - nir, denom, out=np.zeros_like(denom), where=denom > 0)
    return (ndwi > 0).mean(axis=(1, 2))


def main():
    files = sorted(PROCESSED.glob("*.npz"))
    if not any(f.stem == HELD_OUT_EVENT for f in files):
        sys.exit(f"{PROCESSED / (HELD_OUT_EVENT + '.npz')} is missing: run scripts/build_processed.py")
    rows = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            tile_id = z["tile_id"].astype(str)
            share = ndwi_water_share(z["optical"])
        rows.append(pd.DataFrame({"tile_id": tile_id, "p_flood": np.round(share, 6), "extent": np.round(share, 6)}))
        print(f"  {f.stem}: {len(tile_id)} tiles")
    out = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, lineterminator="\n")
    print(f"wrote {OUT} ({len(out)} tiles; events on disk: {', '.join(f.stem for f in files)})")


if __name__ == "__main__":
    main()
