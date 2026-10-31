"""Build data/processed/<event>.npz from the raw Sen1Floods11 chips, then check it against tiles_base.csv.

For every event: reads LabelHand, JRCWaterHand, S1Hand and S2Hand chips, scales optical (B2, B3, B4, B8)
and radar (VV, VH) to uint8, simulates clouds on the OPTICAL tiles only (radar is never touched), cuts
each 512x512 chip into 16 tiles of 128x128, and saves one NPZ per event with
    tile_id, optical (N,128,128,4) uint8, radar (N,128,128,2) uint8, water_frac, cloud_frac, split, critical

The splits are the official ones (train / valid / test, and Bolivia as holdout). Cloud fractions come from
data/tiles_base.csv. After each event the NPZ is compared with tiles_base.csv: same tiles, splits, labels
and clouds. Any mismatch stops the script with a non-zero exit code.

Needs the imagery:  python scripts/download_flood.py --imagery [--events Bolivia Ghana]
Run from the repo root:
    python scripts/build_processed.py                     # all 11 events (about 670 MB of NPZ)
    python scripts/build_processed.py --events Bolivia    # one event
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import process_event_chips  # noqa: E402
from src.data.split_events import load_official_chip_splits  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TILES_BASE = ROOT / "data" / "tiles_base.csv"


def all_events(raw: Path):
    labels = (raw / "data" / "flood_events" / "HandLabeled" / "LabelHand").glob("*_LabelHand.tif")
    return sorted({p.name.split("_", 1)[0] for p in labels})


def check_event(npz_path: Path, event: str, base: pd.DataFrame) -> list:
    """Problems found when comparing one event's NPZ with tiles_base.csv (empty list = fine)."""
    problems = []
    exp = base[base["event"] == event].copy()
    exp["tile_id"] = exp["chip"] + "_" + exp["row"].astype(str) + "_" + exp["col"].astype(str)
    exp = exp.set_index("tile_id")
    with np.load(npz_path) as z:
        got = pd.DataFrame({"tile_id": z["tile_id"], "split": z["split"], "critical": z["critical"],
                            "water_frac": z["water_frac"], "cloud_frac": z["cloud_frac"]}).set_index("tile_id")
        shapes = (z["optical"].shape[1:], z["radar"].shape[1:], z["optical"].dtype, z["radar"].dtype)
    if shapes != ((128, 128, 4), (128, 128, 2), np.uint8, np.uint8):
        problems.append(f"unexpected array shapes/dtypes {shapes}")
    if set(got.index) != set(exp.index):
        problems.append(f"tile set differs: {len(set(got.index) - set(exp.index))} extra, "
                        f"{len(set(exp.index) - set(got.index))} missing")
        return problems
    got = got.loc[exp.index]
    if (got["split"].to_numpy() != exp["split"].to_numpy()).any():
        problems.append("split labels differ from tiles_base.csv")
    if (got["critical"].to_numpy() != exp["critical"].to_numpy()).any():
        problems.append("critical labels differ from tiles_base.csv")
    if not np.allclose(got["water_frac"], exp["water_frac"], atol=1e-4):
        problems.append("water_frac differs from tiles_base.csv")
    if not np.allclose(got["cloud_frac"], exp["cloud_frac"], atol=1e-4):
        problems.append("cloud_frac differs from tiles_base.csv")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", nargs="+", default=None, help="events to build (default: all)")
    ap.add_argument("--raw-dir", type=Path, default=RAW)
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = ap.parse_args()

    if not TILES_BASE.exists():
        sys.exit("Missing data/tiles_base.csv (run python scripts/build_tiles_base.py)")
    imagery = args.raw_dir / "data" / "flood_events" / "HandLabeled" / "S2Hand"
    if not imagery.is_dir() or not any(imagery.glob("*.tif")):
        sys.exit(f"No imagery in {imagery}. Run: python scripts/download_flood.py --imagery")

    splits = load_official_chip_splits(args.raw_dir / "splits" / "flood_handlabeled")
    base = pd.read_csv(TILES_BASE)
    lookup = {f"{r.chip}_{r.row}_{r.col}": float(r.cloud_frac) for r in base.itertuples()}
    events = args.events or all_events(args.raw_dir)

    failed = False
    for event in events:
        # Only chips whose imagery was downloaded are processed; a partial download is reported, not hidden.
        n_labels = len(list((args.raw_dir / "data" / "flood_events" / "HandLabeled" / "LabelHand").glob(f"{event}_*")))
        n_s2 = len(list(imagery.glob(f"{event}_*")))
        if n_s2 < n_labels:
            print(f"SKIP {event}: imagery for {n_s2} of {n_labels} chips (run scripts/download_flood.py --imagery)")
            failed = True
            continue
        npz, _, counts = process_event_chips(event, raw_dir=args.raw_dir, processed_dir=args.processed_dir,
                                             chip_splits=splits, target_cloud_lookup=lookup)
        problems = check_event(npz, event, base)
        status = "OK " if not problems else "BAD"
        print(f"{status} {event}: {counts['valid_tiles']} tiles ({counts['critical_tiles']} critical), "
              f"dropped {counts['dropped_tiles']} -> {npz.name}")
        for p in problems:
            print(f"      problem: {p}")
        failed |= bool(problems)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
