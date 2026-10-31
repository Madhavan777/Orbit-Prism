"""Quantise the trained float models to INT8, score every tile, and rebuild the memory table.

Inputs (made by the training steps):
    models/optical_float.keras      scripts/train.py (Shenbaga)
    models/radar_float.keras        src.models.radar.train_radar_pipeline (Harish)
    results/scores_optical_float.csv, results/scores_radar_float.csv   float scores (for the float rows)
    data/processed/*.npz            scripts/build_processed.py

Outputs:
    models/optical_int8.tflite, models/radar_int8.tflite
    results/scores_optical_int8.csv, results/scores_radar_int8.csv   tile_id,p_flood,extent for EVERY tile
    results/memory_table.csv, results/memory_notes.md                weights, peak activation, MACs, and
                                                                     recall and ece MEASURED on the valid split

Calibration images for quantisation are TRAIN tiles only (optical: cloud_frac < 0.5, like its training set;
radar: all train tiles). The holdout is only scored, never used. recall and ece are measured on the `valid`
split, on the tiles each model is routed (optical: cloud_frac < 0.5, radar: cloud_frac >= 0.5), at p >= 0.5.

Run from the repo root:  python scripts/quantize_and_score.py
Use --allow-partial only on a subset of events (a smoke test); results from it are not results.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.metrics import compute_ece  # noqa: E402
from src.compress.predict_int8 import run_optical_int8_predictions, run_radar_int8_predictions  # noqa: E402
from src.compress.profile_memory import NOT_MEASURED, profile_all_models_pipeline  # noqa: E402
from src.compress.quantize import quantize_optical_model, quantize_radar_model  # noqa: E402
from src.constants import CLOUD_SWITCH  # noqa: E402
from src.data.dataset import OrbitPrismDataset  # noqa: E402

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
TILES_BASE = ROOT / "data" / "tiles_base.csv"


def _need(path: Path, hint: str):
    if not path.exists():
        sys.exit(f"Missing {path.relative_to(ROOT)}. {hint}")


def _measure(scores_csv: Path, ds, routed: np.ndarray):
    """(recall at 0.5, ece) on the valid tiles in `routed`, or NOT_MEASURED if the file is missing."""
    if not scores_csv.exists():
        return NOT_MEASURED, NOT_MEASURED
    s = pd.read_csv(scores_csv).set_index("tile_id")["p_flood"]
    sel = (ds.split == "valid") & routed
    ids = ds.tile_id[sel]
    y = ds.critical[sel].astype(int)
    p = s.reindex(ids).to_numpy(dtype=float)
    if np.isnan(p).any() or len(y) == 0 or y.sum() == 0:
        return NOT_MEASURED, NOT_MEASURED
    recall = float(((p >= 0.5) & (y == 1)).sum() / (y == 1).sum())
    return round(recall, 4), round(float(compute_ece(y, p)), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--allow-partial", action="store_true", help="smoke test on a subset of events")
    args = ap.parse_args()

    opt_keras, rad_keras = MODELS / "optical_float.keras", MODELS / "radar_float.keras"
    _need(opt_keras, "Train the optical model first: python scripts/train.py")
    _need(rad_keras, "Train the radar model first (src.models.radar.train_radar_pipeline).")

    ds = OrbitPrismDataset(processed_dir=args.processed_dir, split="all", modality="both")
    train = ds.split == "train"
    print(f"{len(ds.tile_id)} tiles loaded; calibration images for quantisation come from the train split only")

    o = quantize_optical_model(opt_keras, ds.optical[train & (ds.cloud_frac < CLOUD_SWITCH)])
    r = quantize_radar_model(rad_keras, (ds.radar[train] / 255.0).astype("float32"))
    for name, info in (("optical", o), ("radar", r)):
        if info.get("status") != "CONVERTED_INT8":
            sys.exit(f"{name} quantisation failed: {info}")
    print("quantised:", MODELS / "optical_int8.tflite", MODELS / "radar_int8.tflite")

    for name, res in (("optical", run_optical_int8_predictions(optical_images=ds.optical, tile_ids=list(ds.tile_id))),
                      ("radar", run_radar_int8_predictions(dataset_path=args.processed_dir))):
        if res.get("status") != "COMPLETED":
            sys.exit(f"{name} INT8 scoring failed: {res}")

    # Every tile in tiles_base.csv must have a score from both models.
    base = pd.read_csv(TILES_BASE)
    expected = set(base["chip"] + "_" + base["row"].astype(str) + "_" + base["col"].astype(str))
    for name in ("optical", "radar"):
        got = set(pd.read_csv(RESULTS / f"scores_{name}_int8.csv")["tile_id"])
        missing = len(expected - got)
        if missing and not args.allow_partial:
            sys.exit(f"scores_{name}_int8.csv is missing {missing} of {len(expected)} tiles: build all events "
                     f"(python scripts/build_processed.py) and rerun")
        print(f"scores_{name}_int8.csv: {len(got)} tiles ({missing} of the {len(expected)} in tiles_base.csv missing)")

    routed_opt = ds.cloud_frac < CLOUD_SWITCH
    metrics = {}
    for name, routed in (("optical", routed_opt), ("radar", ~routed_opt)):
        for kind in ("float", "int8"):
            rec, ece = _measure(RESULTS / f"scores_{name}_{kind}.csv", ds, routed)
            metrics[f"{name}_{kind}_recall"], metrics[f"{name}_{kind}_ece"] = rec, ece
            print(f"  {name} {kind}: recall@0.5 = {rec}, ece = {ece}   (valid split, routed tiles)")

    profile_all_models_pipeline(cal_metrics=metrics)
    print("done. Next: python -m src.calibration.calibrate, then python scripts/merge_scores.py")


if __name__ == "__main__":
    main()
