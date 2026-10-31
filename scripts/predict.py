"""Inference and Scoring Pipeline for ORBIT-PRISM (Shenbaga's Module).

Generates predictions for all TRAIN, CAL, and TEST tiles, including cloudy tiles.
Outputs scores to `results/scores_optical_float.csv` in the team CSV format:
  tile_id,p_flood,extent        (tile_id = <chip>_<row>_<col>)
This is the FLOAT model's score. The contract file results/scores_optical_int8.csv, the one
scripts/merge_scores.py reads, is produced from the INT8 model by src/compress/predict_int8.py.
Without --model_path pointing at a trained model, or without processed data, this script
writes a clearly marked SYNTHETIC template and never to results/.

Also supports NDWI baseline fallback if CNN is uninitialized or requested.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import tensorflow as tf

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.optical import build_optical_model, compile_optical_model, ndwi_baseline_predict, setup_gpu
from scripts.train import discover_npz_keys


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Optical Predictions for ORBIT-PRISM")
    parser.add_argument("--model_path", type=str, default="models/optical_float.keras", help="Path to trained float model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed NPZ files")
    parser.add_argument("--output_csv", type=str, default="results/scores_optical_float.csv", help="Destination CSV path")
    parser.add_argument("--use_ndwi_fallback", action="store_true", help="Force NDWI water index fallback")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    return parser.parse_args()


def predict():
    args = parse_args()
    print("=" * 70)
    print("ORBIT-PRISM: Optical Flood Prediction & Scoring Pipeline (Shenbaga)")
    print("=" * 70)

    setup_gpu()

    data_dir = Path(args.data_dir)
    npz_files = list(data_dir.glob("*.npz")) if data_dir.exists() else []

    if not npz_files:
        print(f"[INFO] No .npz files found in {data_dir.resolve()}.")
        print("Generating template scores using synthetic tiles to verify CSV schema...")
        if args.output_csv == "results/scores_optical_float.csv":
            args.output_csv = "data/_synthetic_template_scores_optical.csv"  # never into results/
            print(f"SYNTHETIC template: writing to {args.output_csv}, not results/.")
        n_samples = 20
        tile_ids = [f"synthetic_tile_{i:04d}" for i in range(n_samples)]
        images = np.random.randint(0, 256, size=(n_samples, 128, 128, 4), dtype=np.uint8)
    else:
        from src.data.dataset import OrbitPrismDataset

        print(f"Loading all event files from {data_dir} ({len(npz_files)} file(s))...")
        ds = OrbitPrismDataset(processed_dir=data_dir, split="all", modality="optical")
        images = ds.optical
        tile_ids = [str(t) for t in ds.tile_id]  # <chip>_<row>_<col>, the team tile_id

    print(f"Total tiles to score: {len(images)} (includes TRAIN, CAL, and TEST, even cloudy tiles)")

    # Model inference or NDWI fallback
    model_file = Path(args.model_path)
    if model_file.exists() and not args.use_ndwi_fallback:
        print(f"Loading trained optical model from {model_file}...")
        model = tf.keras.models.load_model(str(model_file))
        p_flood_preds, extent_preds = model.predict(images, batch_size=args.batch_size, verbose=1)
    else:
        if args.use_ndwi_fallback:
            print("Using NDWI Water-Index Baseline Fallback as explicitly requested.")
        else:
            print(f"Trained model '{model_file}' not found. Executing NDWI baseline fallback.")
        p_flood_preds, extent_preds = ndwi_baseline_predict(images)

    # Flatten predictions
    p_flood_flat = p_flood_preds.ravel()
    extent_flat = extent_preds.ravel()

    # Create CSV records strictly in Madhavan's required contract
    records = []
    for tid, p, ext in zip(tile_ids, p_flood_flat, extent_flat):
        records.append({
            "tile_id": tid,
            "p_flood": round(float(p), 4),
            "extent": round(float(ext), 4)
        })

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False, lineterminator="\n")

    print(f"\nPredictions exported to: {out_csv.resolve()}")
    print(f"CSV Shape: {df.shape} (Columns: {list(df.columns)})")
    print(f"\nSample Predictions:")
    print(df.head(5).to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    predict()
