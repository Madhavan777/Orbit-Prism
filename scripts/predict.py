"""Inference and Scoring Pipeline for ORBIT-PRISM (Shenbaga's Module).

Generates predictions for all TRAIN, CAL, and TEST tiles, including cloudy tiles.
Outputs scores to `results/scores_optical.csv` in Madhavan's required CSV format:
  tile_id,p_flood,extent

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
    parser.add_argument("--output_csv", type=str, default="results/scores_optical.csv", help="Destination CSV path")
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
        n_samples = 20
        tile_ids = [f"synthetic_tile_{i:04d}" for i in range(n_samples)]
        images = np.random.randint(0, 256, size=(n_samples, 128, 128, 4), dtype=np.uint8)
    else:
        print(f"Loading dataset from {npz_files[0]}...")
        data = np.load(npz_files[0], allow_pickle=True)
        key_map = discover_npz_keys(data)

        images = data[key_map.get("images", "images")]
        if "tile_id" in key_map:
            tile_ids = [str(tid) for tid in data[key_map["tile_id"]]]
        else:
            tile_ids = [f"tile_{i:05d}" for i in range(len(images))]

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
    df.to_csv(out_csv, index=False)

    print(f"\nPredictions exported to: {out_csv.resolve()}")
    print(f"CSV Shape: {df.shape} (Columns: {list(df.columns)})")
    print(f"\nSample Predictions:")
    print(df.head(5).to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    predict()
