"""Seed sensitivity study for ORBIT-PRISM optical flood model.

Trains the optical model across seeds 1 to 8 to assess run-to-run noise:
- EARLY-STOPPING set: train split tiles of Spain and Somalia with cloud_frac < 0.5 (300 tiles, 15 critical).
- CALIBRATION (valid) set: valid split tiles (1,354 tiles, 118 critical).

Outputs:
- results/optical_seed_study.csv
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import INTERNAL_VAL_EVENTS
from src.data.dataset import OrbitPrismDataset

RESULTS_DIR = ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "optical_seed_study.csv"


def load_evaluation_data():
    """Load early-stopping set and valid (cal) set tiles and labels."""
    ds_all = OrbitPrismDataset(split="all", modality="optical")

    # Early stopping set: train split, cloud_frac < 0.5, events in INTERNAL_VAL_EVENTS
    es_mask = (
        (ds_all.split == "train")
        & (ds_all.cloud_frac < 0.5)
        & np.isin(ds_all.event, INTERNAL_VAL_EVENTS)
    )
    x_es = ds_all.optical[es_mask]
    y_es_crit = ds_all.critical[es_mask]

    # Valid set (calibration split)
    val_mask = ds_all.split == "valid"
    x_val = ds_all.optical[val_mask]
    y_val_crit = ds_all.critical[val_mask]

    print(f"Loaded Early-Stopping set: {len(x_es)} tiles ({int(y_es_crit.sum())} critical)")
    print(f"Loaded Valid (CAL) set:    {len(x_val)} tiles ({int(y_val_crit.sum())} critical)")

    return x_es, y_es_crit, x_val, y_val_crit


def run_seed_study(seeds: List[int], epochs: int = 40):
    print("=" * 70)
    print("ORBIT-PRISM: Optical Seed Sensitivity Study (Task B)")
    print("=" * 70)

    x_es, y_es_crit, x_val, y_val_crit = load_evaluation_data()

    temp_dir = Path(tempfile.gettempdir()) / "orbit_prism_seeds"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temporary model weights directory (outside repo): {temp_dir}")

    results: List[Dict] = []

    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---")
        out_model_path = temp_dir / f"optical_seed_{seed}.keras"

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--output_model",
            str(out_model_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Training failed for seed {seed}:\n{proc.stderr}")
            continue

        # Parse epochs run from stdout
        # e.g., Epoch 00013: early stopping or Epoch 13/40
        epochs_run = None
        for line in proc.stdout.splitlines():
            if "Epoch " in line and "early stopping" in line:
                # e.g. "Epoch 00013: early stopping"
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1].replace(":", "").isdigit():
                    epochs_run = int(parts[1].replace(":", ""))
            elif "Epoch " in line and "/" in line and line.strip().startswith("Epoch "):
                # fallback: track maximum epoch seen
                try:
                    ep_str = line.strip().split()[1].split("/")[0]
                    epochs_run = max(epochs_run or 0, int(ep_str))
                except Exception:
                    pass

        if epochs_run is None:
            epochs_run = epochs

        # Load trained model to evaluate
        model = keras.models.load_model(str(out_model_path))

        # Evaluate on early stopping set
        p_es_preds, _ = model.predict(x_es, batch_size=64, verbose=0)
        p_es = p_es_preds.flatten()
        auroc_es = float(roc_auc_score(y_es_crit, p_es))

        # Evaluate on valid set
        p_val_preds, _ = model.predict(x_val, batch_size=64, verbose=0)
        p_val = p_val_preds.flatten()
        auroc_val = float(roc_auc_score(y_val_crit, p_val))

        # 90% recall threshold on valid split
        # Threshold at 10th percentile of critical positives
        val_positives = p_val[y_val_crit == 1]
        thr_90 = float(np.quantile(val_positives, 0.10))
        recall_at_90 = float(((p_val >= thr_90) & (y_val_crit == 1)).sum() / (y_val_crit == 1).sum())

        print(
            f"Seed {seed:2d} | Epochs: {epochs_run:2d} | "
            f"AUROC ES: {auroc_es:.4f} | AUROC Valid: {auroc_val:.4f} | "
            f"Thr@90: {thr_90:.4f} | Recall@90: {recall_at_90:.4f}"
        )

        results.append({
            "seed": seed,
            "epochs_run": epochs_run,
            "auroc_early_stopping": round(auroc_es, 4),
            "auroc_valid": round(auroc_val, 4),
            "threshold_90": round(thr_90, 4),
            "recall_valid_at_90": round(recall_at_90, 4),
        })

    df = pd.DataFrame(results)

    # Save results/optical_seed_study.csv
    df.to_csv(RESULTS_CSV, index=False, lineterminator="\n")
    print(f"\nSaved {RESULTS_CSV}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SEED STUDY SUMMARY STATISTICS (N = 8)")
    print("=" * 70)
    print(f"AUROC Early Stopping: Mean = {df['auroc_early_stopping'].mean():.4f}, Std = {df['auroc_early_stopping'].std():.4f}")
    print(f"AUROC Valid (CAL):    Mean = {df['auroc_valid'].mean():.4f}, Std = {df['auroc_valid'].std():.4f}")
    print(f"Recall @ 90% Valid:   Mean = {df['recall_valid_at_90'].mean():.4f}, Std = {df['recall_valid_at_90'].std():.4f}")
    print(f"Epochs Run:           Mean = {df['epochs_run'].mean():.1f}, Std = {df['epochs_run'].std():.1f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 9)), help="Seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=40, help="Max training epochs per seed")
    args = parser.parse_args()

    run_seed_study(args.seeds, epochs=args.epochs)


if __name__ == "__main__":
    main()
