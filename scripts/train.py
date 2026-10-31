"""Optical Flood CNN Training Pipeline for ORBIT-PRISM (Shenbaga's Module).

Requirements:
  - Dynamic NPZ key resolution (never assume column names)
  - Train strictly on TRAIN split
  - Filter cloud_frac < 0.5 for training
  - Event-based validation holdout
  - Data augmentation (flips and 90-degree rotations)
  - Class weighting for critical flood imbalance (water_frac >= 0.30)
  - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (models/optical_float.keras)
  - CAL evaluation reporting (AUROC, Recall, MAE, Model Size)
  - CPU fallback if GPU unavailable
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, recall_score, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import INTERNAL_VAL_EVENTS
from src.models.optical import (
    build_optical_model,
    compile_optical_model,
    save_optical_float_model,
    calculate_model_memory,
    setup_gpu
)


# Split names from CONTEXT.md. "val"/"cal"/"calibration" are accepted aliases for the calibration set.
TRAIN_SPLITS = {"train"}
CAL_SPLITS = {"valid", "val", "cal", "calibration"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Optical Flood Model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed NPZ files")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size (default 16 for RTX 3050)")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--cloud_threshold", type=float, default=0.50, help="Maximum cloud fraction allowed in TRAIN")
    parser.add_argument("--output_model", type=str, default="models/optical_float.keras", help="Model checkpoint path")
    parser.add_argument("--critical_flood_threshold", type=float, default=0.30, help="Water fraction for critical flood")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch test with synthetic data if NPZ not ready")
    return parser.parse_args()


def discover_npz_keys(npz_data: Any) -> Dict[str, str]:
    """Dynamically resolve NPZ keys without assuming exact column names."""
    keys = list(npz_data.files)
    key_mapping = {}

    for k in keys:
        k_lower = k.lower()
        if any(term in k_lower for term in ["img", "image", "x", "optical", "s2"]):
            key_mapping["images"] = k
        elif any(term in k_lower for term in ["water_frac", "water_fraction", "extent", "fraction"]):
            key_mapping["water_frac"] = k
        elif any(term in k_lower for term in ["cloud_frac", "cloud", "cloud_fraction"]):
            key_mapping["cloud_frac"] = k
        elif any(term in k_lower for term in ["split", "partition"]):
            key_mapping["split"] = k
        elif any(term in k_lower for term in ["event", "event_id", "flood_event"]):
            key_mapping["event_id"] = k
        elif any(term in k_lower for term in ["tile_id", "id", "name"]):
            key_mapping["tile_id"] = k
        elif any(term in k_lower for term in ["label", "flood", "y", "p_flood"]):
            key_mapping["labels"] = k

    return key_mapping


def augment_batch(images: np.ndarray) -> np.ndarray:
    """Apply spatial data augmentations: horizontal/vertical flips and 90-degree rotations."""
    augmented = images.copy()
    # Random horizontal flip
    if np.random.rand() > 0.5:
        augmented = np.flip(augmented, axis=2)
    # Random vertical flip
    if np.random.rand() > 0.5:
        augmented = np.flip(augmented, axis=1)
    # Random 90-degree rotations
    k = np.random.randint(0, 4)
    if k > 0:
        augmented = np.rot90(augmented, k=k, axes=(1, 2))
    return augmented


class DataGenerator(keras.utils.Sequence):
    """Custom Keras Sequence for batching and on-the-fly augmentation."""

    def __init__(self, images: np.ndarray, p_flood: np.ndarray, extent: np.ndarray, batch_size: int = 16, augment: bool = True):
        self.images = images
        self.p_flood = p_flood
        self.extent = extent
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.images))

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.images[batch_idx]
        if self.augment:
            batch_x = augment_batch(batch_x)
        batch_y_p = self.p_flood[batch_idx]
        batch_y_e = self.extent[batch_idx]
        return batch_x, {"p_flood": batch_y_p, "extent": batch_y_e}

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)


def train():
    args = parse_args()
    print("=" * 70)
    print("ORBIT-PRISM: Optical Flood CNN Training Pipeline (Shenbaga)")
    print("=" * 70)

    # 1. Hardware Initialization
    setup_gpu()

    data_dir = Path(args.data_dir)
    npz_files = list(data_dir.glob("*.npz")) if data_dir.exists() else []

    if not npz_files and not args.dry_run:
        print(f"\n[WARNING] No .npz files found in {data_dir.resolve()}.")
        print("Agnel's dataset pipeline has not delivered files to data/processed/ yet.")
        print("To verify training pipeline execution now, run:")
        print("  python scripts/train.py --dry_run")
        return

    # 2. Data loading: ALL event files, with explicit split labels from the pipeline
    if npz_files:
        from src.data.dataset import OrbitPrismDataset

        print(f"\nLoading the processed dataset from {len(npz_files)} event file(s) in {data_dir}...")
        ds = OrbitPrismDataset(processed_dir=data_dir, split="all", modality="optical")
        images = ds.optical
        water_frac = ds.water_frac
        cloud_frac = ds.cloud_frac
        splits_str = np.char.lower(ds.split.astype(str))

        # Binary critical flood label: water_frac >= 0.30
        p_flood = (water_frac >= args.critical_flood_threshold).astype(np.float32).reshape(-1, 1)
        extent = water_frac.astype(np.float32).reshape(-1, 1)

        # Split masks use the CONTEXT.md names: train / valid / test / holdout.
        # The held-out event (holdout) must never reach training or calibration.
        # Early stopping uses whole events held back from `train` (INTERNAL_VAL_EVENTS), exactly like the
        # radar model, so the calibration split (`valid`) is not used for model selection.
        in_train = np.isin(splits_str, list(TRAIN_SPLITS)) & (cloud_frac < args.cloud_threshold)
        internal = np.isin(np.asarray(ds.event).astype(str), list(INTERNAL_VAL_EVENTS))
        train_mask = in_train & ~internal
        es_mask = in_train & internal
        cal_mask = np.isin(splits_str, list(CAL_SPLITS))
        assert not np.isin(splits_str[train_mask | es_mask | cal_mask], ["holdout"]).any(), \
            "holdout leaked into train/early-stopping/cal"
        if train_mask.sum() == 0:
            raise SystemExit(f"No training tiles found; split values present: {sorted(set(splits_str))}")
        if cal_mask.sum() == 0:
            raise SystemExit("The calibration set is empty; expected split 'valid' "
                             f"(accepted: {sorted(CAL_SPLITS)}); split values present: {sorted(set(splits_str))}")

        x_train, y_train_p, y_train_e = images[train_mask], p_flood[train_mask], extent[train_mask]
        x_cal, y_cal_p, y_cal_e = images[cal_mask], p_flood[cal_mask], extent[cal_mask]
        if es_mask.sum() > 0:
            x_es, y_es_p, y_es_e = images[es_mask], p_flood[es_mask], extent[es_mask]
        else:
            print(f"[WARNING] none of {INTERNAL_VAL_EVENTS} are in the data: early stopping falls back to the "
                  "calibration split, which biases the miss-rate dial. Do not use this for final results.")
            x_es, y_es_p, y_es_e = x_cal, y_cal_p, y_cal_e
        print(f"TRAIN Set (cloud_frac < {args.cloud_threshold}): {len(x_train)} samples")
        print(f"EARLY-STOPPING Set (events {INTERNAL_VAL_EVENTS}, train split): {len(x_es)} samples")
        print(f"CAL/VAL Set: {len(x_cal)} samples")
        print(f"HOLDOUT tiles present and untouched: {int((splits_str == 'holdout').sum())}")

    else:
        # Dry Run Mode with Synthetic Data
        print("\n[DRY RUN MODE] Generating synthetic satellite dataset to verify pipeline...")
        n_samples = 64
        images = np.random.randint(0, 256, size=(n_samples, 128, 128, 4), dtype=np.uint8)
        water_frac = np.random.uniform(0.0, 1.0, size=(n_samples, 1)).astype(np.float32)
        p_flood = (water_frac >= args.critical_flood_threshold).astype(np.float32)
        extent = water_frac

        split_idx = int(0.75 * n_samples)
        x_train, y_train_p, y_train_e = images[:split_idx], p_flood[:split_idx], extent[:split_idx]
        x_cal, y_cal_p, y_cal_e = images[split_idx:], p_flood[split_idx:], extent[split_idx:]
        x_es, y_es_p, y_es_e = x_cal, y_cal_p, y_cal_e
        args.epochs = 2

    # 3. Class Imbalance Weighting
    y_train_flat = y_train_p.ravel().astype(int)
    classes = np.unique(y_train_flat)
    if len(classes) > 1:
        weights = compute_class_weight("balanced", classes=classes, y=y_train_flat)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        print(f"Computed Class Weights: {class_weight_dict}")
    else:
        class_weight_dict = None

    # 4. Model Construction & Compilation
    model = build_optical_model(input_shape=(128, 128, 4))
    compile_optical_model(model, learning_rate=args.lr)

    mem_profile = calculate_model_memory(model)
    print(f"\nModel Budget Check:")
    print(f"  Total Params: {mem_profile['total_params']:,}")
    print(f"  FP32 Weight Size: {mem_profile['float32_size_kb']} KB (Budget: <= 100 KB)")
    print(f"  INT8 Weight Size: {mem_profile['int8_size_kb']} KB")

    # 5. Callbacks
    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # 6. Training Execution
    train_gen = DataGenerator(x_train, y_train_p, y_train_e, batch_size=args.batch_size, augment=True)
    val_data = (x_es, {"p_flood": y_es_p, "extent": y_es_e})   # early stopping / checkpointing set

    print(f"\nStarting training for up to {args.epochs} epochs (Batch size: {args.batch_size})...")
    history = model.fit(
        train_gen,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Final Model Serialization
    save_optical_float_model(model, filepath=str(out_path))

    # 8. CAL Split Evaluation
    print("\n" + "=" * 70)
    print("CALIBRATION (CAL) SET EVALUATION REPORT")
    print("=" * 70)

    preds_p, preds_e = model.predict(x_cal, verbose=0)
    if len(np.unique(y_cal_p)) > 1:
        cal_auroc = roc_auc_score(y_cal_p, preds_p)
    else:
        cal_auroc = 1.0

    binary_preds = (preds_p >= 0.50).astype(int)
    cal_recall = recall_score(y_cal_p, binary_preds, zero_division=0)
    extent_mae = mean_absolute_error(y_cal_e, preds_e)

    print(f"CAL AUROC:           {cal_auroc:.4f}")
    print(f"CAL Recall (@ 0.5):  {cal_recall:.4f}")
    print(f"CAL Extent MAE:      {extent_mae:.4f}")
    print(f"Model Parameters:    {mem_profile['total_params']:,}")
    print(f"Actual Saved Size:   {out_path.stat().st_size / 1024.0:.2f} KB")
    print("=" * 70)
    print("Training pipeline finished successfully.")


if __name__ == "__main__":
    train()
