"""Cloud sensitivity analysis for ORBIT-PRISM optical flood model.

Analyzes the impact of synthetic cloud opacity on optical flood detection performance:
1. Validates soft cloud properties across cloud fraction bins on the valid split.
2. Compares model performance (AUROC) under soft continuous clouds vs. opaque step clouds.
3. Produces example visualisations of critical flood tiles under varying cloud opacities.

Outputs:
- results/optical_by_cloud.csv
- results/optical_cloud_sensitivity.csv
- results/cloud_examples.png
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import TILE, CLOUD_SWITCH
from src.data.cloud_simulation import (
    blend_clouds_optical,
    compute_tile_cloud_metrics,
    simulate_chip_cloud_mask,
)
from src.data.preprocess import preprocess_optical_chip

RAW_S2_DIR = ROOT / "data" / "raw" / "data" / "flood_events" / "HandLabeled" / "S2Hand"
TILES_BASE_CSV = ROOT / "data" / "tiles_base.csv"
MODEL_PATH = ROOT / "models" / "optical_float.keras"
RESULTS_DIR = ROOT / "results"

CLOUD_BINS = [
    ("(0,0.25]", 0.0, 0.25),
    ("(0.25,0.5]", 0.25, 0.50),
    ("(0.5,0.75]", 0.50, 0.75),
    ("(0.75,1.0]", 0.75, 1.00),
]


def load_valid_tiles(tiles_base_path: Path) -> pd.DataFrame:
    """Load valid split tile metadata from tiles_base.csv."""
    df = pd.read_csv(tiles_base_path)
    valid = df[df["split"] == "valid"].copy()
    valid["tile_id"] = (
        valid["chip"] + "_" + valid["row"].astype(str) + "_" + valid["col"].astype(str)
    )
    return valid


def build_valid_optical_variations(
    valid_df: pd.DataFrame,
    raw_s2_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Reconstruct clean, soft-cloud, and opaque-cloud optical tiles for valid split."""
    chips = valid_df["chip"].unique()
    soft_tiles: List[np.ndarray] = []
    opaque_tiles: List[np.ndarray] = []
    records: List[Dict] = []

    for chip in chips:
        s2_path = raw_s2_dir / f"{chip}_S2Hand.tif"
        if not s2_path.exists():
            raise FileNotFoundError(f"Missing S2 raw file: {s2_path}")

        s2_raw = tifffile.imread(s2_path)
        clean = preprocess_optical_chip(s2_raw)
        alpha = simulate_chip_cloud_mask(chip)
        alpha_opaque = (alpha >= 0.5).astype(np.float32)

        soft_chip = blend_clouds_optical(clean, alpha)
        opaque_chip = blend_clouds_optical(clean, alpha_opaque)

        chip_rows = valid_df[valid_df["chip"] == chip]
        for _, row in chip_rows.iterrows():
            r, c = int(row["row"]), int(row["col"])
            r_sl = slice(r * TILE, (r + 1) * TILE)
            c_sl = slice(c * TILE, (c + 1) * TILE)

            t_soft = soft_chip[r_sl, c_sl, :]
            t_opaque = opaque_chip[r_sl, c_sl, :]
            t_alpha = alpha[r_sl, c_sl]

            cf, _ = compute_tile_cloud_metrics(t_alpha)
            soft_tiles.append(t_soft)
            opaque_tiles.append(t_opaque)

            records.append({
                "tile_id": row["tile_id"],
                "chip": chip,
                "row": r,
                "col": c,
                "cloud_frac": float(row["cloud_frac"]),
                "sim_cf": float(cf),
                "critical": int(row["critical"]),
                "water_frac": float(row["water_frac"]),
                "mean_alpha": float(np.mean(t_alpha)),
                "share_alpha_lt_0_1": float(np.mean(t_alpha < 0.1)),
                "share_alpha_gt_0_9": float(np.mean(t_alpha > 0.9)),
            })

    df_out = pd.DataFrame(records)
    X_soft = np.array(soft_tiles, dtype=np.uint8)
    X_opaque = np.array(opaque_tiles, dtype=np.uint8)
    return X_soft, X_opaque, df_out


def run_cloud_analysis():
    print("=" * 70)
    print("ORBIT-PRISM: Optical Cloud Sensitivity Analysis (Task A)")
    print("=" * 70)

    if not MODEL_PATH.exists():
        sys.exit(f"Missing model: {MODEL_PATH}")
    if not TILES_BASE_CSV.exists():
        sys.exit(f"Missing tiles base: {TILES_BASE_CSV}")

    valid_df = load_valid_tiles(TILES_BASE_CSV)
    print(f"Loaded {len(valid_df)} valid tiles across {valid_df['chip'].nunique()} chips.")

    X_soft, X_opaque, df_records = build_valid_optical_variations(valid_df, RAW_S2_DIR)
    print(f"Rebuilt optical arrays: X_soft {X_soft.shape}, X_opaque {X_opaque.shape}")

    # Load float model
    model = keras.models.load_model(MODEL_PATH)
    print("Scoring soft-cloud and opaque-cloud valid tiles with models/optical_float.keras...")
    p_soft_preds, _ = model.predict(X_soft, batch_size=64, verbose=0)
    p_opaque_preds, _ = model.predict(X_opaque, batch_size=64, verbose=0)

    df_records["p_flood_soft"] = p_soft_preds.flatten()
    df_records["p_flood_opaque"] = p_opaque_preds.flatten()

    # Task A2: results/optical_by_cloud.csv
    # Columns: cloud_bin, tiles, critical, auroc, mean_alpha, share_alpha_lt_0_1, share_alpha_gt_0_9
    rows_by_cloud = []
    rows_sensitivity = []

    print("\n--- Cloud Bin Analysis ---")
    for name, b_low, b_high in CLOUD_BINS:
        if b_low == 0.0:
            mask = (df_records["cloud_frac"] >= b_low) & (df_records["cloud_frac"] <= b_high)
        else:
            mask = (df_records["cloud_frac"] > b_low) & (df_records["cloud_frac"] <= b_high)

        sub = df_records[mask]
        n_tiles = len(sub)
        n_crit = int(sub["critical"].sum())

        auc_soft = float(roc_auc_score(sub["critical"], sub["p_flood_soft"]))
        auc_opaque = float(roc_auc_score(sub["critical"], sub["p_flood_opaque"]))

        mean_alpha = float(sub["mean_alpha"].mean())
        share_lt = float(sub["share_alpha_lt_0_1"].mean())
        share_gt = float(sub["share_alpha_gt_0_9"].mean())

        print(
            f"Bin {name:12s} | Tiles: {n_tiles:4d} (Crit: {n_crit:2d}) | "
            f"AUROC Soft: {auc_soft:.4f} | AUROC Opaque: {auc_opaque:.4f} | "
            f"Mean Alpha: {mean_alpha:.4f} | <0.1: {share_lt:.4f} | >0.9: {share_gt:.4f}"
        )

        rows_by_cloud.append({
            "cloud_bin": name,
            "tiles": n_tiles,
            "critical": n_crit,
            "auroc": round(auc_soft, 4),
            "mean_alpha": round(mean_alpha, 4),
            "share_alpha_lt_0_1": round(share_lt, 4),
            "share_alpha_gt_0_9": round(share_gt, 4),
        })

        rows_sensitivity.append({
            "cloud_bin": name,
            "tiles": n_tiles,
            "critical": n_crit,
            "auroc_soft": round(auc_soft, 4),
            "auroc_opaque": round(auc_opaque, 4),
        })

    # Save results/optical_by_cloud.csv
    df_by_cloud = pd.DataFrame(rows_by_cloud)
    out_by_cloud = RESULTS_DIR / "optical_by_cloud.csv"
    df_by_cloud.to_csv(out_by_cloud, index=False, lineterminator="\n")
    print(f"\nSaved {out_by_cloud}")

    # Save results/optical_cloud_sensitivity.csv
    df_sens = pd.DataFrame(rows_sensitivity)
    out_sens = RESULTS_DIR / "optical_cloud_sensitivity.csv"
    df_sens.to_csv(out_sens, index=False, lineterminator="\n")
    print(f"Saved {out_sens}")

    # Task A4: Generate results/cloud_examples.png
    generate_cloud_examples(df_records, RAW_S2_DIR, RESULTS_DIR / "cloud_examples.png")


def find_cover_for_tile(chip_id: str, r: int, c: int, target_tile_cf: float) -> float:
    """Find chip target_cover parameter that yields desired tile cloud fraction."""
    if target_tile_cf <= 0.001:
        return 0.0
    r_sl = slice(r * TILE, (r + 1) * TILE)
    c_sl = slice(c * TILE, (c + 1) * TILE)
    best_cov = target_tile_cf
    best_diff = 1.0
    for cov in np.linspace(0.01, 0.99, 100):
        a = simulate_chip_cloud_mask(chip_id, target_cover=float(cov))
        cf, _ = compute_tile_cloud_metrics(a[r_sl, c_sl])
        diff = abs(cf - target_tile_cf)
        if diff < best_diff:
            best_diff = diff
            best_cov = float(cov)
    return best_cov


def generate_cloud_examples(df_records: pd.DataFrame, raw_s2_dir: Path, out_path: Path):
    """Save 3x5 grid demonstrating clean, soft (0.50, 0.75, 0.95), and opaque (0.95) clouds."""
    print("\nGenerating cloud example images...")

    # Choose 3 representative critical valid tiles from different events
    example_chips = [
        ("Ghana_868803", 2, 0),
        ("India_1018317", 2, 1),
        ("Paraguay_581976", 0, 2),
    ]

    target_cfs = [0.0, 0.50, 0.75, 0.95]
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    for row_idx, (chip, r, c) in enumerate(example_chips):
        s2_path = raw_s2_dir / f"{chip}_S2Hand.tif"
        s2_raw = tifffile.imread(s2_path)
        clean_chip = preprocess_optical_chip(s2_raw)

        r_sl = slice(r * TILE, (r + 1) * TILE)
        c_sl = slice(c * TILE, (c + 1) * TILE)
        tile_clean = clean_chip[r_sl, c_sl, :]

        # Retrieve water fraction for context
        match = df_records[(df_records["chip"] == chip) & (df_records["row"] == r) & (df_records["col"] == c)]
        wf = float(match["water_frac"].iloc[0]) if len(match) > 0 else 0.0
        event_name = chip.split("_")[0]

        # 1. Clean (0.0)
        rgb_clean = tile_clean[:, :, [2, 1, 0]]
        axes[row_idx, 0].imshow(rgb_clean)
        axes[row_idx, 0].set_title(f"{event_name} (WF: {wf:.2f})\nClean (CF: 0.00)", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Soft clouds for ~0.50, ~0.75, ~0.95
        last_alpha_tile = None
        for col_idx, tgt in enumerate([0.50, 0.75, 0.95], start=1):
            cov = find_cover_for_tile(chip, r, c, tgt)
            alpha_chip = simulate_chip_cloud_mask(chip, target_cover=cov)
            alpha_tile = alpha_chip[r_sl, c_sl]
            actual_cf, _ = compute_tile_cloud_metrics(alpha_tile)

            blended_tile = blend_clouds_optical(tile_clean, alpha_tile)
            rgb_soft = blended_tile[:, :, [2, 1, 0]]
            axes[row_idx, col_idx].imshow(rgb_soft)
            axes[row_idx, col_idx].set_title(f"Soft Cloud\n(CF: {actual_cf:.2f})", fontsize=10)
            axes[row_idx, col_idx].axis("off")

            if tgt == 0.95:
                last_alpha_tile = alpha_tile
                last_cf = actual_cf

        # Opaque variant at ~0.95
        alpha_opaque = (last_alpha_tile >= 0.5).astype(np.float32)
        blended_opaque = blend_clouds_optical(tile_clean, alpha_opaque)
        rgb_opaque = blended_opaque[:, :, [2, 1, 0]]
        axes[row_idx, 4].imshow(rgb_opaque)
        axes[row_idx, 4].set_title(f"Opaque Variant\n(CF: {last_cf:.2f})", fontsize=10)
        axes[row_idx, 4].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_cloud_analysis()


if __name__ == "__main__":
    main()
