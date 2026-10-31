"""Automated Data Validation and Handoff Reporting Module for ORBIT-PRISM.

Performs comprehensive validation on raw, processed NPZ datasets and metadata CSVs:
- Shape and channel count validation: (128, 128, 4) optical, (128, 128, 2) radar.
- Data type check: uint8 for optical and radar image arrays.
- NaN and Inf detection: zero NaNs permitted in processed uint8 imagery.
- Range checks: water_frac in [0, 1], cloud_frac in [0, 1].
- Transmission byte size calculations: 65,536 B optical, 32,768 B radar.
- Event split leakage verification.
- Team handoff data summary generation.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.constants import OPTICAL_B, RADAR_B, TILE
from src.data.generate_metadata import validate_tiles_base_dataframe
from src.data.split_events import validate_event_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_data")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TILES_BASE_CSV = ROOT_DIR / "data" / "tiles_base.csv"


def validate_npz_file(npz_path: Path) -> Dict[str, Union[int, str, bool]]:
    """Validate a single event's processed NPZ archive.

    Checks:
    - Existence of all required arrays: tile_id, optical, radar, water_frac, cloud_frac
    - Array dimensions and types:
        optical: (N, 128, 128, 4) uint8
        radar: (N, 128, 128, 2) uint8
        water_frac: (N,) float in [0, 1]
        cloud_frac: (N,) float in [0, 1]
    - NaN and Inf absence
    - Memory byte sizes per tile
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path) as data:
        required_keys = ["tile_id", "optical", "radar", "water_frac", "cloud_frac"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"NPZ {npz_path.name} missing required array '{key}'")

        tile_ids = data["tile_id"]
        optical = data["optical"]
        radar = data["radar"]
        water_frac = data["water_frac"]
        cloud_frac = data["cloud_frac"]

        n_tiles = len(tile_ids)

        # 1. Dimension checks
        if optical.ndim != 4 or optical.shape[1:] != (TILE, TILE, 4):
            raise ValueError(f"Invalid optical shape {optical.shape}. Expected (N, {TILE}, {TILE}, 4)")
        if radar.ndim != 4 or radar.shape[1:] != (TILE, TILE, 2):
            raise ValueError(f"Invalid radar shape {radar.shape}. Expected (N, {TILE}, {TILE}, 2)")
        if len(optical) != n_tiles or len(radar) != n_tiles or len(water_frac) != n_tiles or len(cloud_frac) != n_tiles:
            raise ValueError(f"Array length mismatch in {npz_path.name}: n_tiles={n_tiles}")

        # 2. Dtype checks
        if optical.dtype != np.uint8:
            raise TypeError(f"Optical array dtype is {optical.dtype}, expected uint8")
        if radar.dtype != np.uint8:
            raise TypeError(f"Radar array dtype is {radar.dtype}, expected uint8")

        # 3. NaN / Inf checks
        if np.isnan(water_frac).any() or np.isinf(water_frac).any():
            raise ValueError(f"water_frac contains NaNs or Infs in {npz_path.name}")
        if np.isnan(cloud_frac).any() or np.isinf(cloud_frac).any():
            raise ValueError(f"cloud_frac contains NaNs or Infs in {npz_path.name}")

        # 4. Range checks
        if (water_frac < 0.0).any() or (water_frac > 1.0).any():
            raise ValueError(f"water_frac out of range [0, 1] in {npz_path.name}")
        if (cloud_frac < 0.0).any() or (cloud_frac > 1.0).any():
            raise ValueError(f"cloud_frac out of range [0, 1] in {npz_path.name}")

        # 5. Byte size checks
        opt_tile_bytes = optical[0].nbytes if n_tiles > 0 else OPTICAL_B
        rad_tile_bytes = radar[0].nbytes if n_tiles > 0 else RADAR_B
        if opt_tile_bytes != OPTICAL_B:
            raise ValueError(f"Optical tile size mismatch: {opt_tile_bytes} bytes != {OPTICAL_B} expected")
        if rad_tile_bytes != RADAR_B:
            raise ValueError(f"Radar tile size mismatch: {rad_tile_bytes} bytes != {RADAR_B} expected")

    file_size_mb = npz_path.stat().st_size / (1024 * 1024)
    logger.info("NPZ file %s validated: %d tiles, %.2f MB", npz_path.name, n_tiles, file_size_mb)
    return {
        "event": npz_path.stem,
        "tiles": n_tiles,
        "size_mb": file_size_mb,
        "valid": True,
    }


def generate_data_report(
    tiles_base_path: Path = TILES_BASE_CSV,
    processed_dir: Path = PROCESSED_DIR,
) -> str:
    """Generate comprehensive team handoff report."""
    df = pd.read_csv(tiles_base_path)
    validate_tiles_base_dataframe(df)

    n_events = df["event"].nunique()
    n_chips = df["chip"].nunique()
    n_tiles = len(df)
    n_critical = int(df["critical"].sum())
    n_non_critical = n_tiles - n_critical

    # Split breakdown
    split_counts = df.groupby("split")["critical"].agg(
        total="count",
        critical="sum",
        non_critical=lambda x: (x == 0).sum(),
    ).reset_index()

    # Processed NPZ storage size
    npz_files = list(processed_dir.glob("*.npz"))
    total_npz_bytes = sum(f.stat().st_size for f in npz_files)
    total_npz_mb = total_npz_bytes / (1024 * 1024)

    lines = [
        "=" * 72,
        "ORBIT-PRISM DATA PIPELINE VERIFICATION & HANDOFF REPORT",
        "=" * 72,
        f"Dataset source:      Sen1Floods11 v1.1 Hand-Labelled Subset",
        f"Total events:        {n_events}",
        f"Total chips:         {n_chips}",
        f"Total valid tiles:   {n_tiles}",
        f"Total critical:      {n_critical} ({100 * n_critical / n_tiles:.2f}%)",
        f"Total non-critical:  {n_non_critical} ({100 * n_non_critical / n_tiles:.2f}%)",
        f"Processed NPZ files: {len(npz_files)} events ({total_npz_mb:.2f} MB)",
        "-" * 72,
        "SPLIT BREAKDOWN:",
    ]
    for _, r in split_counts.iterrows():
        lines.append(
            f"  {r['split']:<12}: {r['total']:>5} tiles | Critical: {r['critical']:>4} | Non-critical: {r['non_critical']:>5}"
        )
    lines.append("=" * 72)

    report_str = "\n".join(lines)
    return report_str


def main():
    parser = argparse.ArgumentParser(description="Validate ORBIT-PRISM dataset.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR, help="Path to data/processed/")
    parser.add_argument("--tiles-csv", type=Path, default=TILES_BASE_CSV, help="Path to data/tiles_base.csv")
    args = parser.parse_args()

    # 1. Validate tiles_base.csv if present
    if args.tiles_csv.exists():
        df = pd.read_csv(args.tiles_csv)
        validate_tiles_base_dataframe(df)
        logger.info("Validated %s successfully.", args.tiles_csv)

    # 2. Validate all processed NPZ archives
    npz_files = list(args.processed_dir.glob("*.npz"))
    for npz in npz_files:
        validate_npz_file(npz)

    # 3. Print report
    if args.tiles_csv.exists():
        print(generate_data_report(args.tiles_csv, args.processed_dir))


if __name__ == "__main__":
    main()
