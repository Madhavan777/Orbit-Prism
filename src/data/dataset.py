"""Dataset container, NPZ serialization, and event processor for ORBIT-PRISM.

Saves processed event datasets under data/processed/<event>.npz with arrays:
- tile_id: (N,) string array
- optical: (N, 128, 128, 4) uint8
- radar: (N, 128, 128, 2) uint8
- water_frac: (N,) float32
- cloud_frac: (N,) float32
"""

import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import tifffile
import pandas as pd

from src.data.tiling import (
    TILE_SIZE,
    GRID_SIZE,
    generate_tile_id,
    tile_chip_2d,
    tile_chip_hwc,
)
from src.data.preprocess import (
    preprocess_optical_chip,
    preprocess_radar_chip,
    preprocess_labels_and_stats,
)
from src.data.cloud_simulation import (
    generate_chip_clouds,
    derive_seed_from_chip_id,
)
from src.data.generate_metadata import build_tile_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def process_event_chips(
    event_name: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    chip_splits: Optional[Dict[str, str]] = None,
    drop_over_50_nodata: bool = False,
    target_cloud_lookup: Optional[Dict[str, float]] = None,
) -> Tuple[Path, List[Dict[str, Union[str, int, float]]], Dict[str, int]]:
    """Process all chips for a single flood event into a compressed NPZ file and metadata records.

    Args:
        event_name: e.g. 'Bolivia' or 'Nigeria'
        raw_dir: data/raw path
        processed_dir: data/processed path
        chip_splits: mapping of chip_id to split name
        drop_over_50_nodata: if True, drop tiles with >50% no-data; if False, drop only zero-valid tiles (matching tiles_base.csv)
        target_cloud_lookup: optional tile_id -> cloud_frac lookup (e.g. from existing tiles_base.csv)

    Returns:
        npz_path: Path to data/processed/<event>.npz
        metadata_records: list of metadata dicts for all valid tiles
        counts: dict with counts of generated, valid, dropped, critical, non-critical tiles
    """
    labels_dir = raw_dir / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    jrc_dir = raw_dir / "data" / "flood_events" / "HandLabeled" / "JRCWaterHand"
    s1_dir = raw_dir / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    s2_dir = raw_dir / "data" / "flood_events" / "HandLabeled" / "S2Hand"

    chip_label_files = sorted(labels_dir.glob(f"{event_name}_*_LabelHand.tif"))
    if not chip_label_files:
        raise FileNotFoundError(f"No LabelHand files found for event '{event_name}' in {labels_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_npz = processed_dir / f"{event_name}.npz"

    tile_ids: List[str] = []
    optical_list: List[np.ndarray] = []
    radar_list: List[np.ndarray] = []
    water_frac_list: List[float] = []
    cloud_frac_list: List[float] = []
    metadata_records: List[Dict[str, Union[str, int, float]]] = []

    total_generated = 0
    dropped_count = 0
    critical_count = 0
    non_critical_count = 0

    for lab_path in chip_label_files:
        chip_id = lab_path.name.replace("_LabelHand.tif", "")

        jrc_path = jrc_dir / f"{chip_id}_JRCWaterHand.tif"
        s1_path = s1_dir / f"{chip_id}_S1Hand.tif"
        s2_path = s2_dir / f"{chip_id}_S2Hand.tif"

        if not (jrc_path.exists() and s1_path.exists() and s2_path.exists()):
            logger.warning("Skipping chip %s: missing one or more component files.", chip_id)
            continue

        # Load raw GeoTIFF chips
        label_raw = tifffile.imread(lab_path)
        jrc_raw = tifffile.imread(jrc_path)
        s1_raw = tifffile.imread(s1_path)
        s2_raw = tifffile.imread(s2_path)

        # Preprocess optical and radar chips (512, 512, C) uint8
        optical_clean = preprocess_optical_chip(s2_raw)
        radar_chip = preprocess_radar_chip(s1_raw)

        # Synthetic cloud simulation on OPTICAL ONLY
        optical_cloudy, alpha_mask, cloud_stats = generate_chip_clouds(chip_id, optical_clean)

        # Preprocess labels
        label_stats = preprocess_labels_and_stats(
            label_raw,
            jrc_raw,
            chip_id,
            max_nodata_ratio=0.50,
            drop_zero_valid_only=not drop_over_50_nodata,
        )

        # Split 512x512 into 16 tiles
        opt_tiles = tile_chip_hwc(optical_cloudy)
        rad_tiles = tile_chip_hwc(radar_chip)

        opt_dict = {generate_tile_id(chip_id, r, c): arr for r, c, arr in opt_tiles}
        rad_dict = {generate_tile_id(chip_id, r, c): arr for r, c, arr in rad_tiles}

        split_name = chip_splits.get(chip_id, "holdout" if event_name == "Bolivia" else "train") if chip_splits else "holdout"

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                t_id = generate_tile_id(chip_id, r, c)
                total_generated += 1

                t_stat = label_stats[t_id]
                if t_stat["dropped"]:
                    dropped_count += 1
                    continue

                sim_cf, sim_sensor = cloud_stats[t_id]
                # If target cloud lookup is given (e.g. from existing tiles_base.csv), sync cloud_frac
                if target_cloud_lookup and t_id in target_cloud_lookup:
                    sim_cf = target_cloud_lookup[t_id]
                    sim_sensor = "radar" if sim_cf >= 0.5 else "optical"

                wf = t_stat["water_frac"]
                crit = t_stat["critical"]

                if crit == 1:
                    critical_count += 1
                else:
                    non_critical_count += 1

                tile_ids.append(t_id)
                optical_list.append(opt_dict[t_id])
                radar_list.append(rad_dict[t_id])
                water_frac_list.append(wf)
                cloud_frac_list.append(sim_cf)

                rec = build_tile_record(
                    event=event_name,
                    chip=chip_id,
                    row=r,
                    col=c,
                    split=split_name,
                    water_frac=wf,
                    critical=crit,
                    cloud_frac=sim_cf,
                    sensor_used=sim_sensor,
                    include_predictions_placeholders=True,
                )
                metadata_records.append(rec)

    # Convert to NumPy arrays matching required shapes
    arr_tile_id = np.array(tile_ids, dtype=str)
    arr_optical = np.stack(optical_list, axis=0) if optical_list else np.empty((0, 128, 128, 4), dtype=np.uint8)
    arr_radar = np.stack(radar_list, axis=0) if radar_list else np.empty((0, 128, 128, 2), dtype=np.uint8)
    arr_water_frac = np.array(water_frac_list, dtype=np.float32)
    arr_cloud_frac = np.array(cloud_frac_list, dtype=np.float32)

    # Save compressed NPZ
    np.savez_compressed(
        out_npz,
        tile_id=arr_tile_id,
        optical=arr_optical,
        radar=arr_radar,
        water_frac=arr_water_frac,
        cloud_frac=arr_cloud_frac,
    )

    counts = {
        "total_generated": total_generated,
        "valid_tiles": len(tile_ids),
        "dropped_tiles": dropped_count,
        "critical_tiles": critical_count,
        "non_critical_tiles": non_critical_count,
    }

    logger.info(
        "Event '%s' processed -> %s (Valid: %d, Dropped: %d, Critical: %d, Non-critical: %d)",
        event_name,
        out_npz.name,
        counts["valid_tiles"],
        counts["dropped_tiles"],
        counts["critical_tiles"],
        counts["non_critical_tiles"],
    )

    return out_npz, metadata_records, counts


class EventDataset:
    """Loader for processed event NPZ files."""

    def __init__(self, npz_path: Union[str, Path]):
        self.path = Path(npz_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Processed NPZ file not found: {self.path}")
        self._data = np.load(self.path)

    @property
    def tile_id(self) -> np.ndarray:
        return self._data["tile_id"]

    @property
    def optical(self) -> np.ndarray:
        return self._data["optical"]

    @property
    def radar(self) -> np.ndarray:
        return self._data["radar"]

    @property
    def water_frac(self) -> np.ndarray:
        return self._data["water_frac"]

    @property
    def cloud_frac(self) -> np.ndarray:
        return self._data["cloud_frac"]

    def __len__(self) -> int:
        return len(self.tile_id)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, np.ndarray, float]]:
        return {
            "tile_id": str(self.tile_id[idx]),
            "optical": self.optical[idx],
            "radar": self.radar[idx],
            "water_frac": float(self.water_frac[idx]),
            "cloud_frac": float(self.cloud_frac[idx]),
        }
