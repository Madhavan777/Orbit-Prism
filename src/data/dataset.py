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
    split_list: List[str] = []
    critical_list: List[int] = []
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
                split_list.append(split_name)
                critical_list.append(crit)

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
    arr_split = np.array(split_list, dtype=str)
    arr_critical = np.array(critical_list, dtype=np.int32)

    # Save compressed NPZ
    np.savez_compressed(
        out_npz,
        tile_id=arr_tile_id,
        optical=arr_optical,
        radar=arr_radar,
        water_frac=arr_water_frac,
        cloud_frac=arr_cloud_frac,
        split=arr_split,
        critical=arr_critical,
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

    @property
    def split(self) -> Optional[np.ndarray]:
        return self._data["split"] if "split" in self._data else None

    @property
    def critical(self) -> Optional[np.ndarray]:
        return self._data["critical"] if "critical" in self._data else (self.water_frac >= 0.30).astype(np.int32)

    def __len__(self) -> int:
        return len(self.tile_id)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, np.ndarray, float, int]]:
        item = {
            "tile_id": str(self.tile_id[idx]),
            "optical": self.optical[idx],
            "radar": self.radar[idx],
            "water_frac": float(self.water_frac[idx]),
            "cloud_frac": float(self.cloud_frac[idx]),
            "critical": int(self.critical[idx]) if self.critical is not None else int(self.water_frac[idx] >= 0.30),
        }
        if self.split is not None:
            item["split"] = str(self.split[idx])
        return item


class OrbitPrismDataset:
    """Unified multi-event dataset loader for CNN training and calibration.

    Aggregates tiles across processed event NPZ files and filters by split:
    - 'train': Training split
    - 'valid' or 'calibration': Validation/calibration split
    - 'holdout' or 'test': Held-out evaluation (Bolivia)
    - 'all': All available tiles

    Provides easy channel access:
    - .vv: (N, 128, 128) uint8 Sentinel-1 VV channel
    - .vh: (N, 128, 128) uint8 Sentinel-1 VH channel
    - .radar: (N, 128, 128, 2) uint8 [VV, VH]
    - .optical: (N, 128, 128, 4) uint8 [B2, B3, B4, B8]
    - .water_frac: (N,) float32 continuous flood label
    - .critical: (N,) int32 binary label (water_frac >= 0.30)
    """

    def __init__(
        self,
        processed_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
        split: Optional[str] = None,
        modality: str = "radar",
    ):
        self.processed_dir = Path(processed_dir)
        self.split_filter = split
        self.modality = modality

        tile_ids = []
        radar_list = []
        optical_list = []
        water_fracs = []
        critical_list = []
        event_list = []
        split_list = []

        npz_files = sorted(self.processed_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No processed NPZ files found in {self.processed_dir}")

        for npz_path in npz_files:
            ev_name = npz_path.stem
            ev_data = np.load(npz_path)

            ev_splits = ev_data["split"] if "split" in ev_data else np.array(["holdout" if ev_name == "Bolivia" else "train"] * len(ev_data["tile_id"]))

            if self.split_filter is not None and self.split_filter != "all":
                if self.split_filter == "test":
                    mask = np.isin(ev_splits, ["test", "holdout"])
                elif self.split_filter == "calibration":
                    mask = np.isin(ev_splits, ["valid", "calibration"])
                else:
                    mask = (ev_splits == self.split_filter)
            else:
                mask = np.ones(len(ev_splits), dtype=bool)

            if not np.any(mask):
                continue

            tile_ids.extend(ev_data["tile_id"][mask].tolist())
            water_fracs.extend(ev_data["water_frac"][mask].tolist())
            if "critical" in ev_data:
                critical_list.extend(ev_data["critical"][mask].tolist())
            else:
                critical_list.extend((ev_data["water_frac"][mask] >= 0.30).astype(int).tolist())
            event_list.extend([ev_name] * int(mask.sum()))
            split_list.extend(ev_splits[mask].tolist())

            if modality in ["radar", "both"]:
                radar_list.append(ev_data["radar"][mask])
            if modality in ["optical", "both"]:
                optical_list.append(ev_data["optical"][mask])

        self.tile_id = np.array(tile_ids, dtype=str)
        self.water_frac = np.array(water_fracs, dtype=np.float32)
        self.critical = np.array(critical_list, dtype=np.int32)
        self.event = np.array(event_list, dtype=str)
        self.split = np.array(split_list, dtype=str)

        self.radar = np.concatenate(radar_list, axis=0) if radar_list else np.empty((0, 128, 128, 2), dtype=np.uint8)
        self.optical = np.concatenate(optical_list, axis=0) if optical_list else np.empty((0, 128, 128, 4), dtype=np.uint8)

    @property
    def vv(self) -> np.ndarray:
        """Sentinel-1 VV polarization channel (N, 128, 128) uint8."""
        return self.radar[:, :, :, 0]

    @property
    def vh(self) -> np.ndarray:
        """Sentinel-1 VH polarization channel (N, 128, 128) uint8."""
        return self.radar[:, :, :, 1]

    def __len__(self) -> int:
        return len(self.tile_id)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, np.ndarray, float, int]]:
        res = {
            "tile_id": self.tile_id[idx],
            "water_frac": self.water_frac[idx],
            "critical": self.critical[idx],
            "event": self.event[idx],
            "split": self.split[idx],
        }
        if len(self.radar) > 0:
            res["radar"] = self.radar[idx]
            res["vv"] = self.vv[idx]
            res["vh"] = self.vh[idx]
        if len(self.optical) > 0:
            res["optical"] = self.optical[idx]
        return res
