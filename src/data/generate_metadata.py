"""Metadata generation and CSV schema validation for ORBIT-PRISM.

Generates data/tiles_base.csv and computes telemetry packet transmission byte costs:
- bytes_l0: 32 bytes (alert packet)
- bytes_l1: 2048 bytes (preview thumbnail)
- bytes_l2: 65,536 bytes (optical tile) or 32,768 bytes (radar tile)
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from src.constants import OPTICAL_B, RADAR_B, L0_B
from src.data.tiling import generate_tile_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_metadata")

L1_DEFAULT_BYTES = 2048

BASE_SCHEMA_COLUMNS = [
    "event",
    "chip",
    "row",
    "col",
    "split",
    "water_frac",
    "critical",
    "cloud_frac",
    "sensor_used",
]


def compute_byte_sizes(sensor_used: str) -> Tuple[int, int, int]:
    """Calculate transmission byte sizes for L0, L1, and L2 packets.

    Returns:
        (bytes_l0, bytes_l1, bytes_l2)
    """
    bytes_l0 = L0_B  # 32
    bytes_l1 = L1_DEFAULT_BYTES  # 2048
    bytes_l2 = OPTICAL_B if sensor_used == "optical" else RADAR_B  # 65536 or 32768
    return bytes_l0, bytes_l1, bytes_l2


def build_tile_record(
    event: str,
    chip: str,
    row: int,
    col: int,
    split: str,
    water_frac: float,
    critical: int,
    cloud_frac: float,
    sensor_used: str,
    include_predictions_placeholders: bool = False,
    include_byte_sizes: bool = False,
) -> Dict[str, Union[str, int, float]]:
    """Build a single tile metadata record dictionary."""
    rec = {
        "event": event,
        "chip": chip,
        "row": int(row),
        "col": int(col),
        "split": split,
        "water_frac": round(float(water_frac), 6),
        "critical": int(critical),
        "cloud_frac": round(float(cloud_frac), 4),
        "sensor_used": sensor_used,
    }

    if include_predictions_placeholders:
        rec["p_flood"] = ""
        rec["extent"] = ""

    if include_byte_sizes:
        l0, l1, l2 = compute_byte_sizes(sensor_used)
        rec["bytes_l0"] = l0
        rec["bytes_l1"] = l1
        rec["bytes_l2"] = l2

    return rec


def validate_tiles_base_dataframe(df: pd.DataFrame) -> bool:
    """Validate DataFrame against CONTEXT.md required schema and ranges."""
    for col in BASE_SCHEMA_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column in tiles_base: '{col}'")

    # Range validations
    if not df["row"].isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid row values; must be 0..3")
    if not df["col"].isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid col values; must be 0..3")
    if not df["critical"].isin([0, 1]).all():
        raise ValueError("Invalid critical values; must be 0 or 1")
    if not df["sensor_used"].isin(["optical", "radar"]).all():
        raise ValueError("Invalid sensor_used values; must be 'optical' or 'radar'")

    # Value ranges
    if (df["water_frac"] < 0.0).any() or (df["water_frac"] > 1.0).any():
        raise ValueError("water_frac values out of range [0, 1]")
    if (df["cloud_frac"] < 0.0).any() or (df["cloud_frac"] > 1.0).any():
        raise ValueError("cloud_frac values out of range [0, 1]")

    # Consistency check: sensor_used must be radar iff cloud_frac >= 0.5
    inconsistent = df[(df["cloud_frac"] >= 0.5) & (df["sensor_used"] != "radar")]
    if not inconsistent.empty:
        raise ValueError(f"Inconsistent sensor_used when cloud_frac >= 0.5:\n{inconsistent.head()}")

    inconsistent_opt = df[(df["cloud_frac"] < 0.5) & (df["sensor_used"] != "optical")]
    if not inconsistent_opt.empty:
        raise ValueError(f"Inconsistent sensor_used when cloud_frac < 0.5:\n{inconsistent_opt.head()}")

    logger.info("tiles_base schema and range constraints verified (%d rows).", len(df))
    return True


def save_tiles_base_csv(
    records: List[Dict[str, Union[str, int, float]]],
    output_path: Path,
) -> None:
    """Save records to tiles_base.csv preserving exact schema."""
    if not records:
        raise ValueError("No records to save.")

    df = pd.DataFrame(records)
    validate_tiles_base_dataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, lineterminator="\n")
    logger.info("Saved %d records to %s", len(df), output_path)
