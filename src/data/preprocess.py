"""Optical, radar, and label preprocessing pipelines for ORBIT-PRISM.

Handles:
- Sentinel-2 13-band to 4-band [B2, B3, B4, B8] uint8 scaling [0, 4000] -> [0, 255].
- Sentinel-1 2-channel [VV, VH] dB clipping [-30, 0] -> [0, 255] uint8.
- LabelHand and JRCWaterHand flood fraction calculation and critical tile labeling.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tifffile

from src.constants import CRITICAL_THRESHOLD, OPTICAL_B, RADAR_B, TILE, GRID
from src.data.tiling import generate_tile_id

# Optical band indices in the Sentinel-2 13-band GeoTIFF:
# Index 1: B2 (Blue, 490 nm)
# Index 2: B3 (Green, 560 nm)
# Index 3: B4 (Red, 665 nm)
# Index 7: B8 (NIR, 842 nm)
S2_BAND_INDICES = [1, 2, 3, 7]
S2_MAX_DN = 4000.0

# Radar channels in the Sentinel-1 2-band GeoTIFF:
# Index 0: VV
# Index 1: VH
S1_CHANNEL_INDICES = [0, 1]
S1_MIN_DB = -30.0
S1_MAX_DB = 0.0


def preprocess_optical_chip(s2_raw: np.ndarray) -> np.ndarray:
    """Preprocess raw 13-band Sentinel-2 chip into 4-band uint8 (512, 512, 4).

    1. Extracts bands [B2, B3, B4, B8] at indices [1, 2, 3, 7].
    2. Transposes from (13, 512, 512) to (512, 512, 4) HWC.
    3. Scales digital numbers from 0-4000 to 0-255 uint8, clipping out-of-range values.

    Returns:
        np.ndarray: shape (512, 512, 4), dtype uint8.
    """
    if s2_raw.ndim != 3:
        raise ValueError(f"Expected 3D S2 array, got shape {s2_raw.shape}")

    # Verify band dimension
    if s2_raw.shape[0] == 13:
        # CHW
        selected = s2_raw[S2_BAND_INDICES, :, :]
        selected = np.transpose(selected, (1, 2, 0))
    elif s2_raw.shape[2] == 13:
        # HWC
        selected = s2_raw[:, :, S2_BAND_INDICES]
    else:
        raise ValueError(f"Expected 13 bands in S2 data, got shape {s2_raw.shape}")

    # Handle NaNs or negative reflectance values
    clean = np.nan_to_num(selected, nan=0.0)
    clipped = np.clip(clean, 0.0, S2_MAX_DN)

    # Scale to uint8 [0, 255]
    scaled = np.round(clipped * (255.0 / S2_MAX_DN))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def preprocess_radar_chip(s1_raw: np.ndarray) -> np.ndarray:
    """Preprocess raw 2-channel Sentinel-1 chip into uint8 (512, 512, 2).

    1. Identifies VV (index 0) and VH (index 1) channels.
    2. Transposes to (512, 512, 2) HWC.
    3. Handles NaNs by replacing them with -30.0 dB (scaled to 0 uint8).
    4. Clips dB values to [-30, 0].
    5. Scales to uint8 range [0, 255].

    Returns:
        np.ndarray: shape (512, 512, 2), dtype uint8.
    """
    if s1_raw.ndim != 3:
        raise ValueError(f"Expected 3D S1 array, got shape {s1_raw.shape}")

    if s1_raw.shape[0] == 2:
        selected = s1_raw[S1_CHANNEL_INDICES, :, :]
        selected = np.transpose(selected, (1, 2, 0))
    elif s1_raw.shape[2] == 2:
        selected = s1_raw[:, :, S1_CHANNEL_INDICES]
    else:
        raise ValueError(f"Expected 2 channels in S1 data, got shape {s1_raw.shape}")

    # Handle NaNs safely: invalid or no-data border pixels
    clean = np.nan_to_num(selected, nan=S1_MIN_DB)
    clipped = np.clip(clean, S1_MIN_DB, S1_MAX_DB)

    # Scale [-30, 0] to [0, 255]
    scaled = np.round((clipped - S1_MIN_DB) / (S1_MAX_DB - S1_MIN_DB) * 255.0)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def preprocess_labels_and_stats(
    label_chip: np.ndarray,
    jrc_chip: np.ndarray,
    chip_id: str,
    max_nodata_ratio: float = 0.50,
    drop_zero_valid_only: bool = True,
) -> Dict[str, Dict[str, Union[float, int, bool]]]:
    """Compute per-tile flood fractions, validity, and critical status.

    Rules:
    - Label value -1 means no-data.
    - Valid pixels: label != -1.
    - Flood pixels: (label == 1) & (jrc != 1) (permanent water subtracted).
    - water_frac = flood_pixels / valid_pixels.
    - critical = 1 if water_frac >= 0.30 else 0.
    - Drop condition:
        - If drop_zero_valid_only is True (matching tiles_base.csv contract): drop if valid_pixels == 0.
        - If drop_zero_valid_only is False: drop if (nodata_pixels / total_pixels) > max_nodata_ratio.

    Returns:
        dict: tile_id -> {
            'row': r,
            'col': c,
            'valid_pixels': int,
            'flood_pixels': int,
            'water_frac': float,
            'critical': int,
            'dropped': bool,
            'nodata_ratio': float
        }
    """
    valid_mask = (label_chip != -1)
    flood_mask = (label_chip == 1) & (jrc_chip != 1)

    tile_total_pixels = TILE * TILE  # 16,384

    results = {}
    for r in range(GRID):
        for c in range(GRID):
            tile_id = generate_tile_id(chip_id, r, c)
            sl = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))

            n_valid = int(valid_mask[sl].sum())
            n_flood = int(flood_mask[sl].sum())
            nodata_ratio = 1.0 - (n_valid / tile_total_pixels)

            if n_valid == 0:
                wf = np.nan
                crit = 0
                dropped = True
            else:
                wf = float(n_flood / n_valid)
                crit = 1 if wf >= CRITICAL_THRESHOLD else 0
                if drop_zero_valid_only:
                    dropped = False
                else:
                    dropped = nodata_ratio > max_nodata_ratio

            results[tile_id] = {
                "row": r,
                "col": c,
                "valid_pixels": n_valid,
                "flood_pixels": n_flood,
                "water_frac": wf,
                "critical": crit,
                "dropped": dropped,
                "nodata_ratio": nodata_ratio,
            }

    return results
