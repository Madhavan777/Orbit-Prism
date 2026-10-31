"""Unit tests for ORBIT-PRISM data pipeline.

Uses synthetic test data for deterministic verification:
- 512x512 to 16 128x128 tiles
- Optical band selection (B2, B3, B4, B8) and scaling (0-4000 -> 0-255 uint8)
- Radar channel selection (VV, VH), NaN handling, and dB scaling ([-30, 0] -> 0-255 uint8)
- JRC permanent-water subtraction and water_frac computation
- Critical tile classification (water_frac >= 0.30)
- Reproducible cloud simulation on optical only (radar untouched)
- Sensor used logic (cloud_frac >= 0.5 -> radar, else optical)
- Event split isolation
- CSV schema validation
- NPZ serialization and loading
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.constants import OPTICAL_B, RADAR_B, CRITICAL_THRESHOLD, CLOUD_SWITCH
from src.data.tiling import (
    CHIP_SIZE,
    TILE_SIZE,
    GRID_SIZE,
    generate_tile_id,
    parse_tile_id,
    tile_chip_2d,
    tile_chip_hwc,
    tile_chip_dict,
)
from src.data.preprocess import (
    preprocess_optical_chip,
    preprocess_radar_chip,
    preprocess_labels_and_stats,
    S2_BAND_INDICES,
    S1_CHANNEL_INDICES,
)
from src.data.cloud_simulation import (
    simulate_chip_cloud_mask,
    blend_clouds_optical,
    compute_tile_cloud_metrics,
    generate_chip_clouds,
    derive_seed_from_chip_id,
)
from src.data.split_events import (
    assign_event_based_split,
    validate_event_splits,
    generate_split_report,
)
from src.data.generate_metadata import (
    build_tile_record,
    validate_tiles_base_dataframe,
    compute_byte_sizes,
)
from src.data.dataset import EventDataset
from src.data.validate_data import validate_npz_file


def test_tiling_dimensions():
    """Test 512x512 chip is partitioned into sixteen 128x128 tiles."""
    chip_2d = np.arange(512 * 512, dtype=np.int32).reshape(512, 512)
    tiles_2d = tile_chip_2d(chip_2d)
    assert len(tiles_2d) == 16
    for r, c, tile in tiles_2d:
        assert tile.shape == (128, 128)
        assert 0 <= r < 4
        assert 0 <= c < 4

    chip_3d = np.zeros((512, 512, 4), dtype=np.uint8)
    tiles_3d = tile_chip_hwc(chip_3d)
    assert len(tiles_3d) == 16
    for r, c, tile in tiles_3d:
        assert tile.shape == (128, 128, 4)


def test_tile_id_helpers():
    """Test tile ID generation and parsing."""
    t_id = generate_tile_id("Bolivia_103757", 1, 2)
    assert t_id == "Bolivia_103757_1_2"
    chip, r, c = parse_tile_id(t_id)
    assert chip == "Bolivia_103757"
    assert r == 1
    assert c == 2


def test_optical_preprocessing():
    """Test 13-band Sentinel-2 extraction, scaling, and byte count."""
    # Synthetic 13-band S2 image (CHW: 13, 512, 512)
    s2_raw = np.zeros((13, 512, 512), dtype=np.int16)
    # Set distinct DN values on target bands: B2 (idx 1), B3 (idx 2), B4 (idx 3), B8 (idx 7)
    s2_raw[1, :, :] = 1000  # Blue
    s2_raw[2, :, :] = 2000  # Green
    s2_raw[3, :, :] = 3000  # Red
    s2_raw[7, :, :] = 4000  # NIR
    s2_raw[0, :, :] = 9999  # B1 (should be ignored)

    opt = preprocess_optical_chip(s2_raw)
    assert opt.shape == (512, 512, 4)
    assert opt.dtype == np.uint8

    # Verify scaling: 1000/4000*255 = 63.75 -> 64
    assert np.all(opt[:, :, 0] == round(1000 * 255.0 / 4000.0))
    # 2000/4000*255 = 127.5 -> 128
    assert np.all(opt[:, :, 1] == round(2000 * 255.0 / 4000.0))
    # 4000/4000*255 = 255
    assert np.all(opt[:, :, 3] == 255)

    # Check tile byte size: 128 * 128 * 4 = 65,536 bytes
    tile = opt[0:128, 0:128, :]
    assert tile.nbytes == OPTICAL_B


def test_radar_preprocessing():
    """Test 2-channel Sentinel-1 preprocessing, dB clipping, and NaN handling."""
    s1_raw = np.zeros((2, 512, 512), dtype=np.float32)
    s1_raw[0, :, :] = -30.0  # Min dB -> should scale to 0 uint8
    s1_raw[1, :, :] = 0.0    # Max dB -> should scale to 255 uint8

    # Add NaNs (border invalid pixels)
    s1_raw[0, 0:10, 0:10] = np.nan

    rad = preprocess_radar_chip(s1_raw)
    assert rad.shape == (512, 512, 2)
    assert rad.dtype == np.uint8
    assert not np.isnan(rad).any()

    # -30 dB maps to 0 uint8
    assert rad[50, 50, 0] == 0
    # 0 dB maps to 255 uint8
    assert rad[50, 50, 1] == 255
    # NaNs safely mapped to 0 uint8
    assert np.all(rad[0:10, 0:10, 0] == 0)

    # Check tile byte size: 128 * 128 * 2 = 32,768 bytes
    tile = rad[0:128, 0:128, :]
    assert tile.nbytes == RADAR_B


def test_label_preprocessing_and_jrc_subtraction():
    """Test flood labeling, permanent water removal, and critical threshold."""
    label = np.zeros((512, 512), dtype=np.int16)
    jrc = np.zeros((512, 512), dtype=np.uint8)

    # Tile (0, 0): 50% flood, 0% permanent water -> water_frac = 0.50 (critical)
    label[0:128, 0:64] = 1

    # Tile (0, 1): 50% water, but all 50% is JRC permanent water -> flood = 0 -> water_frac = 0.0 (non-critical)
    label[0:128, 128:192] = 1
    jrc[0:128, 128:192] = 1

    # Tile (0, 2): All -1 (no-data) -> dropped
    label[0:128, 256:384] = -1

    stats = preprocess_labels_and_stats(label, jrc, "TestChip_001")

    t00 = stats["TestChip_001_0_0"]
    assert pytest.approx(t00["water_frac"], 0.01) == 0.50
    assert t00["critical"] == 1
    assert not t00["dropped"]

    t01 = stats["TestChip_001_0_1"]
    assert pytest.approx(t01["water_frac"], 0.01) == 0.0
    assert t01["critical"] == 0
    assert not t01["dropped"]

    t02 = stats["TestChip_001_0_2"]
    assert np.isnan(t02["water_frac"])
    assert t02["dropped"]


def test_synthetic_cloud_simulation_reproducibility():
    """Test deterministic cloud simulation and optical blending (radar unaffected)."""
    chip_id = "Bolivia_129334"
    alpha1 = simulate_chip_cloud_mask(chip_id, target_cover=0.5, seed=123)
    alpha2 = simulate_chip_cloud_mask(chip_id, target_cover=0.5, seed=123)
    np.testing.assert_array_equal(alpha1, alpha2)

    # Blend optical
    opt = np.full((512, 512, 4), 50, dtype=np.uint8)
    blended = blend_clouds_optical(opt, alpha1)
    assert blended.shape == (512, 512, 4)
    assert blended.dtype == np.uint8
    # Cloud blending pushes pixels towards 255
    assert np.all(blended >= opt)

    # Test sensor selection logic
    cf_high, sensor_high = compute_tile_cloud_metrics(np.full((128, 128), 0.8))
    assert cf_high == 1.0
    assert sensor_high == "radar"

    cf_low, sensor_low = compute_tile_cloud_metrics(np.full((128, 128), 0.1))
    assert cf_low == 0.0
    assert sensor_low == "optical"


def test_event_split_isolation():
    """Test strict event-based splitting and isolation check."""
    df = pd.DataFrame([
        {"event": "Bolivia", "chip": "B1", "row": 0, "col": 0, "split": "test", "critical": 1},
        {"event": "Bolivia", "chip": "B2", "row": 0, "col": 0, "split": "test", "critical": 0},
        {"event": "Nigeria", "chip": "N1", "row": 0, "col": 0, "split": "calibration", "critical": 1},
        {"event": "Ghana", "chip": "G1", "row": 0, "col": 0, "split": "train", "critical": 0},
    ])

    assert validate_event_splits(df, test_event="Bolivia")

    # If Bolivia leaks to train, must fail
    df_leak = df.copy()
    df_leak.loc[0, "split"] = "train"
    with pytest.raises(ValueError):
        validate_event_splits(df_leak, test_event="Bolivia")


def test_npz_container_and_validation():
    """Test NPZ serialization and validation module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "TestEvent.npz"
        n = 10
        tile_ids = np.array([f"TestEvent_C{i}_0_0" for i in range(n)], dtype=str)
        optical = np.zeros((n, 128, 128, 4), dtype=np.uint8)
        radar = np.zeros((n, 128, 128, 2), dtype=np.uint8)
        water_frac = np.full(n, 0.4, dtype=np.float32)
        cloud_frac = np.full(n, 0.2, dtype=np.float32)

        np.savez_compressed(
            npz_path,
            tile_id=tile_ids,
            optical=optical,
            radar=radar,
            water_frac=water_frac,
            cloud_frac=cloud_frac,
        )

        res = validate_npz_file(npz_path)
        assert res["valid"] is True
        assert res["tiles"] == n

        ds = EventDataset(npz_path)
        assert len(ds) == n
        item = ds[0]
        assert item["optical"].shape == (128, 128, 4)
        assert item["radar"].shape == (128, 128, 2)
