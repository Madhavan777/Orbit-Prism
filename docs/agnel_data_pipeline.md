# ORBIT-PRISM Data Pipeline Documentation
**Author:** Agnel Vishwak (Data Pipeline + Calibration Engineer)  
**Branch:** `agnell-ship-it`  
**System Objective:** TinyML satellite visual triage and confidence-aware bandwidth optimization for constrained telemetry links (256 kbps downlink, 4.2 MB SRAM).

---

## 1. Overview and Architecture

The data pipeline provides the foundation for the ORBIT-PRISM visual triage system:
- Ingests raw Sentinel-1 and Sentinel-2 hand-labelled imagery from Sen1Floods11 (v1.1).
- Preprocesses multispectral optical and dual-polarization SAR radar chips into memory-efficient, standard uint8 format.
- Partitions 512×512 chips into 4×4 grids of sixteen 128×128 tiles.
- Simulates realistic cloud occlusion reproducibly on optical data without touching radar data.
- Enforces strict event-based dataset splitting (Bolivia held-out test event, calibration events, training events).
- Exports per-event compressed NPZ containers; the validated tile metadata (`data/tiles_base.csv`) is made by `scripts/build_tiles_base.py`.

```
Raw GeoTIFF Chips (512x512)
├── Sentinel-2 L1C (13 bands) ──> [B2, B3, B4, B8] ──> Scaled [0, 255] uint8 ──> Synthetic Clouds ──> (16) 128x128x4 uint8 Tiles (65,536 B)
├── Sentinel-1 GRD (2 bands)  ──> [VV, VH] dB [-30, 0] ──> Scaled [0, 255] uint8 ──────────────────> (16) 128x128x2 uint8 Tiles (32,768 B)
└── Labels (LabelHand, JRC)   ──> Flood & Non-Perm Water ──> water_frac & critical ─────────────────> Tile Metadata Records
```

---

## 2. Dataset Acquisition

The pipeline uses the official **Sen1Floods11 v1.1 Hand-Labelled Subset**:
- Source bucket: `gs://sen1floods11/v1.1/` / `https://storage.googleapis.com/sen1floods11/v1.1/`
- Hand-labelled subset only (weakly-labelled data is excluded):
  - `LabelHand`: 446 files, ~2.44 MB (QC ground truth masks: -1 invalid, 0 not water, 1 water)
  - `JRCWaterHand`: 446 files, ~0.49 MB (JRC global surface water occurrence / permanent water)
  - `S1Hand`: 446 files, ~695.70 MB (Sentinel-1 IW GRD radar images in dB)
  - `S2Hand`: 446 files, ~971.42 MB (Sentinel-2 MSI Level-1C top-of-atmosphere reflectance scaled by 10,000)
  - `splits/flood_handlabeled/`: Official split definition CSV files (4 files, ~24 KB)
- Total raw hand-labelled download size: ~1.67 GB across 11 flood events.
- Storage destination: `data/raw/` (git-ignored, preserving original GeoTIFF files intact).

To run download:
```bash
# Download first two priority events (Bolivia and Nigeria)
python -m src.data.download --events Bolivia Nigeria

# Download all 11 hand-labelled events
python -m src.data.download
```

---

## 3. Optical Preprocessing

- **Input:** Raw Sentinel-2 13-band GeoTIFF chip (`shape: (13, 512, 512)`, dtype `int16`/`uint16`).
- **Verified Band Ordering:**
  - Band 0: B1 (Coastal, 443 nm)
  - Band 1: **B2 (Blue, 490 nm)**
  - Band 2: **B3 (Green, 560 nm)**
  - Band 3: **B4 (Red, 665 nm)**
  - Band 4: B5 (RedEdge-1)
  - Band 5: B6 (RedEdge-2)
  - Band 6: B7 (RedEdge-3)
  - Band 7: **B8 (NIR, 842 nm)**
  - Bands 8-12: B8A, B9, B10, B11, B12
- **Channel Extraction:** Extracts indices `[1, 2, 3, 7]` corresponding to `[B2, B3, B4, B8]`.
- **Scaling & Normalization:**
  - Digital Numbers scaled from `[0, 4000]` to `[0, 255]` uint8:
    $$\text{DN}_{\text{uint8}} = \text{clip}\left(\text{round}\left(\text{DN} \cdot \frac{255.0}{4000.0}\right), 0, 255\right)$$
  - Values $> 4000$ are clipped; negative values or NaNs are set to 0.
- **Output:** `(128, 128, 4)` uint8 per tile.
- **Payload byte size:** $128 \times 128 \times 4 = 65,536$ bytes.

---

## 4. Radar Preprocessing

- **Input:** Raw Sentinel-1 IW GRD 2-band GeoTIFF chip (`shape: (2, 512, 512)`, dtype `float32`).
- **Verified Channels:**
  - Channel 0: **VV** polarization
  - Channel 1: **VH** polarization
- **Scaling & Normalization:**
  - Raw pixel values are in decibels (dB).
  - Valid signal clipped to $[-30.0, 0.0]\text{ dB}$.
  - Scaled linearly to $[0, 255]$ uint8:
    $$\text{DN}_{\text{uint8}} = \text{clip}\left(\text{round}\left(\frac{\text{dB} - (-30.0)}{0.0 - (-30.0)} \cdot 255.0\right), 0, 255\right)$$
  - Border and invalid NaN pixels are safely replaced by $-30.0\text{ dB}$, mapping to `0` uint8.
- **Output:** `(128, 128, 2)` uint8 per tile.
- **Payload byte size:** $128 \times 128 \times 2 = 32,768$ bytes.
- **Radar Invariance:** Radar images are strictly never modified by synthetic cloud simulation.

---

## 5. Label Preprocessing and Critical Thresholding

- **Inputs:** `LabelHand` (-1: no-data, 0: not water, 1: water) and `JRCWaterHand` (1: permanent water).
- **Rules:**
  - Valid pixels: $\text{valid} = (\text{label} \neq -1)$.
  - Flood water (non-permanent): $\text{flood} = (\text{label} == 1) \land (\text{jrc} \neq 1)$.
  - Permanent water pixels $((\text{label} == 1) \land (\text{jrc} == 1))$ are subtracted from flood water.
  - Per-tile water fraction:
    $$\text{water\_frac} = \frac{\sum \text{flood\_pixels}}{\sum \text{valid\_pixels}}$$
  - Tile criticality:
    $$\text{critical} = \begin{cases} 1 & \text{if } \text{water\_frac} \ge 0.30 \\ 0 & \text{otherwise} \end{cases}$$
- **Filtering Contract:**
  - Tiles with zero valid pixels are dropped (311 tiles across the dataset).
  - All 6,825 valid tiles match the exact keys in `data/tiles_base.csv`.

---

## 6. Synthetic Cloud Simulation

```python
# SYNTHETIC CLOUDS: These clouds are simulated for experimentation and are NOT real cloud observations.
```

- **Mechanism:**
  1. Deterministic random seed derived from `chip_id` (e.g. SHA-256 hash).
  2. Gaussian white noise field smoothed with a 2D Gaussian filter ($\sigma = 24.0$).
  3. Chip-level target cloud cover drawn from $[0.0, 0.90]$.
  4. Continuous alpha opacity mask generated in [0, 1] with soft boundary sigmoid transitions.
  5. **Clouds are OPAQUE (project default, decided 2026-10-31).** Pixels with alpha >= 0.5 become pure white (255) in every optical band, so optical sees nothing there, as under dense real cloud; pixels outside are untouched. The earlier semi-transparent haze, I_cloudy = (1 - alpha) * I_optical + alpha * 255, is kept as a sensitivity setting (`opaque=False`, `scripts/build_processed.py --soft-clouds` into its own folder). Under haze an optical model read through 75-100% cloud tiles at AUROC 0.93, against 0.79 when opaque (results/optical_cloud_sensitivity.csv), which would have hidden the reason for the radar fallback.
  6. Per-tile cloud fraction: fraction of pixels with alpha >= 0.5. This set of pixels is the same for opaque and soft clouds, so `cloud_frac`, `sensor_used`, the splits and `data/tiles_base.csv` do not change between the two.
  7. Sensor selection contract:
     $$\text{sensor\_used} = \begin{cases} \text{radar} & \text{if } \text{cloud\_frac} \ge 0.5 \\ \text{optical} & \text{otherwise} \end{cases}$$

---

## 7. Event-Based Splitting

- **Default (official) mode, the one the team uses:** the split names are exactly `train`, `valid`, `test`, `holdout` (CONTEXT.md). The official train/valid/test sets are chip-level splits that mix events; only Bolivia is held out as a whole event.
  - **`holdout`:** `Bolivia` (15 chips, 231 tiles, 42 critical). The headline test set. Never used in training or calibration. It is never merged into `test`; request it as `holdout`.
  - **`valid`:** the calibration set (1,354 tiles across events). Calibration and the miss-rate dial fit here only.
  - **`test`:** a secondary in-distribution report (1,378 tiles), never used for tuning.
  - **`train`:** the rest.
- **Optional event-based mode** (`calibration` instead of `valid`, using Nigeria and Spain as calibration events) exists in `src/data/split_events.py` for experiments. It is not the default, and its split name `calibration` is accepted by the calibration code and the trainer.
- **Validation:** an automated test checks there is no event overlap between the holdout and the other splits.

---

## 8. Data Outputs and Artifacts

- **Processed Event Archives:** `data/processed/<event>.npz`
  - `tile_id`: (N,) string identifiers (`<chip>_<row>_<col>`)
  - `optical`: (N, 128, 128, 4) uint8
  - `radar`: (N, 128, 128, 2) uint8
  - `water_frac`: (N,) float32
  - `cloud_frac`: (N,) float32
  - `split`: (N,) string, one of `train`, `valid`, `test`, `holdout`
  - `critical`: (N,) int32
- **Metadata CSV:** `data/tiles_base.csv`, made by `python scripts/build_tiles_base.py` from the real labels and the simulated clouds. It needs no imagery, because the cloud mask depends only on the chip id.
  - Columns (9): `event, chip, row, col, split, water_frac, critical, cloud_frac, sensor_used`. The model scores `p_flood` and `extent` are added later by `scripts/merge_scores.py`, from real model outputs only.
  - Packets: `bytes_l0 = 32`, `bytes_l1 = 2048` (provisional until the scheduler owner confirms it in `src/constants.py`), `bytes_l2 = 65536` (optical) or `32768` (radar).
- **Cloud contract:** `cloud_frac` in `tiles_base.csv` is the measured fraction of the simulated cloud mask that the pixels get (alpha >= 0.5). A test checks that it matches `src/data/cloud_simulation.py`.

---

## 9. Verification Commands

```bash
# Run complete unit test suite
python -m pytest tests/data/ -v

# Validate all processed NPZ archives and metadata
python -m src.data.validate_data
```
