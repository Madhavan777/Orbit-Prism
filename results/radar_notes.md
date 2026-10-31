# ORBIT-PRISM: Sentinel-1 C-band SAR Radar AI Notes
**Author**: Harish D  
**Branch**: `harish/radar-compress`  
**Date**: 2026-10-31  

---

## 1. Satellite Sensor & Radar Polarization Technology
- **Satellite System**: European Space Agency (ESA) Sentinel-1 C-band Synthetic Aperture Radar (SAR) (5.405 GHz).
- **Polarimetric Channels**:
  - **Channel 1 (Index 0)**: **VV** (Vertical transmit, Vertical receive)
  - **Channel 2 (Index 1)**: **VH** (Vertical transmit, Horizontal receive)
- **SAR Physical Backscatter Principles**:
  - **Specular Reflection**: Calm flood waters act as smooth specular surfaces, reflecting radar pulses away from the antenna. This results in distinct dark, low-backscatter values (typically $-22$ dB to $-16$ dB in calibrated $\sigma^0$).
  - **Diffuse / Volume Scattering**: Rough dry land, vegetation, and urban infrastructure produce high diffuse or volume backscatter (typically $-14$ dB to $-6$ dB), creating sharp contrast even through dense optical cloud cover.
- **Dataset**: Sen1Floods11 — `S1Hand` (hand-labeled flood events).
- **Active Scopes**: Flood detection (Sen1Floods11 `S1Hand`, loaded). Forest fire is the second hazard but **no fire radar data is loaded yet**; whether radar helps for fire is a hypothesis to be measured.
- **Out of scope**: hazards other than floods and forest fires (see "Scope" in CONTEXT.md).

---

## 2. ESA SNAP (Sentinel Application Platform) Ground-Side Integration
- **Tool**: ESA SNAP with **Sentinel-1 Microwave Toolbox (S1TBX)**.
- **Download**: [ESA SNAP Download](https://step.esa.int/main/download/snap-download/)
- **System Check**: Executed system scan via `src/models/snap_workflow.py`: `installed: False`.
- **Role & Boundary**:
  - ESA SNAP is a **ground-side preprocessing tool**. It is **not** part of the flight software or onboard TinyML runtime.
  - **Raw GRD Ingestion Pipeline**: When ingesting raw Sentinel-1 Level-1 GRD/SLC products, SNAP executes the 7-step graph:
    1. `Read` Sentinel-1 SAR GRD product.
    2. `Apply-Orbit-File` with Precise Orbit Ephemerides (POEORB).
    3. `ThermalNoiseRemoval` to strip the instrument thermal noise floor.
    4. `Calibration` to radiometric $\sigma^0$ (Sigma0) backscatter.
    5. `Speckle-Filter` (Refined Lee or Lee 5x5).
    6. `Terrain-Correction` (Range-Doppler with SRTM 1-sec HGT DEM).
    7. `BandSelect` & `Write` to extract VV and VH GeoTIFFs for Python ingestion.
  - **Duplicate Processing Prevention**: Sen1Floods11 `S1Hand` images are **already** calibrated to $\sigma^0$ and orthorectified. Therefore, running SNAP on `S1Hand` is redundant and unnecessary.

---

## 3. Onboard Radar CNN Architecture (`src/models/radar.py`)
- **Input Dimensions**: `(128, 128, 2)` (Channel 0: VV, Channel 1: VH, normalized to $[0.0, 1.0]$).
- **Layers**:
  1. `Conv2D(8, 3x3, stride=2, padding='same', use_bias=False)` $\to$ `BatchNorm` $\to$ `ReLU` $\to (64, 64, 8)$
  2. `Conv2D(16, 3x3, stride=2, padding='same', use_bias=False)` $\to$ `BatchNorm` $\to$ `ReLU` $\to (32, 32, 16)$
  3. `Conv2D(32, 3x3, stride=2, padding='same', use_bias=False)` $\to$ `BatchNorm` $\to$ `ReLU` $\to (16, 16, 32)$
  4. `Conv2D(32, 3x3, stride=1, padding='same', use_bias=False)` $\to$ `BatchNorm` $\to$ `ReLU` $\to (16, 16, 32)$
  5. `GlobalAveragePooling2D` $\to$ vector length `32`
  6. `Head 1 (p_flood)`: `Dense(1, activation='sigmoid')` (critical threshold `water_frac >= 0.30`)
  7. `Head 2 (extent)`: `Dense(1, activation='sigmoid')` (continuous water fraction $[0, 1]$)
- **Total Parameters**: **15,538** parameters.
- **Quantization Compatibility**: Pure INT8 operations (`TFLITE_BUILTINS_INT8`), verified scale and zero-point parameters.

---

## 4. Memory Profiling & 4.2 MB SRAM Budget
- **Budget**: `4,200,000` bytes (4.2 MB raw ceiling, ~4,101.6 KiB).
- **Weight Footprint (Flash)**:
  - FLOAT32: **59.54 KB**
  - INT8: **15.18 KB** (**74.5% reduction**)
- **Peak Activation Memory (SRAM — ESTIMATED)**:
  - Model: Ping-pong arena lifetime model (Conv1 input buffer + output buffer).
  - FLOAT32: **256.00 KB** (6.2% of SRAM budget).
  - INT8: **64.00 KB** (1.5% of SRAM budget, **75.0% reduction**).
  - **Verdict**: Strictly meets the 4.2 MB design constraint with ample headroom for runtime queues and radio DMA buffers.

---

## 5. Dataset Audit & Safety
- **Flood Dataset**: Sen1Floods11 hand-labelled subset, downloaded by `python scripts/download_flood.py` into `data/raw/data/flood_events/HandLabeled/` (git-ignored). Processed tiles go to `data/processed/<event>.npz`.
- **Forest-Fire Dataset**: **not loaded.** There is no `data/raw/forest_fire/` and no fire radar data in the repository. See the Hazards section of CONTEXT.md.
- **Out-of-scope hazards**: no code or data for them exists in the repository (checked by `test_15` on code and data files).
- **Numbers above**: weights and activations in section 4 are the radar-only figures from an untrained architecture. The combined, regenerated table is `results/memory_table.csv`.

---

## 6. Measured results

### Training Run Details
- **Random Seed**: Fixed at `42` (`np.random.seed(42)`, `tf.random.set_seed(42)`).
- **Epochs Trained**: Configured for `40` epochs with early stopping (`patience=7` on validation loss). Best weights restored from epoch 25; training terminated at epoch 32.
- **Wall-Clock Time**: ~3.5 minutes total (~6.5 seconds per epoch).
- **Hardware Used**: Native Windows CPU (Intel Core i5, oneDNN AVX2 optimizations enabled; GPU disabled per TF >= 2.11 Windows limitation).

### Valid-Split Metrics (Float Radar Model)
Evaluated on the full `valid` split (1,354 tiles, 118 critical):
- **AUROC**: `0.9426` (on routed tiles with `cloud_frac >= 0.5`: `0.9605`)
- **Recall @ 0.90 Target Threshold**: `0.9068` (measured at threshold `p_flood >= 0.0623`; reaches `0.9407` at `0.0600`)
- **Extent MAE**: `0.1271`

### INT8 vs. Float Comparison (Valid Split)
- **AUROC**:
  - Float: `0.9426` (Routed: `0.9605`)
  - INT8: `0.9281` (Routed: `0.9558`)
  - Drop: `-0.0145` AUROC under full-integer quantization.
- **Recall @ 0.5**:
  - Float: `0.0000` (0 / 118 detected at fixed 0.5 threshold due to uncalibrated probability skew on rare-class floods)
  - INT8: `0.0000` (Consistent across both models before calibration; calibrated operating threshold is ~0.06)
- **Max Absolute Error**:
  - `p_flood` max absolute error: `0.1441` (mean error: `0.0798`)
  - `extent` max absolute error: `0.1486` (mean error: `0.0791`)
- **Expected Calibration Error (ECE)**:
  - Float: `0.0402` (Routed: `0.0362`)
  - INT8: `0.1009` (Routed: `0.1071`)

### Radar Memory Profile Summary
From `results/memory_table.csv`:
| Model | Dtype | Weights (B) | Peak Activation (B) | Tile Buffer (B) | Total SRAM (B) | MACs | Recall @ 0.5 | ECE | Basis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Radar** | FLOAT32 | 61,104 | 262,144 | 32,768 | 356,016 | 5,308,480 | 0.0000 | 0.0362 | `trained_model_file` |
| **Radar** | INT8 | 15,552 | 65,536 | 32,768 | 113,856 | 5,308,480 | 0.0000 | 0.1071 | `trained_model_file` |

- **Weight Compression**: 74.5% reduction (61,104 B $\to$ 15,552 B).
- **Peak Activation Memory**: 75.0% reduction (262,144 B $\to$ 65,536 B).
- **Total SRAM Requirement**: 113,856 B (~111.2 KiB), consuming only **2.71%** of the 4.2 MB SRAM ceiling.

### Optical Rescaling Finding & Proposed Architecture Fix
- **Observation**:
  In `results/memory_table.csv`, the Optical INT8 model has a peak activation of `327,680 bytes` (only a ~16.7% drop from FP32's `393,216 bytes`). In comparison, Radar INT8 drops from `262,144 bytes` to `65,536 bytes` (a 75.0% reduction).
- **Root Cause in `src/models/optical.py`**:
  The optical model accepts `uint8` inputs and uses `layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)`. In TFLite, this `Rescaling` operator executes in `float32`, casting the entire input tile tensor `(128, 128, 4)` to 32-bit floats (`128 * 128 * 4 * 4 = 262,144 bytes`) and keeping it resident in the dynamic activation arena alongside layer activations.
- **Proposed Fix**:
  Normalize optical tiles to float `[0.0, 1.0]` (or int8 `[-128, 127]`) in the data loader before model entry, matching the radar pipeline design. Removing `layers.Rescaling` from the Keras graph allows TFLite converter to quantize the entire graph to 8-bit integers without float intermediate buffers, dropping Optical INT8 peak activation from `327,680 bytes` to `65,536 bytes`.

### Strict Holdout Isolation Confirmation
- **Holdout Event**: `Bolivia` (231 tiles total, 42 critical flood tiles).
- **Audit**: Bolivia tiles were strictly excluded from:
  1. Training batches
  2. Early stopping validation splits
  3. Calibration / threshold selection
  4. Quantization calibration representative datasets
  5. Validation metric reporting
- Holdout tiles were purely scored and written to `results/scores_radar_float.csv`, `results/scores_radar_int8.csv`, `results/scores_optical_float.csv`, and `results/scores_optical_int8.csv` for downstream evaluation by Madhavan as dictated by competition rules.


