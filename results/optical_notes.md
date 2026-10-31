# Optical Flood Classification & Water Extent Module Notes

**Module Lead:** Shenbaga (Senior Deep Learning & TinyML Research Engineer)  
**Branch:** `shenbaga/optical`  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) / Edge MCU TinyML Deployment  

---

## 1. System & Hardware Verification (Hour 0 - 1.5)

* **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM)
* **Driver / CUDA:** Driver 610.64 / CUDA 13.3
* **TensorFlow Version:** 2.21.0
* **Keras Version:** 3.15.1
* **Memory Management Strategy:**
  * GPU memory growth dynamically enabled via `tf.config.experimental.set_memory_growth`.
  * Initial batch size target: 16 (FP32 training).
  * Seamless CPU fallback built into `setup_gpu()` to ensure unblocked execution across diverse environments.

---

## 2. TinyML Model Architecture (`src/models/optical.py`)

A dual-output lightweight Convolutional Neural Network designed for edge deployment on microcontrollers and embedded vision devices.

### Architecture Details:
1. **Input Layer:** `(None, 128, 128, 4)` uint8 satellite imagery (B, G, R, NIR).
2. **Preprocessing:** `Rescaling(1.0 / 255.0)` scaling pixel intensities to `[0.0, 1.0]`.
3. **Feature Extractor:**
   * **Stage 1:** `Conv2D(8, 3x3, stride=2, padding='same')` -> `BatchNorm` -> `ReLU` $\to (64, 64, 8)$
   * **Stage 2:** `Conv2D(16, 3x3, stride=2, padding='same')` -> `BatchNorm` -> `ReLU` $\to (32, 32, 16)$
   * **Stage 3:** `Conv2D(32, 3x3, stride=2, padding='same')` -> `BatchNorm` -> `ReLU` $\to (16, 16, 32)$
   * **Stage 4:** `Conv2D(32, 3x3, stride=2, padding='same')` -> `BatchNorm` -> `ReLU` $\to (8, 8, 32)$
   * **Pooling:** `GlobalAveragePooling2D()` $\to (32,)$
4. **Output Heads:**
   * **`p_flood` Head:** `Dense(1, activation='sigmoid')` with Binary Cross-Entropy loss. Predicts critical flood occurrence (`water_frac >= 0.30`).
   * **`extent` Head:** `Dense(1, activation='sigmoid')` with Mean Squared Error loss. Predicts fractional water coverage in $[0.0, 1.0]$.

### TinyML Profile & Budget Verification:
* **Total Parameters:** 15,682
* **Trainable Parameters:** 15,506
* **FP32 Weight Footprint:** 61.26 KB (strictly within the $\le 100\text{ KB}$ project budget)
* **Projected INT8 Weight Footprint:** ~15.31 KB
* **INT8 Compatibility:** All operators are standard TFLite INT8 quantizable layers (Conv2D, BatchNorm, ReLU, GAP, Dense, Sigmoid).

---

## 3. Test Suite & Verification (`tests/test_optical.py`)

A comprehensive 5-point unit test suite was implemented and validated:
1. `test_model_build_and_shapes`: Verified input `(None, 128, 128, 4)` and dual heads `p_flood` and `extent` `(None, 1)`.
2. `test_model_parameter_budget`: Confirmed FP32 model size is 61.26 KB ($\le 100\text{ KB}$ ceiling).
3. `test_random_tensor_forward_pass`: Validated end-to-end forward pass using synthetic uint8 satellite imagery; verified sigmoid range $[0, 1]$.
4. `test_train_step_random_data`: Validated loss computation and gradient backpropagation for 1 epoch.
5. `test_ndwi_baseline_fallback`: Validated NDWI water-index formula:
   $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR} + 10^{-7}}$$
   and the critical flood thresholding curve.

**Test Result:** All 5 unit tests passed cleanly.

---

## 5. Experimental extras (`src/disasters/`), floods and fires only

Scope is floods and forest fires (CONTEXT.md). These modules are experiments and are **not used by the evaluation**; the reference scheduler is `src/scheduler` and the reference priority score is calibrated flood probability x flood extent.

1. **Hazard registry (`registry.py`):** two profiles, flood and wildfire. Statuses describe what is in the repo: flood is `SIMULATED` (architecture and tests, not trained or evaluated on real data), wildfire is `PLANNED` (no fire data is loaded). Nothing is `IMPLEMENTED` yet. The Sen1Floods11 licence is recorded as unconfirmed.
2. **Priority engine (`risk_engine.py`):** an experimental confidence-aware priority formula. Its weights and per-hazard urgency multipliers are unvalidated assumptions. Missing exposure is flagged `UNKNOWN` instead of being treated as zero.
3. **Scheduler sketch (`scheduler_interface.py`):** a two-tier queue (32 B alert before a full tile) using the packet sizes in `src/constants.py`, with carry-over and de-duplication. Its CSV exporter also writes the team `tile_id,p_flood,extent` format.
4. **Wildfire architecture (`wildfire.py`):** an untrained 6,338-parameter network (24.76 KB float32) plus an NBR formula. It has not seen any data.
5. **Tests:** `tests/test_optical.py` and `tests/test_multi_disaster.py` run in a clean environment; `tests/test_training_data_guards.py` checks that training reads every event and never touches the holdout.

## 6. Data and outputs

* `scripts/train.py` and `scripts/predict.py` read **all** event files in `data/processed/` through `OrbitPrismDataset`, using the split names `train / valid / test / holdout`. Training uses `train` tiles with `cloud_frac < 0.5`; `valid` is the calibration set; `holdout` (Bolivia) is never read for training or calibration.
* `scripts/predict.py` writes `results/scores_optical_float.csv` (`tile_id,p_flood,extent`, `tile_id = <chip>_<row>_<col>`). The contract file `results/scores_optical_int8.csv` comes from the INT8 model (`src/compress/predict_int8.py`).
* Early stopping monitors the internal validation set (`Spain` and `Somalia` train split), keeping the `valid` calibration set completely independent.

## 7. Measured results

The optical model was trained on the processed Sen1Floods11 dataset using `scripts/train.py`:

* **Random seed:** 42
* **Hardware:** Native Windows CPU fallback (Intel x86_64, TensorFlow 2.21.0)
* **Epochs run:** 13 epochs (early stopping triggered; best weights restored from epoch 8)
* **Wall-clock training time:** ~25 seconds
* **Set sizes:**
  * TRAIN set (`cloud_frac < 0.5`, excluding Spain/Somalia): 1,984 tiles (135 critical)
  * Early-stopping set (`Spain` and `Somalia` train tiles): 300 tiles (15 critical)
  * CAL/VAL set (`valid` split): 1,354 tiles
  * HOLDOUT set (`Bolivia` event): 231 tiles (untouched, scored only)
* **Calibration split (valid) evaluation metrics:**
  * **CAL AUROC:** 0.9507
  * **CAL Recall (@ 0.5):** 0.4831
  * **CAL Extent MAE:** 0.0797
  * **Total parameters:** 15,682
  * **FP32 weight size:** 61.26 KB (checkpoint file `models/optical_float.keras`: 260.90 KB)

These numbers reflect simulated-cloud results from a single training run on the Sen1Floods11 dataset. They are not flight hardware performance measurements and are not compared against holdout metrics.

## 8. Cloud transparency finding

An analysis of optical model discrimination across cloud fraction bins on the valid split (1,354 tiles: 118 critical) was performed using `scripts/cloud_sensitivity.py`. Results are recorded in `results/optical_by_cloud.csv` and `results/optical_cloud_sensitivity.csv`, with sample visualizations in `results/cloud_examples.png`.

### Measured values by cloud bin (valid split)

| Cloud Bin | Tiles | Critical | AUROC (Soft) | AUROC (Opaque) | Mean Alpha | Pixels Alpha < 0.1 | Pixels Alpha > 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `(0, 0.25]` | 503 | 50 | 0.9637 | 0.9588 | 0.1580 | 57.67% | 0.57% |
| `(0.25, 0.5]` | 273 | 26 | 0.9562 | 0.9533 | 0.4160 | 18.49% | 7.29% |
| `(0.5, 0.75]` | 282 | 17 | 0.9747 | 0.9847 | 0.5971 | 6.44% | 19.73% |
| `(0.75, 1.0]` | 296 | 25 | 0.9261 | 0.7863 | 0.7914 | 0.88% | 43.78% |

### Analysis and plain-words conclusion

1. **Why optical appears resilient under soft clouds:** In `src/data/cloud_simulation.py`, a tile's `cloud_frac` is defined as the fraction of pixels with `alpha >= 0.5`. However, pixel blending is continuous: `blended = (1 - alpha) * optical + alpha * 255`. In the highest bin `(0.75, 1.0]`, the average pixel opacity is only `mean_alpha = 0.7914`, and only 43.78% of pixels have `alpha > 0.9`. The remaining pixels transmit 20% to 50% of the surface reflectance. Because water strongly absorbs near-infrared (B8) and shortwave-infrared bands, this residual spectral attenuation remains easily discernible to the CNN, yielding an artificially high AUROC of 0.9261 even under heavy simulated cloud cover.
2. **Impact of opaque occlusion:** When the exact same masks are made strictly opaque (`alpha_opaque = 1.0` where `alpha >= 0.5`, else `0.0`), the ground signal is genuinely blocked from view. In the `(0.75, 1.0]` bin, AUROC drops significantly from **0.9261 to 0.7863** (a reduction of ~0.14 AUROC).
3. **Recommendation:**
   - If the intent of cloud simulation is to realistically model opaque cloud cover where optical inspection fails and forces hand-off to Sentinel-1 SAR radar, the simulation should use opaque masks (`alpha = 1.0` where `alpha >= 0.5`).
   - If the intent is to model thin semi-transparent cirrus or haze, the current continuous sigmoid formulation is appropriate, but optical models will retain partial ground visibility.
   - The architectural decision to retain the current continuous simulation or transition to opaque masks is left to Madhavan.

## 9. Seed sensitivity study (Run-to-run noise)

To evaluate run-to-run variance, the optical CNN was trained across 8 distinct random seeds (seeds 1 to 8) using `scripts/optical_seed_study.py`. All temporary weights were stored outside the repository.

### Measured results per seed

| Seed | Epochs Run | AUROC (Early-Stopping) | AUROC (Valid / CAL) | Threshold (90% Recall) | Recall @ 90% (Valid) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 19 | 0.8758 | 0.9635 | 0.1176 | 0.8983 |
| 2 | 16 | 0.9053 | 0.9624 | 0.0827 | 0.8983 |
| 3 | 9 | 0.9296 | 0.9380 | 0.0802 | 0.8983 |
| 4 | 15 | 0.8922 | 0.9602 | 0.1151 | 0.8983 |
| 5 | 21 | 0.9488 | 0.9650 | 0.1279 | 0.8983 |
| 6 | 9 | 0.9336 | 0.8858 | 0.0171 | 0.8983 |
| 7 | 28 | 0.9263 | 0.9634 | 0.0936 | 0.8983 |
| 8 | 16 | 0.9661 | 0.9615 | 0.0505 | 0.8983 |

### Summary statistics (N = 8)

* **AUROC Early-Stopping (300 tiles, 15 critical):** Mean = 0.9222, Std = 0.0297
* **AUROC Valid / CAL (1,354 tiles, 118 critical):** Mean = 0.9500, Std = 0.0274
* **Recall @ 90% Valid:** Mean = 0.8983 (106 / 118 critical tiles), Std = 0.0000
* **Epochs Run:** Mean = 16.6, Std = 6.3

### Noise framing

The early-stopping set contains only 15 critical tiles and the calibration split contains 118 critical tiles. Consequently, AUROC fluctuations on the order of $\pm 0.01$ to $0.02$ are expected statistical noise rather than genuine algorithmic differences. The committed production model (trained with seed 42, AUROC 0.9507) aligns closely with the 8-seed empirical mean (0.9500) and remains unchanged.


