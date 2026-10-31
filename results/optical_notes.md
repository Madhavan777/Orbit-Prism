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

## 4. Agnel's Dataset Pipeline Status

* Inspection script: `scripts/inspect_data.py`
* Target folder: `data/processed/*.npz`
* Current status: Awaiting Agnel's processed dataset files.
* Pipeline rules established:
  * Inspect actual NPZ keys dynamically without hardcoded assumptions.
  * Train exclusively on `TRAIN` split with cloud filter `cloud_frac < 0.5`.
  * Event-based validation for out-of-distribution evaluation.
  * Data augmentation: Random horizontal/vertical flips and 90-degree rotations.
  * `TEST` split strictly isolated (never used for model selection).
