# ORBIT-PRISM Statistical Calibration System
**Author:** Agnel Vishwak (Data Pipeline + Calibration Engineer)  
**Branch:** `agnell-ship-it`  
**System Objective:** Confidence-aware bandwidth optimization through path-separated isotonic probability calibration and split-conformal miss-rate dials.

---

## 1. Overview and Core Philosophy

In the ORBIT-PRISM visual triage architecture, satellite image tiles must be prioritized for downlink under severe bandwidth (256 kbps, 19.2 MB raw ceiling per 10-minute pass) and on-device SRAM (4.2 MB) constraints.

To enable satellite operators to make principled trade-offs, the calibration system provides two critical stages:
1. **Isotonic Probability Calibration:** Maps uncalibrated model scores (optical and radar) into empirical probabilities of critical flooding.
2. **Split-Conformal Miss-Rate Dial:** Selects decision thresholds that provably control the critical-tile miss rate to at most a target $\alpha \in \{1\%, 5\%, 10\%\}$ on exchangeable data.

---

## 2. Optical and Radar Path Separation

The optical and radar sensing modalities face fundamentally different error structures:
- Optical models are degraded by cloud cover, atmospheric haze, and cloud shadows.
- Radar (SAR) is cloud-invariant, but experiences false alarms on smooth surfaces (specular reflection from flat runways, deserts) and radar shadow in rugged terrain.

**Strict Separation Rule:**
- The optical calibrator fits strictly on optical tiles (`sensor_used == "optical"`).
- The radar calibrator fits strictly on radar tiles (`sensor_used == "radar"`).
- Optical and radar calibration score distributions are **never** pooled together.

---

## 3. Stage 1: Isotonic Probability Calibration

- **Algorithm:** Non-parametric monotonic step function fitted using `sklearn.isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")`.
- **Training Data:** Calibration split only (`split == "valid"` in official mode, or `split == "calibration"` in event mode). Never sees Bolivia held-out test data.
- **Metrics Evaluated:**
  - **Expected Calibration Error (ECE):**
    $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  - **Brier Score:**
    $$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$
- **Integration Function:**
  ```python
  from src.calibration.calibrate import apply
  df_calibrated = apply(df, score_col="p_flood", out_col="p_calibrated")
  ```
  Preserves original DataFrame columns, tile IDs, and row order.

---

## 4. Stage 2: Split-Conformal Miss-Rate Dial

The operator can specify an allowable miss-rate budget $\alpha \in \{0.01, 0.05, 0.10\}$ (1%, 5%, 10%).

### 4.1 Statistical Formulation & Finite-Sample Correction
Let $S_{\text{crit}} = \{s_{(1)}, s_{(2)}, \dots, s_{(n)}\}$ be the sorted calibration scores of critical flood tiles ($y_i = 1$) in ascending order:
$$s_{(1)} \le s_{(2)} \le \dots \le s_{(n)}$$

A tile is downlinked if $s \ge \tau$. A critical tile is missed if $s < \tau$.
To bound the probability of missed critical tiles by $\alpha$, the project specification establishes the finite-sample correction index:
$$k = \left\lfloor (n + 1) \cdot \alpha \right\rfloor$$

- **1-based indexing:** The threshold $\tau$ is the $k$-th smallest critical score $s_{(k)}$.
- **0-based indexing (Python):** The array index is $\text{idx} = k - 1$.
- **Why this works:** There are at most $k - 1$ critical calibration samples with score strictly less than $s_{(k)}$. Thus the empirical miss rate is at most:
  $$\frac{k - 1}{n} \le \frac{(n + 1)\alpha - 1}{n} < \alpha$$

### 4.2 Edge Cases and Fallback Protocol
1. **$k = 0$ ($n < \lceil 1/\alpha \rceil - 1$):**
   - For $\alpha = 0.01$, exact non-parametric quantile estimation requires $n \ge 99$ critical calibration tiles.
   - For $\alpha = 0.05$, requires $n \ge 19$.
   - For $\alpha = 0.10$, requires $n \ge 9$.
   - When $k = 0$, index $k - 1 = -1$ is invalid.
   - **Conservative Fallback:** The threshold is set to $s_{(1)}$ (index 0, the minimum critical calibration score) or $0.0$. This ensures the miss rate remains $\le \alpha$ (0 missed in calibration).
   - The condition is explicitly logged and recorded in `results/calibration.json` with status `"conservative_fallback"`.
2. **$n = 0$ (Empty critical set):**
   - Status set to `"unavailable"`, threshold defaults to 0.0 with warning.

---

## 5. Output Specification (`results/calibration.json`)

The output matches the team file contract specified in CONTEXT.md:
```json
{
  "version": 1,
  "calibration_split": "valid",
  "method": "split_conformal",
  "optical": {
    "0.01": 0.3542,
    "0.05": 0.4812,
    "0.1": 0.5510
  },
  "radar": {
    "0.01": 0.2814,
    "0.05": 0.4120,
    "0.1": 0.4950
  },
  "critical_calibration_tiles": {
    "optical": 48,
    "radar": 54
  },
  "decision_rule": "keep_tile if p_flood >= threshold"
}
```

---

## 6. Held-Out Test Evaluation Protocol (Bolivia Only)

- **Test Set:** `Bolivia` (`split == "holdout"`, 15 chips, 231 tiles, 42 critical flood tiles).
- **Strict Isolation:** Bolivia data is never used to fit isotonic regression, select conformal thresholds, or adjust hyperparameters.
- **Reporting Format:**
  ```
  Path    | Target | Achieved | Critical count | Missed count | Threshold
  Optical | 1%     | 0.00%    | 12             | 0            | 0.3542
  Optical | 5%     | 4.17%    | 12             | 0            | 0.4812
  ...
  ```

---

## 7. How to Consume Calibration in Downstream Modules

### For Harish (Embedded / INT8 Model Evaluation):
Evaluate memory footprint against the uncalibrated scores, then verify calibrated ECE using:
```python
from src.calibration.isotonic import PathIsotonicCalibrator
cal = PathIsotonicCalibrator()
cal.fit(scores_df)
calibrated_df = cal.apply(scores_df)
```

### For Shenbaga and Akhilesh (Telemetry Scheduler / Sim):
Load the operational miss-rate thresholds directly from `results/calibration.json`:
```python
import json
with open("results/calibration.json") as f:
    calib = json.load(f)

# Select threshold for 5% target miss rate
target_miss_rate = "0.05"
optical_threshold = calib["optical"][target_miss_rate]
radar_threshold = calib["radar"][target_miss_rate]

# Apply triage filter:
tile_kept = (tile.p_flood >= (radar_threshold if tile.sensor_used == "radar" else optical_threshold))
```

### For Madhavan (Evaluation & Pitch):
Run evaluation from the repo root:
```bash
python -m src.calibration.calibrate
python -m src.calibration.evaluate
```
