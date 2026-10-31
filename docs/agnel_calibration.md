# ORBIT-PRISM Statistical Calibration System
**Author:** Agnel Vishwak (Data Pipeline + Calibration Engineer)  
**Branch:** `agnell-ship-it`  
**System Objective:** Confidence-aware bandwidth optimization through path-separated isotonic probability calibration and split-conformal miss-rate dials.

---

## 1. Overview and Core Philosophy

In the ORBIT-PRISM visual triage architecture, satellite image tiles must be prioritized for downlink under severe bandwidth (256 kbps, 19.2 MB raw ceiling per 10-minute pass) and on-device SRAM (4.2 MB) constraints.

To enable satellite operators to make principled trade-offs, the calibration system provides two critical stages:
1. **Isotonic Probability Calibration:** Maps uncalibrated model scores (optical and radar) into empirical probabilities of critical flooding.
2. **Split-Conformal Miss-Rate Dial:** Selects decision thresholds that bound the expected critical-tile miss rate by a target $\alpha \in \{1\%, 5\%, 10\%\}$ **when the test tiles are exchangeable with the calibration tiles**. The Bolivia held-out event is a different event and may not be exchangeable, so the miss rate actually achieved there is measured and reported, never assumed.

**Thresholds are in raw `p_flood` units.** The dial runs on the raw model score, so the rule `keep_tile if p_flood >= threshold` in `calibration.json` is literally what to apply. The isotonic map is used for calibrated probabilities (for example the priority score), and its knots are saved in the JSON.

---

## 2. Optical and Radar Path Separation

The optical and radar sensing modalities face fundamentally different error structures:
- Optical models are degraded by cloud cover, atmospheric haze, and cloud shadows.
- Radar (SAR) is cloud-invariant, but experiences false alarms on smooth surfaces (specular reflection from flat runways, deserts) and radar shadow in rugged terrain.

**Strict Separation Rule:**
- The optical calibrator fits strictly on optical tiles (`sensor_used == "optical"`).
- The radar calibrator fits strictly on radar tiles (`sensor_used == "radar"`).
- Optical and radar calibration score distributions are **never** pooled together.

**Radar dial basis (decided 2026-10-31):** each model keeps its own cut-offs, but the radar *dial* calibrates on the radar model's scores for all 118 critical `valid` tiles, not only the 42 routed to radar. Simulated clouds touch optical bands only, so a radar score does not depend on whether its tile was cloudy (clear vs cloudy critical tiles: median 0.213 vs 0.232, KS p = 0.40). The optical dial cannot do this because the optical model is blind under cloud, so it stays on its 76 clear tiles and its 1% target still falls back to threshold 0. The basis is recorded as `radar_calibration_basis` in `calibration.json`. This change was proposed after seeing the Bolivia dial results, so Bolivia is no longer fully independent evidence for it; the `test` split, which nothing was tuned on, is the cleaner check (`results/dial_test.csv`).

---

## 3. Stage 1: Isotonic Probability Calibration

- **Algorithm:** Non-parametric monotonic step function fitted using `sklearn.isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")`.
- **Training Data:** Calibration split only (`split == "valid"` in official mode, or `split == "calibration"` in event mode). Never sees Bolivia held-out test data.
- **Metrics Evaluated:**
  - **Expected Calibration Error (ECE):**
    $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  - **Brier Score:**
    $$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$
  - **These are in-sample.** They are computed on the tiles the isotonic map was fitted on, so ECE is about 0 by construction. The JSON stores them as `probability_calibration_metrics_in_sample`. A real calibration claim needs the same metrics on held-out tiles.
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
   - **Conservative Fallback:** The threshold is set to $0.0$, i.e. **keep every tile** (no bandwidth saving, miss rate 0). Using the minimum critical score $s_{(1)}$ instead would NOT be safe: a new critical tile falls below $s_{(1)}$ with probability $1/(n+1)$, which is larger than $\alpha$ whenever $n + 1 < 1/\alpha$ (for example 1.8% against a 1% target at $n = 54$).
   - The condition is logged and recorded in `results/calibration.json` with status `"conservative_fallback"`.
2. **$n = 0$ (Empty critical set):**
   - Status set to `"unavailable"`, threshold defaults to 0.0 with warning.

---

## 5. Output Specification (`results/calibration.json`)

The output matches the team file contract in CONTEXT.md. **The numbers below are made up to show the shape of the file; they are not results.**
```json
{
  "version": 1,
  "calibration_split": "valid",
  "method": "split_conformal",
  "optical": {"0.01": 0.0, "0.05": 0.4812, "0.1": 0.5510},
  "radar":   {"0.01": 0.0, "0.05": 0.4120, "0.1": 0.4950},
  "critical_calibration_tiles": {"optical": 48, "radar": 54},
  "decision_rule": "keep_tile if p_flood >= threshold",
  "threshold_score_space": "raw_p_flood",
  "score_source": "real_model_predictions (scores_optical_int8.csv + scores_radar_int8.csv)",
  "isotonic_knots": {"optical": {"x": [], "y": []}, "radar": {"x": [], "y": []}}
}
```
`score_source` says what the scores were. A file whose `score_source` starts with `synthetic_placeholder` came from a plumbing test and must not be used for results. `python -m src.calibration.calibrate` refuses to run without real scores unless `--allow-placeholder` is passed, and `results/calibration.json` is not committed until it is made from real scores.

---

## 6. Held-Out Test Evaluation Protocol (Bolivia Only)

- **Test Set:** `Bolivia` (`split == "holdout"`, 15 chips, 231 tiles, 42 critical flood tiles).
- **Strict Isolation:** Bolivia data is never used to fit isotonic regression, select conformal thresholds, or adjust hyperparameters.
- **Reporting Format** (layout only; these rows are invented to show the columns, not measured):
  ```
  Path    | Target | Achieved | Critical count | Missed count | Threshold
  Optical | 1%     | 0.00%    | 12             | 0            | 0.0000
  Optical | 5%     | 4.17%    | 12             | 0            | 0.4812
  ...
  ```
  Report the achieved miss rate exactly as measured, including when it exceeds the target.

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
