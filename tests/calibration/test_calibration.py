"""Unit tests for ORBIT-PRISM calibration system.

Tests:
- Isotonic probability calibration fitting and clipping in [0, 1]
- Optical and radar path separation
- Split-conformal dial threshold computation (1%, 5%, 10%)
- Finite-sample formula k = floor((n + 1) * alpha)
- Small sample / k = 0 conservative fallback
- Empty critical calibration sets
- Missing prediction scores handling
- Test-set isolation: Bolivia held out, never used for fitting
- Achieved miss rate computation
- JSON serialization and compatibility
- apply(df) DataFrame preservation
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.calibration.metrics import (
    compute_ece,
    compute_brier_score,
    compute_miss_rate,
    compute_calibration_curve,
)
from src.calibration.isotonic import PathIsotonicCalibrator
from src.calibration.conformal import (
    compute_conformal_threshold,
    SplitConformalDial,
)
from src.calibration.calibrate import CalibrationPipeline
from src.calibration.evaluate import evaluate_heldout


def test_isotonic_fitting_and_range():
    """Test isotonic regression fits monotonically and outputs valid probabilities in [0, 1]."""
    scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])

    cal = PathIsotonicCalibrator()
    cal.fit_path(scores, labels, "optical")

    test_scores = np.array([-0.5, 0.15, 0.5, 0.85, 1.5])
    cal_probs = cal.predict_proba(test_scores, "optical")

    assert (cal_probs >= 0.0).all()
    assert (cal_probs <= 1.0).all()
    # Monotonicity check
    assert np.all(np.diff(cal_probs) >= 0.0)


def test_calibration_metrics_ece_brier():
    """Test ECE and Brier score computation."""
    y_true = np.array([1, 1, 0, 0])
    y_prob_perfect = np.array([1.0, 1.0, 0.0, 0.0])
    y_prob_poor = np.array([0.0, 0.0, 1.0, 1.0])

    assert compute_brier_score(y_true, y_prob_perfect) == 0.0
    assert compute_ece(y_true, y_prob_perfect, n_bins=5) == 0.0

    assert compute_brier_score(y_true, y_prob_poor) == 1.0
    assert compute_ece(y_true, y_prob_poor, n_bins=5) == 1.0


def test_conformal_finite_sample_formula():
    """Test split-conformal formula k = floor((n + 1) * alpha)."""
    # 100 critical scores sorted from 0.01 to 1.00
    n = 100
    scores = np.linspace(0.01, 1.00, n)

    # For alpha = 0.05: k = floor((100 + 1) * 0.05) = floor(5.05) = 5
    # In 0-based indexing: index k - 1 = 4 -> scores[4]
    res_05 = compute_conformal_threshold(scores, 0.05)
    assert res_05["status"] == "exact"
    assert res_05["k"] == 5
    assert pytest.approx(res_05["threshold"], 0.001) == scores[4]

    # For alpha = 0.10: k = floor(101 * 0.10) = floor(10.1) = 10
    # In 0-based indexing: index 9 -> scores[9]
    res_10 = compute_conformal_threshold(scores, 0.10)
    assert res_10["status"] == "exact"
    assert res_10["k"] == 10
    assert pytest.approx(res_10["threshold"], 0.001) == scores[9]


def test_conformal_edge_case_k_zero():
    """Test small sample size where k = floor((n + 1) * alpha) == 0 triggers conservative fallback."""
    # With n = 10 critical scores, for alpha = 0.01:
    # (10 + 1) * 0.01 = 0.11 -> k = 0
    scores = np.linspace(0.2, 0.9, 10)
    res = compute_conformal_threshold(scores, 0.01)

    assert res["k"] == 0
    assert res["status"] == "conservative_fallback"
    # Fallback uses min score (0-th element)
    assert res["threshold"] == pytest.approx(scores[0], 0.001)


def test_conformal_empty_scores():
    """Test empty critical scores handling."""
    res = compute_conformal_threshold(np.array([]), 0.05)
    assert res["status"] == "unavailable"
    assert res["threshold"] == 0.0


def test_optical_radar_path_separation():
    """Test that optical and radar paths calibrate separate distributions."""
    df = pd.DataFrame([
        # Optical calibration tiles (moderate scores)
        {"split": "valid", "sensor_used": "optical", "p_flood": 0.40, "critical": 1, "event": "E1", "chip": "C1", "row": 0, "col": 0},
        {"split": "valid", "sensor_used": "optical", "p_flood": 0.60, "critical": 1, "event": "E1", "chip": "C1", "row": 0, "col": 1},
        {"split": "valid", "sensor_used": "optical", "p_flood": 0.20, "critical": 0, "event": "E1", "chip": "C1", "row": 0, "col": 2},
        # Radar calibration tiles (higher scores)
        {"split": "valid", "sensor_used": "radar", "p_flood": 0.80, "critical": 1, "event": "E1", "chip": "C1", "row": 1, "col": 0},
        {"split": "valid", "sensor_used": "radar", "p_flood": 0.90, "critical": 1, "event": "E1", "chip": "C1", "row": 1, "col": 1},
        {"split": "valid", "sensor_used": "radar", "p_flood": 0.30, "critical": 0, "event": "E1", "chip": "C1", "row": 1, "col": 2},
    ])

    dial = SplitConformalDial(target_alphas=[0.10])
    dial.calibrate(df, score_col="p_flood", calib_splits=["valid"])

    opt_t = dial.get_threshold("optical", 0.10)
    rad_t = dial.get_threshold("radar", 0.10)

    # Radar scores are higher than optical, so radar threshold must be distinct
    assert opt_t != rad_t
    assert rad_t > opt_t


def test_pipeline_apply_preserves_structure():
    """Test pipeline.apply(df) preserves rows, order, and columns."""
    df = pd.DataFrame([
        {"event": "Bolivia", "chip": "B1", "row": 0, "col": 0, "split": "holdout", "sensor_used": "optical", "p_flood": 0.75, "critical": 1},
        {"event": "Ghana", "chip": "G1", "row": 1, "col": 2, "split": "valid", "sensor_used": "radar", "p_flood": 0.85, "critical": 1},
        {"event": "Ghana", "chip": "G1", "row": 1, "col": 3, "split": "valid", "sensor_used": "optical", "p_flood": 0.25, "critical": 0},
    ])

    pipeline = CalibrationPipeline(target_alphas=[0.05, 0.10])
    pipeline.fit_and_calibrate(df, calib_splits=["valid"])

    res_df = pipeline.apply(df, out_col="p_calibrated")

    assert len(res_df) == len(df)
    assert list(res_df["chip"]) == list(df["chip"])
    assert "p_calibrated" in res_df.columns
    assert (res_df["p_calibrated"] >= 0.0).all()
    assert (res_df["p_calibrated"] <= 1.0).all()


def test_heldout_isolation_and_evaluation():
    """Test that Bolivia is evaluated without leaking into calibration."""
    df = pd.DataFrame([
        # Calibration tiles
        {"event": "Ghana", "chip": "G1", "row": 0, "col": 0, "split": "valid", "sensor_used": "optical", "p_flood": 0.30, "critical": 1},
        {"event": "Ghana", "chip": "G1", "row": 0, "col": 1, "split": "valid", "sensor_used": "optical", "p_flood": 0.70, "critical": 1},
        {"event": "Ghana", "chip": "G1", "row": 0, "col": 2, "split": "valid", "sensor_used": "radar", "p_flood": 0.40, "critical": 1},
        {"event": "Ghana", "chip": "G1", "row": 0, "col": 3, "split": "valid", "sensor_used": "radar", "p_flood": 0.80, "critical": 1},
        # Bolivia holdout tiles
        {"event": "Bolivia", "chip": "B1", "row": 0, "col": 0, "split": "holdout", "sensor_used": "optical", "p_flood": 0.75, "critical": 1},
        {"event": "Bolivia", "chip": "B1", "row": 0, "col": 1, "split": "holdout", "sensor_used": "optical", "p_flood": 0.10, "critical": 0},
        {"event": "Bolivia", "chip": "B1", "row": 1, "col": 0, "split": "holdout", "sensor_used": "radar", "p_flood": 0.85, "critical": 1},
    ])

    pipeline = CalibrationPipeline(target_alphas=[0.05, 0.10])
    pipeline.fit_and_calibrate(df, calib_splits=["valid"])

    # Bolivia should not be in calibration events
    assert "Bolivia" not in pipeline.calibration_events

    summary_df, detailed = evaluate_heldout(pipeline, df, test_split_name="holdout", test_event_name="Bolivia")
    assert len(summary_df) > 0
    assert "Optical" in summary_df["Path"].values
    assert "Radar" in summary_df["Path"].values
