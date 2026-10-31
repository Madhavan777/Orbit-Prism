"""Isotonic probability calibration module for ORBIT-PRISM.

Fits independent isotonic regression models for optical and radar sensor paths:
- Uses calibration split only (never fits on test/holdout data).
- Converts raw scores into calibrated probabilities in [0, 1].
- Evaluates calibration error: ECE and Brier score.
- Provides apply(df) preserving row order and all existing columns.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.calibration.metrics import compute_ece, compute_brier_score, compute_calibration_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("isotonic_calibration")

MIN_CALIBRATION_SAMPLES = 20


class PathIsotonicCalibrator:
    """Independent isotonic calibrator for optical and radar detection paths."""

    def __init__(self):
        self.calibrators: Dict[str, IsotonicRegression] = {}
        self.fitted_counts: Dict[str, int] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False

    def fit_path(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        sensor_path: str,
    ) -> None:
        """Fit isotonic regression for a single sensor path ('optical' or 'radar')."""
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        n_samples = len(scores)
        self.fitted_counts[sensor_path] = n_samples

        if n_samples < MIN_CALIBRATION_SAMPLES:
            logger.warning(
                "Calibration dataset for '%s' path is small (N=%d < %d). "
                "Calibration may be statistically noisy.",
                sensor_path,
                n_samples,
                MIN_CALIBRATION_SAMPLES,
            )

        if n_samples == 0:
            logger.error("Cannot fit isotonic regression for '%s': 0 calibration samples.", sensor_path)
            self.calibrators[sensor_path] = None
            self.metrics[sensor_path] = {"ece": 0.0, "brier": 0.0, "samples": 0}
            return

        # Fit IsotonicRegression with clipping in [0, 1]
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(scores, labels)
        self.calibrators[sensor_path] = ir

        # Compute in-sample calibration diagnostics
        cal_probs = ir.predict(scores)
        ece = compute_ece(labels, cal_probs)
        brier = compute_brier_score(labels, cal_probs)

        self.metrics[sensor_path] = {
            "ece": round(ece, 6),
            "brier": round(brier, 6),
            "samples": n_samples,
        }
        logger.info(
            "Fitted '%s' path isotonic calibrator (N=%d, ECE=%.4f, Brier=%.4f)",
            sensor_path,
            n_samples,
            ece,
            brier,
        )

    def fit(
        self,
        df: pd.DataFrame,
        score_col: str = "p_flood",
        label_col: str = "critical",
        sensor_col: str = "sensor_used",
        split_col: str = "split",
        calib_splits: Optional[List[str]] = None,
    ) -> "PathIsotonicCalibrator":
        """Fit optical and radar isotonic calibrators on calibration data.

        Args:
            df: DataFrame containing tiles and predictions.
            calib_splits: Split values to use for calibration (default ['valid', 'calibration']).
        """
        if calib_splits is None:
            calib_splits = ["valid", "calibration"]

        calib_df = df[df[split_col].isin(calib_splits)].copy()
        if calib_df.empty:
            raise ValueError(
                f"No calibration data found for splits: {calib_splits} in column '{split_col}'"
            )

        for path in ["optical", "radar"]:
            sub = calib_df[calib_df[sensor_col] == path]
            scores = sub[score_col].values
            labels = sub[label_col].values
            self.fit_path(scores, labels, path)

        self.is_fitted = True
        return self

    def export_knots(self) -> Dict[str, Optional[Dict[str, List[float]]]]:
        """Piecewise-linear knots of each fitted path, so a calibrated probability can be
        recomputed without refitting: p = np.interp(score, x, y) (clipped at the ends)."""
        out: Dict[str, Optional[Dict[str, List[float]]]] = {}
        for path, ir in self.calibrators.items():
            if ir is None:
                out[path] = None
            else:
                out[path] = {
                    "x": [round(float(v), 6) for v in ir.X_thresholds_],
                    "y": [round(float(v), 6) for v in ir.y_thresholds_],
                }
        return out

    def predict_proba(
        self,
        scores: np.ndarray,
        sensor_path: str,
    ) -> np.ndarray:
        """Predict calibrated probabilities for a given sensor path."""
        ir = self.calibrators.get(sensor_path)
        if ir is None:
            logger.warning("Calibrator for '%s' not available; returning raw scores.", sensor_path)
            return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
        return ir.predict(np.asarray(scores, dtype=np.float64))

    def apply(
        self,
        df: pd.DataFrame,
        score_col: str = "p_flood",
        sensor_col: str = "sensor_used",
        out_col: str = "p_calibrated",
    ) -> pd.DataFrame:
        """Apply path-specific calibrator to DataFrame, preserving row order and all columns.

        Returns:
            pd.DataFrame: Copy of original DataFrame with out_col populated.
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before calling apply().")

        res_df = df.copy()
        calibrated_scores = np.zeros(len(res_df), dtype=np.float64)

        for path in ["optical", "radar"]:
            mask = (res_df[sensor_col] == path)
            if mask.any():
                raw = res_df.loc[mask, score_col].values
                calibrated_scores[mask.values] = self.predict_proba(raw, path)

        res_df[out_col] = calibrated_scores
        return res_df
