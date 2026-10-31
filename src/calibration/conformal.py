"""Split-conformal miss-rate dial for ORBIT-PRISM.

Implements operator-selectable target miss rates: 1% (0.01), 5% (0.05), 10% (0.10).
Uses separate calibration distributions for optical and radar paths on critical calibration tiles.

Threshold Formula:
    k = floor((n + 1) * alpha)
    where:
      - n is the number of critical calibration tiles for that path.
      - alpha is the target miss rate in {0.01, 0.05, 0.10}.
      - Threshold is the k-th smallest critical score (index k - 1 in 0-based Python indexing).

Edge Cases & Fallbacks:
    - If k == 0 (occurs when (n + 1) * alpha < 1, e.g. n < ceil(1/alpha) - 1):
      Cannot pick a valid 1-based index k. Fallback: threshold = 0.0, i.e. keep every tile
      (miss rate 0, no bandwidth saving). The minimum critical score is NOT safe here: its
      expected miss rate is up to 1/(n+1), which exceeds alpha when n + 1 < 1/alpha.

Scores: thresholds are in the units of the scores passed in. The pipeline passes RAW p_flood,
so the rule is "keep a tile if p_flood >= threshold". The bound holds only if the test
data is exchangeable with the calibration data (a held-out event may not be).
    - If n == 0: Reports unavailable with fallback threshold.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("conformal_calibration")

SUPPORTED_TARGET_ALPHAS = [0.01, 0.05, 0.10]


def compute_conformal_threshold(
    critical_scores: np.ndarray,
    alpha: float,
) -> Dict[str, Union[float, int, str, bool]]:
    """Compute split-conformal threshold for a specific target miss rate alpha.

    Args:
        critical_scores: 1D array of prediction scores for critical calibration tiles (water_frac >= 0.30).
        alpha: Target miss rate in (0, 1), e.g. 0.01, 0.05, 0.10.

    Returns:
        dict containing:
            - 'threshold': float threshold value
            - 'k': computed index k = floor((n + 1) * alpha)
            - 'n': number of critical samples
            - 'status': 'exact', 'conservative_fallback', or 'unavailable'
            - 'alpha': target alpha
    """
    clean_scores = np.asarray(critical_scores, dtype=np.float64)
    clean_scores = clean_scores[~np.isnan(clean_scores)]
    clean_scores.sort()  # Ascending order: s_(1) <= s_(2) <= ... <= s_(n)

    n = len(clean_scores)

    if n == 0:
        logger.error("No valid critical calibration scores available. Threshold cannot be computed.")
        return {
            "threshold": 0.0,
            "k": 0,
            "n": 0,
            "status": "unavailable",
            "alpha": alpha,
            "note": "Empty critical calibration set.",
        }

    # Project specified finite-sample correction: k = floor((n + 1) * alpha)
    k = math.floor((n + 1) * alpha)

    if k < 1:
        # Sample size is insufficient for the exact finite-sample quantile at this alpha
        # Example: for alpha=0.01, requires (n + 1) * 0.01 >= 1 => n >= 99
        min_required_n = math.ceil(1.0 / alpha) - 1
        # Keep every tile (threshold 0): the only choice that cannot exceed alpha. Using the
        # minimum critical score instead would give an expected miss rate up to 1/(n+1),
        # which is above alpha whenever n + 1 < 1/alpha.
        threshold = 0.0
        logger.warning(
            "Finite-sample correction k = floor((%d + 1) * %.2f) = 0. "
            "Requires n >= %d for an exact quantile. Falling back to threshold=0.0 (keep every tile).",
            n,
            alpha,
            min_required_n,
        )
        return {
            "threshold": threshold,
            "k": 0,
            "n": n,
            "status": "conservative_fallback",
            "alpha": alpha,
            "min_required_n": min_required_n,
            "note": f"Sample size n={n} < {min_required_n}; no exact quantile exists, so every tile is kept (threshold 0.0).",
        }

    # k >= 1: 1-based index k corresponds to 0-based index k - 1 in sorted array
    # Clamp to at most n - 1
    idx = min(k - 1, n - 1)
    threshold = float(clean_scores[idx])

    return {
        # Stored exactly: INT8 scores sit on a coarse grid (multiples of 1/256), so rounding the
        # cut-off up by even 5e-7 would push every tile tied at the cut-off below it and count it missed.
        "threshold": threshold,
        "k": k,
        "n": n,
        "status": "exact",
        "alpha": alpha,
        "note": f"k-th smallest score with 1-based k={k} (0-based index {idx}).",
    }


class SplitConformalDial:
    """Split-conformal calibration dial managing independent optical and radar thresholds."""

    def __init__(self, target_alphas: Optional[List[float]] = None):
        if target_alphas is None:
            target_alphas = SUPPORTED_TARGET_ALPHAS
        self.target_alphas = target_alphas
        self.thresholds: Dict[str, Dict[str, float]] = {"optical": {}, "radar": {}}
        self.metadata: Dict[str, Dict[str, dict]] = {"optical": {}, "radar": {}}
        self.n_critical: Dict[str, int] = {"optical": 0, "radar": 0}
        self.radar_basis = "radar_routed_critical_tiles"

    def calibrate_path(
        self,
        critical_scores: np.ndarray,
        sensor_path: str,
    ) -> None:
        """Calibrate thresholds for one path across all target alphas."""
        clean = np.asarray(critical_scores, dtype=np.float64)
        clean = clean[~np.isnan(clean)]
        self.n_critical[sensor_path] = len(clean)

        for alpha in self.target_alphas:
            alpha_key = str(alpha)
            res = compute_conformal_threshold(clean, alpha)
            self.thresholds[sensor_path][alpha_key] = res["threshold"]
            self.metadata[sensor_path][alpha_key] = res

        logger.info(
            "Calibrated '%s' split-conformal dial on N=%d critical tiles: %s",
            sensor_path,
            len(clean),
            self.thresholds[sensor_path],
        )

    def calibrate(
        self,
        df: pd.DataFrame,
        score_col: str = "p_flood",
        label_col: str = "critical",
        sensor_col: str = "sensor_used",
        split_col: str = "split",
        calib_splits: Optional[List[str]] = None,
        radar_all_col: Optional[str] = None,
    ) -> "SplitConformalDial":
        """Calibrate both optical and radar dials using critical calibration tiles.

        radar_all_col names a column with the radar model's score on EVERY tile. When present, the radar dial
        uses all critical calibration tiles, not only the cloudy ones routed to radar: simulated clouds touch
        optical bands only, so a radar score does not depend on whether its tile happened to be cloudy. The
        optical dial cannot do this, because the optical model is blind under cloud.
        """
        if calib_splits is None:
            calib_splits = ["valid", "calibration"]

        calib_df = df[df[split_col].isin(calib_splits)].copy()
        if calib_df.empty:
            raise ValueError(f"No calibration tiles found for splits: {calib_splits}")

        # Filter critical calibration tiles only: critical == 1
        crit_df = calib_df[calib_df[label_col] == 1]
        logger.info("Found %d critical calibration tiles across all paths.", len(crit_df))

        for path in ["optical", "radar"]:
            if path == "radar" and radar_all_col and radar_all_col in crit_df.columns:
                scores = crit_df[radar_all_col].values
                self.radar_basis = "all_critical_tiles (radar scores do not depend on simulated clouds)"
            else:
                scores = crit_df[crit_df[sensor_col] == path][score_col].values
            self.calibrate_path(scores, path)

        return self

    def get_threshold(self, sensor_path: str, alpha: Union[float, str]) -> float:
        """Retrieve threshold for a given sensor path and target miss rate."""
        alpha_str = str(alpha)
        if sensor_path not in self.thresholds:
            raise KeyError(f"Unknown sensor path: '{sensor_path}'")
        if alpha_str not in self.thresholds[sensor_path]:
            raise KeyError(f"Threshold for alpha={alpha_str} not calibrated for {sensor_path}")
        return self.thresholds[sensor_path][alpha_str]
