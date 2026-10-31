"""Calibration Pipeline Orchestration for ORBIT-PRISM.

Integrates:
1. Optical and Radar Path Separation
2. Isotonic Probability Calibration
3. Split-Conformal Miss-Rate Dial (1%, 5%, 10%)
4. Export to results/calibration.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.calibration.isotonic import PathIsotonicCalibrator
from src.calibration.conformal import SplitConformalDial, SUPPORTED_TARGET_ALPHAS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibrate")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_CSV = ROOT_DIR / "data" / "fake_tiles.csv"
DEFAULT_TILES_BASE = ROOT_DIR / "data" / "tiles_base.csv"
DEFAULT_OPTICAL_SCORES = ROOT_DIR / "results" / "scores_optical_int8.csv"
DEFAULT_RADAR_SCORES = ROOT_DIR / "results" / "scores_radar_int8.csv"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "results" / "calibration.json"


def load_dataset_for_calibration(
    tiles_base_path: Path = DEFAULT_TILES_BASE,
    optical_scores_path: Path = DEFAULT_OPTICAL_SCORES,
    radar_scores_path: Path = DEFAULT_RADAR_SCORES,
    fallback_fake_path: Path = DEFAULT_INPUT_CSV,
) -> Tuple[pd.DataFrame, str]:
    """Load and merge real model predictions if available, or fall back to synthetic placeholder tiles.

    Returns:
        (df, source_description)
    """
    if optical_scores_path.exists() and radar_scores_path.exists() and tiles_base_path.exists():
        logger.info("Integrating real model predictions: %s and %s", optical_scores_path.name, radar_scores_path.name)
        base_df = pd.read_csv(tiles_base_path)
        opt_df = pd.read_csv(optical_scores_path)
        rad_df = pd.read_csv(radar_scores_path)

        # Merge optical and radar scores
        # optical scores apply where sensor_used == 'optical', radar where sensor_used == 'radar'
        merged = base_df.copy()
        opt_map = {(r["chip"], int(r["row"]), int(r["col"])): float(r["p_flood"]) for _, r in opt_df.iterrows()}
        rad_map = {(r["chip"], int(r["row"]), int(r["col"])): float(r["p_flood"]) for _, r in rad_df.iterrows()}

        p_floods = []
        for _, row in merged.iterrows():
            key = (row["chip"], int(row["row"]), int(row["col"]))
            if row["sensor_used"] == "optical":
                p_floods.append(opt_map.get(key, np.nan))
            else:
                p_floods.append(rad_map.get(key, np.nan))

        merged["p_flood"] = p_floods
        return merged, "real_model_predictions (scores_optical_int8.csv + scores_radar_int8.csv)"

    if fallback_fake_path.exists():
        logger.info("Real model prediction CSVs not yet available. Using synthetic placeholder: %s", fallback_fake_path.name)
        df = pd.read_csv(fallback_fake_path)
        return df, "synthetic_placeholder (data/fake_tiles.csv)"

    raise FileNotFoundError("Neither real prediction CSVs nor fallback fake_tiles.csv found.")


class CalibrationPipeline:
    """End-to-end statistical calibration pipeline."""

    def __init__(self, target_alphas: Optional[List[float]] = None):
        if target_alphas is None:
            target_alphas = SUPPORTED_TARGET_ALPHAS
        self.target_alphas = target_alphas
        self.isotonic_calibrator = PathIsotonicCalibrator()
        self.conformal_dial = SplitConformalDial(target_alphas=target_alphas)
        self.is_calibrated = False
        self.calibration_split_name = "valid"
        self.calibration_events: List[str] = []

    def fit_and_calibrate(
        self,
        df: pd.DataFrame,
        score_col: str = "p_flood",
        label_col: str = "critical",
        sensor_col: str = "sensor_used",
        split_col: str = "split",
        calib_splits: Optional[List[str]] = None,
        event_col: str = "event",
    ) -> Dict[str, Union[dict, str, int]]:
        """Run complete calibration on calibration split.

        Args:
            df: DataFrame containing tile scores, splits, and labels.
        """
        if calib_splits is None:
            # Check what splits exist in df
            existing_splits = set(df[split_col].unique())
            if "valid" in existing_splits:
                calib_splits = ["valid"]
            elif "calibration" in existing_splits:
                calib_splits = ["calibration"]
            else:
                raise ValueError(f"Neither 'valid' nor 'calibration' split found in {existing_splits}")

        self.calibration_split_name = ",".join(calib_splits)
        calib_df = df[df[split_col].isin(calib_splits)].copy()
        if event_col in calib_df.columns:
            self.calibration_events = sorted(calib_df[event_col].unique().tolist())

        logger.info(
            "Fitting calibration pipeline on split '%s' (%d tiles, events: %s)",
            self.calibration_split_name,
            len(calib_df),
            self.calibration_events,
        )

        # 1. Fit Isotonic probability calibration
        self.isotonic_calibrator.fit(
            df,
            score_col=score_col,
            label_col=label_col,
            sensor_col=sensor_col,
            split_col=split_col,
            calib_splits=calib_splits,
        )

        # Apply calibration to get calibrated probabilities for conformal step
        df_calibrated = self.isotonic_calibrator.apply(
            df,
            score_col=score_col,
            sensor_col=sensor_col,
            out_col="p_calibrated",
        )

        # 2. Calibrate split-conformal miss-rate dial on critical calibration tiles
        self.conformal_dial.calibrate(
            df_calibrated,
            score_col="p_calibrated",
            label_col=label_col,
            sensor_col=sensor_col,
            split_col=split_col,
            calib_splits=calib_splits,
        )

        self.is_calibrated = True

        # Build results/calibration.json structure
        output_data = self.to_dict()
        return output_data

    def apply(
        self,
        df: pd.DataFrame,
        score_col: str = "p_flood",
        sensor_col: str = "sensor_used",
        out_col: str = "p_calibrated",
    ) -> pd.DataFrame:
        """Apply fitted isotonic calibrator to any DataFrame, preserving columns and row order."""
        if not self.is_calibrated:
            raise RuntimeError("Pipeline must be calibrated before calling apply().")
        return self.isotonic_calibrator.apply(
            df,
            score_col=score_col,
            sensor_col=sensor_col,
            out_col=out_col,
        )

    def get_threshold(self, sensor_path: str, alpha: Union[float, str]) -> float:
        """Get miss-rate threshold for a given path and target alpha."""
        return self.conformal_dial.get_threshold(sensor_path, alpha)

    def to_dict(self) -> Dict[str, Union[dict, str, int]]:
        """Export calibration results matching CONTEXT.md schema and team contracts."""
        # Top-level thresholds formatted as string keys per CONTEXT.md
        # e.g. {"0.01": t1, "0.05": t5, "0.1": t10}
        opt_thresh = {k: v for k, v in self.conformal_dial.thresholds["optical"].items()}
        rad_thresh = {k: v for k, v in self.conformal_dial.thresholds["radar"].items()}

        out = {
            "version": 1,
            "calibration_split": self.calibration_split_name,
            "method": "split_conformal",
            "optical": opt_thresh,
            "radar": rad_thresh,
            "target_alphas": [str(a) for a in self.target_alphas],
            "critical_calibration_tiles": self.conformal_dial.n_critical,
            "calibration_events": self.calibration_events,
            "score_direction": "higher_score_indicates_greater_flood_probability",
            "threshold_convention": "k_th_smallest_critical_score (1-based k = floor((n+1)*alpha))",
            "decision_rule": "keep_tile if p_flood >= threshold",
            "probability_calibration_metrics": self.isotonic_calibrator.metrics,
            "conformal_diagnostics": self.conformal_dial.metadata,
        }
        return out

    def save_json(self, output_path: Path) -> None:
        """Save calibration data to JSON."""
        data = self.to_dict()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved calibration configuration to %s", output_path)


# Module-level convenience functions
_DEFAULT_PIPELINE: Optional[CalibrationPipeline] = None


def apply(
    df: pd.DataFrame,
    score_col: str = "p_flood",
    sensor_col: str = "sensor_used",
    out_col: str = "p_calibrated",
) -> pd.DataFrame:
    """Module-level apply function to calibrate DataFrame."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None or not _DEFAULT_PIPELINE.is_calibrated:
        _DEFAULT_PIPELINE = CalibrationPipeline()
        _DEFAULT_PIPELINE.fit_and_calibrate(df, score_col=score_col, sensor_col=sensor_col)
    return _DEFAULT_PIPELINE.apply(df, score_col=score_col, sensor_col=sensor_col, out_col=out_col)


def get_threshold(sensor_path: str, alpha: Union[float, str]) -> float:
    """Module-level threshold retrieval."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None or not _DEFAULT_PIPELINE.is_calibrated:
        raise RuntimeError("Calibration pipeline has not been run.")
    return _DEFAULT_PIPELINE.get_threshold(sensor_path, alpha)


def main():
    parser = argparse.ArgumentParser(description="ORBIT-PRISM Calibration Pipeline.")
    parser.add_argument("--input", type=Path, default=None, help="Input prediction CSV (optional)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON, help="Output calibration.json")
    args = parser.parse_args()

    if args.input is not None and args.input.exists():
        df = pd.read_csv(args.input)
        source_name = f"manual_input ({args.input.name})"
    else:
        df, source_name = load_dataset_for_calibration()

    logger.info("Running calibration on: %s (%d rows)", source_name, len(df))
    pipe = CalibrationPipeline()
    pipe.fit_and_calibrate(df)
    pipe.save_json(args.output)
    print(f"Calibration successfully completed and saved to {args.output}")


if __name__ == "__main__":
    main()
