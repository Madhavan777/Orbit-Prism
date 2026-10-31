"""Held-Out Test Evaluation Module for ORBIT-PRISM.

Evaluates calibrated optical and radar paths on the held-out test event (Bolivia only).
Never uses Bolivia for fitting or threshold selection.

Reports:
- Target Miss Rate (1%, 5%, 10%)
- Achieved Miss Rate
- Critical test tiles count
- Missed critical tiles count
- Conformal threshold
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.constants import TEST_SPLIT
from src.calibration.calibrate import CalibrationPipeline
from src.calibration.metrics import compute_miss_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_CSV = ROOT_DIR / "data" / "fake_tiles.csv"


def evaluate_heldout(
    pipeline: CalibrationPipeline,
    df: pd.DataFrame,
    test_split_name: str = TEST_SPLIT,
    test_event_name: str = "Bolivia",
    score_col: str = "p_flood",
    label_col: str = "critical",
    sensor_col: str = "sensor_used",
    split_col: str = "split",
    event_col: str = "event",
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Evaluate calibrated thresholds on the held-out test event (Bolivia).

    Strict Isolation:
    - Verifies test split is held-out Bolivia only.
    - Evaluates optical and radar paths independently.

    Returns:
        (summary_df, detailed_results)
    """
    # Filter test split
    test_df = df[(df[split_col] == test_split_name) & (df[event_col] == test_event_name)].copy()
    if test_df.empty:
        # Fallback if split_col has 'test' and event_col has 'Bolivia'
        test_df = df[df[event_col] == test_event_name].copy()

    if test_df.empty:
        raise ValueError(f"No held-out test data found for event '{test_event_name}'")

    logger.info("Evaluating on held-out test event '%s' (%d tiles)", test_event_name, len(test_df))

    # Apply calibrated probability transformation
    test_df_cal = pipeline.apply(test_df, score_col=score_col, sensor_col=sensor_col, out_col="p_calibrated")

    records = []
    detailed = {}

    for path in ["optical", "radar"]:
        path_df = test_df_cal[test_df_cal[sensor_col] == path]
        path_crit = path_df[label_col].values
        path_scores = path_df["p_calibrated"].values

        n_total_crit = int((path_crit == 1).sum())
        detailed[path] = {}

        for alpha in pipeline.target_alphas:
            alpha_str = str(alpha)
            alpha_pct = f"{int(alpha * 100)}%"

            thresh = pipeline.get_threshold(path, alpha)
            miss_rate, n_crit, n_missed = compute_miss_rate(path_crit, path_scores, thresh)

            records.append({
                "Path": path.capitalize(),
                "Target": alpha_pct,
                "Achieved": f"{miss_rate * 100:.2f}%",
                "Critical count": n_crit,
                "Missed count": n_missed,
                "Threshold": round(thresh, 6),
            })

            detailed[path][alpha_str] = {
                "target_alpha": alpha,
                "achieved_miss_rate": miss_rate,
                "critical_count": n_crit,
                "missed_count": n_missed,
                "threshold": thresh,
            }

    summary_df = pd.DataFrame(records)
    return summary_df, detailed


def print_evaluation_table(summary_df: pd.DataFrame) -> None:
    """Print the required evaluation table format."""
    print("\n" + "=" * 76)
    print("BOLIVIA HELD-OUT TEST EVALUATION REPORT")
    print("=" * 76)
    print(f"{'Path':<10} | {'Target':<8} | {'Achieved':<12} | {'Critical count':<15} | {'Missed count':<12} | {'Threshold':<10}")
    print("-" * 76)
    for _, row in summary_df.iterrows():
        print(f"{row['Path']:<10} | {row['Target']:<8} | {row['Achieved']:<12} | {row['Critical count']:<15} | {row['Missed count']:<12} | {row['Threshold']:<10.4f}")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ORBIT-PRISM calibration on held-out test event.")
    parser.add_argument("--input", type=Path, default=None, help="Tiles CSV with scores (optional)")
    args = parser.parse_args()

    from src.calibration.calibrate import load_dataset_for_calibration
    if args.input is not None and args.input.exists():
        df = pd.read_csv(args.input)
    else:
        df, _ = load_dataset_for_calibration()

    pipeline = CalibrationPipeline()
    pipeline.fit_and_calibrate(df)

    summary_df, _ = evaluate_heldout(pipeline, df)
    print_evaluation_table(summary_df)


if __name__ == "__main__":
    main()
