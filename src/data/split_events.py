"""Event-based and official dataset splitting module for ORBIT-PRISM.

Implements strict event-based splitting:
- TEST: Bolivia (held-out test event, never leaked)
- CALIBRATION: One or two events with sufficient critical tiles (e.g. Nigeria, Spain)
- TRAIN: All remaining eligible events

Also provides official split mapping (holdout, valid, train, test) for compatibility
with CONTEXT.md locked decisions.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("split_events")

DEFAULT_TEST_EVENT = "Bolivia"
DEFAULT_CALIBRATION_EVENTS = ["Nigeria", "Spain"]  # Rich in critical flood tiles (71 + 46 = 117)

OFFICIAL_SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "holdout": "flood_bolivia_data.csv",
}


def load_official_chip_splits(splits_dir: Path) -> Dict[str, str]:
    """Load official Sen1Floods11 chip-level splits from CSV files."""
    chip_to_split = {}
    for split_name, filename in OFFICIAL_SPLIT_FILES.items():
        file_path = splits_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required split file: {file_path}")
        with open(file_path, newline="") as f:
            for row in csv.reader(f):
                if row:
                    chip_id = row[1].replace("_LabelHand.tif", "").replace("_S1Hand.tif", "")
                    chip_to_split[chip_id] = split_name
    return chip_to_split


def assign_event_based_split(
    event_name: str,
    test_event: str = DEFAULT_TEST_EVENT,
    calib_events: Optional[List[str]] = None,
) -> str:
    """Assign split strictly based on event name.

    Returns: 'test', 'calibration', or 'train'
    """
    if calib_events is None:
        calib_events = DEFAULT_CALIBRATION_EVENTS

    if event_name == test_event:
        return "test"
    elif event_name in calib_events:
        return "calibration"
    else:
        return "train"


def validate_event_splits(
    df: pd.DataFrame,
    split_col: str = "split",
    event_col: str = "event",
    test_event: str = DEFAULT_TEST_EVENT,
) -> bool:
    """Strict automated validation of event split isolation.

    Validates:
    - No event overlap between any splits.
    - Test event is held out and exclusive.
    - All tiles of any event belong to exactly one split.
    """
    event_split_counts = df.groupby(event_col)[split_col].nunique()
    multi_split_events = event_split_counts[event_split_counts > 1]
    if not multi_split_events.empty:
        raise ValueError(
            f"Event split leak detected! Events spanning multiple splits: {dict(multi_split_events)}"
        )

    splits = df[split_col].unique()
    event_sets = {s: set(df[df[split_col] == s][event_col].unique()) for s in splits}

    # Pairwise overlap check
    split_names = list(event_sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = event_sets[s1].intersection(event_sets[s2])
            if overlap:
                raise ValueError(f"Event overlap between {s1} and {s2}: {overlap}")

    # Check test event
    if test_event in df[event_col].values:
        test_splits = df[df[event_col] == test_event][split_col].unique()
        if len(test_splits) != 1 or test_splits[0] not in ["test", "holdout"]:
            raise ValueError(
                f"Test event {test_event} assigned to non-test split: {test_splits}"
            )

    logger.info("Event split isolation successfully validated. No leaks.")
    return True


def generate_split_report(
    df: pd.DataFrame,
    split_col: str = "split",
    event_col: str = "event",
    crit_col: str = "critical",
) -> pd.DataFrame:
    """Generate formatted summary report:
    Split | Number of events | Total tiles | Critical tiles | Non-critical tiles
    """
    records = []
    for split_val, group in df.groupby(split_col):
        n_events = group[event_col].nunique()
        total_tiles = len(group)
        crit_tiles = int(group[crit_col].sum())
        non_crit_tiles = total_tiles - crit_tiles
        records.append({
            "Split": split_val,
            "Number of events": n_events,
            "Total tiles": total_tiles,
            "Critical tiles": crit_tiles,
            "Non-critical tiles": non_crit_tiles,
        })
    report_df = pd.DataFrame(records)
    return report_df


def print_split_report(report_df: pd.DataFrame) -> None:
    """Print markdown-style report table."""
    print("\n" + "=" * 70)
    print("EVENT SPLIT REPORT")
    print("=" * 70)
    print(f"{'Split':<15} | {'Events':>8} | {'Total tiles':>12} | {'Critical':>10} | {'Non-critical':>12}")
    print("-" * 70)
    for _, row in report_df.iterrows():
        print(f"{row['Split']:<15} | {row['Number of events']:>8} | {row['Total tiles']:>12} | {row['Critical tiles']:>10} | {row['Non-critical tiles']:>12}")
    print("=" * 70 + "\n")


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    tiles_path = root_dir / "data" / "tiles_base.csv"
    if not tiles_path.exists():
        raise FileNotFoundError(f"Missing {tiles_path}")

    df = pd.read_csv(tiles_path)
    print("1. Official Sen1Floods11 Split Report (CONTEXT.md locked):")
    rep_official = generate_split_report(df)
    print_split_report(rep_official)

    print("2. Strict Event-Based Split Report (TEST=Bolivia, CALIBRATION=Nigeria+Spain, TRAIN=Remaining):")
    df_event = df.copy()
    df_event["event_split"] = df_event["event"].apply(assign_event_based_split)
    validate_event_splits(df_event, split_col="event_split", event_col="event", test_event="Bolivia")
    rep_event = generate_split_report(df_event, split_col="event_split")
    print_split_report(rep_event)


if __name__ == "__main__":
    main()
