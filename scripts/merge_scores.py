"""Merge model predictions into data/tiles.csv (the 11-column schema).

Per CONTEXT.md:
Takes data/tiles_base.csv and merges raw model prediction scores:
- results/scores_optical_int8.csv (for optical tiles)
- results/scores_radar_int8.csv (for radar tiles)
If real score CSVs are not yet available, uses data/fake_tiles.csv placeholders
with explicit logging.

Output: data/tiles.csv
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_scores")

ROOT = Path(__file__).resolve().parent.parent
TILES_BASE = ROOT / "data" / "tiles_base.csv"
SCORES_OPTICAL = ROOT / "results" / "scores_optical_int8.csv"
SCORES_RADAR = ROOT / "results" / "scores_radar_int8.csv"
FAKE_TILES = ROOT / "data" / "fake_tiles.csv"
OUT_TILES = ROOT / "data" / "tiles.csv"


def merge_scores(
    tiles_base_path: Path = TILES_BASE,
    optical_scores_path: Path = SCORES_OPTICAL,
    radar_scores_path: Path = SCORES_RADAR,
    fake_tiles_path: Path = FAKE_TILES,
    out_path: Path = OUT_TILES,
) -> pd.DataFrame:
    if not tiles_base_path.exists():
        raise FileNotFoundError(f"Missing base tiles file: {tiles_base_path}")

    base_df = pd.read_csv(tiles_base_path)

    if optical_scores_path.exists() and radar_scores_path.exists():
        logger.info("Merging real model prediction scores from %s and %s", optical_scores_path.name, radar_scores_path.name)
        opt_df = pd.read_csv(optical_scores_path)
        rad_df = pd.read_csv(radar_scores_path)

        opt_scores = {(r["chip"], int(r["row"]), int(r["col"])): (float(r["p_flood"]), float(r.get("extent", 0.0))) for _, r in opt_df.iterrows()}
        rad_scores = {(r["chip"], int(r["row"]), int(r["col"])): (float(r["p_flood"]), float(r.get("extent", 0.0))) for _, r in rad_df.iterrows()}

        p_list, ext_list = [], []
        for _, row in base_df.iterrows():
            key = (row["chip"], int(row["row"]), int(row["col"]))
            if row["sensor_used"] == "optical":
                vals = opt_scores.get(key, (np.nan, np.nan))
            else:
                vals = rad_scores.get(key, (np.nan, np.nan))
            p_list.append(vals[0])
            ext_list.append(vals[1])

        base_df["p_flood"] = p_list
        base_df["extent"] = ext_list
    elif fake_tiles_path.exists():
        logger.info("Real model scores not found. Merging placeholder scores from %s", fake_tiles_path.name)
        fake_df = pd.read_csv(fake_tiles_path)
        fake_map = {(r["chip"], int(r["row"]), int(r["col"])): (float(r["p_flood"]), float(r["extent"])) for _, r in fake_df.iterrows()}

        p_list, ext_list = [], []
        for _, row in base_df.iterrows():
            key = (row["chip"], int(row["row"]), int(row["col"]))
            vals = fake_map.get(key, (0.0, 0.0))
            p_list.append(vals[0])
            ext_list.append(vals[1])

        base_df["p_flood"] = p_list
        base_df["extent"] = ext_list
    else:
        base_df["p_flood"] = np.nan
        base_df["extent"] = np.nan

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(out_path, index=False)
    logger.info("Wrote %s with %d rows and %d columns", out_path, len(base_df), len(base_df.columns))
    return base_df


def main():
    parser = argparse.ArgumentParser(description="Merge model prediction scores into data/tiles.csv.")
    parser.add_argument("--out", type=Path, default=OUT_TILES, help="Output tiles.csv path")
    args = parser.parse_args()
    merge_scores(out_path=args.out)


if __name__ == "__main__":
    main()
