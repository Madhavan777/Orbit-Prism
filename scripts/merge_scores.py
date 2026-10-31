"""Merge model predictions into data/tiles.csv (the 11-column schema).

Per CONTEXT.md, takes data/tiles_base.csv and merges the raw model scores:
- results/scores_optical_int8.csv (used for tiles with sensor_used == "optical")
- results/scores_radar_int8.csv   (used for tiles with sensor_used == "radar")

data/tiles.csv holds REAL model scores only. If a score file is missing, or any tile has no
score, this script stops with an error. For a plumbing test only, pass --allow-placeholder:
the synthetic scores of data/fake_tiles.csv are merged and written to
data/tiles_placeholder.csv (git-ignored), never to data/tiles.csv.

Run from the repo root:  python scripts/merge_scores.py
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_scores")

ROOT = Path(__file__).resolve().parent.parent
TILES_BASE = ROOT / "data" / "tiles_base.csv"
SCORES_OPTICAL = ROOT / "results" / "scores_optical_int8.csv"
SCORES_RADAR = ROOT / "results" / "scores_radar_int8.csv"
SCORES_NDWI = ROOT / "results" / "scores_ndwi.csv"
FAKE_TILES = ROOT / "data" / "fake_tiles.csv"
OUT_TILES = ROOT / "data" / "tiles.csv"
OUT_PLACEHOLDER = ROOT / "data" / "tiles_placeholder.csv"
KEY = ["chip", "row", "col"]
COLUMNS = ["event", "chip", "row", "col", "split", "water_frac", "critical",
           "cloud_frac", "sensor_used", "p_flood", "extent"]


def _score_table(path: Path) -> pd.DataFrame:
    """Read a score file. Accepted layouts: tile_id,p_flood,extent (the team format, with
    tile_id = <chip>_<row>_<col>) or chip,row,col,p_flood,extent."""
    df = pd.read_csv(path)
    if "tile_id" in df.columns and not set(KEY) <= set(df.columns):
        parts = df["tile_id"].astype(str).str.rsplit("_", n=2, expand=True)
        if parts.shape[1] != 3 or parts.isna().any().any():
            raise SystemExit(f"{path.name}: tile_id must look like <chip>_<row>_<col>")
        df["chip"], df["row"], df["col"] = parts[0], parts[1].astype(int), parts[2].astype(int)
    missing = [c for c in KEY + ["p_flood", "extent"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns {missing}; "
                         f"expected tile_id,p_flood,extent or chip,row,col,p_flood,extent")
    if df.duplicated(KEY).any():
        raise SystemExit(f"{path.name} has duplicate (chip,row,col) keys")
    return df[KEY + ["p_flood", "extent"]]


def merge_scores(
    tiles_base_path: Path = TILES_BASE,
    optical_scores_path: Path = SCORES_OPTICAL,
    radar_scores_path: Path = SCORES_RADAR,
    fake_tiles_path: Path = FAKE_TILES,
    out_path: Path = OUT_TILES,
    allow_placeholder: bool = False,
) -> pd.DataFrame:
    if not tiles_base_path.exists():
        raise SystemExit(f"Missing base tiles file: {tiles_base_path} (run scripts/build_tiles_base.py)")
    base = pd.read_csv(tiles_base_path)

    if optical_scores_path.exists() and radar_scores_path.exists():
        logger.info("Merging real model scores from %s and %s",
                    optical_scores_path.name, radar_scores_path.name)
        opt = _score_table(optical_scores_path).rename(columns={"p_flood": "p_opt", "extent": "e_opt"})
        rad = _score_table(radar_scores_path).rename(columns={"p_flood": "p_rad", "extent": "e_rad"})
        merged = base.merge(opt, on=KEY, how="left").merge(rad, on=KEY, how="left")
        is_radar = merged["sensor_used"] == "radar"
        merged["p_flood"] = merged["p_rad"].where(is_radar, merged["p_opt"])
        merged["extent"] = merged["e_rad"].where(is_radar, merged["e_opt"])
        n_missing = int(merged[["p_flood", "extent"]].isna().any(axis=1).sum())
        if n_missing:
            raise SystemExit(f"{n_missing} tiles have no score for their sensor path; "
                             f"the score files must cover every tile in tiles_base.csv")
        result, target = merged[COLUMNS], out_path
        # tiles_all.csv: the same tiles plus every model's score on EVERY tile (not only on the
        # tiles routed to it), which the baselines and ablations in scripts/evaluate.py need.
        extra = merged[COLUMNS].copy()
        extra["p_optical"], extra["extent_optical"] = merged["p_opt"], merged["e_opt"]
        extra["p_radar"], extra["extent_radar"] = merged["p_rad"], merged["e_rad"]
        if SCORES_NDWI.exists():
            nd = _score_table(SCORES_NDWI).rename(columns={"p_flood": "p_ndwi"})[KEY + ["p_ndwi"]]
            extra = extra.merge(nd, on=KEY, how="left")
            if extra["p_ndwi"].isna().any():
                raise SystemExit("scores_ndwi.csv does not cover every tile")
    elif allow_placeholder:
        extra = None
        logger.warning("PLACEHOLDER: real scores not found; using SYNTHETIC scores from %s. "
                       "Writing %s, not %s.", fake_tiles_path.name, OUT_PLACEHOLDER.name, out_path.name)
        fake = pd.read_csv(fake_tiles_path)[KEY + ["p_flood", "extent"]]
        extra = None
        result = base.merge(fake, on=KEY, how="left")[COLUMNS]
        target = OUT_PLACEHOLDER
    else:
        raise SystemExit(
            f"Real model scores not found ({optical_scores_path.name}, {radar_scores_path.name} "
            f"in results/). Refusing to write {out_path.name} from synthetic scores. "
            f"Use --allow-placeholder to test the plumbing only."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False, lineterminator="\n")
    logger.info("Wrote %s with %d rows and %d columns", target, len(result), len(result.columns))
    if extra is not None:
        all_path = target.with_name("tiles_all.csv")
        extra.to_csv(all_path, index=False, lineterminator="\n")
        logger.info("Wrote %s with %d columns", all_path, len(extra.columns))
    return result


def main():
    parser = argparse.ArgumentParser(description="Merge model scores into data/tiles.csv.")
    parser.add_argument("--out", type=Path, default=OUT_TILES, help="Output tiles.csv path")
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="Plumbing test only: merge synthetic scores into data/tiles_placeholder.csv")
    args = parser.parse_args()
    merge_scores(out_path=args.out, allow_placeholder=args.allow_placeholder)


if __name__ == "__main__":
    sys.exit(main())
