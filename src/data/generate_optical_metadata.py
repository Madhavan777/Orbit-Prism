"""Optical Module Metadata Generation for ORBIT-PRISM.

Generates data/optical_tiles.csv with the exact schema requested by the optical module:
- image_id: Unique image/tile ID
- image_path: Location of satellite image / processed NPZ
- disaster_type: Flood / Normal
- latitude: WGS-84 latitude of tile center
- longitude: WGS-84 longitude of tile center
- timestamp: Acquisition timestamp (ISO-8601)
- event_id: Disaster event identifier
- B2: Blue band info (channel 0 in processed optical array, 490 nm)
- B3: Green band info (channel 1, 560 nm)
- B4: Red band info (channel 2, 665 nm)
- B8: NIR band info (channel 3, 842 nm)
- water_mask_path: Flood water mask file
- burn_mask_path: Wildfire burned-area mask file (None for flood benchmark)
- water_fraction: Fraction of water-covered valid pixels [0, 1]
- burn_fraction: Fraction of burned area (0.0)
- cloud_fraction: Fraction of cloud-covered area [0, 1]
- split: train / valid / test / holdout (CONTEXT.md values; holdout = Bolivia, kept separate)

cloud_fraction is the SIMULATED cloud cover from data/tiles_base.csv (src/data/cloud_simulation.py).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import tifffile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_optical_metadata")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TILES_BASE_CSV = ROOT_DIR / "data" / "tiles_base.csv"
GEOJSON_PATH = RAW_DIR / "Sen1Floods11_Metadata.geojson"
OUT_CSV = ROOT_DIR / "data" / "optical_tiles.csv"

# Mapping location names from GeoJSON to event prefixes
LOCATION_MAP = {
    "Cambodia": "Mekong",
}


def load_event_timestamps(geojson_path: Path = GEOJSON_PATH) -> Dict[str, str]:
    """Load Sentinel-2 acquisition date per event from GeoJSON."""
    timestamps = {}
    if not geojson_path.exists():
        logger.warning("GeoJSON metadata not found at %s. Using default dates.", geojson_path)
        return timestamps

    with open(geojson_path) as f:
        data = json.load(f)

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        loc = props.get("location")
        if not loc:
            continue
        event_name = LOCATION_MAP.get(loc, loc)
        s2_date = props.get("s2_date")
        if s2_date:
            # Format: '2018/02/15' -> '2018-02-15T00:00:00Z'
            iso_date = s2_date.replace("/", "-") + "T00:00:00Z"
            timestamps[event_name] = iso_date

    return timestamps


def extract_chip_georeferencing(label_path: Path) -> Tuple[float, float, float, float]:
    """Extract origin (lon, lat) and pixel scales from GeoTIFF tags.

    Returns:
        (origin_lon, origin_lat, scale_x, scale_y)
    """
    with tifffile.TiffFile(label_path) as tif:
        page = tif.pages[0]
        tiepoint = page.tags.get("ModelTiepointTag")
        scale = page.tags.get("ModelPixelScaleTag")

        if tiepoint is not None and scale is not None:
            # ModelTiepointTag: (I, J, K, X, Y, Z) -> (0, 0, 0, origin_lon, origin_lat, 0)
            origin_lon = float(tiepoint.value[3])
            origin_lat = float(tiepoint.value[4])
            scale_x = float(scale.value[0])
            scale_y = float(scale.value[1])
            return origin_lon, origin_lat, scale_x, scale_y

    return 0.0, 0.0, 8.983e-5, 8.983e-5


def compute_tile_lat_lon(
    origin_lon: float,
    origin_lat: float,
    scale_x: float,
    scale_y: float,
    row: int,
    col: int,
    tile_size: int = 128,
) -> Tuple[float, float]:
    """Compute center latitude and longitude for a 128x128 tile inside a 512x512 chip."""
    center_px_x = (col + 0.5) * tile_size
    center_px_y = (row + 0.5) * tile_size

    lon = origin_lon + center_px_x * scale_x
    lat = origin_lat - center_px_y * scale_y
    return round(lat, 6), round(lon, 6)


def generate_optical_dataset_metadata(
    tiles_base_path: Path = TILES_BASE_CSV,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    out_csv: Path = OUT_CSV,
) -> pd.DataFrame:
    """Generate the optical module metadata CSV matching the user's specification."""
    if not tiles_base_path.exists():
        raise FileNotFoundError(f"Missing {tiles_base_path}")

    base_df = pd.read_csv(tiles_base_path)
    timestamps = load_event_timestamps()

    chip_geo_cache = {}
    labels_dir = raw_dir / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    s2_dir = raw_dir / "data" / "flood_events" / "HandLabeled" / "S2Hand"

    records = []
    for _, row in base_df.iterrows():
        event = str(row["event"])
        chip = str(row["chip"])
        r = int(row["row"])
        c = int(row["col"])
        split_raw = str(row["split"])
        wf = float(row["water_frac"])
        cf = float(row["cloud_frac"])

        tile_id = f"{chip}_{r}_{c}"

        # Georeferencing
        if chip not in chip_geo_cache:
            lab_path = labels_dir / f"{chip}_LabelHand.tif"
            if lab_path.exists():
                chip_geo_cache[chip] = extract_chip_georeferencing(lab_path)
            else:
                chip_geo_cache[chip] = (0.0, 0.0, 8.983e-5, 8.983e-5)

        origin_lon, origin_lat, scale_x, scale_y = chip_geo_cache[chip]
        tile_lat, tile_lon = compute_tile_lat_lon(origin_lon, origin_lat, scale_x, scale_y, r, c)

        # Split values follow CONTEXT.md exactly (train / valid / test / holdout). The held-out
        # event (Bolivia, "holdout") is never merged into "test".
        split_std = split_raw

        # Disaster classification
        disaster_type = "Flood" if wf >= 0.05 else "Normal"

        # Paths
        img_path = f"data/processed/{event}.npz"
        water_mask_path = f"data/raw/data/flood_events/HandLabeled/LabelHand/{chip}_LabelHand.tif"

        # Band specifications
        b2_info = "Channel 0 (490 nm Blue, scaled uint8)"
        b3_info = "Channel 1 (560 nm Green, scaled uint8)"
        b4_info = "Channel 2 (665 nm Red, scaled uint8)"
        b8_info = "Channel 3 (842 nm NIR, scaled uint8)"

        records.append({
            "image_id": tile_id,
            "image_path": img_path,
            "disaster_type": disaster_type,
            "latitude": tile_lat,
            "longitude": tile_lon,
            "timestamp": timestamps.get(event, "2018-01-01T00:00:00Z"),
            "event_id": event,
            "B2": b2_info,
            "B3": b3_info,
            "B4": b4_info,
            "B8": b8_info,
            "water_mask_path": water_mask_path,
            "burn_mask_path": "",
            "water_fraction": wf,
            "burn_fraction": 0.0,
            "cloud_fraction": cf,
            "split": split_std,
        })

    out_df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, lineterminator="\n")
    logger.info("Generated %s with %d rows and %d columns", out_csv, len(out_df), len(out_df.columns))
    return out_df


def main():
    df = generate_optical_dataset_metadata()
    print("\nOptical Metadata Summary:")
    print("Total rows:", len(df))
    print("Split breakdown:\n", df["split"].value_counts())
    print("Disaster types:\n", df["disaster_type"].value_counts())
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
