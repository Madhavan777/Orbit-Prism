"""Radar Module Metadata Generation for ORBIT-PRISM.

Generates data/radar_tiles.csv specifically structured for the Radar CNN developer:
- image_id: Unique tile identifier ({chip}_{row}_{col})
- image_path: Location of satellite image in data/processed/<event>.npz
- disaster_type: Flood (primary worked example) with wildfire extension schema
- latitude: WGS-84 tile center latitude
- longitude: WGS-84 tile center longitude
- timestamp: Sentinel-1 SAR acquisition timestamp (ISO-8601)
- event_id: Disaster event name
- VV: Sentinel-1 SAR VV polarization channel (uint8 scaled [-30, 0] dB)
- VH: Sentinel-1 SAR VH polarization channel (uint8 scaled [-30, 0] dB)
- water_mask_path: Ground truth water mask path
- burn_mask_path: Placeholder for wildfire burn mask (NASA FIRMS / ESA FireCCI51)
- water_fraction: Continuous flood water fraction [0, 1]
- burn_fraction: Burned area fraction (0.0 for flood benchmark)
- cloud_fraction: Simulated cloud fraction (radar is cloud-invariant)
- split: Train / Validation / Test
- source_dataset: Sen1Floods11 S1Hand (v1.1)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import tifffile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_radar_metadata")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TILES_BASE_CSV = ROOT_DIR / "data" / "tiles_base.csv"
GEOJSON_PATH = RAW_DIR / "Sen1Floods11_Metadata.geojson"
OUT_CSV = ROOT_DIR / "data" / "radar_tiles.csv"

LOCATION_MAP = {
    "Cambodia": "Mekong",
}


def load_s1_event_timestamps(geojson_path: Path = GEOJSON_PATH) -> Dict[str, str]:
    """Load Sentinel-1 SAR acquisition dates per event from GeoJSON."""
    timestamps = {}
    if not geojson_path.exists():
        return timestamps

    with open(geojson_path) as f:
        data = json.load(f)

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        loc = props.get("location")
        if not loc:
            continue
        event_name = LOCATION_MAP.get(loc, loc)
        s1_date = props.get("s1_date")
        if s1_date:
            iso_date = s1_date.replace("/", "-") + "T00:00:00Z"
            timestamps[event_name] = iso_date

    return timestamps


def extract_chip_georeferencing(label_path: Path) -> Tuple[float, float, float, float]:
    """Extract origin (lon, lat) and pixel scales from GeoTIFF tags."""
    with tifffile.TiffFile(label_path) as tif:
        page = tif.pages[0]
        tiepoint = page.tags.get("ModelTiepointTag")
        scale = page.tags.get("ModelPixelScaleTag")

        if tiepoint is not None and scale is not None:
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


def generate_radar_dataset_metadata(
    tiles_base_path: Path = TILES_BASE_CSV,
    raw_dir: Path = RAW_DIR,
    out_csv: Path = OUT_CSV,
) -> pd.DataFrame:
    """Generate the radar module metadata CSV."""
    if not tiles_base_path.exists():
        raise FileNotFoundError(f"Missing {tiles_base_path}")

    base_df = pd.read_csv(tiles_base_path)
    timestamps = load_s1_event_timestamps()

    chip_geo_cache = {}
    labels_dir = raw_dir / "data" / "flood_events" / "HandLabeled" / "LabelHand"

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

        if chip not in chip_geo_cache:
            lab_path = labels_dir / f"{chip}_LabelHand.tif"
            if lab_path.exists():
                chip_geo_cache[chip] = extract_chip_georeferencing(lab_path)
            else:
                chip_geo_cache[chip] = (0.0, 0.0, 8.983e-5, 8.983e-5)

        origin_lon, origin_lat, scale_x, scale_y = chip_geo_cache[chip]
        tile_lat, tile_lon = compute_tile_lat_lon(origin_lon, origin_lat, scale_x, scale_y, r, c)

        if split_raw == "train":
            split_std = "Train"
        elif split_raw in ["valid", "calibration"]:
            split_std = "Validation"
        else:
            split_std = "Test"

        disaster_type = "Flood" if wf >= 0.05 else "Normal"
        img_path = f"data/processed/{event}.npz"
        water_mask_path = f"data/raw/data/flood_events/HandLabeled/LabelHand/{chip}_LabelHand.tif"

        vv_info = "Channel 0 (VV, [-30, 0] dB scaled to uint8 [0, 255])"
        vh_info = "Channel 1 (VH, [-30, 0] dB scaled to uint8 [0, 255])"

        records.append({
            "image_id": tile_id,
            "image_path": img_path,
            "disaster_type": disaster_type,
            "latitude": tile_lat,
            "longitude": tile_lon,
            "timestamp": timestamps.get(event, "2018-01-01T00:00:00Z"),
            "event_id": event,
            "VV": vv_info,
            "VH": vh_info,
            "water_mask_path": water_mask_path,
            "burn_mask_path": "",
            "water_fraction": wf,
            "burn_fraction": 0.0,
            "cloud_fraction": cf,
            "split": split_std,
            "source_dataset": "Sen1Floods11 S1Hand",
        })

    out_df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logger.info("Generated %s with %d rows and %d columns", out_csv, len(out_df), len(out_df.columns))
    return out_df


def main():
    df = generate_radar_dataset_metadata()
    print("\nRadar Metadata Summary:")
    print("Total rows:", len(df))
    print("Split breakdown:\n", df["split"].value_counts())
    print("Disaster types:\n", df["disaster_type"].value_counts())
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
