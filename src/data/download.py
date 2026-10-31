"""Sen1Floods11 Hand-Labelled Dataset Acquisition Module.

Downloads only the required components of the hand-labelled subset:
- S1Hand (raw Sentinel-1 IW GRD radar images in dB, 2 bands)
- S2Hand (raw Sentinel-2 L1C optical images, 13 bands)
- LabelHand (hand-labelled ground truth masks)
- JRCWaterHand (JRC global surface water permanent water masks)
- Official split CSV files

Target location: data/raw/
Dataset source: https://storage.googleapis.com/sen1floods11/ (v1.1)
"""

import argparse
import concurrent.futures
import logging
import os
import shutil
import sys
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sen1floods11_download")

GCS_BASE_URL = "https://storage.googleapis.com/sen1floods11"
GCS_XML_NS = "{http://doc.s3.amazonaws.com/2006-03-01}"

HANDLABELED_COMPONENTS = ["LabelHand", "JRCWaterHand", "S1Hand", "S2Hand"]
SPLIT_FILES = [
    "flood_bolivia_data.csv",
    "flood_train_data.csv",
    "flood_valid_data.csv",
    "flood_test_data.csv",
]

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def check_disk_storage(target_dir: Path, required_gb: float = 2.0) -> bool:
    """Check available disk storage on the target volume."""
    target_dir.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(target_dir)
    free_gb = free / (1024**3)
    logger.info("Storage check: %.2f GB available at %s (required: %.2f GB)", free_gb, target_dir, required_gb)
    if free_gb < required_gb:
        logger.error("Insufficient disk storage: %.2f GB available, %.2f GB required.", free_gb, required_gb)
        return False
    return True


def list_bucket_prefix(prefix: str) -> List[Tuple[str, int]]:
    """List objects and sizes under a GCS prefix via the XML API."""
    items = []
    marker = ""
    while True:
        url = f"{GCS_BASE_URL}/?prefix={prefix}"
        if marker:
            url += f"&marker={marker}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Orbit-Prism/1.0)"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                xml_data = resp.read()
        except urllib.error.URLError as e:
            logger.error("Failed to query GCS bucket prefix '%s': %s", prefix, e)
            raise

        root = ET.fromstring(xml_data)
        for contents in root.findall(f".//{GCS_XML_NS}Contents"):
            key_elem = contents.find(f"{GCS_XML_NS}Key")
            size_elem = contents.find(f"{GCS_XML_NS}Size")
            if key_elem is not None and size_elem is not None:
                items.append((key_elem.text, int(size_elem.text)))

        is_truncated = root.find(f"{GCS_XML_NS}IsTruncated")
        if is_truncated is not None and is_truncated.text == "true" and items:
            marker = items[-1][0]
        else:
            break
    return items


def download_split_csvs(raw_dir: Path = DEFAULT_RAW_DIR) -> Dict[str, Path]:
    """Download the official Sen1Floods11 split CSV files."""
    dest_dir = raw_dir / "splits" / "flood_handlabeled"
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for filename in SPLIT_FILES:
        target_path = dest_dir / filename
        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info("Split file already present: %s", target_path)
            downloaded[filename] = target_path
            continue

        url = f"{GCS_BASE_URL}/v1.1/splits/flood_handlabeled/{filename}"
        logger.info("Downloading split %s...", filename)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(target_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        downloaded[filename] = target_path
    return downloaded


def download_file(gcs_key: str, dest_path: Path, expected_size: Optional[int] = None) -> bool:
    """Download a single file from GCS bucket if not already present and valid."""
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if expected_size is None or actual_size == expected_size:
            return True
        logger.warning("File %s size mismatch (expected %s, got %s), redownloading...", dest_path.name, expected_size, actual_size)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    url = f"{GCS_BASE_URL}/{gcs_key}"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        temp_path.replace(dest_path)
        return True
    except Exception as e:
        logger.error("Failed downloading %s: %s", gcs_key, e)
        if temp_path.exists():
            temp_path.unlink()
        return False


def download_sen1floods11(
    raw_dir: Path = DEFAULT_RAW_DIR,
    events: Optional[List[str]] = None,
    components: Optional[List[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """Download hand-labelled Sen1Floods11 dataset components.

    Args:
        raw_dir: Root raw data directory (data/raw/).
        events: List of event prefixes (e.g. ['Bolivia', 'Nigeria']). If None, downloads all events.
        components: List of components to download. Defaults to all 4 hand-labelled components.

    Returns:
        Summary dict containing counts and byte sizes per component.
    """
    if components is None:
        components = HANDLABELED_COMPONENTS

    if not check_disk_storage(raw_dir, required_gb=2.0 if events is None else 0.5):
        raise RuntimeError("Insufficient storage space for download.")

    # 1. Ensure splits are downloaded
    download_split_csvs(raw_dir)

    target_events = set(events) if events else None
    summary = {}

    for comp in components:
        prefix = f"v1.1/data/flood_events/HandLabeled/{comp}/"
        dest_dir = raw_dir / "data" / "flood_events" / "HandLabeled" / comp
        dest_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Listing objects for %s...", comp)
        keys = list_bucket_prefix(prefix)

        downloaded_count = 0
        total_bytes = 0

        tasks = []
        for key, size in keys:
            filename = key.split("/")[-1]
            if not filename.endswith(".tif"):
                continue
            event_name = filename.split("_")[0]
            if target_events and event_name not in target_events:
                continue

            dest_file = dest_dir / filename
            tasks.append((key, dest_file, size))

        downloaded_count = 0
        total_bytes = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_item = {
                executor.submit(download_file, key, dest_file, size): (key, dest_file, size)
                for key, dest_file, size in tasks
            }
            for future in concurrent.futures.as_completed(future_to_item):
                key, dest_file, size = future_to_item[future]
                try:
                    success = future.result()
                    if success:
                        downloaded_count += 1
                        total_bytes += size
                except Exception as e:
                    logger.error("Exception downloading %s: %s", key, e)

        summary[comp] = {
            "files_downloaded": downloaded_count,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
        }
        logger.info("Finished %s: %d files (%.2f MB)", comp, downloaded_count, total_bytes / (1024 * 1024))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Download Sen1Floods11 hand-labelled subset.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Target data/raw directory")
    parser.add_argument("--events", nargs="+", default=None, help="Specific events to download (e.g. Bolivia Nigeria)")
    parser.add_argument("--components", nargs="+", default=None, choices=HANDLABELED_COMPONENTS, help="Components to download")
    args = parser.parse_args()

    logger.info("Initiating Sen1Floods11 download. Events: %s", args.events or "ALL")
    summary = download_sen1floods11(args.raw_dir, events=args.events, components=args.components)
    for comp, stats in summary.items():
        print(f"Component: {comp:<14} Files: {stats['files_downloaded']:>4}  Size: {stats['total_mb']:>8.2f} MB")


if __name__ == "__main__":
    main()
