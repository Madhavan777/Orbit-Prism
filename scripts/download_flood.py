"""Download the hand-labelled Sen1Floods11 subset from the public bucket gs://sen1floods11.

Needs no gsutil and no extra packages (standard library only). Files land in
data/raw/ with the layout the other scripts expect:

    data/raw/data/flood_events/HandLabeled/{LabelHand,JRCWaterHand,S1Hand,S2Hand}/
    data/raw/splits/flood_handlabeled/

By default only labels, the JRC permanent-water layer and the split lists are
fetched (about 3 MB). Add --imagery for S1Hand and S2Hand (about 1.7 GB).
Existing files are skipped, so an interrupted run can simply be repeated.

Run from the repo root:  python scripts/download_flood.py [--imagery]
"""
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
API = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
FILE = "https://storage.googleapis.com/sen1floods11/"
HAND = "v1.1/data/flood_events/HandLabeled/"
SMALL = [HAND + "LabelHand/", HAND + "JRCWaterHand/", "v1.1/splits/flood_handlabeled/",
         "v1.1/Sen1Floods11_Metadata.geojson"]  # event dates, used by src/data/generate_*_metadata.py
IMAGERY = [HAND + "S1Hand/", HAND + "S2Hand/"]


def list_objects(prefix):
    token = None
    while True:
        q = {"prefix": prefix, "maxResults": 1000}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=60) as r:
            page = json.load(r)
        yield from page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return


def main():
    prefixes = SMALL + (IMAGERY if "--imagery" in sys.argv else [])
    objs = [o for p in prefixes for o in list_objects(p)]
    total = sum(int(o["size"]) for o in objs)
    print(f"{len(objs)} files, {total / 1e6:.1f} MB")
    todo = [o for o in objs if not _have(o)]
    print(f"{len(todo)} to download")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(_fetch, todo), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
    bad = [o["name"] for o in objs if not _have(o)]
    if bad:
        sys.exit(f"{len(bad)} files missing or truncated, rerun the script. First: {bad[0]}")
    print(f"done: {RAW}")


def _dest(o):
    return RAW / o["name"][len("v1.1/"):]


def _have(o):
    d = _dest(o)
    return d.exists() and d.stat().st_size == int(o["size"])


def _fetch(o):
    d = _dest(o)
    d.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FILE + urllib.parse.quote(o["name"]), d)


if __name__ == "__main__":
    main()
