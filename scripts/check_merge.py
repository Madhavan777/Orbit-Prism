"""Merge gate: run before asking Madhavan to merge a branch. Exit code 1 if anything fails.

Checks the tracked files of the current checkout:
  1. no UTF-8 byte-order mark, and LF line endings in .py/.csv/.json/.md files
  2. no raw data, big files or model dumps (tif, npz, zip, data/raw, anything over 5 MB)
  3. no banned claims (outperforms, guarantees, real-time on satellite hardware, ...)
  4. CSV contracts for data/tiles_base.csv and data/fake_tiles.csv
  5. provenance: nothing synthetic sits under a real-looking name
  6. no code or data for hazards outside the scope (floods and forest fires)

Run from the repo root:  python scripts/check_merge.py
(The pytest suite is a separate step:  python -m pytest tests -q)
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEXT = {".py", ".csv", ".json", ".md", ".txt", ".yml", ".yaml"}
FORBIDDEN_EXT = {".tif", ".tiff", ".npz", ".zip", ".h5", ".hdf5", ".pt", ".pth", ".npy"}
MAX_BYTES = 5_000_000
BANNED = re.compile(r"outperform|guarantee|real-time on satellite|danger score|disaster management system", re.I)
OUT_OF_SCOPE = re.compile("cy" + r"clone|land" + r"slide|earth" + r"quake|vol" + r"cano", re.I)
BASE_COLUMNS = ["event", "chip", "row", "col", "split", "water_frac", "critical", "cloud_frac", "sensor_used"]
FULL_COLUMNS = BASE_COLUMNS + ["p_flood", "extent"]
# Files that legitimately talk about the banned words or the excluded hazards.
POLICY_FILES = {"CONTEXT.md", "scripts/check_merge.py"}

failures = []


def fail(msg):
    failures.append(msg)
    print("FAIL", msg)


def ok(msg):
    print("ok  ", msg)


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if (ROOT / p).is_file()]


def main():
    files = tracked()
    start = len(failures)

    for p in files:
        path = ROOT / p
        data = path.read_bytes()
        if data[:3] == b"\xef\xbb\xbf":
            fail(f"{p}: has a UTF-8 byte-order mark")
        if path.suffix in {".py", ".csv", ".json", ".md"} and b"\r\n" in data and path.name != "satellite_data.csv":
            fail(f"{p}: has CRLF line endings (use LF)")
        if path.suffix.lower() in FORBIDDEN_EXT or p.startswith("data/raw/") or p.startswith("data/processed/"):
            fail(f"{p}: raw/processed data or model dump must not be committed")
        if len(data) > MAX_BYTES:
            fail(f"{p}: {len(data):,} bytes is over the {MAX_BYTES:,}-byte limit")
    if len(failures) == start:
        ok("file hygiene (BOM, line endings, forbidden files, size)")

    start = len(failures)
    for p in files:
        if Path(p).suffix not in TEXT or p in POLICY_FILES or p.startswith("docs/merge_review"):
            continue
        text = (ROOT / p).read_text(encoding="utf-8", errors="ignore")
        for n, line in enumerate(text.splitlines(), 1):
            if BANNED.search(line):
                fail(f"{p}:{n}: banned claim: {line.strip()[:90]}")
    if len(failures) == start:
        ok("no banned claims")

    start = len(failures)
    for p in files:
        parts = Path(p).parts
        if Path(p).suffix not in {".py", ".csv", ".json"} or "tests" in parts or "docs" in parts or p in POLICY_FILES:
            continue
        text = (ROOT / p).read_text(encoding="utf-8", errors="ignore")
        for n, line in enumerate(text.splitlines(), 1):
            if OUT_OF_SCOPE.search(line) and "excluded from active scope" not in line.lower():
                fail(f"{p}:{n}: out-of-scope hazard: {line.strip()[:90]}")
    if len(failures) == start:
        ok("no out-of-scope hazards in code or data")

    start = len(failures)
    base_p, fake_p = ROOT / "data" / "tiles_base.csv", ROOT / "data" / "fake_tiles.csv"
    for path, cols in [(base_p, BASE_COLUMNS), (fake_p, FULL_COLUMNS)]:
        if not path.exists():
            fail(f"{path.name}: missing")
            continue
        df = pd.read_csv(path)
        if list(df.columns) != cols:
            fail(f"{path.name}: columns {list(df.columns)} != contract {cols}")
            continue
        if df.duplicated(["chip", "row", "col"]).any():
            fail(f"{path.name}: duplicate tile keys")
        if not set(df["split"]) <= {"train", "valid", "test", "holdout"}:
            fail(f"{path.name}: split values {sorted(set(df['split']))}")
        if set(df.loc[df["event"] == "Bolivia", "split"]) != {"holdout"} or set(df.loc[df["split"] == "holdout", "event"]) != {"Bolivia"}:
            fail(f"{path.name}: Bolivia and holdout must be the same set of tiles")
        if (df["critical"] != (df["water_frac"] >= 0.30).astype(int)).any():
            fail(f"{path.name}: critical != (water_frac >= 0.30)")
        if (df["sensor_used"] != df["cloud_frac"].ge(0.5).map({True: "radar", False: "optical"})).any():
            fail(f"{path.name}: sensor_used != radar iff cloud_frac >= 0.5")
    if base_p.exists() and fake_p.exists():
        b, f = pd.read_csv(base_p), pd.read_csv(fake_p)
        if not b[BASE_COLUMNS].equals(f[BASE_COLUMNS]):
            fail("fake_tiles.csv does not agree with tiles_base.csv on the shared columns")
    if len(failures) == start:
        ok("CSV contracts (tiles_base.csv, fake_tiles.csv)")

    start = len(failures)
    cal = ROOT / "results" / "calibration.json"
    if "results/calibration.json" in files:
        import json
        src = json.loads(cal.read_text()).get("score_source", "")
        if not src or src.startswith("synthetic") or src == "unspecified":
            fail(f"results/calibration.json: score_source is '{src}' (needs real model scores)")
    tiles = ROOT / "data" / "tiles.csv"
    if "data/tiles.csv" in files and fake_p.exists():
        t, f = pd.read_csv(tiles), pd.read_csv(fake_p)
        if len(t) == len(f) and t["p_flood"].equals(f["p_flood"]):
            fail("data/tiles.csv holds the synthetic scores of fake_tiles.csv")
    for p in files:
        if p.startswith("results/") and re.search(r"synthetic|dry.?run|placeholder", Path(p).name, re.I):
            fail(f"{p}: synthetic/dry-run artifact under results/")
    if len(failures) == start:
        ok("provenance (no synthetic file under a real-looking name)")

    print()
    if failures:
        print(f"GATE FAILED: {len(failures)} problem(s)")
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
