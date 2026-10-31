"""Simple baselines on data/fake_tiles.csv (synthetic scores; placeholder numbers).

All systems send whole tiles and share one SCALED byte budget per pass (20% of the
event's all-optical volume; the real pass is 19.2 MB). Evaluated on the held-out
test event only (split == "holdout", Bolivia).

1 capture order      - tiles in file order (alphabetical by chip, an arbitrary order,
                        not acquisition time), optical payload
2 discard cloudy     - drop tiles with cloud_frac >= 0.5, then capture order
3 optical-only       - rank clear tiles by p_flood; cloudy tiles are unscored (blind).
                        Simplification: the real baseline scores cloudy tiles with the
                        actual optical model instead.
4 colour-index       - NOT computable here: needs real band imagery (NDWI/Otsu)
5 ORBIT-PRISM sketch - rank by p_flood * extent per byte, radar payload under cloud
5b same, p_flood only - drops the extent term. The synthetic extent is the true water
                        fraction plus noise (nearly the label), so 5 vs 5b shows how
                        much of the gap comes from that leak.

Metric: critical-tile recall against bytes sent. Results are "to be measured"
on real model outputs; this only checks the evaluation plumbing. Synthetic numbers
move a lot with the random seed: never quote them.

Run from the repo root:  python -I scripts/baselines.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.constants import (  # noqa: E402
    CLOUD_SWITCH, OPTICAL_B, PASS_BYTES as REAL_PASS_BYTES, RADAR_B,
    SCALED_PASS_FRACTION as SCALED_FRACTION, TEST_SPLIT)

SRC = ROOT / "data" / "fake_tiles.csv"
# The Bolivia event (~15 MB) fits in one real pass, so the evaluation uses a SCALED pass:
# 20% of the event's all-optical volume. Always label it "scaled" in plots and slides.
PASS_BYTES = None  # set in main() from the event size


def load():
    with open(SRC, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == TEST_SPLIT]
    return {
        "crit": np.array([int(r["critical"]) for r in rows]),
        "cloud": np.array([float(r["cloud_frac"]) for r in rows]),
        "p": np.array([float(r["p_flood"]) for r in rows]),
        "ext": np.array([float(r["extent"]) for r in rows]),
        "radar": np.array([r["sensor_used"] == "radar" for r in rows]),
    }


def evaluate(name, order, nbytes, d):
    """Print bytes to 95% recall, recall within one (scaled) pass, and the same recall
    restricted to cloudy critical tiles. Recall is over ALL critical tiles in the event."""
    n_crit = d["crit"].sum()
    n_crit_cloudy = (d["crit"] * (d["cloud"] >= CLOUD_SWITCH)).sum()
    sent = np.cumsum(nbytes[order])
    in_pass = sent <= PASS_BYTES  # the tiles that fit in one pass; sent is increasing
    crit = d["crit"][order]
    cc = crit * (d["cloud"][order] >= CLOUD_SWITCH)
    rec = np.cumsum(crit) / n_crit if n_crit else np.zeros(len(crit))
    hit = np.searchsorted(rec, 0.95) if n_crit else len(rec)
    b95 = f"{sent[hit] / 1e6:8.1f} MB" if hit < len(rec) else "     never"
    first = crit[in_pass].sum() / n_crit if n_crit else float("nan")
    cloud_rec = cc[in_pass].sum() / n_crit_cloudy if n_crit_cloudy else float("nan")
    print(f"{name:<22} {b95:>12} {first:>16.3f} {cloud_rec:>16.3f}")


def main():
    d = load()
    global PASS_BYTES
    n = len(d["crit"])
    PASS_BYTES = SCALED_FRACTION * n * OPTICAL_B
    opt_b = np.full(n, OPTICAL_B, dtype=float)
    mix_b = np.where(d["radar"], RADAR_B, OPTICAL_B).astype(float)
    cloudy = d["cloud"] >= CLOUD_SWITCH
    print(f"{n} eval tiles, {d['crit'].sum()} critical, {cloudy.mean():.2f} cloudy; "
          f"scaled pass = {PASS_BYTES / 1e6:.2f} MB ({SCALED_FRACTION:.0%} of event volume; "
          f"real pass is {REAL_PASS_BYTES / 1e6:.1f} MB)")
    print(f"{'system':<22} {'bytes to 95%':>12} {'recall scaled P1':>16} {'recall cloudy P1':>16}")
    evaluate("1 capture order", np.arange(n), opt_b, d)
    keep = np.flatnonzero(~cloudy)
    evaluate("2 discard cloudy", keep, opt_b, d)
    score3 = np.where(cloudy, -1.0, d["p"])
    evaluate("3 optical-only triage", np.argsort(-score3, kind="stable"), opt_b, d)
    print("4 colour-index         n/a (needs real imagery)")
    score5 = d["p"] * d["ext"] / mix_b
    evaluate("5 ORBIT-PRISM sketch", np.argsort(-score5, kind="stable"), mix_b, d)
    score5b = d["p"] / mix_b
    evaluate("5b same, p_flood only", np.argsort(-score5b, kind="stable"), mix_b, d)


if __name__ == "__main__":
    main()
