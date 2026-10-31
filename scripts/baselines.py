"""Simple baselines on data/fake_tiles.csv (synthetic scores; placeholder numbers).

All systems send whole tiles and share one byte budget per pass (19.2 MB).
Evaluated on the held-out test event only (split == "holdout", Bolivia).

1 capture order      - tiles in file order, optical payload
2 discard cloudy     - drop tiles with cloud_frac >= 0.5, then capture order
3 optical-only       - rank clear tiles by p_flood; cloudy tiles are unscored (optical is blind)
4 colour-index       - NOT computable here: needs real band imagery (NDWI/Otsu)
5 ORBIT-PRISM sketch - rank by p_flood * extent per byte, radar payload under cloud

Metric: critical-tile recall against bytes sent. Results are "to be measured"
on real model outputs; this only checks the evaluation plumbing.

Run from the repo root:  python -I scripts/baselines.py
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "fake_tiles.csv"
TEST_SPLIT = "holdout"  # Bolivia, the held-out test event (locked in CONTEXT.md)
PASS_BYTES = 19_200_000
OPTICAL_B, RADAR_B = 65_536, 32_768
CLOUD_SWITCH = 0.5


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
    crit = d["crit"][order]
    sent = np.cumsum(nbytes[order])
    rec = np.cumsum(crit) / d["crit"].sum()
    hit = np.searchsorted(rec, 0.95)
    b95 = f"{sent[hit] / 1e6:8.1f} MB" if hit < len(rec) else "     never"
    first = rec[np.searchsorted(sent, PASS_BYTES) - 1]
    cloudy = d["cloud"][order] >= CLOUD_SWITCH
    cc = d["crit"][order] * cloudy
    in_pass = sent <= PASS_BYTES
    cloud_rec = cc[in_pass].sum() / max((d["crit"] * (d["cloud"] >= CLOUD_SWITCH)).sum(), 1)
    print(f"{name:<22} {b95:>12} {first:>14.3f} {cloud_rec:>16.3f}")


def main():
    d = load()
    n = len(d["crit"])
    opt_b = np.full(n, OPTICAL_B, dtype=float)
    mix_b = np.where(d["radar"], RADAR_B, OPTICAL_B).astype(float)
    cloudy = d["cloud"] >= CLOUD_SWITCH
    print(f"{n} eval tiles, {d['crit'].sum()} critical, {cloudy.mean():.2f} cloudy; "
          f"{PASS_BYTES / 1e6:.1f} MB per pass, one pass assumed")
    print(f"{'system':<22} {'bytes to 95%':>12} {'recall pass 1':>14} {'recall cloudy P1':>16}")
    evaluate("1 capture order", np.arange(n), opt_b, d)
    keep = np.flatnonzero(~cloudy)
    evaluate("2 discard cloudy", keep, opt_b, d)
    score3 = np.where(cloudy, -1.0, d["p"])
    evaluate("3 optical-only triage", np.argsort(-score3, kind="stable"), opt_b, d)
    print("4 colour-index         n/a (needs real imagery)")
    score5 = d["p"] * d["ext"] / mix_b
    evaluate("5 ORBIT-PRISM sketch", np.argsort(-score5, kind="stable"), mix_b, d)


if __name__ == "__main__":
    main()
