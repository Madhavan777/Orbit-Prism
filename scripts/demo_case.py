"""Pick the demo tile: a critical flood tile under cloud that ORBIT-PRISM delivers and the baselines do not.

Reads data/tiles_all.csv and results/calibration.json (real scores only), replays the held-out event
through the scheduler at the scaled pass, and chooses the cloudy critical tile that:
  - cloud_frac >= CLOUD_SWITCH (routed to radar),
  - is sent as a full tile by ORBIT-PRISM within the scaled budget,
  - is never sent by "discard cloudy", and is not sent by optical-only triage within the budget.
and whose hand label covers at least MIN_LABELLED of the tile, so the picture really shows the flood (some
tiles are critical on a few hundred labelled pixels). Among those, the tile with the most flood water is chosen.
Writes results/demo_case.json, which GET /api/demo-case serves. Every number comes from the run;
nothing is typed in. The tile is an illustration chosen from the held-out event, not a typical tile:
the file also reports how many cloudy critical tiles each system delivered, so the demo can be shown
next to the aggregate.

Run from the repo root:  python scripts/demo_case.py [--alpha 0.05]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.constants import (CLOUD_SWITCH, OPTICAL_B, SCALED_PASS_FRACTION, TEST_SPLIT)  # noqa: E402
from src.scheduler import sim  # noqa: E402

MIN_LABELLED = 0.9

SYSTEMS = {"capture_order": "Capture order", "discard_cloudy": "Discard cloudy",
           "optical_only": "Optical-only triage", "orbit_prism": "ORBIT-PRISM"}


def _labelled_share(chip: str, row: int, col: int) -> float:
    path = ROOT / "data/raw/data/flood_events/HandLabeled/LabelHand" / f"{chip}_LabelHand.tif"
    return float((tifffile.imread(path)[row * 128:(row + 1) * 128, col * 128:(col + 1) * 128] != -1).mean())


def _full_tile_rank(log: pd.DataFrame, chip: str, row: int, col: int, budget: int):
    """(order, cum_MB) of the tile's full-tile packet if it fits in the budget, else None."""
    hit = log[(log["chip"] == chip) & (log["row"] == row) & (log["col"] == col) & (log["layer"] == "L2")]
    if hit.empty:
        return None
    r = hit.iloc[0]
    if r["cum_bytes"] > budget:
        return None
    return {"order": int(r["order"]), "cum_MB": round(float(r["cum_bytes"]) / 1e6, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    tiles = pd.read_csv(ROOT / "data" / "tiles_all.csv")
    cal = json.loads((ROOT / "results" / "calibration.json").read_text())
    if "synthetic" in json.dumps(cal).lower():
        sys.exit("calibration.json is marked synthetic; the demo case must come from real scores")
    thr = {p: float(cal[p][str(args.alpha)]) for p in ("optical", "radar")}
    test = tiles[tiles["split"] == TEST_SPLIT].reset_index(drop=True)
    budget = int(SCALED_PASS_FRACTION * len(test) * OPTICAL_B)

    logs = {k: sim.run(test, k, budget, thresholds=thr) for k in SYSTEMS}
    cloudy_crit = test[(test["critical"] == 1) & (test["cloud_frac"] >= CLOUD_SWITCH)]

    def delivered(key):
        return int(sum(_full_tile_rank(logs[key], r.chip, r.row, r.col, budget) is not None
                       for r in cloudy_crit.itertuples()))

    cands = []
    for r in cloudy_crit.itertuples():
        pos = {k: _full_tile_rank(logs[k], r.chip, r.row, r.col, budget) for k in SYSTEMS}
        if (pos["orbit_prism"] and not pos["discard_cloudy"] and not pos["optical_only"]
                and _labelled_share(r.chip, int(r.row), int(r.col)) >= MIN_LABELLED):
            cands.append((r.water_frac, r.p_radar, r, pos))
    if not cands:
        sys.exit("no cloudy critical tile is delivered by ORBIT-PRISM and missed by both discard-cloudy "
                 "and optical-only at this budget; widen the criteria or pick a tile by hand")
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    _, _, t, pos = cands[0]

    alert = sim.build_l0_alert_packet(t.chip, int(t.row), int(t.col), 1, float(t.p_radar),
                                      float(t.extent_radar), pass_number=0, seq=0)
    out = {
        "tile_id": f"{t.chip}_{int(t.row)}_{int(t.col)}",
        "event": t.event, "chip": t.chip, "row": int(t.row), "col": int(t.col), "split": t.split,
        "dial_target": args.alpha, "thresholds": thr,
        "truth": {"water_frac": round(float(t.water_frac), 4), "critical": True},
        "cloud_frac": round(float(t.cloud_frac), 4), "routed_to": "radar",
        "scores": {"p_optical": round(float(t.p_optical), 4), "p_radar": round(float(t.p_radar), 4),
                   "extent_radar": round(float(t.extent_radar), 4)},
        "kept_by_dial": bool(float(t.p_radar) >= thr["radar"]),
        "alert_hex": alert.hex(),
        "scaled_budget_MB": round(budget / 1e6, 3),
        "full_tile_delivery": {SYSTEMS[k]: pos[k] for k in SYSTEMS},
        "cloudy_critical_tiles_delivered": {"total": int(len(cloudy_crit)),
                                            **{SYSTEMS[k]: delivered(k) for k in SYSTEMS}},
        "selection_rule": "highest flood share among cloudy critical tiles that ORBIT-PRISM delivers and "
                          "discard-cloudy and optical-only do not, at the scaled pass",
        "caveats": ["Chosen from the held-out event to illustrate the radar fallback; not a typical tile.",
                    "Clouds are simulated.", "Scaled pass, not the real 19.2 MB pass."],
        "n_candidates": len(cands),
    }
    path = ROOT / "results" / "demo_case.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
