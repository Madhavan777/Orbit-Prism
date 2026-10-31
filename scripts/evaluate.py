"""Evaluate the five systems and two ablations on the held-out test event.

Outputs (in --out, default results/ for real runs and results/synthetic/ for placeholder runs):
    headline.png     critical-tile recall vs cumulative MB, one line per system
    ablations.png    ORBIT-PRISM vs no radar vs no dial
    summary.csv      per system: bytes to 95% recall, recall at the scaled pass, cloudy recall,
                     final recall, alert recall, time to first critical alert
    dial.csv         target miss rate vs achieved miss rate per path
    curves.csv       the plotted curves as a table (the table view of headline.png)
    run_info.json    where the scores came from, the split, counts, and whether it is synthetic

Real run (needs data/tiles_all.csv from scripts/merge_scores.py and a real results/calibration.json):
    python scripts/evaluate.py
Plumbing test on the synthetic placeholder (output is marked SYNTHETIC and goes to results/synthetic/):
    python scripts/evaluate.py --allow-placeholder

The headline recall is FULL-TILE recall (CONTEXT.md). The byte budget is the SCALED pass: 20% of the
event's all-optical volume, because Bolivia fits inside one real 19.2 MB pass. Always label it so.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import OPTICAL_B, SCALED_PASS_FRACTION, TEST_SPLIT  # noqa: E402
from src.evaluation import metrics, plotting  # noqa: E402
from src.evaluation.reference_policies import KEY, NotAvailable, kept_mask, run_policy  # noqa: E402

SYSTEMS = {  # policy -> display name
    "orbit_prism": "ORBIT-PRISM",
    "capture_order": "Capture order",
    "discard_cloudy": "Discard cloudy",
    "optical_only": "Optical-only triage",
    "colour_index": "Colour-index triage",
}
ABLATIONS = {"orbit_prism": "ORBIT-PRISM", "orbit_no_radar": "No radar (ablation)",
             "orbit_no_dial": "No dial (ablation)"}
SYNTHETIC_SEED = 11


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def synthetic_extra_scores(tiles: pd.DataFrame) -> pd.DataFrame:
    """PLACEHOLDER only: invented optical and NDWI scores for every tile, built from the real labels.

    Real runs read these from the score files instead. Cloudy tiles get a noisy optical score,
    which is what the 'no radar' and 'optical-only' systems would see.
    """
    rng = np.random.default_rng(SYNTHETIC_SEED)
    signal = 5.0 * (tiles["water_frac"].to_numpy() - 0.30)
    cloud = tiles["cloud_frac"].to_numpy()
    sd_opt = 1.0 + 4.0 * cloud
    out = tiles.copy()
    out["p_optical"] = _sigmoid(signal + rng.normal(0, sd_opt))
    out["p_ndwi"] = _sigmoid(0.8 * signal + rng.normal(0, 1.5 + 3.5 * cloud))
    return out


def load_tiles(args):
    if args.allow_placeholder:
        fake = ROOT / "data" / "fake_tiles.csv"
        print("*** SYNTHETIC PLACEHOLDER SCORES (data/fake_tiles.csv): these numbers are NOT results. ***")
        return synthetic_extra_scores(pd.read_csv(fake)), "synthetic_placeholder (data/fake_tiles.csv)", True
    path = Path(args.tiles)
    if not path.exists():
        sys.exit(f"Missing {path}. Make it with scripts/merge_scores.py once real score files exist, "
                 f"or pass --allow-placeholder to test the plumbing.")
    return pd.read_csv(path), f"real ({path.name})", False


def load_thresholds(args, all_tiles: pd.DataFrame, synthetic: bool):
    key = str(args.alpha)
    if synthetic:  # calibrate in-process on the placeholder; nothing is written
        from src.calibration.calibrate import CalibrationPipeline
        pipe = CalibrationPipeline(target_alphas=[args.alpha])
        pipe.score_source = "synthetic_placeholder (in-process)"
        pipe.fit_and_calibrate(all_tiles, calib_splits=["valid"])
        return pipe.to_dict()
    path = Path(args.calibration)
    if not path.exists():
        sys.exit(f"Missing {path}: run python -m src.calibration.calibrate on real scores first.")
    cal = json.loads(path.read_text())
    src = str(cal.get("score_source", ""))
    if not src or src.startswith("synthetic") or src == "unspecified":
        sys.exit(f"{path} has score_source '{src}': it was not made from real model scores.")
    if key not in cal["optical"]:
        sys.exit(f"{path} has no threshold for target miss rate {key}")
    return cal


def dial_table(test: pd.DataFrame, cal: dict) -> pd.DataFrame:
    rows = []
    for path in ("optical", "radar"):
        sub = test[test["sensor_used"] == path]
        crit = sub[sub["critical"] == 1]
        for alpha, thr in cal[path].items():
            missed = int((crit["p_flood"] < float(thr)).sum())
            rows.append({"path": path, "target_miss_rate": float(alpha), "threshold": float(thr),
                         "critical_tiles": len(crit), "missed": missed,
                         "achieved_miss_rate": missed / len(crit) if len(crit) else float("nan")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", default=str(ROOT / "data" / "tiles_all.csv"))
    ap.add_argument("--calibration", default=str(ROOT / "results" / "calibration.json"))
    ap.add_argument("--split", default=TEST_SPLIT, help="split to evaluate (default: holdout = Bolivia)")
    ap.add_argument("--alpha", type=float, default=0.05, help="target miss rate for the dial (example setting)")
    ap.add_argument("--scheduler", choices=["reference", "sim"], default="reference",
                    help="reference = src/evaluation/reference_policies.py; sim = src.scheduler.sim.run")
    ap.add_argument("--allow-placeholder", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_tiles, source, synthetic = load_tiles(args)
    cal = load_thresholds(args, all_tiles, synthetic)
    out = Path(args.out) if args.out else ROOT / "results" / ("synthetic" if synthetic else "")
    out.mkdir(parents=True, exist_ok=True)

    test = all_tiles[all_tiles["split"] == args.split].reset_index(drop=True)
    if test.empty:
        sys.exit(f"No tiles with split == {args.split!r}")
    thresholds = {p: float(cal[p][str(args.alpha)]) for p in ("optical", "radar")}
    scaled_budget = SCALED_PASS_FRACTION * len(test) * OPTICAL_B
    n_crit = int(test["critical"].sum())
    print(f"split={args.split}: {len(test)} tiles, {n_crit} critical, scaled pass {scaled_budget / 1e6:.2f} MB, "
          f"dial target {args.alpha:.0%} -> thresholds {thresholds}")

    if args.scheduler == "sim":
        try:
            from src.scheduler.sim import run as sim_run
        except ImportError:
            sys.exit("src.scheduler.sim.run does not exist yet; use --scheduler reference")

    logs, skipped = {}, {}
    for policy in dict.fromkeys(list(SYSTEMS) + list(ABLATIONS)):
        try:
            if args.scheduler == "sim":
                # sim.run(tiles, policy, budget_bytes, thresholds=None): see CONTEXT.md. `thresholds`
                # is {"optical": t, "radar": t}, the raw-p_flood cut-offs of the miss-rate dial.
                logs[policy] = sim_run(test, policy, scaled_budget, thresholds=thresholds)
            else:
                logs[policy] = run_policy(test, policy, thresholds)
        except NotAvailable as e:
            skipped[policy] = str(e)
            print(f"  skipped {policy}: {e}")

    names = {**SYSTEMS, **ABLATIONS}
    summary = pd.DataFrame([metrics.summarise(names[p], logs[p], test, scaled_budget) for p in logs])
    summary["realised_miss_rate"] = [metrics.realised_miss_rate(test, kept_mask(test, logs[p])) for p in logs]
    summary.to_csv(out / "summary.csv", index=False, lineterminator="\n", float_format="%.4f")
    dial_table(test, cal).to_csv(out / "dial.csv", index=False, lineterminator="\n", float_format="%.4f")

    curves = {names[p]: metrics.recall_curve(logs[p], test) for p in logs}
    long = pd.concat([c.assign(system=n, cum_MB=c["cum_bytes"] / 1e6)[["system", "cum_MB", "recall"]]
                      for n, c in curves.items()])
    long.to_csv(out / "curves.csv", index=False, lineterminator="\n", float_format="%.5f")

    event = "Bolivia" if args.split == TEST_SPLIT else args.split
    subtitle = (f"Held-out event {event}: {len(test)} tiles, {n_crit} critical. Dial target "
                f"{args.alpha:.0%} (example setting). Clouds are simulated.\n"
                f"Vertical line: the scaled pass (20% of the event), not the real 19.2 MB pass.")
    mb = scaled_budget / 1e6
    for theme, suffix in (("light", ""), ("dark", "_dark")):
        plotting.recall_plot({names[p]: curves[names[p]] for p in SYSTEMS if p in logs},
                             out / f"headline{suffix}.png", "Critical-tile recall against data sent",
                             subtitle, mb, synthetic, theme=theme)
        plotting.recall_plot({names[p]: curves[names[p]] for p in ABLATIONS if p in logs},
                             out / f"ablations{suffix}.png", "Ablations: what each part contributes",
                             subtitle, mb, synthetic, theme=theme)
    (out / "run_info.json").write_text(json.dumps({
        "score_source": source, "synthetic": synthetic, "split": args.split, "tiles": len(test),
        "critical_tiles": n_crit, "scaled_pass_MB": round(mb, 4), "dial_target": args.alpha,
        "thresholds": thresholds, "scheduler": args.scheduler, "skipped": skipped,
    }, indent=2) + "\n")

    print("\n" + summary.round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
