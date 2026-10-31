"""Tests for the evaluation metrics, the reference policies and scripts/evaluate.py.

Metrics are checked on small hand-made send logs where the right answer is obvious.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics
from src.evaluation.reference_policies import NotAvailable, kept_mask, run_policy

ROOT = Path(__file__).resolve().parent.parent


def _tiles():
    # 6 tiles; 2 critical (A0, A1); A1 is cloudy
    return pd.DataFrame({
        "chip": list("AAAAAA"), "row": [0, 0, 0, 0, 0, 0], "col": list(range(6)),
        "critical": [1, 1, 0, 0, 0, 0], "cloud_frac": [0.1, 0.8, 0.1, 0.1, 0.9, 0.1],
        "sensor_used": ["optical", "radar", "optical", "optical", "radar", "optical"],
        "p_flood": [0.9, 0.8, 0.3, 0.2, 0.7, 0.1], "extent": [0.6, 0.5, 0.1, 0.1, 0.1, 0.0],
        "p_optical": [0.9, 0.2, 0.3, 0.2, 0.4, 0.1], "split": "holdout",
    })


def _log(rows):
    """rows: (col, layer, bytes) in transmission order."""
    cum = np.cumsum([r[2] for r in rows])
    return pd.DataFrame({"order": range(len(rows)), "chip": "A", "row": 0,
                         "col": [r[0] for r in rows], "layer": [r[1] for r in rows],
                         "bytes": [r[2] for r in rows], "cum_bytes": cum, "pass_index": 0})


def test_recall_curve_and_bytes_to_recall():
    log = _log([(2, "L2", 100), (0, "L2", 100), (3, "L2", 100), (1, "L2", 100)])
    curve = metrics.recall_curve(log, _tiles())
    assert metrics.recall_at_bytes(curve, 99) == 0.0
    assert metrics.recall_at_bytes(curve, 200) == 0.5      # critical tile A0 arrived at 200 B
    assert metrics.recall_at_bytes(curve, 400) == 1.0
    assert metrics.bytes_to_recall(curve, 0.95) == 400
    assert metrics.bytes_to_recall(curve, 0.5) == 200


def test_bytes_to_recall_is_nan_when_never_reached():
    log = _log([(0, "L2", 100), (2, "L2", 100)])            # critical tile A1 never sent
    curve = metrics.recall_curve(log, _tiles())
    assert metrics.final_recall(curve) == 0.5
    assert np.isnan(metrics.bytes_to_recall(curve, 0.95))


def test_curve_extends_to_total_bytes_sent():
    log = _log([(0, "L2", 100), (1, "L2", 100), (2, "L2", 100)])
    curve = metrics.recall_curve(log, _tiles())
    assert curve["cum_bytes"].iloc[-1] == 300 and curve["recall"].iloc[-1] == 1.0


def test_alert_recall_is_separate_from_full_tile_recall():
    log = _log([(0, "L0", 32), (1, "L0", 32), (0, "L2", 100)])
    tiles = _tiles()
    assert metrics.final_recall(metrics.recall_curve(log, tiles, "L0")) == 1.0
    assert metrics.final_recall(metrics.recall_curve(log, tiles, "L2")) == 0.5


def test_time_to_first_critical_alert_uses_the_link_rate():
    alert_first = _log([(0, "L0", 32), (0, "L2", 100)])
    full_only = _log([(2, "L2", 100), (0, "L2", 100)])
    t = _tiles()
    assert metrics.time_to_first_critical_alert(alert_first, t) == pytest.approx(32 * 8 / 256_000)
    assert metrics.time_to_first_critical_alert(full_only, t) == pytest.approx(200 * 8 / 256_000)
    assert np.isnan(metrics.time_to_first_critical_alert(_log([(2, "L2", 100)]), t))


def test_cloudy_recall_only_counts_cloudy_critical_tiles():
    log = _log([(0, "L2", 100), (1, "L2", 100)])
    # Only A1 is a cloudy critical tile; it arrives at 200 B.
    assert metrics.cloudy_recall_at_bytes(log, _tiles(), 150) == 0.0
    assert metrics.cloudy_recall_at_bytes(log, _tiles(), 200) == 1.0


def test_orbit_drops_tiles_below_the_threshold_and_sends_alerts_first():
    tiles = _tiles()
    log = run_policy(tiles, "orbit_prism", {"optical": 0.5, "radar": 0.5})
    layers = list(log["layer"])
    assert layers == sorted(layers)                         # all L0 before any L2
    kept = kept_mask(tiles, log)
    assert kept.tolist() == [True, True, False, False, True, False]  # p_flood >= 0.5 per path
    assert log.loc[log["layer"] == "L2", "bytes"].tolist().count(32768) == 2   # radar tiles are 32 KB


def test_baselines_and_missing_inputs():
    tiles = _tiles()
    assert len(run_policy(tiles, "capture_order")) == 6
    assert len(run_policy(tiles, "discard_cloudy")) == 4    # cloud_frac >= 0.5 dropped
    with pytest.raises(NotAvailable):
        run_policy(tiles, "colour_index")                  # no p_ndwi column
    with pytest.raises(NotAvailable):
        run_policy(tiles, "orbit_prism")                   # no thresholds


def test_evaluate_refuses_without_real_scores_and_runs_on_placeholder(tmp_path):
    env_args = [sys.executable, str(ROOT / "scripts" / "evaluate.py")]
    refuse = subprocess.run(env_args + ["--tiles", str(tmp_path / "nope.csv")], capture_output=True,
                            text=True, cwd=ROOT)
    assert refuse.returncode != 0 and "allow-placeholder" in (refuse.stdout + refuse.stderr)

    run = subprocess.run(env_args + ["--allow-placeholder", "--out", str(tmp_path / "out")],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    assert run.returncode == 0, (run.stdout + run.stderr)[-1500:]
    out = tmp_path / "out"
    for name in ("headline.png", "headline_dark.png", "ablations.png", "summary.csv", "dial.csv",
                 "curves.csv", "run_info.json"):
        assert (out / name).exists(), name
    summary = pd.read_csv(out / "summary.csv")
    assert {"ORBIT-PRISM", "Capture order", "Discard cloudy"} <= set(summary["system"])
    assert '"synthetic": true' in (out / "run_info.json").read_text()
