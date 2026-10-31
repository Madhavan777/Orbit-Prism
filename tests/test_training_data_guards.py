"""The trainer must read every event file, use the CONTEXT.md split names, and never touch the holdout.

Builds a tiny processed dataset (random pixels, the exact NPZ layout of src/data/dataset.py) in a
temp folder and runs scripts/train.py on it. No real imagery is involved.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _write_event(folder: Path, event: str, splits, n_each: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_each * len(splits)
    wf = rng.uniform(0, 1, n).astype(np.float32)
    np.savez_compressed(
        folder / f"{event}.npz",
        tile_id=np.array([f"{event}_C{i}_0_0" for i in range(n)]),
        optical=rng.integers(0, 256, (n, 128, 128, 4), dtype=np.uint8),
        radar=rng.integers(0, 256, (n, 128, 128, 2), dtype=np.uint8),
        water_frac=wf,
        cloud_frac=np.full(n, 0.1, dtype=np.float32),
        split=np.array([s for s in splits for _ in range(n_each)]),
        critical=(wf >= 0.30).astype(np.int32),
    )


def _run_train(data_dir: Path, tmp_path: Path):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--data_dir", str(data_dir),
         "--epochs", "1", "--output_model", str(tmp_path / "m.keras")],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3",
             "PYTHONIOENCODING": "utf-8"},
    )


def test_trainer_reads_all_events_and_leaves_holdout_alone(tmp_path):
    data = tmp_path / "processed"
    data.mkdir()
    # "Bolivia" sorts first alphabetically, like the real held-out event.
    _write_event(data, "Bolivia", ["holdout"], seed=1)
    _write_event(data, "Ghana", ["train", "valid", "test"], seed=2)
    _write_event(data, "India", ["train", "valid"], seed=3)

    res = _run_train(data, tmp_path)
    out = res.stdout + res.stderr
    assert res.returncode == 0, out[-2000:]
    assert "TRAIN Set" in out and "CAL/VAL Set: 16 samples" in out   # Ghana 8 + India 8 valid tiles
    assert "TRAIN Set (cloud_frac < 0.5): 16 samples" in out        # Ghana 8 + India 8 train tiles
    assert "HOLDOUT tiles present and untouched: 8" in out


def test_radar_loader_reads_every_event_file(tmp_path, monkeypatch):
    """The radar loader must combine all per-event NPZ files, not just the first one."""
    import pandas as pd
    from src.models import radar

    data = tmp_path / "processed"
    data.mkdir()
    rows = []
    for event, splits, seed in [("Bolivia", ["holdout"], 1), ("Ghana", ["train", "valid"], 2)]:
        _write_event(data, event, splits, n_each=4, seed=seed)
        for i in range(4 * len(splits)):
            rows.append({"event": event, "chip": f"{event}_C{i}", "row": 0, "col": 0,
                         "split": [s for s in splits for _ in range(4)][i],
                         "water_frac": 0.5, "critical": 1, "cloud_frac": 0.1, "sensor_used": "optical"})
    base = tmp_path / "tiles_base.csv"
    pd.DataFrame(rows).to_csv(base, index=False)
    monkeypatch.setattr(radar, "TILES_BASE_CSV", base)

    data_dict, matched = radar.load_radar_data_for_training(dataset_path=data)
    assert len(matched) == 12                                   # 4 holdout + 8 Ghana, none dropped
    assert set(matched["event"]) == {"Bolivia", "Ghana"}
    assert data_dict["images"].shape == (12, 128, 128, 2)
    assert data_dict["images"].dtype == np.float32 and data_dict["images"].max() <= 1.0
    # The training function only selects split == "train" and calibrates on "valid".
    assert not matched[matched["split"] == "train"]["event"].eq("Bolivia").any()


def test_early_stopping_uses_held_back_events_not_the_calibration_split(tmp_path):
    from src.constants import INTERNAL_VAL_EVENTS
    from src.models import radar
    import inspect
    radar_default = inspect.signature(radar.train_radar_pipeline).parameters["internal_val_events"].default
    assert tuple(radar_default) == tuple(INTERNAL_VAL_EVENTS)         # both models hold back the same events

    data = tmp_path / "processed"
    data.mkdir()
    _write_event(data, "Bolivia", ["holdout"], seed=1)
    _write_event(data, "Ghana", ["train", "valid"], seed=2)
    _write_event(data, "Spain", ["train", "valid"], seed=3)          # an internal-validation event
    res = _run_train(data, tmp_path)
    out = res.stdout + res.stderr
    assert res.returncode == 0, out[-2000:]
    assert "TRAIN Set (cloud_frac < 0.5): 8 samples" in out          # Ghana train only; Spain train held back
    assert "EARLY-STOPPING Set" in out and ": 8 samples" in out.split("EARLY-STOPPING Set")[1].splitlines()[0]
    assert "CAL/VAL Set: 16 samples" in out                          # both events' valid tiles, untouched
    assert "falls back" not in out


def test_trainer_stops_when_calibration_split_is_missing(tmp_path):
    data = tmp_path / "processed"
    data.mkdir()
    _write_event(data, "Ghana", ["train", "test"], seed=2)           # no valid / calibration tiles
    res = _run_train(data, tmp_path)
    assert res.returncode != 0
    assert "calibration set is empty" in (res.stdout + res.stderr).lower()
