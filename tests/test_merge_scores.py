"""scripts/merge_scores.py: real-score merge, tile_id keys, tiles_all.csv, and the refusals."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("merge_scores", ROOT / "scripts" / "merge_scores.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)

BASE_COLS = ["event", "chip", "row", "col", "split", "water_frac", "critical", "cloud_frac", "sensor_used"]


def _setup(tmp_path, n=6):
    base = pd.DataFrame({
        "event": "Ghana", "chip": [f"Ghana_{i}" for i in range(n)], "row": 0, "col": 1,
        "split": "valid", "water_frac": 0.4, "critical": 1,
        "cloud_frac": [0.1, 0.9] * (n // 2), "sensor_used": ["optical", "radar"] * (n // 2),
    })
    base.to_csv(tmp_path / "base.csv", index=False)
    tid = [f"{c}_{r}_{k}" for c, r, k in zip(base.chip, base.row, base.col)]
    pd.DataFrame({"tile_id": tid, "p_flood": 0.11, "extent": 0.21}).to_csv(tmp_path / "opt.csv", index=False)
    pd.DataFrame({"tile_id": tid, "p_flood": 0.77, "extent": 0.88}).to_csv(tmp_path / "rad.csv", index=False)
    return base


def _merge(tmp_path, **kw):
    return ms.merge_scores(tiles_base_path=tmp_path / "base.csv", optical_scores_path=tmp_path / "opt.csv",
                           radar_scores_path=tmp_path / "rad.csv", fake_tiles_path=tmp_path / "none.csv",
                           out_path=tmp_path / "tiles.csv", **kw)


def test_each_tile_gets_the_score_of_its_sensor_and_tiles_all_is_written(tmp_path):
    _setup(tmp_path)
    out = _merge(tmp_path)
    assert list(out.columns) == BASE_COLS + ["p_flood", "extent"]
    assert out.loc[out["sensor_used"] == "optical", "p_flood"].eq(0.11).all()
    assert out.loc[out["sensor_used"] == "radar", "p_flood"].eq(0.77).all()
    allp = pd.read_csv(tmp_path / "tiles_all.csv")
    assert {"p_optical", "extent_optical", "p_radar", "extent_radar"} <= set(allp.columns)
    assert allp["p_optical"].eq(0.11).all() and allp["p_radar"].eq(0.77).all()  # both scores on every tile


def test_missing_tile_scores_stop_the_merge(tmp_path):
    _setup(tmp_path)
    pd.read_csv(tmp_path / "rad.csv").iloc[:-1].to_csv(tmp_path / "rad.csv", index=False)
    with pytest.raises(SystemExit, match="no score"):
        _merge(tmp_path)


def test_synthetic_scores_are_refused_unless_allowed(tmp_path):
    _setup(tmp_path)
    (tmp_path / "opt.csv").unlink()
    with pytest.raises(SystemExit, match="Refusing"):
        _merge(tmp_path)
    assert not (tmp_path / "tiles.csv").exists()
