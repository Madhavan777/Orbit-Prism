"""The website's cloud redraw and pass accounting, against the real data when it is on disk."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CHIP = "Bolivia_242570"
if not (ROOT / "data/raw/data/flood_events/HandLabeled/S2Hand" / f"{CHIP}_S2Hand.tif").exists() \
        or not (ROOT / "data/tiles_all.csv").exists():
    pytest.skip("needs Bolivia imagery and data/tiles_all.csv", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.api import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(ROOT))


def _sim(client, **kw):
    r = client.post("/api/simulate", json={"chip": CHIP, "redraw_clouds": True, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_seed_42_reproduces_the_stored_clouds_and_routing(client):
    stored = pd.read_csv(ROOT / "data/tiles_all.csv").query("chip == @CHIP")
    r = _sim(client, cloud_seed=42)
    assert r["weather"]["cloud_seed"] == 42
    assert r["weather"]["mean_cloud"] == pytest.approx(stored["cloud_frac"].mean(), abs=1e-3)
    assert r["weather"]["radar_routed"] == int((stored["sensor_used"] == "radar").sum())


def test_new_clouds_change_the_outcome_and_the_same_seed_repeats(client):
    a, b = _sim(client, cloud_seed=1), _sim(client, cloud_seed=2)
    assert a["weather"]["mean_cloud"] != b["weather"]["mean_cloud"]
    again = _sim(client, cloud_seed=1)
    assert again["weather"] == a["weather"]
    assert _sim(client)["weather"]["cloud_seed"] != _sim(client)["weather"]["cloud_seed"]


def test_no_pass_carries_more_than_the_budget(client):
    r = client.post("/api/simulate", json={"chip": CHIP}).json()
    orbit = r["systems"]["ORBIT-PRISM"]
    assert all(p["MB"] <= r["scaled_budget_MB"] + 1e-9 for p in orbit["passes"])
    assert all(q["in_pass_bytes"] / 1e6 <= r["scaled_budget_MB"] + 1e-9 for q in orbit["queue_preview"])


def test_events_other_than_the_held_out_one_are_listed_and_have_pictures(client):
    chips = client.get("/api/chips", params={"event": "Bolivia"}).json()["chips"]
    assert {c["split"] for c in chips} == {"holdout"}
    assert {e["event"] for e in client.get("/api/status").json()["events"]} >= {"Bolivia", "India", "USA"}
    tile = client.get("/api/tiles", params={"chip": CHIP}).json()["tiles"][0]["tile_id"]
    for kind in ("clear", "optical", "radar", "label"):
        img = client.get(f"/api/tile/{tile}/image", params={"kind": kind, "scale": 2})
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"
