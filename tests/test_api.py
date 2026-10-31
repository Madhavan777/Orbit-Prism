"""The web API, tested on a small self-contained project folder (placeholder results, untrained models)."""
import io
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import ALERT_FORMAT, create_app, pack_alert

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    (root / "data" / "processed").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "fake_tiles.csv", root / "data" / "fake_tiles.csv")
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--allow-placeholder",
                          "--out", str(root / "results" / "synthetic")], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", cwd=ROOT)
    assert run.returncode == 0, run.stdout[-800:] + run.stderr[-800:]
    # a processed file with the first 6 holdout tiles (random pixels)
    fake = pd.read_csv(ROOT / "data" / "fake_tiles.csv")
    hold = fake[fake.split == "holdout"].head(6)
    ids = (hold.chip + "_" + hold.row.astype(str) + "_" + hold.col.astype(str)).to_numpy().astype(str)  # like the real NPZ
    rng = np.random.default_rng(0)
    np.savez_compressed(root / "data" / "processed" / "Bolivia.npz", tile_id=ids,
                        optical=rng.integers(0, 256, (6, 128, 128, 4), dtype=np.uint8),
                        radar=rng.integers(0, 256, (6, 128, 128, 2), dtype=np.uint8))
    from src.models.optical import build_optical_model
    from src.models.radar import build_radar_model
    (root / "models").mkdir()
    build_optical_model().save(root / "models" / "optical_float.keras")
    build_radar_model().save(root / "models" / "radar_float.keras")
    return root, ids, hold


@pytest.fixture(scope="module")
def client(project):
    return TestClient(create_app(project[0]))


def test_status_says_placeholder_and_what_is_available(client):
    s = client.get("/api/status").json()
    assert s["mode"] == "placeholder" and s["synthetic"] is True
    assert s["banner"] == "SYNTHETIC PLACEHOLDER SCORES: NOT RESULTS"
    assert s["models"] == {"optical": "float", "radar": "float"} and s["images"] is True
    assert s["link"]["pass_MB"] == 19.2 and s["packets"]["L0"] == 32 and s["packets"]["L2_radar"] == 32768
    assert s["dial_targets"] == [0.01, 0.05, 0.1]


def test_results_endpoints_return_the_evaluation_tables(client):
    rows = client.get("/api/summary").json()["rows"]
    assert {"ORBIT-PRISM", "Capture order", "Discard cloudy"} <= {r["system"] for r in rows}
    curves = client.get("/api/curves").json()
    assert curves["synthetic"] is True and len(curves["curves"]["ORBIT-PRISM"]) > 3
    dial = client.get("/api/dial").json()
    assert [t["target_miss_rate"] for t in dial["targets"]] == [0.01, 0.05, 0.1]
    assert all(t["thresholds"]["optical"] >= 0 for t in dial["targets"])
    assert client.get("/api/memory").json()["rows"] == []              # no memory table in this project


def test_chips_and_tiles_carry_the_dial_decision(client, project):
    chips = client.get("/api/chips", params={"split": "holdout"}).json()["chips"]
    assert chips and all(c["event"] == "Bolivia" for c in chips)
    t = client.get("/api/tiles", params={"chip": chips[0]["chip"]}).json()["tiles"]
    assert t and set(t[0]["kept"]) == {"0.01", "0.05", "0.1"}
    assert client.get("/api/tiles", params={"chip": "nope"}).status_code == 404


def test_images_are_pngs_and_missing_pixels_are_a_clear_404(client, project):
    _, ids, hold = project
    for kind in ("optical", "radar"):
        r = client.get(f"/api/tile/{ids[0]}/image", params={"kind": kind, "scale": 2})
        assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    assert Image.open(io.BytesIO(client.get(f"/api/tile/{ids[0]}/image", params={"scale": 3}).content)).size == (384, 384)
    other = pd.read_csv(ROOT / "data" / "fake_tiles.csv").query("split == 'train'").iloc[0]
    missing = f"{other.chip}_{other.row}_{other.col}"
    assert client.get(f"/api/tile/{missing}/image").status_code == 404
    assert client.get(f"/api/tile/{ids[0]}/image", params={"kind": "label"}).status_code == 404   # no raw labels here
    assert client.get("/api/tile/nope/image").status_code == 404


def test_triage_runs_the_models_and_makes_the_dial_decision(client, project):
    _, ids, _ = project
    r = client.post("/api/triage", json={"tile_id": ids[0], "alpha": 0.05}).json()
    assert r["synthetic"] is True and r["routed_to"] in ("optical", "radar")
    assert set(r["both_models"]) == {"optical", "radar"}               # both models ran on the tile
    assert r["score_source"] == "live model" and 0 <= r["p_flood"] <= 1
    assert r["decision"] == ("KEEP" if r["p_flood"] >= r["threshold"] else "SKIP")
    assert r["alert_bytes"] == 32 and len(bytes.fromhex(r["alert_hex"])) == 32
    assert r["l2_bytes"] == (32768 if r["routed_to"] == "radar" else 65536)
    assert r["outcome"] and r["truth"]["critical"] in (0, 1)
    assert client.post("/api/triage", json={"tile_id": ids[0], "alpha": 0.07}).status_code == 422
    assert client.post("/api/triage", json={"tile_id": "nope", "alpha": 0.05}).status_code == 404


def test_triage_falls_back_to_stored_scores_without_pixels(client, project):
    other = pd.read_csv(ROOT / "data" / "fake_tiles.csv").query("split == 'holdout'").iloc[10]
    r = client.post("/api/triage", json={"tile_id": f"{other.chip}_{other.row}_{other.col}", "alpha": 0.1}).json()
    assert r["score_source"] == "stored score" and r["both_models"] == {}
    assert abs(r["p_flood"] - other.p_flood) < 1e-9


def test_alert_packet_is_32_bytes_and_decodes():
    pkt = pack_alert("Bolivia_1", 2, 3, "radar", True, 0.5, 0.25, 0.125, 32768)
    assert len(pkt) == 32 == struct.calcsize(ALERT_FORMAT)
    magic, ver, flags, chiphash, row, col, p, e, pr, l2, _, crc = struct.unpack(ALERT_FORMAT, pkt)
    assert magic == b"OP" and ver == 1 and flags == 3 and (row, col) == (2, 3)
    assert chiphash == zlib.crc32(b"Bolivia_1") and l2 == 32768
    assert abs(p / 65535 - 0.5) < 1e-4 and abs(e / 65535 - 0.25) < 1e-4
    assert crc == zlib.crc32(pkt[:28]) & 0xFFFFFFFF


def _npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_upload_triage_validates_the_file(client):
    good = _npz(optical=np.zeros((128, 128, 4), np.uint8), radar=np.zeros((128, 128, 2), np.uint8))
    up = lambda data, **form: client.post("/api/triage/upload", files={"file": ("t.npz", data)}, data=form)  # noqa: E731
    r = up(good, cloud_frac="0.9", alpha="0.05")
    assert r.status_code == 200 and r.json()["routed_to"] == "radar" and "truth" not in r.json()
    assert up(good, cloud_frac="0.1").json()["routed_to"] == "optical"
    assert up(_npz(optical=np.zeros((64, 64, 4), np.uint8)), cloud_frac="0.1").status_code == 422
    assert up(_npz(optical=np.zeros((128, 128, 4), np.float32)), cloud_frac="0.1").status_code == 422
    assert up(b"not an npz", cloud_frac="0.1").status_code == 422
    assert up(good, cloud_frac="1.5").status_code == 422
    assert up(_npz(radar=np.zeros((128, 128, 2), np.uint8)), cloud_frac="0.1").status_code == 422  # optical routed, no optical
    assert up(b"0" * 2_100_000, cloud_frac="0.1").status_code == 413


def test_simulate_returns_curves_queue_and_a_scaled_budget(client):
    r = client.post("/api/simulate", json={"split": "holdout", "alpha": 0.05}).json()
    assert r["selection"]["tiles"] > 100 and r["scaled_budget_MB"] == pytest.approx(0.2 * r["selection"]["tiles"] * 65536 / 1e6)
    assert "ORBIT-PRISM" in r["systems"] and "Capture order" in r["systems"]
    orbit = r["systems"]["ORBIT-PRISM"]
    assert orbit["queue_preview"][0]["layer"] == "L0" and orbit["curve"][0] == {"cum_MB": 0.0, "recall": 0.0}
    assert "scaled pass" in r["scaled_note"]
    one = client.post("/api/simulate", json={"split": "holdout", "chip": client.get("/api/chips").json()["chips"][0]["chip"],
                                              "policies": ["capture_order"]}).json()
    assert one["selection"]["tiles"] <= 16
    assert client.post("/api/simulate", json={"split": "holdout", "policies": ["nope"]}).status_code == 422
    assert client.post("/api/simulate", json={"split": "nowhere"}).status_code == 404


def test_demo_case_is_reported_unavailable_until_it_exists(client):
    assert client.get("/api/demo-case").json()["available"] is False


def test_static_website_is_served_when_web_exists(project):
    root = project[0]
    (root / "web").mkdir(exist_ok=True)
    (root / "web" / "index.html").write_text("<html><title>ok</title></html>")
    c = TestClient(create_app(root))
    assert c.get("/").status_code == 200 and "<title>ok</title>" in c.get("/").text
    assert c.get("/api/status").status_code == 200                       # the API still wins over static files


def test_every_endpoint_the_website_calls_exists_and_nothing_needs_the_internet():
    """The site's JavaScript must only call routes that exist, and load nothing from the network
    (the demo has to work without wifi)."""
    import re
    routes = {r.path for r in create_app(Path("/nonexistent")).routes}
    web = ROOT / "web"
    js = "\n".join(p.read_text(encoding="utf-8") for p in (web / "js").glob("*.js"))
    called = set(re.findall(r'(?:get|post)\(\s*["`](/[a-z\-]+)', js))       # get("/status"), post("/triage", ...)
    called |= set(re.findall(r'/api(/[a-z\-]+(?:/[a-z\-]+)?)', js))           # fetch("/api/triage/upload"), image URLs
    for path in called - {"/tile"}:          # /api/tile/<id>/image is a template URL, checked below
        assert "/api" + path in routes, f"web/js calls /api{path}, which the API does not serve"
    assert "/api/tile/{tile_id}/image" in routes and "/api/triage/upload" in routes
    assert {"/status", "/summary", "/curves", "/dial", "/memory", "/chips", "/tiles", "/triage", "/simulate", "/demo-case"} <= called
    for page in [web / "index.html", *web.glob("css/*.css"), *web.glob("js/*.js")]:
        text = page.read_text(encoding="utf-8")
        assert not re.search(r'(?:src|href)=["\']https?://', text) and "@import" not in text, f"{page.name} loads from the network"


def test_every_element_the_javascript_looks_up_exists_in_the_html():
    """Restyling must not rename the ids the page logic depends on (ids created at run time are listed)."""
    import re
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', html))
    used = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', (ROOT / "web" / "js" / "app.js").read_text(encoding="utf-8")))
    created_at_runtime = {"simchart"}
    missing = used - ids - created_at_runtime
    assert not missing, f"web/js/app.js uses ids that web/index.html does not define: {sorted(missing)}"


def test_everything_degrades_cleanly_with_no_results(tmp_path):
    c = TestClient(create_app(tmp_path))
    s = c.get("/api/status").json()
    assert s["mode"] == "unavailable" and s["events"] == [] and s["banner"] is not None
    for path in ("/api/summary", "/api/curves", "/api/dial", "/api/chips"):
        assert c.get(path).status_code == 503
