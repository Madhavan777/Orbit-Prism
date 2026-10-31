"""The pass budget is link capacity and never exceeds one real pass (19.2 MB); on-board memory is a separate limit."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "data/tiles_all.csv").exists():
    pytest.skip("needs data/tiles_all.csv", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from backend.api import create_app  # noqa: E402
from src.constants import PASS_BYTES  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(ROOT))


def test_a_large_event_cannot_be_given_more_than_one_real_pass(client):
    r = client.post("/api/simulate", json={"event": "Sri-Lanka", "scaled_fraction": 1.0, "policies": ["capture_order"]}).json()
    assert r["selection_volume_MB"] > 40                      # 663 tiles x 64 KiB
    assert r["scaled_budget_MB"] == pytest.approx(PASS_BYTES / 1e6)
    assert r["budget_capped_at_real_pass"] is True
    assert "scaled pass" in r["scaled_note"] and "capped at the real 19.2 MB pass" in r["scaled_note"]


def test_the_default_budget_is_a_fifth_of_the_event_and_below_one_pass(client):
    r = client.post("/api/simulate", json={"event": "Sri-Lanka", "policies": ["capture_order"]}).json()
    assert r["scaled_budget_MB"] == pytest.approx(0.2 * r["selection_volume_MB"], rel=1e-3)
    assert r["scaled_budget_MB"] < PASS_BYTES / 1e6 and r["budget_capped_at_real_pass"] is False
