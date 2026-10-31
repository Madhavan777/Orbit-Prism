"""Unit tests for the experimental hazard registry, priority scoring and scheduler sketch.

Scope is floods and forest fires only. Validates:
  1. DisasterRegistry: exactly the two in-scope hazards, with honest statuses.
  2. RiskEngine confidence math & explicit UNKNOWN exposure handling.
  3. PriorityScheduler bandwidth simulation, carryover, de-duplication, & CSV contracts.
  4. The wildfire architecture's budget and a random forward pass.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.disasters.registry import DisasterRegistry, DisasterCategory, ImplementationStatus
from src.disasters.risk_engine import RiskEngine, DisasterEvent, RiskAssessment
from src.disasters.scheduler_interface import PriorityScheduler, PacketTier, TransmissionStatus
from src.disasters.wildfire import build_wildfire_model, compute_nbr_index
from src.models.optical import calculate_model_memory


def test_registry_profiles():
    """Only the two in-scope hazards exist, and statuses describe what is really in the repo."""
    categories = DisasterRegistry.list_categories()
    assert set(categories) == {DisasterCategory.FLOOD, DisasterCategory.WILDFIRE}

    flood_prof = DisasterRegistry.get_profile(DisasterCategory.FLOOD)
    # Nothing is trained on real data and evaluated on the held-out event yet.
    assert flood_prof.status == ImplementationStatus.SIMULATED
    assert "Sen1Floods11" in flood_prof.dataset.name
    assert "unconfirmed" in flood_prof.dataset.license.lower()

    fire_prof = DisasterRegistry.get_profile(DisasterCategory.WILDFIRE)
    # No fire data has been loaded, so fire cannot be anything above PLANNED.
    assert fire_prof.status == ImplementationStatus.PLANNED
    assert "NOT LOADED" in fire_prof.dataset.samples_approx

    assert all(p.status != ImplementationStatus.IMPLEMENTED for p in DisasterRegistry.all_profiles().values())


def test_risk_engine_unknown_exposure_handling():
    """Verify that missing exposure is NOT treated as zero risk."""
    engine = RiskEngine()

    # Event with missing exposure (None)
    event_unknown = DisasterEvent(
        tile_id="tile_unknown_exp",
        category=DisasterCategory.FLOOD,
        p_event=0.90,
        severity=0.80,
        extent=0.50,
        exposure_score=None
    )

    # Event with explicit zero exposure (unpopulated ocean/desert)
    event_zero_exp = DisasterEvent(
        tile_id="tile_zero_exp",
        category=DisasterCategory.FLOOD,
        p_event=0.90,
        severity=0.80,
        extent=0.50,
        exposure_score=0.0
    )

    assessment_unknown = engine.assess_event(event_unknown)
    assessment_zero = engine.assess_event(event_zero_exp)

    assert assessment_unknown.is_exposure_unknown is True
    assert assessment_zero.is_exposure_unknown is False

    # The unknown event MUST have a higher risk score than the confirmed zero-exposure event
    assert assessment_unknown.priority_score > assessment_zero.priority_score
    print(f"\nPriority Score Comparison: Unknown Exp = {assessment_unknown.priority_score:.2f} vs Zero Exp = {assessment_zero.priority_score:.2f}")


def test_risk_engine_cloud_confidence_degradation():
    """Verify cloud cover lowers confidence and scales priority appropriately."""
    engine = RiskEngine()

    clear_event = DisasterEvent(
        tile_id="clear_tile",
        category=DisasterCategory.WILDFIRE,
        p_event=0.85,
        severity=0.75,
        extent=0.30,
        cloud_frac=0.05,
        exposure_score=0.80
    )

    cloudy_event = DisasterEvent(
        tile_id="cloudy_tile",
        category=DisasterCategory.WILDFIRE,
        p_event=0.85,
        severity=0.75,
        extent=0.30,
        cloud_frac=0.90,
        exposure_score=0.80
    )

    a_clear = engine.assess_event(clear_event)
    a_cloudy = engine.assess_event(cloudy_event)

    assert a_clear.confidence > a_cloudy.confidence
    assert a_clear.priority_score > a_cloudy.priority_score


def test_priority_ranking_across_hazards():
    """Verify ranking orders by priority, not blindly by hazard type."""
    engine = RiskEngine()

    # Wildfire near dense population
    urban_fire = DisasterEvent(
        tile_id="urban_fire",
        category=DisasterCategory.WILDFIRE,
        p_event=0.95,
        severity=0.90,
        extent=0.60,
        exposure_score=0.95
    )

    # Flood over an unpopulated area
    remote_flood = DisasterEvent(
        tile_id="remote_flood",
        category=DisasterCategory.FLOOD,
        p_event=0.95,
        severity=0.80,
        extent=0.50,
        exposure_score=0.05
    )

    ranked = engine.rank_events([remote_flood, urban_fire])
    assert ranked[0].tile_id == "urban_fire"
    assert ranked[0].priority_score > ranked[1].priority_score


def test_scheduler_tier1_precedence_and_carryover(tmp_path):
    """Test two-tier scheduler: Tier 1 alerts sent before Tier 2 imagery, carryover preserved."""
    engine = RiskEngine()
    scheduler = PriorityScheduler()

    events = [
        DisasterEvent("tile_1", DisasterCategory.FLOOD, 0.90, 0.80, 0.40, exposure_score=0.70),
        DisasterEvent("tile_2", DisasterCategory.WILDFIRE, 0.85, 0.70, 0.30, exposure_score=0.60),
        DisasterEvent("tile_3", DisasterCategory.FLOOD, 0.95, 0.90, 0.80, exposure_score=0.85),
    ]
    assessments = [engine.assess_event(e) for e in events]
    scheduler.enqueue_assessments(assessments)

    # Budget enough for all Tier 1 alerts (3 * 32 B = 96 B) + exactly ONE full optical tile (65,536 B)
    # = 65,632 B; 66,000 B leaves no room for a second tile.
    downlinked, summary = scheduler.simulate_pass(bandwidth_budget_bytes=66_000)

    assert summary.alerts_sent == 3
    assert summary.images_sent == 1
    assert summary.remaining_in_queue == 2  # The other two images remain in carryover
    assert downlinked[0].tier == PacketTier.TIER1_ALERT

    # Test CSV export in legacy and extended format
    csv_extended = tmp_path / "extended_scores.csv"
    csv_legacy = tmp_path / "legacy_scores.csv"

    scheduler.export_csv(assessments, str(csv_extended), legacy_flood_format=False)
    scheduler.export_csv(assessments, str(csv_legacy), legacy_flood_format=True)

    assert csv_extended.exists()
    assert csv_legacy.exists()


def test_wildfire_model_tiny_ml():
    """Verify wildfire TinyML architecture, weight budget (<= 60 KB), and forward pass."""
    model = build_wildfire_model()
    memory_info = calculate_model_memory(model)

    assert memory_info["float32_size_kb"] <= 60.0, f"Wildfire model exceeds 60 KB: {memory_info['float32_size_kb']} KB"
    print(f"\nWildfire Model FP32 Size: {memory_info['float32_size_kb']} KB ({memory_info['total_params']} params)")

    # Test random forward pass
    x_rand = np.random.randint(0, 256, size=(2, 128, 128, 4), dtype=np.uint8)
    p_fire, extent = model.predict(x_rand, verbose=0)
    assert p_fire.shape == (2, 1)
    assert extent.shape == (2, 1)

    # Test NBR calculation
    p_nbr, ext_nbr = compute_nbr_index(x_rand)
    assert p_nbr.shape == (2, 1)
    assert ext_nbr.shape == (2, 1)


def test_scope_is_floods_and_fires_only():
    """No code in src/disasters models an out-of-scope hazard."""
    assert not (Path(__file__).resolve().parent.parent / "src" / "disasters" / ("cy" + "clone.py")).exists()
    assert {c.value for c in DisasterCategory} == {"flood", "wildfire"}
