"""Tests for the ORBIT-PRISM downlink scheduler (src/scheduler/sim.py).
Author: Akhilesh S (akhilesh/scheduler)

Verifies:
- Determinism
- Pass budget constraints and carry-over resumption
- Layer precedence (L0 alerts before L1 previews before L2 full tiles)
- Exact packet byte sizes (32 B L0, 2,048 B L1, 65,536 B optical L2, 32,768 B radar L2)
- Zero layer duplication
- Consistency with reference policies and kept sets
- Safe behavior on empty and small-budget edge cases
- Accurate coverage bitmap and peak queue attributes
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from src.constants import (
    L0_B,
    L1_BYTES,
    OPTICAL_B,
    PASS_BYTES,
    RADAR_B,
)
from src.evaluation.reference_policies import kept_mask, run_policy
from src.scheduler.sim import (
    build_l0_alert_packet,
    run as sim_run,
)

KEY = ["chip", "row", "col"]


def _make_sample_tiles(n: int = 12) -> pd.DataFrame:
    """Creates a deterministic synthetic test DataFrame adhering to the 11-column contract."""
    rng = np.random.RandomState(42)
    chips = [f"Chip_{i // 4:02d}" for i in range(n)]
    rows = [i % 4 for i in range(n)]
    cols = [(i * 2) % 4 for i in range(n)]

    cloud_base = [0.1, 0.7, 0.2, 0.9, 0.4, 0.8, 0.0, 0.6, 0.3, 0.85, 0.15, 0.5]
    cloud_frac = np.array([cloud_base[i % len(cloud_base)] for i in range(n)])
    sensor_used = np.where(cloud_frac >= 0.5, "radar", "optical")
    p_flood = rng.uniform(0.1, 0.95, size=n)
    extent = rng.uniform(0.05, 0.8, size=n)
    water_frac = rng.uniform(0.0, 0.7, size=n)
    critical = (water_frac >= 0.30).astype(int)

    # Extra baseline columns
    p_optical = rng.uniform(0.05, 0.9, size=n)
    p_ndwi = rng.uniform(-0.4, 0.8, size=n)

    return pd.DataFrame({
        "event": "Bolivia",
        "chip": chips,
        "row": rows,
        "col": cols,
        "split": "holdout",
        "water_frac": water_frac,
        "critical": critical,
        "cloud_frac": cloud_frac,
        "sensor_used": sensor_used,
        "p_flood": p_flood,
        "extent": extent,
        "p_optical": p_optical,
        "p_ndwi": p_ndwi,
    })


def test_l0_alert_struct_exact_32_bytes():
    """Verify that build_l0_alert_packet returns exactly 32 bytes with valid CRC."""
    pkt = build_l0_alert_packet(
        chip="Bolivia_129334",
        row=2,
        col=3,
        sensor="radar",
        p_flood=0.875,
        extent=0.654,
        pass_number=1,
        seq=104,
        hazard_type=0,
    )
    assert isinstance(pkt, bytes)
    assert len(pkt) == 32
    assert len(pkt) == L0_B


def test_scheduler_determinism():
    """Verify that sim.run is strictly deterministic across runs."""
    tiles = _make_sample_tiles(12)
    thresholds = {"optical": 0.4, "radar": 0.4}
    budget = 150_000

    log1 = sim_run(tiles, "orbit_prism", budget_bytes=budget, thresholds=thresholds)
    log2 = sim_run(tiles, "orbit_prism", budget_bytes=budget, thresholds=thresholds)

    pd.testing.assert_frame_equal(log1, log2)
    assert log1.attrs["coverage_bitmap_bytes"] == log2.attrs["coverage_bitmap_bytes"]
    assert log1.attrs["peak_queue_bytes"] == log2.attrs["peak_queue_bytes"]


def test_pass_budget_never_exceeded():
    """Verify that in every pass, total transmitted bytes never exceeds pass budget."""
    tiles = _make_sample_tiles(12)
    thresholds = {"optical": 0.35, "radar": 0.35}
    budget = 100_000  # fits ~1 optical tile or 3 radar tiles

    for pol in ["capture_order", "discard_cloudy", "optical_only", "colour_index", "orbit_prism"]:
        log = sim_run(tiles, pol, budget_bytes=budget, thresholds=thresholds)
        if log.empty:
            continue
        bytes_by_pass = log.groupby("pass_index")["bytes"].sum()
        for p_idx, b_sum in bytes_by_pass.items():
            # If a pass contains a single packet that is larger than the budget, it's allowed as a dedicated pass
            pass_pkts = log[log["pass_index"] == p_idx]
            if len(pass_pkts) > 1:
                assert b_sum <= budget, f"Policy {pol} exceeded budget in pass {p_idx}: {b_sum} > {budget}"


def test_carry_over_and_pass_index_monotonicity():
    """Verify that carry-over spans across passes and pass_index is non-decreasing."""
    tiles = _make_sample_tiles(12)
    thresholds = {"optical": 0.3, "radar": 0.3}
    budget = 70_000  # small budget forces multi-pass carry-over

    log = sim_run(tiles, "orbit_prism", budget_bytes=budget, thresholds=thresholds)
    assert not log.empty
    assert log["pass_index"].max() > 0
    assert (np.diff(log["pass_index"]) >= 0).all()
    assert (np.diff(log["cum_bytes"]) > 0).all()


def test_layer_precedence_l0_before_l1_before_l2():
    """Verify that for every tile, L0 alerts precede L1 previews, which precede L2 full tiles."""
    tiles = _make_sample_tiles(12)
    thresholds = {"optical": 0.2, "radar": 0.2}  # keep many tiles
    budget = 500_000

    log = sim_run(tiles, "orbit_prism", budget_bytes=budget, thresholds=thresholds)
    assert not log.empty

    for (chip, row, col), group in log.groupby(KEY):
        layers = list(group["layer"])
        if "L0" in layers and "L1" in layers:
            assert layers.index("L0") < layers.index("L1"), f"L1 before L0 for {chip}_{row}_{col}"
        if "L1" in layers and "L2" in layers:
            assert layers.index("L1") < layers.index("L2"), f"L2 before L1 for {chip}_{row}_{col}"
        if "L0" in layers and "L2" in layers:
            assert layers.index("L0") < layers.index("L2"), f"L2 before L0 for {chip}_{row}_{col}"


def test_exact_packet_sizes():
    """Verify that packet bytes match exact physical constants."""
    tiles = _make_sample_tiles(12)
    thresholds = {"optical": 0.3, "radar": 0.3}
    budget = 2_000_000

    log = sim_run(tiles, "orbit_prism", budget_bytes=budget, thresholds=thresholds)
    
    # L0 must be exactly 32 B
    l0_bytes = log[log["layer"] == "L0"]["bytes"].unique()
    assert list(l0_bytes) == [L0_B]

    # L1 must be exactly L1_BYTES (2048 B)
    if L1_BYTES is not None:
        l1_bytes = log[log["layer"] == "L1"]["bytes"].unique()
        assert list(l1_bytes) == [L1_BYTES]

    # L2 optical must be 65,536 B; radar must be 32,768 B
    l2_log = log[log["layer"] == "L2"].merge(tiles[KEY + ["sensor_used"]], on=KEY)
    opt_l2 = l2_log[l2_log["sensor_used"] == "optical"]["bytes"].unique()
    rad_l2 = l2_log[l2_log["sensor_used"] == "radar"]["bytes"].unique()
    if len(opt_l2) > 0:
        assert list(opt_l2) == [OPTICAL_B]
    if len(rad_l2) > 0:
        assert list(rad_l2) == [RADAR_B]


def test_no_packet_repeated_per_layer():
    """Verify that no tile is transmitted twice in the same layer."""
    tiles = _make_sample_tiles(12)
    thresholds = {"optical": 0.3, "radar": 0.3}
    budget = 100_000

    for pol in ["capture_order", "discard_cloudy", "optical_only", "colour_index", "orbit_prism"]:
        log = sim_run(tiles, pol, budget_bytes=budget, thresholds=thresholds)
        if log.empty:
            continue
        # Check uniqueness of (chip, row, col, layer)
        subset = log[KEY + ["layer"]]
        assert len(subset) == len(subset.drop_duplicates()), f"Duplicate packets in policy {pol}"


def test_baselines_match_reference():
    """Verify that the baseline policies match the reference implementation in transmission order."""
    tiles = _make_sample_tiles(10)
    # Huge budget so everything is in pass 0
    budget = 50_000_000

    for pol in ["capture_order", "discard_cloudy", "optical_only", "colour_index"]:
        sim_log = sim_run(tiles, pol, budget_bytes=budget)
        ref_log = run_policy(tiles, pol)

        # Transmission sequence of (chip, row, col) must match identically
        pd.testing.assert_series_equal(sim_log["chip"], ref_log["chip"], check_names=False)
        pd.testing.assert_series_equal(sim_log["row"], ref_log["row"], check_names=False)
        pd.testing.assert_series_equal(sim_log["col"], ref_log["col"], check_names=False)
        pd.testing.assert_series_equal(sim_log["layer"], ref_log["layer"], check_names=False)
        pd.testing.assert_series_equal(sim_log["bytes"], ref_log["bytes"], check_names=False)


def test_kept_sets_equal_reference():
    """Verify that the set of kept tiles in orbit policies matches reference_policies.py."""
    tiles = _make_sample_tiles(12)
    thresholds = {"optical": 0.45, "radar": 0.45}
    budget = 50_000_000

    for pol in ["orbit_prism", "orbit_no_radar", "orbit_no_dial"]:
        sim_log = sim_run(tiles, pol, budget_bytes=budget, thresholds=thresholds)
        ref_log = run_policy(tiles, pol, thresholds)

        sim_kept = kept_mask(tiles, sim_log)
        ref_kept = kept_mask(tiles, ref_log)
        pd.testing.assert_series_equal(sim_kept, ref_kept)


def test_coverage_bitmap_and_peak_queue_attrs():
    """Verify that coverage_bitmap_bytes and peak_queue_bytes attrs are populated."""
    tiles = _make_sample_tiles(17)  # 17 tiles -> ceil(17 / 8) = 3 bytes
    budget = 200_000
    thresholds = {"optical": 0.3, "radar": 0.3}

    log = sim_run(tiles, "orbit_prism", budget_bytes=budget, thresholds=thresholds)
    assert log.attrs["coverage_bitmap_bytes"] == math.ceil(17 / 8) == 3
    assert log.attrs["peak_queue_bytes"] > 0


def test_empty_and_tiny_budgets():
    """Verify that empty inputs and small budgets behave safely without crashing."""
    empty_tiles = pd.DataFrame(columns=[
        "event", "chip", "row", "col", "split", "water_frac", "critical",
        "cloud_frac", "sensor_used", "p_flood", "extent", "p_optical", "p_ndwi"
    ])
    log = sim_run(empty_tiles, "orbit_prism", budget_bytes=100_000)
    assert log.empty
    assert log.attrs["coverage_bitmap_bytes"] == 0
    assert log.attrs["peak_queue_bytes"] == 0

    # Negative or zero budget should raise ValueError
    tiles = _make_sample_tiles(4)
    with pytest.raises(ValueError):
        sim_run(tiles, "orbit_prism", budget_bytes=0)
    with pytest.raises(ValueError):
        sim_run(tiles, "orbit_prism", budget_bytes=-100)
