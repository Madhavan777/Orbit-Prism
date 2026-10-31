"""
ORBIT-PRISM Downlink Scheduler & Link Simulator
Module: src/scheduler/sim.py
Author: Akhilesh S (akhilesh/scheduler)

Simulates constrained satellite-to-ground downlink passes:
- Link parameters: 256 kbps x 600 s = 19,200,000 B per pass (PASS_BYTES)
- Packet layers:
    L0: Emergency Alert (32 B exact, packed with struct)
    L1: Spatial / Confidence Preview (2,048 B, L1_BYTES in src/constants.py)
    L2: Full Tile Payload (Optical: 65,536 B, Radar: 32,768 B)
- Strict discrete packet packing: No packet splitting.
- Multi-pass carry-over: Unsent packets resume in pass_index + 1 without layer repetition.
- Policies:
    capture_order, discard_cloudy, optical_only, colour_index,
    orbit_prism, orbit_no_radar, orbit_no_dial

Note: This is a discrete numerical simulation of satellite link, queue,
and memory constraints, evaluated on held-out events. It does not run on flight hardware.
"""

from __future__ import annotations

import binascii
import math
import struct
from typing import Dict, List, Optional, Tuple, NamedTuple
import numpy as np
import pandas as pd

from src.constants import (
    CLOUD_SWITCH,
    EVENT_INDEX,
    L0_B,
    L1_BYTES,
    OPTICAL_B,
    PASS_BYTES,
    QUEUE_ENTRY_B,
    RADAR_B,
)
from src.evaluation.reference_policies import NotAvailable

# Supported policy set
POLICIES = [
    "capture_order",
    "discard_cloudy",
    "optical_only",
    "colour_index",
    "orbit_prism",
    "orbit_no_radar",
    "orbit_no_dial",
]

KEY = ["chip", "row", "col"]
FIXED_THRESHOLD = 0.5


ALERT_FORMAT = ">BBIBBBBHHHHIIIH"   # big-endian, exactly 32 bytes (checked below and in tests)
ALERT_VERSION = 1
assert struct.calcsize(ALERT_FORMAT) == L0_B == 32


def _chip_fields(chip: str) -> Tuple[int, int]:
    """(event index, chip number) for a chip id '<Event>_<number>', for example 'Sri-Lanka_1049830'.

    Lossless for every Sen1Floods11 chip. The earlier 16-byte text field truncated the two 17-character
    Sri-Lanka ids. An unknown name gives (255, 0)."""
    event, _, number = chip.rpartition("_")
    if event in EVENT_INDEX and number.isdigit() and int(number) < 2 ** 32:
        return EVENT_INDEX.index(event), int(number)
    return 255, 0


def build_l0_alert_packet(
    chip: str,
    row: int,
    col: int,
    sensor: str,
    p_flood: float,
    extent: float,
    pass_number: int = 0,
    seq: int = 0,
    hazard_type: int = 0,
) -> bytes:
    """
    Constructs the exact 32-byte binary L0 alert (big-endian, struct format ALERT_FORMAT).

      offset  size  field
         0     B    version (1)
         1     B    event index (EVENT_INDEX; 255 = unknown)
         2     I    chip number (the digits of '<Event>_<number>')
         6     B    tile row
         7     B    tile column
         8     B    sensor (0 optical, 1 radar)
         9     B    hazard (0 flood, 1 fire)
        10     H    flood probability, x 65535
        12     H    extent, x 65535
        14     H    priority (probability x extent), x 65535
        16     H    pass number
        18     I    sequence number
        22     I    size of the full tile (L2) in bytes
        26     I    reserved (0)
        30     H    CRC-16/XMODEM (binascii.crc_hqx) over the first 30 bytes
    Decode with decode_l0_alert_packet.
    """
    event, number = _chip_fields(chip)
    q = lambda v: int(round(float(np.clip(v, 0.0, 1.0)) * 65535))  # noqa: E731
    radar = sensor == "radar"
    body = struct.pack(
        ALERT_FORMAT[:-1], ALERT_VERSION, event, number, int(row) & 0xFF, int(col) & 0xFF,
        1 if radar else 0, int(hazard_type) & 0xFF, q(p_flood), q(extent), q(float(p_flood) * float(extent)),
        int(pass_number) & 0xFFFF, int(seq) & 0xFFFFFFFF, RADAR_B if radar else OPTICAL_B, 0,
    )
    packet = body + struct.pack(">H", binascii.crc_hqx(body, 0))
    assert len(packet) == L0_B == 32, f"L0 packet size mismatch: {len(packet)} != 32"
    return packet


def decode_l0_alert_packet(packet: bytes) -> Dict[str, object]:
    """Inverse of build_l0_alert_packet (ground side). Raises ValueError on a wrong size or checksum."""
    if len(packet) != L0_B:
        raise ValueError(f"an alert is {L0_B} bytes, got {len(packet)}")
    ver, ev, num, row, col, sensor, hazard, p, ext, prio, pas, seq, l2, _, crc = struct.unpack(ALERT_FORMAT, packet)
    if crc != binascii.crc_hqx(packet[:30], 0):
        raise ValueError("alert checksum mismatch")
    event = EVENT_INDEX[ev] if ev < len(EVENT_INDEX) else None
    return {
        "version": ver, "event": event, "chip_number": num, "chip": f"{event}_{num}" if event else None,
        "row": row, "col": col, "sensor": "radar" if sensor else "optical", "hazard": "fire" if hazard else "flood",
        "p_flood": p / 65535, "extent": ext / 65535, "priority": prio / 65535,
        "pass_number": pas, "seq": seq, "l2_bytes": l2,
    }

class PacketItem(NamedTuple):
    idx: int
    chip: str
    row: int
    col: int
    layer: str
    nbytes: int
    priority: float
    ppb: float


def _tile_bytes(sensor: pd.Series) -> np.ndarray:
    return np.where(sensor == "radar", RADAR_B, OPTICAL_B).astype(float)


def _pack_queue_into_passes(
    packets: List[PacketItem],
    budget_bytes: float,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Packs a discrete queue of packets into communication passes of size budget_bytes.
    
    Properties:
    - No packet is split across pass boundaries.
    - An unsent packet carries over to pass_index + 1.
    - cum_bytes tracks cumulative transmitted data monotonically.
    - Peak queue bytes is tracked across queue lifecycle.
    """
    if not packets:
        empty_df = pd.DataFrame(
            columns=["order", "chip", "row", "col", "layer", "bytes", "cum_bytes", "pass_index"]
        )
        return empty_df, 0, 0

    if budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")

    records = []
    pass_index = 0
    cum_bytes = 0.0
    order = 0
    current_pass_bytes = 0.0

    # Total queue size in bytes
    queue_bytes = sum(p.nbytes for p in packets)
    peak_queue_payload_bytes = queue_bytes   # payload waiting in FLASH, not SRAM

    i = 0
    n = len(packets)
    while i < n:
        pkt = packets[i]
        b = float(pkt.nbytes)

        # Check if packet fits in the current pass budget
        if current_pass_bytes + b <= budget_bytes:
            # Transmit packet in current pass
            current_pass_bytes += b
            cum_bytes += b
            records.append({
                "order": order,
                "chip": pkt.chip,
                "row": pkt.row,
                "col": pkt.col,
                "layer": pkt.layer,
                "bytes": b,
                "cum_bytes": cum_bytes,
                "pass_index": pass_index,
            })
            order += 1
            queue_bytes -= b
            i += 1
        else:
            # Packet does not fit in remaining pass budget:
            # If the pass has already sent packets, close this pass and carry over to pass_index + 1
            if current_pass_bytes > 0:
                pass_index += 1
                current_pass_bytes = 0.0
            else:
                # Edge case: The packet itself is larger than the entire pass budget!
                # We transmit it in its own dedicated pass so the system doesn't stall indefinitely.
                current_pass_bytes = b
                cum_bytes += b
                records.append({
                    "order": order,
                    "chip": pkt.chip,
                    "row": pkt.row,
                    "col": pkt.col,
                    "layer": pkt.layer,
                    "bytes": b,
                    "cum_bytes": cum_bytes,
                    "pass_index": pass_index,
                })
                order += 1
                queue_bytes -= b
                i += 1
                pass_index += 1
                current_pass_bytes = 0.0

    df_out = pd.DataFrame(records)
    return df_out, len(packets), peak_queue_payload_bytes


def run(
    tiles: pd.DataFrame,
    policy: str,
    budget_bytes: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Executes satellite downlink scheduling for `tiles` under `policy` and `budget_bytes`.

    Parameters:
    - tiles: DataFrame with the 11-column contract (chip, row, col, sensor_used, cloud_frac, p_flood, extent, ...)
             plus p_optical / p_ndwi if required by the policy.
    - policy: capture_order | discard_cloudy | optical_only | colour_index | orbit_prism | orbit_no_radar | orbit_no_dial
    - budget_bytes: Available bytes per ground station pass.
    - thresholds: {"optical": t, "radar": t} raw p_flood calibration thresholds (for orbit policies).

    Returns:
    - DataFrame with exact columns:
        order, chip, row, col, layer, bytes, cum_bytes, pass_index
    - attrs:
        log.attrs["coverage_bitmap_bytes"] = ceil(n_tiles / 8)
        log.attrs["peak_queue_bytes"]         = SRAM used by the pending queue: peak entries x QUEUE_ENTRY_B
                                                (the queue holds references; pixels stay in flash)
        log.attrs["peak_queue_payload_bytes"] = payload bytes waiting in flash (NOT SRAM)
        log.attrs["onboard_scheduler_bytes"]  = peak_queue_bytes + coverage_bitmap_bytes
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")

    if budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")

    tiles = tiles.reset_index(drop=True)
    n = len(tiles)
    coverage_bitmap_bytes = int(math.ceil(n / 8)) if n > 0 else 0

    if n == 0:
        empty = pd.DataFrame(
            columns=["order", "chip", "row", "col", "layer", "bytes", "cum_bytes", "pass_index"]
        )
        empty.attrs["coverage_bitmap_bytes"] = 0
        empty.attrs["peak_queue_bytes"] = 0
        empty.attrs["peak_queue_payload_bytes"] = 0
        empty.attrs["onboard_scheduler_bytes"] = 0
        return empty

    sensor = tiles["sensor_used"]
    cloudy = (tiles["cloud_frac"] >= CLOUD_SWITCH).to_numpy()
    opt_bytes = np.full(n, float(OPTICAL_B))
    mix_bytes = _tile_bytes(sensor)

    chips = tiles["chip"].to_numpy()
    rows = tiles["row"].to_numpy()
    cols = tiles["col"].to_numpy()

    # -------------------------------------------------------------
    # 1. BASELINE POLICIES (L2 full tiles only, optical bytes)
    # -------------------------------------------------------------
    if policy == "capture_order":
        order_indices = np.arange(n)
        packets = [
            PacketItem(idx=i, chip=chips[i], row=rows[i], col=cols[i], layer="L2",
                       nbytes=int(opt_bytes[i]), priority=0.0, ppb=0.0)
            for i in order_indices
        ]

    elif policy == "discard_cloudy":
        order_indices = np.flatnonzero(~cloudy)
        packets = [
            PacketItem(idx=i, chip=chips[i], row=rows[i], col=cols[i], layer="L2",
                       nbytes=int(opt_bytes[i]), priority=0.0, ppb=0.0)
            for i in order_indices
        ]

    elif policy == "optical_only":
        if "p_optical" not in tiles.columns:
            raise NotAvailable("needs p_optical (optical model score on every tile)")
        score = tiles["p_optical"].to_numpy(dtype=float)
        # Stable descending sort on optical score
        order_indices = np.argsort(-score, kind="stable")
        packets = [
            PacketItem(idx=i, chip=chips[i], row=rows[i], col=cols[i], layer="L2",
                       nbytes=int(opt_bytes[i]), priority=float(score[i]), ppb=0.0)
            for i in order_indices
        ]

    elif policy == "colour_index":
        if "p_ndwi" not in tiles.columns:
            raise NotAvailable("needs p_ndwi (results/scores_ndwi.csv, from real band imagery)")
        score = tiles["p_ndwi"].to_numpy(dtype=float)
        # Stable descending sort on NDWI score
        order_indices = np.argsort(-score, kind="stable")
        packets = [
            PacketItem(idx=i, chip=chips[i], row=rows[i], col=cols[i], layer="L2",
                       nbytes=int(opt_bytes[i]), priority=float(score[i]), ppb=0.0)
            for i in order_indices
        ]

    # -------------------------------------------------------------
    # 2. ORBIT POLICIES (L0 alerts -> L1 previews -> L2 full tiles)
    # -------------------------------------------------------------
    elif policy in ("orbit_prism", "orbit_no_radar", "orbit_no_dial"):
        if policy == "orbit_no_radar":
            if "p_optical" not in tiles.columns:
                raise NotAvailable("needs p_optical on cloudy tiles")
            score = tiles["p_optical"].to_numpy(dtype=float)
            per_tile_bytes = opt_bytes
            path = np.full(n, "optical")
        else:
            score = tiles["p_flood"].to_numpy(dtype=float)
            per_tile_bytes = mix_bytes
            path = sensor.to_numpy()

        # Calibration Thresholds
        if policy == "orbit_no_dial":
            thr = np.full(n, FIXED_THRESHOLD)
        else:
            if not thresholds:
                raise NotAvailable("needs calibrated thresholds for the miss-rate dial")
            thr = np.where(path == "radar", thresholds["radar"], thresholds["optical"]).astype(float)

        extent = tiles["extent"].to_numpy(dtype=float)
        priority = score * extent
        keep_mask = score >= thr
        kept_indices = np.flatnonzero(keep_mask)

        # If no tiles are kept, only the bitmap was examined; send log is empty
        if len(kept_indices) == 0:
            empty = pd.DataFrame(
                columns=["order", "chip", "row", "col", "layer", "bytes", "cum_bytes", "pass_index"]
            )
            empty.attrs["coverage_bitmap_bytes"] = coverage_bitmap_bytes
            empty.attrs["peak_queue_bytes"] = 0
            empty.attrs["peak_queue_payload_bytes"] = 0
            empty.attrs["onboard_scheduler_bytes"] = coverage_bitmap_bytes
            return empty

        # Stage 1: L0 Alerts ranked by priority descending (stable tie-break)
        alert_order = kept_indices[np.argsort(-priority[kept_indices], kind="stable")]
        l0_packets = [
            PacketItem(idx=i, chip=chips[i], row=rows[i], col=cols[i], layer="L0",
                       nbytes=L0_B, priority=float(priority[i]), ppb=0.0)
            for i in alert_order
        ]

        # Stage 2: L1 Previews ranked by priority descending (if L1_BYTES is defined)
        l1_packets = []
        if L1_BYTES is not None and L1_BYTES > 0:
            preview_order = kept_indices[np.argsort(-priority[kept_indices], kind="stable")]
            l1_packets = [
                PacketItem(idx=i, chip=chips[i], row=rows[i], col=cols[i], layer="L1",
                           nbytes=int(L1_BYTES), priority=float(priority[i]), ppb=0.0)
                for i in preview_order
            ]

        # Stage 3: L2 Full Tiles ranked by priority per byte descending (stable tie-break)
        ppb = priority[kept_indices] / per_tile_bytes[kept_indices]
        full_order = kept_indices[np.argsort(-ppb, kind="stable")]
        l2_packets = [
            PacketItem(idx=i, chip=chips[i], row=rows[i], col=cols[i], layer="L2",
                       nbytes=int(per_tile_bytes[i]), priority=float(priority[i]), ppb=float(ppb[k]))
            for k, i in enumerate(full_order)
        ]

        packets = l0_packets + l1_packets + l2_packets

    # Pack into discrete communication passes
    log_df, peak_entries, payload_bytes = _pack_queue_into_passes(packets, budget_bytes)
    log_df.attrs["coverage_bitmap_bytes"] = coverage_bitmap_bytes
    # Conservative: every packet is enqueued before the first pass. The queue holds references, so its SRAM cost
    # is entries x QUEUE_ENTRY_B; the payload (pixels) waits in flash.
    log_df.attrs["peak_queue_bytes"] = peak_entries * QUEUE_ENTRY_B
    log_df.attrs["peak_queue_payload_bytes"] = payload_bytes
    log_df.attrs["onboard_scheduler_bytes"] = peak_entries * QUEUE_ENTRY_B + coverage_bitmap_bytes

    return log_df
