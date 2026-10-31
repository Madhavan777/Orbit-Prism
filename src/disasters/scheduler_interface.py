"""EXPERIMENTAL priority downlink scheduler sketch for ORBIT-PRISM.

Not used by the evaluation: the reference scheduler is src/scheduler (Akhilesh), with the
L0 / L1 / L2 packet design. Packet sizes here come from src/constants.py.

Handles bandwidth-constrained LEO satellite communication passes:
  - Two-tier transmission protocol: Tier 1 (alert, L0 size) before Tier 2 (full tile)
  - Multi-event priority ordering
  - Inter-pass carry-over buffer
  - De-duplication
  - Baseline comparison (Priority-Driven vs FIFO)
  - Shared CSV format export
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Any
import pandas as pd

from src.constants import L0_B, OPTICAL_B
from src.disasters.risk_engine import RiskAssessment
from src.disasters.registry import DisasterCategory


class PacketTier(int, Enum):
    TIER1_ALERT = 1      # Compact alert metadata (< 1 KB)
    TIER2_IMAGERY = 2    # Full tile imagery (65,536 B optical, 32,768 B radar)


class TransmissionStatus(str, Enum):
    PENDING = "pending"
    TRANSMITTED_ALERT = "alert_downlinked"
    TRANSMITTED_FULL = "fully_downlinked"
    CARRIED_OVER = "carried_over"
    DROPPED = "dropped"


@dataclass
class DownlinkPacket:
    """Scheduled packet payload for satellite downlink."""
    assessment: RiskAssessment
    tier: PacketTier
    size_bytes: int
    status: TransmissionStatus = TransmissionStatus.PENDING
    attempts: int = 0


@dataclass
class PassSummary:
    """Summary metrics of a ground station downlink pass."""
    pass_id: int
    total_budget_bytes: int
    used_bytes: int
    alerts_sent: int
    images_sent: int
    remaining_in_queue: int
    critical_events_served: int


class PriorityScheduler:
    """Multi-disaster intelligent queue and downlink manager."""

    # Packet sizes come from the shared constants (CONTEXT.md): L0 alert and an optical tile.
    TIER1_ALERT_SIZE_BYTES = L0_B           # 32 B alert packet
    TIER2_INT8_IMAGE_BYTES = OPTICAL_B      # 65,536 B full optical tile (a radar tile is 32,768 B)

    def __init__(self, critical_threshold: float = 60.0):
        """Initialize PriorityScheduler.

        Args:
            critical_threshold: Priority score above which an event is treated as critical.
        """
        self.critical_threshold = critical_threshold
        self.queue: List[DownlinkPacket] = []
        self.transmitted_tile_ids: Set[str] = set()
        self.pass_history: List[PassSummary] = []
        self.pass_counter = 0

    def enqueue_assessments(self, assessments: List[RiskAssessment], allow_duplicates: bool = False) -> int:
        """Enqueue new risk assessments for satellite transmission.

        Generates both Tier 1 (Alert) and Tier 2 (Imagery) packets for each tile.
        Filters duplicates if allow_duplicates is False.
        """
        added = 0
        for a in assessments:
            if not allow_duplicates and a.tile_id in self.transmitted_tile_ids:
                continue

            # Check if tile already queued
            if any(p.assessment.tile_id == a.tile_id for p in self.queue):
                continue

            # Tier 1 packet: Urgent alert
            p_alert = DownlinkPacket(
                assessment=a,
                tier=PacketTier.TIER1_ALERT,
                size_bytes=self.TIER1_ALERT_SIZE_BYTES
            )
            # Tier 2 packet: Imagery
            p_img = DownlinkPacket(
                assessment=a,
                tier=PacketTier.TIER2_IMAGERY,
                size_bytes=self.TIER2_INT8_IMAGE_BYTES
            )

            self.queue.append(p_alert)
            self.queue.append(p_img)
            added += 1

        return added

    def sort_queue(self) -> None:
        """Sort transmission queue.

        Rules:
          1. Tier 1 (Alerts) ALWAYS precede Tier 2 (Imagery).
          2. Within the same tier, sort descending by priority_score.
          3. Ties broken by confidence score.
        """
        self.queue.sort(
            key=lambda p: (
                p.tier.value,                       # Tier 1 (1) before Tier 2 (2)
                -p.assessment.priority_score,       # High priority first
                -p.assessment.confidence            # High confidence first
            )
        )

    def simulate_pass(self, bandwidth_budget_bytes: int) -> Tuple[List[DownlinkPacket], PassSummary]:
        """Simulate a single satellite ground station pass with a fixed bandwidth quota.

        Preserves untransmitted packets (carry-over) for subsequent passes.
        """
        self.pass_counter += 1
        self.sort_queue()

        remaining_budget = bandwidth_budget_bytes
        downlinked_packets: List[DownlinkPacket] = []
        remaining_queue: List[DownlinkPacket] = []

        alerts_sent = 0
        images_sent = 0
        critical_served = 0

        for packet in self.queue:
            if packet.size_bytes <= remaining_budget:
                remaining_budget -= packet.size_bytes
                if packet.tier == PacketTier.TIER1_ALERT:
                    packet.status = TransmissionStatus.TRANSMITTED_ALERT
                    alerts_sent += 1
                else:
                    packet.status = TransmissionStatus.TRANSMITTED_FULL
                    images_sent += 1
                    self.transmitted_tile_ids.add(packet.assessment.tile_id)

                if packet.assessment.priority_score >= self.critical_threshold:
                    critical_served += 1

                downlinked_packets.append(packet)
            else:
                packet.status = TransmissionStatus.CARRIED_OVER
                packet.attempts += 1
                remaining_queue.append(packet)

        # Update queue with carryover
        self.queue = remaining_queue

        summary = PassSummary(
            pass_id=self.pass_counter,
            total_budget_bytes=bandwidth_budget_bytes,
            used_bytes=bandwidth_budget_bytes - remaining_budget,
            alerts_sent=alerts_sent,
            images_sent=images_sent,
            remaining_in_queue=len(self.queue),
            critical_events_served=critical_served
        )
        self.pass_history.append(summary)
        return downlinked_packets, summary

    def export_csv(self, assessments: List[RiskAssessment], filepath: str, legacy_flood_format: bool = False) -> pd.DataFrame:
        """Export scores to CSV adhering to team contract.

        Args:
            assessments: List of RiskAssessments to export.
            filepath: Destination CSV path.
            legacy_flood_format: If True, writes Madhavan's legacy format `tile_id,p_flood,extent`.
                                 If False, writes extended multi-disaster schema.
        """
        records = []
        for a in assessments:
            if legacy_flood_format:
                records.append({
                    "tile_id": a.tile_id,
                    "p_flood": round(a.raw_event.p_event, 4),
                    "extent": round(a.raw_event.extent, 4)
                })
            else:
                records.append({
                    "tile_id": a.tile_id,
                    "disaster_type": a.category.value,
                    "p_event": round(a.raw_event.p_event, 4),
                    "extent": round(a.raw_event.extent, 4),
                    "severity": round(a.raw_event.severity, 4),
                    "exposure_score": "UNKNOWN" if a.is_exposure_unknown else round(a.vulnerability_score, 4),
                    "confidence": round(a.confidence, 4),
                    "priority_score": round(a.priority_score, 2),
                    "status": "active"
                })

        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        return df
