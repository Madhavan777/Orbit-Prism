"""Hazard registry for ORBIT-PRISM: the two in-scope hazards, their data and honest status.

Scope (CONTEXT.md): floods (primary, measured) and forest fires (second hazard). Nothing else.
Statuses describe what exists in the repository today, not what is planned:

  SIMULATED   model architecture and tests exist; not yet trained or evaluated on real data
  PLANNED     chosen or designed, but no data loaded and nothing trained
  UNVERIFIED  claimed elsewhere but not backed by anything in the repository
  IMPLEMENTED reserved for a hazard with a model trained on real data AND evaluated on the
              held-out test event. No hazard qualifies yet.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class DisasterCategory(str, Enum):
    FLOOD = "flood"
    WILDFIRE = "wildfire"


class ImplementationStatus(str, Enum):
    IMPLEMENTED = "implemented"
    SIMULATED = "simulated"
    PLANNED = "planned"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    source: str
    modality: str
    samples_approx: str
    labels: str
    geographic_coverage: str
    split_strategy: str
    license: str


@dataclass(frozen=True)
class DisasterProfile:
    category: DisasterCategory
    primary_modalities: List[str]
    target_weight_kb: float
    urgency_multiplier: float  # unvalidated assumption used only by the experimental risk engine
    status: ImplementationStatus
    dataset: DatasetMetadata
    description: str


class DisasterRegistry:
    """Registry of the in-scope hazards and their dataset provenance."""

    _PROFILES: Dict[DisasterCategory, DisasterProfile] = {
        DisasterCategory.FLOOD: DisasterProfile(
            category=DisasterCategory.FLOOD,
            primary_modalities=["Optical (B2, B3, B4, B8)", "SAR (Sentinel-1 VV/VH)"],
            target_weight_kb=100.0,
            urgency_multiplier=1.8,
            status=ImplementationStatus.SIMULATED,
            dataset=DatasetMetadata(
                name="Sen1Floods11 (v1.1 hand-labelled subset)",
                source="Cloud to Street (Bonafilia et al., CVPR Workshops 2020)",
                modality="Sentinel-1 SAR (VV/VH) and Sentinel-2 optical (13 bands)",
                samples_approx="446 hand-labelled 512x512 chips, cut into 6,825 non-empty 128x128 tiles",
                labels="Hand-labelled water / not water / no data, minus the JRC permanent-water layer",
                geographic_coverage="11 flood events",
                split_strategy="Official chip-level train/valid/test plus the Bolivia event held out",
                license="Not stated on the GitHub repository (unconfirmed)",
            ),
            description="Flood tile triage: critical when the non-permanent water fraction is >= 0.30.",
        ),
        DisasterCategory.WILDFIRE: DisasterProfile(
            category=DisasterCategory.WILDFIRE,
            primary_modalities=["Optical incl. SWIR (B11, B12)", "SAR (Sentinel-1) - hypothesis, not tested"],
            target_weight_kb=60.0,
            urgency_multiplier=2.0,
            status=ImplementationStatus.PLANNED,
            dataset=DatasetMetadata(
                name="Sentinel imagery with FireCCI51 burned area / FIRMS VIIRS (plan); Torino SBAD (fallback)",
                source="Copernicus Data Space, ESA Fire CCI, NASA FIRMS",
                modality="Sentinel-1/2; labels from a 250 m burned-area product or fire points",
                samples_approx="NOT LOADED: no fire data has been downloaded or processed",
                labels="Not yet defined at tile level",
                geographic_coverage="Not yet chosen",
                split_strategy="Whole fires or folds held out, never tiles",
                license="Per source terms; not yet confirmed",
            ),
            description="Burned-area tile triage. Only an untrained architecture and an NBR formula exist.",
        ),
    }

    @classmethod
    def get_profile(cls, category: DisasterCategory) -> DisasterProfile:
        return cls._PROFILES[category]

    @classmethod
    def list_categories(cls) -> List[DisasterCategory]:
        return list(cls._PROFILES.keys())

    @classmethod
    def all_profiles(cls) -> Dict[DisasterCategory, DisasterProfile]:
        return dict(cls._PROFILES)
