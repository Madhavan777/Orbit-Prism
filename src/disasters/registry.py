"""Disaster Registry and Metadata Registry for ORBIT-PRISM.

Defines the 6 disaster categories, their physical modalities, public datasets,
and implementation status.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class DisasterCategory(str, Enum):
    FLOOD = "flood"
    WILDFIRE = "wildfire"
    CYCLONE = "cyclone"
    LANDSLIDE = "landslide"
    EARTHQUAKE = "earthquake"
    VOLCANO = "volcano"


class ImplementationStatus(str, Enum):
    IMPLEMENTED = "implemented"  # Full TinyML code + verified tests
    SIMULATED = "simulated"      # Architectural prototype + synthetic validation
    PLANNED = "planned"          # Designed in roadmap, awaiting dataset integration
    UNVERIFIED = "unverified"    # Theoretical model, not yet benchmarked on hardware


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
    urgency_multiplier: float
    status: ImplementationStatus
    dataset: DatasetMetadata
    description: str


class DisasterRegistry:
    """Central registry of supported disaster profiles and dataset provenance."""

    _PROFILES: Dict[DisasterCategory, DisasterProfile] = {
        DisasterCategory.FLOOD: DisasterProfile(
            category=DisasterCategory.FLOOD,
            primary_modalities=["Optical (RGB + NIR)", "SAR (Sentinel-1 VV/VH)"],
            target_weight_kb=100.0,
            urgency_multiplier=1.8,
            status=ImplementationStatus.IMPLEMENTED,
            dataset=DatasetMetadata(
                name="Sen1Floods11",
                source="NASA / Cloud to Street (Bonafilia et al., CVPRW 2020)",
                modality="Sentinel-1 SAR (VV/VH) & Sentinel-2 Optical (13 bands)",
                samples_approx="4,831 tiles (512x512 downsampled to 128x128)",
                labels="Permanent water, flood water, non-water pixel masks",
                geographic_coverage="11 global flood events across 6 continents",
                split_strategy="Event-based split (TRAIN, CAL, TEST isolated)",
                license="CC BY-4.0"
            ),
            description="Rapid flood extent mapping and critical inundation classification (water_frac >= 0.30)."
        ),
        DisasterCategory.WILDFIRE: DisasterProfile(
            category=DisasterCategory.WILDFIRE,
            primary_modalities=["Optical (Red/NIR)", "SWIR (2.19 um)", "TIR Thermal Hotspots"],
            target_weight_kb=60.0,
            urgency_multiplier=2.0,
            status=ImplementationStatus.IMPLEMENTED,
            dataset=DatasetMetadata(
                name="NASA FIRMS / Sentinel-2 Wildfire",
                source="NASA Land, Atmosphere Near real-time Capability for EOS (LANCE)",
                modality="Sentinel-2 MSI (B4, B8A, B11, B12) & MODIS/VIIRS Active Fire",
                samples_approx="5,000+ annotated wildfire patches",
                labels="Active fire pixel, burn scar extent, thermal radiative power",
                geographic_coverage="North America, Australia, Mediterranean basin",
                split_strategy="Temporal & regional event-based splits",
                license="Open Public Domain (NASA Open Data Policy)"
            ),
            description="Active fire hotspot identification and post-fire burn scar perimeter regression."
        ),
        DisasterCategory.CYCLONE: DisasterProfile(
            category=DisasterCategory.CYCLONE,
            primary_modalities=["Thermal IR (10.8 um)", "Wide-Swath VIS", "Storm Track Vectors"],
            target_weight_kb=120.0,
            urgency_multiplier=2.2,
            status=ImplementationStatus.IMPLEMENTED,
            dataset=DatasetMetadata(
                name="TC-PRIMED / NASA IMPACT Tropical Cyclone",
                source="NASA IMPACT / NOAA NHC & JTWC Archives",
                modality="Passive Microwave, GOES/Himawari/INSAT Infrared Imagery",
                samples_approx="100,000+ storm-centered multi-channel observations",
                labels="Maximum sustained wind (knots), minimum central pressure (hPa)",
                geographic_coverage="Global tropical cyclone basins (Atlantic, Pacific, Indian Ocean)",
                split_strategy="Per-storm holdout (events never shared across splits)",
                license="Creative Commons Attribution 4.0"
            ),
            description="Cyclone cloud-top intensity estimation and storm structure analysis."
        ),
        DisasterCategory.LANDSLIDE: DisasterProfile(
            category=DisasterCategory.LANDSLIDE,
            primary_modalities=["Optical (RGB + NIR)", "Copernicus/SRTM DEM (Slope, Elevation)"],
            target_weight_kb=95.0,
            urgency_multiplier=1.6,
            status=ImplementationStatus.SIMULATED,
            dataset=DatasetMetadata(
                name="Landslide4Sense",
                source="IEEE GRSS Data Fusion Contest (Ghorbanzadeh et al., 2022)",
                modality="Sentinel-2 Multi-Spectral + ALOS PALSAR DEM (Slope, Aspect)",
                samples_approx="3,799 image patches (128x128)",
                labels="Binary landslide pixel masks",
                geographic_coverage="Global mountainous regions (Himalayas, Alps, Andes)",
                split_strategy="Geographic tile holdout",
                license="Open for academic research"
            ),
            description="Rainfall- and earthquake-triggered landslide scarp and deposit detection."
        ),
        DisasterCategory.EARTHQUAKE: DisasterProfile(
            category=DisasterCategory.EARTHQUAKE,
            primary_modalities=["Bi-temporal Optical (Pre/Post)", "InSAR Coherence Loss", "Seismic Metadata"],
            target_weight_kb=150.0,
            urgency_multiplier=2.4,
            status=ImplementationStatus.SIMULATED,
            dataset=DatasetMetadata(
                name="xBD (xView2 Challenge)",
                source="Defense Innovation Unit (Gupta et al., CVPRW 2019)",
                modality="Pre- and post-disaster high-resolution optical satellite imagery",
                samples_approx="850,000+ building damage polygons",
                labels="Building damage level: 0 (no damage) to 3 (destroyed)",
                geographic_coverage="19 global disaster events across 15 countries",
                split_strategy="Disaster event holdout",
                license="CC BY-NC-SA 4.0"
            ),
            description="Structural collapse and building damage assessment following seismic events."
        ),
        DisasterCategory.VOLCANO: DisasterProfile(
            category=DisasterCategory.VOLCANO,
            primary_modalities=["Thermal IR Radiance", "UV/VIS SO2 Ash Plume Index"],
            target_weight_kb=50.0,
            urgency_multiplier=2.1,
            status=ImplementationStatus.SIMULATED,
            dataset=DatasetMetadata(
                name="MOUNTS / Sentinel-2 Volcanic Hotspot",
                source="Monitoring Unrest from Space (Valade et al., Remote Sens. 2019)",
                modality="Sentinel-2 SWIR/TIR & Sentinel-5P TROPOMI SO2 columns",
                samples_approx="17 active volcanoes worldwide (2018-2023)",
                labels="Radiative heat flux, volcanic unrest state, ash plume mask",
                geographic_coverage="Global active volcanic arcs",
                split_strategy="Volcano-based holdout",
                license="Open Academic Data"
            ),
            description="Effusive lava effusion detection and explosive ash plume atmospheric tracking."
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
