"""ESA SNAP (Sentinel Application Platform) Integration & SAR Preprocessing Workflow.

Author: Harish D (ORBIT-PRISM)
Branch: harish/radar-compress

Role: Ground-Side Preprocessing Specification & Inspection.
Note: SNAP is a ground-side tool and NOT part of the onboard satellite CNN inference runtime.

Supported Sensor: Sentinel-1 C-band Synthetic Aperture Radar (SAR)
Required Toolbox: Sentinel-1 Microwave Toolbox (S1TBX)
Download: https://step.esa.int/main/download/snap-download/
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def check_esa_snap_installed() -> Dict[str, any]:
    """Check if ESA SNAP's Graph Processing Tool (gpt) is available on the system."""
    gpt_in_path = shutil.which("gpt")
    if gpt_in_path:
        return {
            "installed": True,
            "executable": gpt_in_path,
            "location": "System PATH",
            "required_toolbox": "Sentinel-1 Microwave Toolbox (S1TBX)",
        }

    common_locations = [
        Path(r"C:\Program Files\snap\bin\gpt.exe"),
        Path(r"C:\Program Files (x86)\snap\bin\gpt.exe"),
        Path.home() / "snap" / "bin" / "gpt.exe",
    ]
    for loc in common_locations:
        if loc.exists():
            return {
                "installed": True,
                "executable": str(loc),
                "location": "Default Installation Directory",
                "required_toolbox": "Sentinel-1 Microwave Toolbox (S1TBX)",
            }

    return {
        "installed": False,
        "executable": None,
        "location": "Not Found",
        "download_url": "https://step.esa.int/main/download/snap-download/",
        "required_toolbox": "Sentinel-1 Microwave Toolbox (S1TBX)",
        "installation_guide": (
            "1. Download ESA SNAP installer (esa-snap_windows-x64_*.exe).\n"
            "2. Run installer and ensure 'Sentinel-1 Toolbox' is checked.\n"
            "3. Add 'C:\\Program Files\\snap\\bin' to your system PATH.\n"
            "4. Verify via terminal: gpt -h"
        ),
        "status_note": (
            "Sen1Floods11 S1Hand data is already radiometrically calibrated and orthorectified. "
            "Therefore, ground-side SNAP preprocessing is only required if ingesting raw "
            "Level-1 GRD/SLC products from Copernicus SciHub/ASF."
        ),
    }


def get_snap_graph_xml(
    input_safe_path: str = "input_product.SAFE",
    output_tif_path: str = "output_sar.tif",
) -> str:
    """Standard ESA SNAP XML graph for Sentinel-1 GRD SAR preprocessing:

    1. Read Sentinel-1 GRD product
    2. Apply-Orbit-File (Precise Orbit Ephemerides - POEORB)
    3. ThermalNoiseRemoval (subtract thermal noise floor)
    4. Calibration (Radiometric calibration to Sigma0)
    5. Speckle-Filter (Refined Lee filter)
    6. Terrain-Correction (Range-Doppler using SRTM 1-sec HGT DEM)
    7. BandSelect (VV and VH polarizations)
    8. Write (GeoTIFF)
    """
    return f"""<graph id="S1_GRD_Preprocessing">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{input_safe_path}</file>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>false</continueOnFail>
    </parameters>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
    </parameters>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="ThermalNoiseRemoval"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands/>
      <auxFile>Product Auxiliary-File</auxFile>
      <externalAuxFile/>
      <outputImageInComplex>false</outputImageInComplex>
      <outputImageScaleInDb>false</outputImageScaleInDb>
      <createGammaBand>false</createGammaBand>
      <createBetaBand>false</createBetaBand>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
      <outputGammaBand>false</outputGammaBand>
      <outputBetaBand>false</outputBetaBand>
    </parameters>
  </node>
  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <filter>Refined Lee</filter>
      <filterSizeX>3</filterSizeX>
      <filterSizeY>3</filterSizeY>
      <dampingFactor>2</dampingFactor>
      <edgeThreshold>5000.0</edgeThreshold>
      <estimateENL>true</estimateENL>
      <enl>1.0</enl>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Speckle-Filter"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands/>
      <demName>SRTM 1Sec HGT</demName>
      <externalDEMFile/>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
      <externalDEMApplyEGM>true</externalDEMApplyEGM>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <mapProjection>WGS84(DD)</mapProjection>
      <alignToStandardGrid>false</alignToStandardGrid>
      <standardGridOriginX>0.0</standardGridOriginX>
      <standardGridOriginY>0.0</standardGridOriginY>
      <nodataValueAtSea>false</nodataValueAtSea>
      <saveDEM>false</saveDEM>
      <saveLatLon>false</saveLatLon>
      <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <saveProjectedLocalIncidenceAngle>false</saveProjectedLocalIncidenceAngle>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
      <outputComplex>false</outputComplex>
      <applyRadiometricNormalization>false</applyRadiometricNormalization>
      <saveSigmaNought>false</saveSigmaNought>
      <saveGammaNought>false</saveGammaNought>
      <saveBetaNought>false</saveBetaNought>
      <incidenceAngleForSigma0>Use projected local incidence angle from DEM</incidenceAngleForSigma0>
      <incidenceAngleForGamma0>Use projected local incidence angle from DEM</incidenceAngleForGamma0>
      <auxFile>Latest Auxiliary File</auxFile>
      <externalAuxFile/>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{output_tif_path}</file>
      <formatName>GeoTIFF</formatName>
    </parameters>
  </node>
</graph>
"""


def validate_sar_tile_channels(tile_array: np.ndarray) -> bool:
    """Validate that the input tensor strictly adheres to the Sentinel-1 SAR specifications:

    - Shape: (128, 128, 2)
    - Channel 0: VV polarization backscatter
    - Channel 1: VH polarization backscatter
    - Values in [0.0, 1.0]
    """
    if tile_array.shape != (128, 128, 2):
        raise ValueError(f"Expected shape (128, 128, 2), got {tile_array.shape}")

    if np.any(np.isnan(tile_array)) or np.any(np.isinf(tile_array)):
        raise ValueError("SAR tile contains NaN or Inf values!")

    if tile_array.min() < -1e-4 or tile_array.max() > 1.0 + 1e-4:
        raise ValueError(f"SAR tile values out of normalized range [0, 1]: min={tile_array.min()}, max={tile_array.max()}")

    return True


if __name__ == "__main__":
    status = check_esa_snap_installed()
    print("ESA SNAP Status:")
    for k, v in status.items():
        print(f"  {k}: {v}")

