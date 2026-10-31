# ORBIT-PRISM Datasets Reference: Flood & Forest Fire

This document outlines the official datasets integrated into the ORBIT-PRISM visual triage architecture across flood detection (primary hackathon benchmark) and forest-fire mapping (multi-hazard generalization).

---

## 1. Flood Detection Datasets (Active Prototype)

The primary worked example for ORBIT-PRISM PS 1.4 is flood triage using the **Sen1Floods11 v1.1 Hand-Labelled Benchmark**.

| No. | Dataset / Layer | Modality / Sensor | Purpose in ORBIT-PRISM | Local Location |
|---|---|---|---|---|
| **1** | **`S1Hand`** | Sentinel-1 SAR (IW mode, GRD) | Dual-polarization radar inputs (VV + VH), cloud-penetrating imagery | `data/raw/data/flood_events/HandLabeled/S1Hand/` |
| **2** | **`S2Hand`** | Sentinel-2 MSI (Level-1C) | Multispectral optical inputs (B2 Blue, B3 Green, B4 Red, B8 NIR) | `data/raw/data/flood_events/HandLabeled/S2Hand/` |
| **3** | **`LabelHand`** | QC Ground Truth Mask | Hand-labelled pixel masks (-1: no-data, 0: not water, 1: water) | `data/raw/data/flood_events/HandLabeled/LabelHand/` |
| **4** | **`JRCWaterHand`** | JRC Global Surface Water | Permanent water occurrence layer; subtracted to isolate flood water | `data/raw/data/flood_events/HandLabeled/JRCWaterHand/` |
| **5** | **Official Split CSVs** | Metadata | Defines Train, Validation, Test, and Bolivia holdout sets | `data/raw/splits/flood_handlabeled/` |
| **6** | **Sen1Floods11 Metadata** | GeoJSON STAC catalog | Event acquisition dates (`s1_date`, `s2_date`), orbits, and bounds | `data/raw/Sen1Floods11_Metadata.geojson` |

**Source:** Google Cloud Storage `gs://sen1floods11/v1.1/` & [Cloud to Street Sen1Floods11 GitHub](https://github.com/cloudtostreet/Sen1Floods11)  
**Processed Outputs:**
- Compressed Arrays: `data/processed/<event>.npz` (`tile_id`, `optical`, `radar`, `water_frac`, `cloud_frac`, `split`, `critical`)
- Radar Metadata: [`data/radar_tiles.csv`](file:///Users/agnelvishwakamalraj/conscientia_2k26/data/radar_tiles.csv)
- Optical Metadata: [`data/optical_tiles.csv`](file:///Users/agnelvishwakamalraj/conscientia_2k26/data/optical_tiles.csv)
- Base Telemetry: [`data/tiles_base.csv`](file:///Users/agnelvishwakamalraj/conscientia_2k26/data/tiles_base.csv) and [`data/tiles.csv`](file:///Users/agnelvishwakamalraj/conscientia_2k26/data/tiles.csv)

---

## 2. Forest-Fire Datasets (Multi-Hazard Extension)

For broader mission generalization (monitoring wildfires, active fire fronts, and post-fire burn severity through dense smoke plumes), the ORBIT-PRISM system accommodates the following standard Earth Observation datasets:

| No. | Dataset | Sensor / Source | Spatial / Temporal Resolution | Purpose in ORBIT-PRISM |
|---|---|---|---|---|
| **1** | **NASA FIRMS VIIRS 375m** | Suomi-NPP / NOAA-20 VIIRS | 375 m ground resolution, near-real-time active hotspots | Rapid L0 alert packet trigger ($32\text{ B}$ packet) for thermal anomalies |
| **2** | **ESA FireCCI51** | MODIS / Sentinel-3 OLCI | 250 m pixel resolution, global burned-area masks | Ground truth training masks for burned-area segmentation (`burn_mask_path`, `burn_fraction`) |
| **3** | **Sentinel-1 SAR VV/VH** | Sentinel-1 C-Band SAR | 10 m resolution, cloud- and smoke-penetrating radar | Primary radar input: detects surface roughness changes and canopy destruction through heavy smoke plumes |
| **4** | **Sentinel-2 MSI Imagery** | Sentinel-2 MSI | 10–20 m resolution, SWIR bands (B11, B12) + NIR (B8) | Optical triage: Normalized Burn Ratio (NBR = (B8 - B12) / (B8 + B12)) and dNBR burn severity |
| **5** | **ESA CCI Land Cover** | MERIS / Sentinel-3 | 300 m resolution, global vegetation / biome classes | Eliminates false alarms by masking non-burnable water, bare rock, and snow surfaces |

**External Repositories & Access:**
- NASA FIRMS: [https://firms.modaps.eosdis.nasa.gov/](https://firms.modaps.eosdis.nasa.gov/)
- ESA FireCCI: [https://climate.esa.int/en/projects/fire/](https://climate.esa.int/en/projects/fire/)
- Copernicus Data Space Ecosystem: [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)

---

## 3. Schema Alignment Across Optical and Radar Modules

Both modules share unified tile geometries:
- Chip dimensions: $512 \times 512$ pixels
- Grid: $4 \times 4 = 16$ tiles per chip
- Tile size: $128 \times 128$ pixels
- Telemetry payload:
  - Optical tile: $128 \times 128 \times 4\text{ B} = 65,536\text{ Bytes}$
  - Radar tile: $128 \times 128 \times 2\text{ B} = 32,768\text{ Bytes}$
- Disaster types supported in schemas: `Flood`, `Wildfire`, `Normal`
- Shared metadata files:
  - Optical: [`data/optical_tiles.csv`](file:///Users/agnelvishwakamalraj/conscientia_2k26/data/optical_tiles.csv)
  - Radar: [`data/radar_tiles.csv`](file:///Users/agnelvishwakamalraj/conscientia_2k26/data/radar_tiles.csv)
