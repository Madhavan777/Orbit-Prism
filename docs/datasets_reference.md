# ORBIT-PRISM Datasets Reference: Flood and Forest Fire

Scope is floods (primary, measured) and forest fires (second hazard). No other hazard is in scope (see "Scope" in CONTEXT.md).

**Status key:** LOADED = files are in use by the code today. NOT LOADED = chosen or listed, but no data has been downloaded or processed for it.

---

## 1. Flood detection: Sen1Floods11 v1.1 hand-labelled subset (LOADED)

| No. | Dataset / layer | Sensor | Purpose | Location after `python scripts/download_flood.py` |
|---|---|---|---|---|
| 1 | `S1Hand` | Sentinel-1 SAR (IW, GRD) | Radar input (VV, VH), unaffected by cloud | `data/raw/data/flood_events/HandLabeled/S1Hand/` (imagery download: `--imagery`) |
| 2 | `S2Hand` | Sentinel-2 MSI (Level-1C) | Optical input (B2, B3, B4, B8) | `data/raw/data/flood_events/HandLabeled/S2Hand/` (imagery download: `--imagery`) |
| 3 | `LabelHand` | Hand-labelled masks | -1 no data, 0 not water, 1 water | `data/raw/data/flood_events/HandLabeled/LabelHand/` |
| 4 | `JRCWaterHand` | JRC surface water | Permanent water, subtracted to isolate flood water | `data/raw/data/flood_events/HandLabeled/JRCWaterHand/` |
| 5 | Official split CSVs | Metadata | train / valid / test, plus the Bolivia held-out event | `data/raw/splits/flood_handlabeled/` |
| 6 | `Sen1Floods11_Metadata.geojson` | Metadata | Event acquisition dates | `data/raw/Sen1Floods11_Metadata.geojson` |

Source: `gs://sen1floods11/v1.1/` and [cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11). Licence: none is stated on the GitHub repository, so treat it as unconfirmed and keep the credit "contains modified Copernicus Sentinel data".

Generated files (all derived from the above plus simulated clouds):
- `results/real_water_stats.csv`: real per-tile water fractions (`scripts/real_stats.py`).
- `data/tiles_base.csv`: per-tile inputs with simulated `cloud_frac` and `sensor_used` (`scripts/build_tiles_base.py`).
- `data/optical_tiles.csv`, `data/radar_tiles.csv`: per-tile index files for the optical and radar developers (`python -m src.data.generate_optical_metadata`, `python -m src.data.generate_radar_metadata`). Split values are `train / valid / test / holdout`.
- `data/processed/<event>.npz`: tile arrays, git-ignored, made by the pixel pipeline in `src/data/` once the imagery is downloaded.

All clouds are SIMULATED. Sen1Floods11 has no cloud mask.

---

## 2. Forest fire (NOT LOADED)

**No fire data has been downloaded or processed.** The optical and radar index files contain flood and normal tiles only; `burn_mask_path` is empty and `burn_fraction` is 0. The sources below are the plan, not an integrated dataset.

| No. | Source | What it provides (from its catalogue page) | Intended use | Caveats |
|---|---|---|---|---|
| 1 | NASA FIRMS VIIRS 375 m | Active-fire point detections (location, brightness temperature, fire radiative power, confidence 0 to 2) | Hotspot context | Points, not burned-area masks; open data, acknowledgement required |
| 2 | ESA FireCCI51 | Monthly global burned area from MODIS at about 250 m | Tile-level `burn_frac` labels | About 5x5 label pixels per 128x128 tile at 10 m; coverage years must overlap the Sentinel-2 era; terms not yet read in full |
| 3 | Sentinel-1 SAR (Copernicus Data Space) | C-band radar, VV and VH | Radar input | Whether radar helps for fire is a hypothesis to be measured, not a result |
| 4 | Sentinel-2 MSI (Copernicus Data Space) | Optical bands including SWIR (B11, B12) | Optical input; burn index (NBR = (B8 - B12) / (B8 + B12)) | Can be filtered by cloud cover; real clouds, unlike the simulated flood clouds |
| 5 | ESA CCI Land Cover | Land-cover classes at about 300 m | Masking non-burnable surfaces | Not yet used |

Verified fallback: the Torino Satellite Burned Area Dataset (Zenodo 10.5281/zenodo.6597139, CC BY 4.0, 14 GB). Only part 5 has been inspected; see the Hazards section of CONTEXT.md.

External pages: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/), [ESA Fire CCI](https://climate.esa.int/en/projects/fire/), [Copernicus Data Space](https://dataspace.copernicus.eu/).

---

## 3. Shared tile geometry

- Chip 512x512 px, cut into a 4x4 grid of 128x128 tiles (flood chips only; fire areas have their own grid size).
- Payload: optical 128x128x4 bytes = 65,536 B; radar 128x128x2 bytes = 32,768 B.
- Alert packet (L0): 32 B. The preview size (L1) is provisional at 2,048 B in `src/data/generate_metadata.py`; the scheduler owner confirms it in `src/constants.py`.
