# ORBIT-PRISM: project context

Hackathon: Conscientia 2k26 / HackOrbital 2026, PS 1.4 (TinyML visual triage and confidence-aware bandwidth optimizer for constrained telemetry links).

One-liner: a TinyML triage layer that decides what a satellite sends first. You set the miss rate; we minimise the bytes, even through cloud.

Floods (Sen1Floods11) are the worked example, not the product. Clouds are simulated. This is a simulation of the link and memory limits, not flight hardware.

## Key numbers

- Downlink pass: 256 kbps x 600 s = 19.2 MB (raw ceiling).
- SRAM budget: 4.2 MB = 4,200,000 bytes (decimal, the conservative reading; 4.2 MiB would be 4,404,019).
- Chip 512x512 -> 16 tiles of 128x128 (row, col in 0..3).
- Tile size: optical (4 bands, 8-bit) 65,536 B; radar (2 bands, 8-bit) 32,768 B.

## Tile-record CSV schema (DO NOT CHANGE without team agreement)

One row per tile. `data/fake_tiles.csv` and `results/real_water_stats.csv` (first five columns plus `water_frac`) follow this.

| column | type | values | source |
|---|---|---|---|
| event | str | Sen1Floods11 event name, e.g. `Bolivia` | real (filename prefix) |
| chip | str | chip id, e.g. `Bolivia_129334` | real (filename) |
| row | int | 0..3 | real (tile grid) |
| col | int | 0..3 | real (tile grid) |
| split | str | `train` / `valid` / `test` / `holdout` (Bolivia held-out event) | real (official split; chip-level for train/valid/test, Bolivia held out as an event) |
| water_frac | float | 0..1, non-permanent water share of valid pixels | real (hand labels minus JRC permanent water) |
| critical | int | 1 if `water_frac >= CRITICAL_THRESHOLD` (0.30), else 0 | derived from real |
| cloud_frac | float | 0..1 | SYNTHETIC (simulated) |
| sensor_used | str | `radar` if `cloud_frac >= 0.5`, else `optical` | derived from synthetic |
| p_flood | float | 0..1, placeholder model score; optical noise rises with cloud_frac, radar does not | SYNTHETIC |
| extent | float | 0..1, placeholder flood extent estimate | SYNTHETIC |

Design notes:
- `tile_id` is not stored; it is `chip` + `row` + `col`.
- Tile byte cost is not stored; it is derived from `sensor_used` (65,536 B optical, 32,768 B radar; see `src/constants.py`).
- Official train/valid/test are chip-level splits across events; only Bolivia is a held-out event (`holdout`). Report held-out-event results on Bolivia.
- `CRITICAL_THRESHOLD` = 0.30 is a named constant so the 5% sensitivity analysis is a one-line change.

## Locked decisions (Madhavan, branch madhavan/eval)

1. **Test event = Bolivia** (`split == "holdout"`: 15 chips, 231 tiles). Train on `train`, calibrate on `valid`, never tune on `holdout`. The chip-level `test` split is not the headline test set.
2. **Optical model trains on tiles with `cloud_frac < 0.5` and scores all tiles.** The radar model trains on all tiles.
3. **Headline recall = full-tile recall** (critical tiles delivered as full tiles vs cumulative MB sent; this is the main plot). Alert recall (critical tiles alerted by an L0 packet) is reported as a separate metric, never mixed into the main plot.
4. **Phase 2 length is 17 hours, not 36.** Slides, speech and the plan must say 17 hours. Checkpoints: hour 8 merge, hour 12 feature freeze, hour 14 code freeze.

5. **Scaled pass budget.** Bolivia (~15 MB) fits inside one real 19.2 MB pass, so the evaluation uses a scaled pass of **20% of the event's all-optical volume** (231 tiles x 65,536 B x 0.20 = ~3.03 MB). Label it "scaled pass" on every plot and slide, and also show the full recall-vs-cumulative-MB curve. Never present it as the real 19.2 MB pass.

6. **The chip-level `test` split** (1,378 tiles) is a secondary in-distribution report, shown separately from Bolivia and never used for tuning. (Proposal: edit here to veto.)
7. **Optical-only baseline, real pipeline:** the same optical model scores cloudy tiles on the simulated-cloud pixels. That is what a system without radar would do. `scripts/baselines.py` only approximates it by treating cloudy tiles as blind.
8. **Permanent-water removal matters.** Subtracting the JRC layer turns 280 of 6,825 tiles (4.1%) from critical to non-critical, and permanent-water pixels stay in the denominator. Anyone rebuilding labels must do the same.

## Hazards (floods and forest fires)

- **Floods are the primary, fully measured hazard.** Forest fires are the second hazard (decided by Madhavan). The pipeline is the same; the model, the target and the sensor story change. Fire work must not delay the flood headline plot.
- **Schema variant.** Fire files use the same 11 columns, with `water_frac` renamed `burn_frac` (share of valid pixels labelled burned). Flood files are unchanged. Fire files live in `data/fire/` and `results/fire/`; flood files stay where they are.
- **Fire dataset (verified from the Zenodo record and the metadata CSV):** the Satellite Burned Area Dataset (Colomba et al., Politecnico di Torino, CIKM 2022), Zenodo 10.5281/zenodo.6597139, **licence CC BY 4.0**, 14.0 GB in five zips, 73 acquisitions from 27 Copernicus EMS activations, 7 folds (column `fold` in `data/fire/satellite_data.csv`). Only 50 of 73 rows list a pre-fire date and 53 a post-fire date. **The `height` and `width` columns in the CSV are not the raster sizes** (the real rasters are far larger; ignore those columns). Each fold keeps whole activations together (no activation spans two folds), so holding out a fold cannot leak a fire.
- **Verified by inspecting part 5 of 5 only (1.49 GB, 9 areas, 52 TIFFs):** the archives do contain Sentinel-1 radar next to Sentinel-2. Each area folder holds `sentinel1_<date>.tiff` and `sentinel2_<date>.tiff` for two to four dates (pre and post fire; dates are in the file names, so use those, not the CSV), a `*_mask.tiff`, matching PNG previews and `*_coverage.png`, and for some areas a `sentinel2_<date>_cloud_coverage.tiff` (0/1 mask). Sentinel-2 is a 13-channel float32 GeoTIFF (12 reflectance bands plus a constant 1.0 channel). Sentinel-1 is a 4-channel float32 GeoTIFF (channel 3 has up to 16% NaN pixels in some files; what each channel means is not documented, so do not assume VV/VH order). Rasters are 531x647 up to 3207x2995 px, so a 128x128 grid has 20 to 575 tiles per area (1,247 tiles for the 9 areas in part 5). The mask is float64 with 0 plus 2 to 4 damage codes that differ by area (seen: 32, 64, 128, 192, 255); `burn_frac` uses burned = mask > 0.
- **Known data defects:** one Sentinel-1 file in part 5 cannot be decoded (`EMSR237_04ELCASTILLODELASGUARDAS.../sentinel1_2017-07-31.tiff`); load defensively and report skipped files. Not yet checked: parts 1 to 4 (the other 64 areas), the exact S1/S2 band meanings and whether every area has a usable pre/post pair.
- **Class balance is very different from floods.** In the 9 areas of part 5, 52% of tiles have burn_frac >= 0.30 (floods: 8%), driven by two large areas (EMSR298, EMSR365) where the annotated region is mostly burned. This is partial and biased, but it means "critical" is common for fire and triage is less selective. Report the prevalence next to every fire result and rethink the 0.30 threshold once all parts are read.
- **Radar for fire is a hypothesis, not a fact.** The flood story (radar sees water through cloud) does not transfer automatically. Burn scars may change SAR backscatter, but flames and smoke are not what SAR measures. State the radar fallback for fire only after it is measured.
- **Fire label source (Madhavan's decision): Agnel's sources are primary.** NASA FIRMS VIIRS 375 m active-fire detections, ESA FireCCI51 burned area, Sentinel-1 and Sentinel-2 from the Copernicus Data Space, ESA CCI land cover. The Torino dataset above stays as the verified fallback and cross-check (its zips are being downloaded to `data/raw/fire_torino/zips/`). What the sources are, from their catalogue pages (confirm on the official pages before the final write-up):
  - **FireCCI51:** monthly global burned area from MODIS at about 250 m. A 128x128 tile at 10 m is about 1.3 km across, so a tile covers only about 5x5 label pixels: good enough for a rough tile-level `burn_frac`, too coarse for pixel masks. Earth Engine lists 2001 to 2020 (other records say to 2022), so fire events must fall inside the product's coverage and in the Sentinel-2 era. Free to use with acknowledgement and citation; the ESA CCI Fire terms have not been read in full.
  - **FIRMS VIIRS 375 m:** point detections (latitude, longitude, brightness temperature, fire radiative power, confidence 0 to 2, acquisition date), not burned-area masks. Open data, acknowledgement required.
  - **Copernicus Data Space:** free account; Sentinel-2 can be filtered by cloud cover, Sentinel-1 cannot (radar ignores cloud). Credit line: "contains modified Copernicus Sentinel data".
  - **Cloud wording changes for fire.** Agnel's table carries a `cloud_fraction` that can be real (from Sentinel-2 metadata), unlike the flood data where clouds are simulated. Slides must say which hazard uses real and which simulated clouds, and never describe the fire clouds as simulated if they are not.
  - **Agnel's tile table maps to our contract by a converter, not by changing the contract:** `water_fraction`/`burn_fraction`/`cloud_fraction` become `water_frac`/`burn_frac`/`cloud_frac` on a 0 to 1 scale; `Validation` becomes `valid`; the split must hold out whole events; `image_id` must resolve to `chip, row, col`. Open questions for Agnel: what one row is (a 128x128 tile or a whole image), how FireCCI51 labels are resampled to 10 m, which fire events and dates are used, the `Normal` class, and where the data is stored (nothing of it is on GitHub yet).
- **Considered and not chosen:** CaBuAr (Sentinel-2 only, OpenRAIL licence, no radar) and Sen2Fire (Sentinel-2 plus a Sentinel-5P aerosol band, CC BY 4.0, 6.3 GB, 2,466 patches; labels come from a 1 km MODIS active-fire product, so they are coarse, and it has no radar).
- **Fire grid:** `row` and `col` run from 0 to the area's own tile count, not 0 to 3 (that range is for flood chips only).
- **Open decisions for fire:** the held-out test fold (hold out whole folds, never tiles), the burned-area threshold for `critical` (default 0.30, same constant), the per-hazard scaled pass budget, and whether to download parts 1 to 4 (12.5 GB) to complete the statistics. `scripts/fire_stats.py` reads whatever zips are present and prints a PARTIAL warning until all 73 areas are found; its output `results/fire/real_burn_stats.csv` is git-ignored until the dataset is complete.

## Data notes

- **Missing tiles.** 7,136 tiles exist (446 chips x 16); 311 have no valid label pixels (all -1) and are absent from every CSV, leaving 6,825 tiles in 441 chips (five Ghana chips, `Ghana_234935`, `Ghana_26376`, `Ghana_277`, `Ghana_5079`, `Ghana_83483`, have no valid pixels at all and are absent; 395 chips have all 16 tiles). Anyone tiling imagery must drop the same (chip, row, col) tiles. The keys in `data/tiles_base.csv` are the source of truth.
- **Cloud simulation contract.** `cloud_frac` in `data/tiles_base.csv` is the single source of truth. The cloud mask applied to Sentinel-2 pixels (`src/cloud.py`, owner: Agnel) must cover that fraction of the tile and be deterministic from (seed, chip, row, col). `sensor_used` = `radar` iff `cloud_frac >= 0.5`. Until real imagery is in use, `data/fake_tiles.csv` stands in for `tiles_base.csv` plus scores.
- **Sentinel-1 / Sentinel-2 dates can differ** within a chip. Check the metadata and state the gap when pairing the two sensors.
- **Licence.** Sen1Floods11 states no licence on GitHub. Keep the credit "contains modified Copernicus Sentinel data" and do not redistribute raw imagery.

## File contracts (change only with team agreement)

Keys for every per-tile file: `chip, row, col` (`tile_id` = `chip_row_col`). All files are CSV with a header unless noted.

| file | owner | columns |
|---|---|---|
| `data/tiles_base.csv` | Madhavan | `event, chip, row, col, split, water_frac, critical, cloud_frac, sensor_used` (the schema above without `p_flood`, `extent`) |
| `results/scores_optical_int8.csv` | Agnel | `chip, row, col, p_flood, extent`: raw (uncalibrated) INT8 optical-model score, for every tile in every split |
| `results/scores_radar_int8.csv` | Agnel | same columns, radar model |
| `results/scores_ndwi.csv` | Madhavan | same columns, NDWI baseline |
| `results/calibration.json` | Shenbaga | `{"version": 1, "calibration_split": "valid", "method": "split_conformal", "optical": {"0.05": t, ...}, "radar": {"0.05": t, ...}}`. Keys are target miss rates as strings; `t` is the `p_flood` threshold: keep a tile when `p_flood >= t` |
| `results/memory_table.csv` | Harish | `model, dtype, weights_bytes, peak_activation_bytes, tile_buffer_bytes, total_bytes, macs, recall, ece` (float vs INT8 rows; `total_bytes` must be <= 4,200,000) |
| `data/tiles.csv` | Madhavan | the 11-column schema, built by `scripts/merge_scores.py` |
| send log from `src.scheduler.sim.run` | Akhilesh | `order, chip, row, col, layer, bytes, cum_bytes, pass_index`, where `layer` is `L0`, `L1` or `L2` |

`src.scheduler.sim.run(tiles, policy, budget_bytes)` takes a DataFrame in the 11-column schema, a policy in `capture_order | discard_cloudy | optical_only | colour_index | orbit_prism`, and a per-pass byte budget, and returns the send log. Shared constants live in `src/constants.py`. The L1 preview size is not decided yet: Akhilesh sets `L1_BYTES` there and tells the group.

## Repo rules

- One branch per owner (`<name>/<topic>`). Only Madhavan merges to `main`, at checkpoints. GitHub cannot enforce this on a free private repo, so it is an agreement.
- Before asking for a merge: columns unchanged, nothing tuned on `holdout`, no raw data or `.tif` committed, no banned claims, scripts run from a clean clone.
- Synthetic data (`data/fake_tiles.csv`) never appears on slides. Every number shown comes from `results/` on real scores.

## Claims policy

Never say "outperforms", "guarantees" (without "holds when test data resembles calibration data"), "real-time on satellite hardware" or "danger score". All results are "to be measured". Do not claim tiling or cloud filtering is new.
