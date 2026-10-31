# ORBIT-PRISM: project context

Hackathon: Conscientia 2k26 / HackOrbital 2026, PS 1.4 (TinyML visual triage and confidence-aware bandwidth optimizer for constrained telemetry links).

One-liner: a TinyML triage layer that decides what a satellite sends first. You set the miss rate; we minimise the bytes, even through cloud.

Floods (Sen1Floods11) are the worked example, not the product. Clouds are simulated. This is a simulation of the link and memory limits, not flight hardware.

## Key numbers

- Downlink pass: 256 kbps x 600 s = 19.2 MB (raw ceiling).
- SRAM budget: 4.2 MB.
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
