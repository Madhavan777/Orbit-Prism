# ORBIT-PRISM: project context

Hackathon: Conscientia 2k26 / HackOrbital 2026, PS 1.4 (TinyML visual triage and confidence-aware bandwidth optimizer for constrained telemetry links).

One-liner: a TinyML triage layer that decides what a satellite sends first. You set the miss rate; we minimise the bytes, even through cloud.

Floods (Sen1Floods11) are the worked example, not the product. Clouds are simulated. This is a simulation of the link and memory limits, not flight hardware.

## Key numbers

- Downlink pass: 256 kbps x 600 s = 19.2 MB (raw ceiling).
- SRAM budget: 4.2 MB.
- Chip 512x512 -> 16 tiles of 128x128 (row, col in 0..3).
- Tile size: optical (4 bands, 8-bit) 65,536 B; radar (2 bands, 8-bit) ~33 KB.

## Tile-record CSV schema (DO NOT CHANGE without team agreement)

One row per tile. `data/fake_tiles.csv` and `results/real_water_stats.csv` (first five columns plus `water_frac`) follow this.

| column | type | values | source |
|---|---|---|---|
| event | str | Sen1Floods11 event name, e.g. `Bolivia` | real (filename prefix) |
| chip | str | chip id, e.g. `Bolivia_129334` | real (filename) |
| row | int | 0..3 | real (tile grid) |
| col | int | 0..3 | real (tile grid) |
| split | str | `train` / `valid` / `test` / `holdout` (Bolivia held-out event) | real (official split, by event) |
| water_frac | float | 0..1, non-permanent water share of valid pixels | real (hand labels minus JRC permanent water) |
| critical | int | 1 if `water_frac >= CRITICAL_THRESHOLD` (0.30), else 0 | derived from real |
| cloud_frac | float | 0..1 | SYNTHETIC (simulated) |
| sensor_used | str | `radar` if `cloud_frac >= 0.5`, else `optical` | derived from synthetic |
| p_flood | float | 0..1, placeholder model score; optical noise rises with cloud_frac, radar does not | SYNTHETIC |
| extent | float | 0..1, placeholder flood extent estimate | SYNTHETIC |

Design notes:
- `tile_id` is not stored; it is `chip` + `row` + `col`.
- Tile byte cost is not stored; it is derived from `sensor_used` (65,536 B optical, ~33,000 B radar).
- Split is by event, never by chip, to avoid leakage.
- `CRITICAL_THRESHOLD` = 0.30 is a named constant so the 5% sensitivity analysis is a one-line change.

## Claims policy

Never say "outperforms", "guarantees" (without "holds when test data resembles calibration data"), "real-time on satellite hardware" or "danger score". All results are "to be measured". Do not claim tiling or cloud filtering is new.
