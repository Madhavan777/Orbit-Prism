# data/

## fake_tiles.csv: SYNTHETIC, do not quote

Only `water_frac` and `critical` come from real Sen1Floods11 hand labels. `cloud_frac`, `sensor_used`, `p_flood` and `extent` are made-up placeholders for building and testing code. **No number computed from this file may appear on a slide, in the README or in the pitch.** Real results come from `results/` once real model scores exist.

Regenerate it with `python scripts/make_fake_data.py` (needs `results/real_water_stats.csv`).

## raw/ (git-ignored)

Sen1Floods11 download. Never commit it. Layout when downloaded from `gs://sen1floods11`:
`raw/data/flood_events/HandLabeled/{LabelHand,JRCWaterHand,S1Hand,S2Hand}/` and `raw/splits/flood_handlabeled/`.
Labels and JRC are 3 MB; S1Hand and S2Hand are about 1.67 GB together.
