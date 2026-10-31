# data/

## fake_tiles.csv: SYNTHETIC, do not quote

Only `water_frac` and `critical` come from real Sen1Floods11 hand labels. `cloud_frac`, `sensor_used`, `p_flood` and `extent` are made-up placeholders for building and testing code. **No number computed from this file may appear on a slide, in the README or in the pitch.** Real results come from `results/` once real model scores exist.

Regenerate it with `python scripts/make_fake_data.py` (needs `results/real_water_stats.csv`).

## raw/ (git-ignored)

Raw downloads. Never commit them.

### Floods: Sen1Floods11

```bash
python scripts/download_flood.py            # labels + JRC + splits, about 3 MB
python scripts/download_flood.py --imagery  # also S1Hand + S2Hand, about 1.7 GB
```

The script needs no gsutil. It writes `raw/data/flood_events/HandLabeled/{LabelHand,JRCWaterHand,S1Hand,S2Hand}/` and `raw/splits/flood_handlabeled/`, the layout `scripts/real_stats.py` expects. Licence: none stated on GitHub (unconfirmed).

### Fires: Satellite Burned Area Dataset (Politecnico di Torino)

Zenodo record 10.5281/zenodo.6597139, licence CC BY 4.0, 14.0 GB in five zips (`Satellite_burned_area_dataset_part1..5.zip`, not yet downloaded). The small metadata file `satellite_data.csv` (73 rows, one per acquisition, with a `fold` column) is committed as `data/fire/satellite_data.csv`. See the Hazards section of `CONTEXT.md` for what is and is not verified about this dataset.
