# ORBIT-PRISM

A TinyML triage layer that decides what a satellite sends first. You set the miss rate; we minimise the bytes, even through cloud.

Conscientia 2k26 / HackOrbital 2026, PS 1.4. This is a simulation of the link and memory limits, not flight hardware. Clouds are simulated. Floods (Sen1Floods11) are the worked example.

Read [CONTEXT.md](CONTEXT.md) first: it defines the tile CSV schema, the locked decisions, the file contracts and the claims policy.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Python 3.10 to 3.12.

## Run

```bash
python scripts/download_flood.py    # labels + JRC + splits (3 MB); add --imagery for 1.7 GB more
python scripts/real_stats.py        # needs data/raw labels; writes results/real_water_stats.csv
python scripts/build_tiles_base.py  # writes data/tiles_base.csv (real labels + simulated clouds)
python scripts/make_fake_data.py    # writes data/fake_tiles.csv (synthetic scores, see data/README.md)
python scripts/baselines.py         # baselines on the Bolivia held-out event
python scripts/evaluate.py --allow-placeholder   # plumbing test on synthetic scores -> results/synthetic/
python -m uvicorn backend.api:app --port 8000   # the website and its API: http://localhost:8000
python scripts/check_merge.py       # merge gate: run before asking for a merge
python -m pytest tests -q          # full test suite
```

`results/real_water_stats.csv` is committed, so you only need the raw download for `real_stats.py` and for anything using imagery.

## Runbook: from raw data to the headline plot

Run everything from the repo root, in this order. Each step names who owns it and what it writes.

| # | Command | Who | Writes |
|---|---|---|---|
| 0 | `git pull`, `pip install -r requirements.txt` | everyone | |
| 1 | `python scripts/download_flood.py --imagery` (1.7 GB; `--events Bolivia Ghana` for a subset) | everyone who trains | `data/raw/` (git-ignored) |
| 2 | `python scripts/build_processed.py` (about 3 minutes; prints `OK` per event and exits 0 only if every event matches `tiles_base.csv`) | everyone who trains | `data/processed/<event>.npz` (git-ignored, about 450 MB) |
| 3 | `python scripts/train.py` | Shenbaga | `models/optical_float.keras` |
| 4 | `python scripts/predict.py` | Shenbaga | `results/scores_optical_float.csv` |
| 5 | `python -c "from src.models.radar import train_radar_pipeline as t; print(t(epochs=40))"` | Harish | `models/radar_float.keras`, `results/scores_radar_float.csv` |
| 6 | `python scripts/quantize_and_score.py` (needs 3 to 5 pulled into one checkout) | Harish | `models/*_int8.tflite`, `results/scores_*_int8.csv`, `results/memory_table.csv` |
| 7 | `python -m src.calibration.calibrate` | Agnel | `results/calibration.json` |
| 8 | `python scripts/merge_scores.py` | Madhavan | `data/tiles.csv`, `data/tiles_all.csv` |
| 9 | `python scripts/evaluate.py` | Madhavan | `results/headline.png`, `summary.csv`, `dial.csv`, `ablations.png` |
| 10 | `python -m uvicorn backend.api:app --port 8000` | everyone | the website at http://localhost:8000 (placeholder mode until step 9 has real inputs) |

`data/processed/` is not in git, so each person builds it locally (step 2) or receives the NPZ files from Agnel directly.

## Layout

- `scripts/`: data statistics, fake-data generation, baselines
- `src/`: shared code (`constants.py`, `scheduler/`)
- `data/`: tile CSVs (`data/fire/` for the forest-fire hazard); `data/raw/` is git-ignored, never commit it
- `models/`: trained model files (committed, small)
- `results/`: generated CSVs and plots

## Working rules

One branch per owner (`<name>/<topic>`). Only Madhavan merges to `main`. Never commit raw data, `.tif` files or `.tflite` files. Never claim more than `results/` shows. See the claims policy in CONTEXT.md.

## Credit

Contains modified Copernicus Sentinel data. Dataset: Bonafilia et al., Sen1Floods11, CVPR Workshops 2020 (github.com/cloudtostreet/Sen1Floods11). Its licence is unconfirmed.
