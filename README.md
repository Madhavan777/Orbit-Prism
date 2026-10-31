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
python scripts/real_stats.py        # needs data/raw labels; writes results/real_water_stats.csv
python scripts/make_fake_data.py    # writes data/fake_tiles.csv (synthetic, see data/README.md)
python scripts/baselines.py         # baselines on the Bolivia held-out event
```

`results/real_water_stats.csv` is committed, so you only need the raw download for `real_stats.py` and for anything using imagery.

## Layout

- `scripts/`: data statistics, fake-data generation, baselines
- `src/`: shared code (`constants.py`, `scheduler/`)
- `data/`: tile CSVs; `data/raw/` is git-ignored, never commit it
- `results/`: generated CSVs and plots

## Working rules

One branch per owner (`<name>/<topic>`). Only Madhavan merges to `main`. Never commit raw data, `.tif` files or `.tflite` files. Never claim more than `results/` shows. See the claims policy in CONTEXT.md.

## Credit

Contains modified Copernicus Sentinel data. Dataset: Bonafilia et al., Sen1Floods11, CVPR Workshops 2020 (github.com/cloudtostreet/Sen1Floods11). Its licence is unconfirmed.
