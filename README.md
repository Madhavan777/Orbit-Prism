# ORBIT-PRISM

A TinyML triage layer that decides what a satellite sends first. You set the miss rate; we minimise the bytes, even through cloud.

Conscientia 2k26 / HackOrbital 2026, PS 1.4. See [CONTEXT.md](CONTEXT.md) for the project context and the tile-record CSV schema.

Layout:
- `scripts/`: data statistics, fake-data generation, baselines
- `data/raw/`: Sen1Floods11 download (git-ignored, never commit)
- `results/`: generated CSVs and plots
