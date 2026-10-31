# Merge review, 2026-10-31 (Madhavan)

Reviewed: `Agnel_branch` (same commits as `agnell-ship-it`), `harish/radar-compress`, `shenbaga/optical`.
Method: file-by-file read against CONTEXT.md, byte-level checks (BOM, CRLF, size, forbidden files), banned-claims grep, and each branch's own tests run in a clean environment.

**Nothing has been merged.** All three branches fail the merge gate in section "Blockers". There is useful work in each one; the fixes below are mostly small. Tell me when a branch is ready and I will re-run the gate.

| Branch | Own tests | Gate | Main problem |
|---|---|---|---|
| Agnel | 15 pass, 1 fails (Windows temp-file lock, not logic) | FAIL | Synthetic scores sit in files that look real; fire data is not loaded |
| Harish | 11 pass, 1 fails (his cyclone audit fails on his own notes) | FAIL | Memory table is all PENDING and off-contract; claims folders that do not exist |
| Shenbaga | 13 pass | FAIL | Registry marks wildfire and cyclone "implemented" with no data or training; synthetic artifacts committed |

## Good, keep it
- Agnel: tiling, cloud simulation, event splits, validation and the calibration modules are well structured, and the code logs when it falls back to placeholders.
- Harish: peak memory is labelled ESTIMATED with the omitted costs listed. That is the right honesty standard.
- Shenbaga: the model fits the budget (61 KB float), the dry run is flagged as a dry run, and tests are clean.

## Blockers: Agnel
1. **`data/tiles.csv` is synthetic but named as the real merged file.** Its `p_flood` and `extent` equal `data/fake_tiles.csv` exactly. Do not commit it; it must be produced by `scripts/merge_scores.py` from real scores only. Make `merge_scores.py` fail (not fall back to fake data) unless `--allow-placeholder` is passed, and write `score_source` into the output.
2. **`results/calibration.json` was fitted on the synthetic placeholder** (`calibrate.py` falls back to `fake_tiles.csv`; no `scores_*_int8.csv` exists), and the JSON has no field saying so. Thresholds like 0.0159 are meaningless. Add `"score_source"`, and do not commit a calibration until it comes from real scores.
3. **Clouds disagree with the CSV.** `cloud_simulation.py` draws its own cover (uniform 0 to 0.9 per chip) and measures `cloud_frac` from the mask, but `optical_tiles.csv` copies `cloud_fraction` from the old synthetic `tiles_base.csv` (row 1: 0.9547 in both). Regenerate `data/tiles_base.csv` from your pipeline output so `cloud_frac` and `sensor_used` are the ones the pixels actually have, then redo everything that depends on `sensor_used` (calibration, splits).
4. **Holdout lost.** `optical_tiles.csv` and `radar_tiles.csv` map Bolivia into `Test` (1,609 = 1,378 + 231). Keep `holdout` as its own value, or add `event_id`-based filtering in the loader, so Bolivia can always be isolated. Use `valid` not `Validation`.
5. **Fire is not loaded.** Both tile files have only Flood and Normal rows; `burn_mask_path` is empty and `burn_fraction` is 0. `docs/datasets_reference.md` section 2 is a table of intentions, not data. Say "not loaded" in that doc.
6. **ECE 0.0 is in-sample.** Isotonic regression scored on its own fitting data always gives ECE about 0. Report it on the held-out split or drop it.
7. **`.gitignore` reverts my changes**: it removes the fire-stats ignore line and re-adds `*.tflite`, which blocks `models/`. Merge, do not overwrite.
8. **Docs**: remove the `file:///Users/agnelvishwakamalraj/...` links; section 6 of `agnel_calibration.md` shows invented example numbers, label them illustrative; `datasets_reference.md` says FireCCI51 comes from "MODIS / Sentinel-3 OLCI" (the product we checked is MODIS) and states that Sentinel-1 "detects ... canopy destruction through heavy smoke" as fact. For fire that is a hypothesis, not a result.
9. **Test hygiene**: `test_npz_container_and_validation` fails on Windows because the NPZ is still open when the temp folder is deleted. Close it (use a `with np.load(...)` block).

## Blockers: Harish
1. **`results/memory_table.csv` is off-contract and empty.** Required columns: `model, dtype, weights_bytes, peak_activation_bytes, tile_buffer_bytes, total_bytes, macs, recall, ece` (CONTEXT.md). Yours: `precision, weights_KB, ..., fits_4.2MB`, every value `PENDING`. Either use the contract columns or propose a change to CONTEXT.md, and fill real numbers before merge. Your notes quote 59.54 KB / 15.18 KB / 64 KB; put those in the table.
2. **Claims folders that do not exist.** `radar_notes.md` says forest-fire data is "Preserved (`data/raw/forest_fire/`)" and lists fire SAR as an active scope. No such folder or data exists in the repo. Correct it.
3. **Your cyclone audit fails on your own notes** (`test_15`): the notes contain the word it searches for. Also it will fail on any branch that contains cyclone code. See "Decision needed: cyclones".
4. **No trained radar model or scores yet**: there is no `scores_radar_int8.csv`. Merge the quantisation pipeline, but do not call the memory numbers results until they are measured on a trained model.

## Blockers: Shenbaga
1. **`src/disasters/registry.py` is wrong on status and facts.** Wildfire and Cyclone are marked `IMPLEMENTED`, but there is no training data, no training and no result for either (only an architecture and an NBR formula for wildfire). Sample counts like "5,000+ annotated wildfire patches" and "100,000+ storm-centered observations" are not backed by anything in the repo. Sen1Floods11 is listed as "CC BY-4.0" and its GitHub states no licence, so say "unconfirmed". It also says "4,831 tiles (512x512 downsampled to 128x128)": it is 4,831 chips, cut (not downsampled) into 128x128 tiles. Only flood should be anything above `PLANNED`, and even flood only becomes implemented when trained on real data.
2. **Synthetic artifacts committed as if real.** `results/scores_optical.csv` has 20 rows with ids `synthetic_tile_0000...`, p_flood about 0.488 (an untrained model). `models/optical_float.keras` came from the dry run. Remove both, or rename clearly (`*_DRYRUN_SYNTHETIC`) and keep them out of `results/`.
3. **Splits will not connect to Agnel's data.** `train.py` uses `cal` or `val`; the data uses `valid` (official mode) or `calibration` (event mode). With the real NPZ, the calibration set would be empty. Accept `valid`, fail loudly when empty, and assert that `holdout` never appears in train or calibration.
4. **Output names and keys.** Contract: `results/scores_optical_int8.csv` with `chip, row, col, p_flood, extent`. Yours: `scores_optical.csv` keyed by `tile_id`. Also these should be INT8 model scores, per the contract.
5. **Scope and wording.** "Multi-Disaster Detection and Intelligent Prioritisation System" with six hazards (earthquake, volcano, landslide) is not what PS 1.4 asks for and conflicts with our framing (TinyML triage layer, floods as the worked example, not a disaster-management system). Use "priority", not "risk". The scheduler's packet sizes (64 B alert, 25 to 100 KB imagery) conflict with our L0 32 B / L1 / L2 design; use `src/constants.py` and Akhilesh's scheduler.
6. **`requirements.txt` drops the pins** and replaces them with unpinned names. Keep the pinned versions and add the new packages with versions (tensorflow 2.21.0, keras 3.15.1, scikit-learn 1.9.1, pytest were what worked in my test run).

## Decision: cyclones are OUT (Madhavan, 2026-10-31)
Scope is floods and forest fires only (see "Scope" in CONTEXT.md). Consequences:
- **Shenbaga:** delete `src/disasters/cyclone.py`, and the cyclone, landslide, earthquake and volcano entries from the registry and their tests. Keep flood and wildfire only, with honest statuses. Rename or drop the "multi-disaster" framing.
- **Harish:** keep the exclusion, but fix `test_15` so it does not search documentation files (it fails on your own notes). Test code files only, or drop the test.
- **Agnel:** no change, except the reference doc must not list hazards outside scope.

## Order of work
1. Agnel: items 1 to 4 first (they change `sensor_used`, which everyone downstream depends on). Tell Shenbaga and Harish when `tiles_base.csv` is regenerated.
2. Shenbaga and Harish: fix the contract mismatches, then train on the regenerated data.
3. Everyone: rebase on `main`, run `python -m pytest tests -q`, then ask for a re-check.
