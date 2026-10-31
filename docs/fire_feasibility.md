# Forest fire as a second hazard: feasibility memo

Status: **fire is not loaded and no fire number may appear as a result.** This memo says what is verified, what is not, and what it would cost to load fire. Facts come from `data/fire/satellite_data.csv` (all 73 rows) and from part 5 of 5 of the dataset (9 of 73 areas), measured with `scripts/fire_stats.py`.

## 1. Recommendation

Keep fire as a **stated extension, not a result**, unless the team can spend the time below. Floods are fully measured; the fire story has three open links (labels, radar, selectivity). The honest slide line is: "the same pipeline applies; fire statistics are partial and the radar fallback for fire is untested."

## 2. What is verified

| Item | Value | Source |
|---|---|---|
| Dataset | Satellite Burned Area Dataset (Politecnico di Torino), Zenodo 10.5281/zenodo.6597139, CC BY 4.0 | Zenodo record |
| Size | 14.0 GB in 5 zips; 73 areas from 27 Copernicus EMS activations | CSV, record |
| Years | 2017: 60 areas, 2018: 8, 2019: 5 (by activation date) | CSV |
| Folds | 7 colours; each keeps whole activations together | CSV |
| Sensors | Sentinel-2 (13-channel float32) and Sentinel-1 (4-channel float32) | part 5 inspected |
| Dates | Only 50 of 73 rows list a pre-fire date, 53 a post-fire date. Use the dates in the file names | CSV, part 5 |

Folds (areas / activations): purple 8/4, coral 12/3, pink 14/5, grey 9/4, cyan 11/5, lime 10/1, magenta 9/5. Lime is one activation (EMSR213) with 10 areas, so holding it out tests a single fire event.

## 3. What part 5 shows (partial: 9 of 73 areas, 1,247 tiles)

- Share of tiles with `burn_frac >= 0.30`: **52%** (floods: 8%). Share with `burn_frac == 0`: 41%.
- It is driven by two areas: EMSR298 (99% critical) and EMSR365 (71% critical, 575 tiles, almost half of all tiles here). The other areas range from 5% to 34%.
- Part 5 covers only the folds coral, purple, grey, pink and magenta. It has **no cyan and no lime tiles**, so no held-out candidate can be checked from what is downloaded.

These are not dataset statistics. The `fire_stats.py` run says so itself.

## 4. Why fire is harder than flood

1. **Triage is less selective.** When half the tiles are critical, a prioritiser has little to reorder. The 0.30 threshold likely needs rethinking, and every fire result must show prevalence next to it.
2. **Labels are annotated regions, not a full map.** Mask codes are Copernicus damage grades that differ per area (32, 64, 128, 192, 255), so they are collapsed to burned = mask > 0. In large areas the annotated region is mostly burned, which inflates `burn_frac`.
3. **The radar fallback is a hypothesis.** Radar sees water through cloud because water is dark in SAR. Burn scars change backscatter less clearly, and smoke and flame are not what SAR measures. Sentinel-1 channel meaning is undocumented (do not assume VV/VH order) and channel 3 has up to 16% NaN pixels in some files. The radar claim for fire must be measured before it is stated.
4. **Clouds are real for fire.** Some areas ship a Sentinel-2 cloud mask. That differs from the simulated clouds used for floods, so the two cannot share a cloud-handling sentence on a slide.
5. **Grid size differs.** Rasters are 531x647 to 3207x2995 px, giving 20 to 575 tiles per area, not a fixed 4x4. `row` and `col` run to each area's own count.

## 5. What loading fire would take

1. Download parts 1 to 4 (about 12.5 GB; part 5 is already local). Needs the team's go-ahead for disk and time.
2. Run `python -I scripts/fire_stats.py` until it reports all 73 areas, then fix the critical threshold and prevalence.
3. Decide the held-out fold (whole folds only, never tiles). Candidates: grey or cyan (4 to 5 activations, mid-sized). Avoid lime as the sole test, since it is one event.
4. Convert to the 11-column schema with `burn_frac` in place of `water_frac` (`data/fire/`), and add a per-hazard scaled pass budget.
5. Build fire tiles (Sentinel-2 bands to the optical 4-band layout; Sentinel-1 bands chosen after checking what they are), train, quantize, score, calibrate.
6. Measure whether radar helps under cloud for fire. Only then write the fallback claim.

Rough effort: steps 1 to 4 are a few hours, mostly download. Steps 5 and 6 repeat the flood pipeline for a second model pair and are the real cost.

## 6. Agnel's primary sources (FIRMS, FireCCI51)

Per CONTEXT.md these are the primary label sources. FIRMS VIIRS gives point detections, not burned-area masks. FireCCI51 is about 250 m, so a 1.3 km tile holds roughly 5x5 label pixels: usable for a rough tile-level burned share, too coarse to be tile ground truth on its own. Torino masks are 10 m and are the fallback labels. Nothing from these sources is loaded yet.

## 7. Claims allowed today

- Allowed: fire is a second hazard under the same pipeline; Torino dataset facts in section 2; the partial prevalence in section 3 labelled "partial, 9 of 73 areas".
- Not allowed: any fire recall, miss-rate, bytes-saved or radar-under-cloud figure.
