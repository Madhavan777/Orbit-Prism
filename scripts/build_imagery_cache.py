"""Pre-render high-definition realistic satellite imagery cache for all 11 flood events.

Extracts real Sentinel-2 optical and Sentinel-1 SAR tiles from Sen1Floods11 raw chips
and generates realistic, contrast-enhanced, clearly defined imagery:
  1. clear:   natural true-colour satellite RGB with atmospheric haze correction
  2. optical: realistic volumetric cloud-blended optical camera view
  3. radar:   calibrated high-contrast SAR backscatter (water is dark, terrain is textured)
  4. label:   semi-transparent luminous ground truth flood overlay on satellite terrain

Saves to data/imagery/{chip}_{kind}.webp (quality 82, ~20 KB per chip).
"""
import io
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.cloud_simulation import simulate_chip_cloud_mask, derive_seed_from_chip_id

RAW = ROOT / "data" / "raw" / "data" / "flood_events" / "HandLabeled"
OUT = ROOT / "data" / "imagery"


def render_chip(chip: str, s2_path: Path, s1_path: Path, lab_path: Path, jrc_path: Path):
    s2 = tifffile.imread(s2_path).astype(np.float32)
    s1 = tifffile.imread(s1_path).astype(np.float32) if s1_path.exists() else None
    lab = tifffile.imread(lab_path) if lab_path.exists() else None
    jrc = (tifffile.imread(jrc_path) == 1) if jrc_path.exists() else None

    # Sentinel-2 Bands:
    # 1: Blue, 2: Green, 3: Red, 7: NIR
    r = s2[3]
    g = s2[2]
    b = s2[1]
    nir = s2[7]
    valid = (nir > 0) | (r > 0) | (g > 0)

    # 1. Natural True-Color Clear RGB
    def stretch_band(band, valid_mask, p_lo=2, p_hi=98):
        v = band[valid_mask]
        if v.size == 0:
            return np.zeros_like(band)
        lo, hi = np.percentile(v, [p_lo, p_hi])
        return np.clip((band - lo) / max(hi - lo, 1e-4), 0, 1)

    r_n = stretch_band(r, valid, 2, 98)
    g_n = stretch_band(g, valid, 2, 98)
    b_n = stretch_band(b, valid, 2, 98) * 0.95  # Rayleigh scattering correction

    rgb_clear = np.stack([r_n, g_n, b_n], axis=-1)
    rgb_clear = np.clip(rgb_clear ** 0.72 * 1.10, 0, 1)
    rgb_clear[~valid] = [0.06, 0.08, 0.11]
    img_clear = Image.fromarray((rgb_clear * 255).astype(np.uint8))

    # 2. Volumetric Cloud-Blended Optical
    alpha = simulate_chip_cloud_mask(chip, seed=derive_seed_from_chip_id(chip, 42))
    cloud_density = np.clip(alpha * 1.35 - 0.05, 0, 1) ** 0.82
    cloud_color = np.array([244, 248, 253], dtype=np.float32) / 255.0
    optical = rgb_clear * (1.0 - cloud_density[..., None]) + cloud_color * cloud_density[..., None]
    optical = np.clip(optical, 0, 1)
    optical[~valid] = [0.06, 0.08, 0.11]
    img_optical = Image.fromarray((optical * 255).astype(np.uint8))

    # 3. High-Contrast Crisp SAR Radar
    if s1 is not None:
        vv, vh = s1[0], s1[1]
        val_sar = np.isfinite(vv) & np.isfinite(vh)
        vv_c = np.nan_to_num(vv, nan=-35.0)
        vh_c = np.nan_to_num(vh, nan=-40.0)
        vv_norm = np.clip((vv_c - (-25.0)) / 20.0, 0, 1)
        vh_norm = np.clip((vh_c - (-32.0)) / 22.0, 0, 1)
        sar = np.clip((0.65 * vv_norm + 0.35 * vh_norm) ** 1.15, 0, 1)
        rgb_radar = np.stack([sar * 0.88, sar * 0.96, sar], axis=-1)
        rgb_radar[~val_sar] = [0.05, 0.07, 0.10]
        img_radar = Image.fromarray((rgb_radar * 255).astype(np.uint8))
    else:
        img_radar = img_clear

    # 4. Realistic Ground Truth Flood Overlay
    if lab is not None:
        out = (rgb_clear * 255).astype(np.float32)
        is_flood = (lab == 1) & (~jrc if jrc is not None else np.ones_like(lab, dtype=bool))
        is_perm = (lab == 1) & (jrc if jrc is not None else np.zeros_like(lab, dtype=bool))
        nodata = lab == -1

        # Luminous semi-transparent overlay
        out[is_flood] = 0.25 * out[is_flood] + 0.75 * np.array([0, 255, 120], dtype=np.float32)
        out[is_perm] = 0.30 * out[is_perm] + 0.70 * np.array([40, 140, 255], dtype=np.float32)
        yy, xx = np.indices(lab.shape)
        out[nodata] = [18, 22, 28]
        out[nodata & (((yy + xx) // 6) % 2 == 0)] = [32, 38, 48]
        img_label = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    else:
        img_label = img_clear

    return img_clear, img_optical, img_radar, img_label


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "tiles_all.csv")

    chips_to_render = set()
    for event, grp in df.groupby("event"):
        crit_chips = grp[grp["critical"] == 1]["chip"].unique().tolist()
        chips_to_render.update(crit_chips)
        chip_stats = grp.groupby("chip").agg(critical=("critical", "sum"), mean_cloud=("cloud_frac", "mean")).reset_index()
        candidates = chip_stats[(chip_stats["critical"] > 0) & (chip_stats["mean_cloud"] > 0.45)]
        if len(candidates) == 0:
            candidates = chip_stats[chip_stats["critical"] > 0]
        best_chip = candidates.iloc[0]["chip"] if len(candidates) > 0 else chip_stats.iloc[0]["chip"]
        chips_to_render.add(best_chip)
        for c in chip_stats.sort_values(by="critical", ascending=False)["chip"].head(3):
            chips_to_render.add(c)

    print(f"Generating realistic imagery for {len(chips_to_render)} key chips across all 11 events...")
    t0 = time.time()
    total_bytes = 0
    count = 0

    for chip in sorted(chips_to_render):
        s2_p = RAW / "S2Hand" / f"{chip}_S2Hand.tif"
        if not s2_p.exists():
            continue
        s1_p = RAW / "S1Hand" / f"{chip}_S1Hand.tif"
        lab_p = RAW / "LabelHand" / f"{chip}_LabelHand.tif"
        jrc_p = RAW / "JRCWaterHand" / f"{chip}_JRCWaterHand.tif"

        clear_img, opt_img, rad_img, lab_img = render_chip(chip, s2_p, s1_p, lab_p, jrc_p)

        for kind, im in [("clear", clear_img), ("optical", opt_img), ("radar", rad_img), ("label", lab_img)]:
            out_p = OUT / f"{chip}_{kind}.webp"
            im.save(out_p, format="WEBP", quality=82)
            total_bytes += out_p.stat().st_size
            count += 1

    dt = time.time() - t0
    print(f"Done! Created {count} images in {OUT} ({total_bytes / (1024*1024):.2f} MB total in {dt:.2f}s)")


if __name__ == "__main__":
    main()
