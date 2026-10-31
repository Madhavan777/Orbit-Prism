"""Tile pictures and cloud redraws for the website, read from the raw Sen1Floods11 chips.

Pictures (display only, never model input). Each band is stretched over its own chip, so tiles of one chip
match and water stands out:
    clear    infrared false colour (SWIR B11, NIR B8, red B4): water is dark blue, vegetation green
    optical  the same picture with the simulated opaque cloud painted on: what the optical model faces
    radar    VV and VH backscatter, lightly smoothed: smooth water is dark, even under cloud
    label    the clear picture dimmed, with hand-labelled flood in lime, permanent water in blue, no data hatched

redraw_clouds re-simulates the clouds with another seed and re-runs the optical model on the new pixels. Radar
scores are reused because simulated clouds touch optical bands only. Seed 42 reproduces data/tiles_all.csv.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter

from src.constants import CLOUD_SWITCH, TILE
from src.data.cloud_simulation import blend_clouds_optical, derive_seed_from_chip_id, opaque_mask, simulate_chip_cloud_mask
from src.data.preprocess import preprocess_optical_chip

HEADLINE_CLOUD_SEED = 42
LIME, BLUE = np.array([198, 255, 0]), np.array([57, 135, 229])
S2_SWIR, S2_NIR, S2_RED = 11, 7, 3     # positions in the 13-band Sentinel-2 chip


def _hand_dir(root: Path) -> Path:
    return root / "data" / "raw" / "data" / "flood_events" / "HandLabeled"


def has_imagery(root: Path, chip: str) -> bool:
    d = _hand_dir(root)
    return (d / "S2Hand" / f"{chip}_S2Hand.tif").exists() and (d / "S1Hand" / f"{chip}_S1Hand.tif").exists()


def _stretch(band: np.ndarray, valid: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Percentile stretch to [0, 1] over the chip's valid pixels."""
    vals = band[valid]
    if vals.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(vals, [low, high])
    return np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


@functools.lru_cache(maxsize=48)
def _s2(root: str, chip: str) -> np.ndarray:
    return np.nan_to_num(tifffile.imread(Path(root) / "data/raw/data/flood_events/HandLabeled/S2Hand" / f"{chip}_S2Hand.tif"))


@functools.lru_cache(maxsize=48)
def _clear_rgb(root: str, chip: str) -> np.ndarray:
    s2 = _s2(root, chip)
    valid = s2[S2_NIR] > 0
    rgb = np.stack([_stretch(s2[b], valid) for b in (S2_SWIR, S2_NIR, S2_RED)], axis=-1)
    return np.clip(rgb ** 0.7, 0, 1)    # lift dark water so a mostly flooded chip is still readable


@functools.lru_cache(maxsize=48)
def _radar_rgb(root: str, chip: str) -> np.ndarray:
    s1 = tifffile.imread(Path(root) / "data/raw/data/flood_events/HandLabeled/S1Hand" / f"{chip}_S1Hand.tif")
    vv, vh = s1[0].astype(np.float32), s1[1].astype(np.float32)
    valid = np.isfinite(vv) & np.isfinite(vh)
    vv, vh = gaussian_filter(np.nan_to_num(vv, nan=-30.0), 1.0), gaussian_filter(np.nan_to_num(vh, nan=-30.0), 1.0)
    grey = 0.5 * _stretch(vv, valid, 1, 99) + 0.5 * _stretch(vh, valid, 1, 99)
    grey = np.clip((grey - 0.08) / 0.84, 0, 1) ** 1.25     # keep water dark and land bright
    return np.stack([grey * 0.92, grey * 0.97, grey], axis=-1)


@functools.lru_cache(maxsize=48)
def _label(root: str, chip: str):
    d = Path(root) / "data/raw/data/flood_events/HandLabeled"
    return tifffile.imread(d / "LabelHand" / f"{chip}_LabelHand.tif"), tifffile.imread(d / "JRCWaterHand" / f"{chip}_JRCWaterHand.tif") == 1


def _cloud_alpha(chip: str, seed: int = HEADLINE_CLOUD_SEED) -> np.ndarray:
    return simulate_chip_cloud_mask(chip, seed=derive_seed_from_chip_id(chip, seed))


def _crop(arr: np.ndarray, row: int, col: int) -> np.ndarray:
    return arr[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE]


def tile_view(root: Path, chip: str, row: int, col: int, kind: str, scale: int) -> Optional[Image.Image]:
    """The picture of one tile, or None when the raw chip is not on disk."""
    key = str(root)
    if kind in ("clear", "optical", "label") and not (_hand_dir(root) / "S2Hand" / f"{chip}_S2Hand.tif").exists():
        return None
    if kind == "radar" and not has_imagery(root, chip):
        return None
    if kind == "radar":
        rgb = _crop(_radar_rgb(key, chip), row, col)
    else:
        rgb = _crop(_clear_rgb(key, chip), row, col).copy()
        if kind == "optical":
            rgb[_crop(_cloud_alpha(chip), row, col) >= 0.5] = 1.0
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    if kind == "label":
        label, perm = _label(key, chip)
        lab, jrc = _crop(label, row, col), _crop(perm, row, col)
        out = (np.asarray(img, dtype=np.float32) * 0.42)
        out[(lab == 1) & ~jrc] = LIME
        out[(lab == 1) & jrc] = BLUE
        yy, xx = np.indices(lab.shape)
        nodata = lab == -1
        out[nodata] = 18
        out[nodata & (((yy + xx) // 6) % 2 == 0)] = 48
        return Image.fromarray(out.astype(np.uint8)).resize((TILE * scale, TILE * scale), Image.NEAREST)
    return img.resize((TILE * scale, TILE * scale), Image.BICUBIC)


@functools.lru_cache(maxsize=256)
def _clean_optical(root: str, chip: str) -> np.ndarray:
    return preprocess_optical_chip(_s2(root, chip))


def redraw_clouds(root: Path, sel: pd.DataFrame, seed: int, predict_optical, ndwi_share) -> pd.DataFrame:
    """A copy of `sel` under newly simulated clouds: cloud fraction, routing, optical and colour-index scores change."""
    out = sel.copy()
    missing = [c for c in out["chip"].unique() if not (_hand_dir(root) / "S2Hand" / f"{c}_S2Hand.tif").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} chips have no imagery on disk; run python scripts/download_flood.py --imagery")
    cloud, p_opt, e_opt, ndwi = [], [], [], []
    for chip, g in out.groupby("chip", sort=False):
        alpha = _cloud_alpha(chip, seed)
        cloudy = blend_clouds_optical(_clean_optical(str(root), chip), opaque_mask(alpha))
        tiles = np.stack([_crop(cloudy, int(r), int(c)) for r, c in zip(g["row"], g["col"])])
        ndwi_vals = ndwi_share(tiles)
        for tile, r, c, nd, idx in zip(tiles, g["row"], g["col"], ndwi_vals, g.index):
            p, e, _ = predict_optical(tile)
            cloud.append((idx, float((_crop(alpha, int(r), int(c)) >= 0.5).mean())))
            p_opt.append((idx, p)); e_opt.append((idx, e)); ndwi.append((idx, float(nd)))
    out["cloud_frac"] = pd.Series(dict(cloud))
    out["p_optical"], out["extent_optical"] = pd.Series(dict(p_opt)), pd.Series(dict(e_opt))
    out["p_ndwi"] = pd.Series(dict(ndwi))
    radar = out["cloud_frac"] >= CLOUD_SWITCH
    out["sensor_used"] = np.where(radar, "radar", "optical")
    out["p_flood"] = out["p_radar"].where(radar, out["p_optical"])
    out["extent"] = out["extent_radar"].where(radar, out["extent_optical"])
    return out
