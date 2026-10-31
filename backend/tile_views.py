"""Tile pictures and cloud redraws for the website, read from the real Sen1Floods11 chips.

Pictures (display only, never model input):
    clear    natural true-colour satellite RGB with atmospheric haze correction
    optical  the same view under the simulated cloud, painted with the structure of REAL clouds found in the dataset
             (Paraguay and Sri-Lanka chips): billowy edges, shading and a cast shadow. Display only: the model is
             given the cloud as fully occluded (pure white) pixels, see src/data/cloud_simulation.py
    radar    calibrated high-contrast SAR backscatter (water is dark, terrain is textured)
    label    luminous semi-transparent ground truth flood overlay on the real satellite terrain

Fallback hierarchy:
    1. Pre-rendered web-ready cache in data/imagery/{chip}_{kind}.webp (instant response)
    2. Local raw GeoTIFFs in data/raw/... (rendered on the fly and cached)

redraw_clouds re-simulates clouds with another seed and re-runs the optical model on new pixels.
Seed 42 reproduces data/tiles_all.csv.
"""
from __future__ import annotations

import functools
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter, shift

from src.constants import CLOUD_SWITCH, TILE
from src.data.cloud_simulation import blend_clouds_optical, derive_seed_from_chip_id, opaque_mask, simulate_chip_cloud_mask
from src.data.preprocess import preprocess_optical_chip

logger = logging.getLogger("tile_views")
HEADLINE_CLOUD_SEED = 42
# Chips of the dataset that contain genuine cumulus and thin cloud; their cloud structure is the paint for the display.
CLOUD_SOURCE_CHIPS = ("Paraguay_198534", "Paraguay_12870", "Sri-Lanka_135713", "Sri-Lanka_579082")


def _hand_dir(root: Path) -> Path:
    return root / "data" / "raw" / "data" / "flood_events" / "HandLabeled"


def _cache_dir(root: Path) -> Path:
    return root / "data" / "imagery"


def has_imagery(root: Path, chip: str) -> bool:
    """True if pre-rendered cache or raw GeoTIFFs exist for this chip."""
    if (_cache_dir(root) / f"{chip}_optical.webp").exists():
        return True
    d = _hand_dir(root)
    return (d / "S2Hand" / f"{chip}_S2Hand.tif").exists() and (d / "S1Hand" / f"{chip}_S1Hand.tif").exists()


@functools.lru_cache(maxsize=1)
def _real_cloud_structure(root_str: str):
    """(billow, fine, mid) maps, 512x512 and zero-mean unit-variance, measured on real clouds in the dataset.

    billow is the cloud's own opacity pattern (blue-band reflectance, lumps 3 to 30 px across); fine and mid are the
    brightness texture of the cloud itself at two scales, weighted by real cloud opacity so ground detail seen through
    thin cloud does not leak in. Returns [] when none of the source chips is on disk.
    """
    def band(x, lo, hi):
        b = gaussian_filter(x, lo) - gaussian_filter(x, hi)
        return b / (b.std() + 1e-6)

    out = []
    for c in CLOUD_SOURCE_CHIPS:
        path = _hand_dir(Path(root_str)) / "S2Hand" / f"{c}_S2Hand.tif"
        if not path.exists():
            continue
        a = np.nan_to_num(tifffile.imread(path)).astype(np.float32)
        opac = np.clip((a[1] - 700.0) / 2100.0, 0.0, 1.0)           # blue reflectance: clouds are bright, ground is dark
        lum = np.clip(a[[3, 2, 1]].mean(0) / 3000.0, 0.0, 1.0) ** 0.7
        w = gaussian_filter((opac > 0.6).astype(np.float32), 3)
        out.append((band(opac, 2.5, 30), band(lum, 1.2, 6) * w, band(lum, 6, 30) * w))
    return out


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _paint_clouds(root_str: str, chip: str, rgb_clear: np.ndarray) -> np.ndarray:
    """Cloud-covered picture: the simulated mask decides WHERE, real cloud structure decides HOW it looks."""
    alpha = simulate_chip_cloud_mask(chip, seed=derive_seed_from_chip_id(chip, HEADLINE_CLOUD_SEED))
    mask = (alpha >= 0.5).astype(np.float32)
    lib = _real_cloud_structure(root_str)
    if not lib:                                    # no source chips on disk: plain soft white
        op = gaussian_filter(mask, 3.0)[..., None]
        return rgb_clear * (1.0 - op) + np.array([0.95, 0.965, 0.985], dtype=np.float32) * op
    rng = np.random.default_rng(int(hashlib.sha256(chip.encode()).hexdigest()[:8], 16))
    billow, fine, mid = lib[int(rng.integers(len(lib)))]
    k = int(rng.integers(4))
    billow, fine, mid = np.rot90(billow, k), np.rot90(fine, k), np.rot90(mid, k)
    if rng.random() < 0.5:
        billow, fine, mid = billow[:, ::-1], fine[:, ::-1], mid[:, ::-1]
    band = np.clip(1.0 - np.abs(2.0 * gaussian_filter(mask, 16) - 1.0), 0.0, 1.0)     # strongest near cloud edges
    op = _smoothstep(gaussian_filter(mask, 5) + 0.22 * billow * (0.4 + band), 0.18, 0.82)
    op = np.maximum(op, _smoothstep(gaussian_filter(mask, 9), 0.82, 0.95))          # solid core: nothing shows through
    body = gaussian_filter(op, 12)[..., None]
    cloud = np.array([0.965, 0.975, 0.99], dtype=np.float32) * (0.80 + 0.20 * body)
    cloud = cloud + 0.075 * mid[..., None] + 0.045 * fine[..., None]
    cloud = np.clip(cloud - 0.10 * (1.0 - body) * np.array([0.6, 0.45, 0.25], dtype=np.float32), 0.0, 1.0)
    shadow = gaussian_filter(shift(op, (12, -12), order=1, mode="constant"), 6) * (1.0 - op)   # cast shadow on the ground
    ground = rgb_clear * (1.0 - 0.45 * shadow[..., None])
    return ground * (1.0 - op[..., None]) + cloud * op[..., None]


def _render_chip_from_raw(root: Path, chip: str, kind: str) -> Optional[Image.Image]:
    """Render high-definition, realistic satellite imagery directly from raw Sen1Floods11 data."""
    d = _hand_dir(root)
    s2_p = d / "S2Hand" / f"{chip}_S2Hand.tif"
    if not s2_p.exists():
        return None

    if kind == "radar" and not (d / "S1Hand" / f"{chip}_S1Hand.tif").exists():
        return None

    s2 = np.nan_to_num(tifffile.imread(s2_p)).astype(np.float32)
    # Sentinel-2 Bands: B4=3 (Red), B3=2 (Green), B2=1 (Blue), B8=7 (NIR)
    r = s2[3]
    g = s2[2]
    b = s2[1]
    nir = s2[7]
    valid = (nir > 0) | (r > 0) | (g > 0)

    def stretch_band(band: np.ndarray, valid_mask: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
        v = band[valid_mask]
        if v.size == 0:
            return np.zeros_like(band)
        lo, hi = np.percentile(v, [p_lo, p_hi])
        return np.clip((band - lo) / max(hi - lo, 1e-4), 0.0, 1.0)

    # 1. Natural True-Colour Optical (Rayleigh atmospheric haze correction)
    r_n = stretch_band(r, valid, 2.0, 98.0)
    g_n = stretch_band(g, valid, 2.0, 98.0)
    b_n = stretch_band(b, valid, 2.0, 98.0) * 0.95
    rgb_clear = np.stack([r_n, g_n, b_n], axis=-1)
    rgb_clear = np.clip(rgb_clear ** 0.72 * 1.10, 0.0, 1.0)
    rgb_clear[~valid] = [0.06, 0.08, 0.11]

    if kind == "clear":
        return Image.fromarray((rgb_clear * 255).astype(np.uint8))

    if kind == "optical":
        optical = np.clip(_paint_clouds(str(root), chip, rgb_clear), 0.0, 1.0)
        optical[~valid] = [0.06, 0.08, 0.11]
        return Image.fromarray((optical * 255).astype(np.uint8))

    if kind == "radar":
        s1_p = d / "S1Hand" / f"{chip}_S1Hand.tif"
        if not s1_p.exists():
            return None
        s1 = tifffile.imread(s1_p).astype(np.float32)
        vv, vh = s1[0], s1[1]
        val_sar = np.isfinite(vv) & np.isfinite(vh)
        vv_c = np.nan_to_num(vv, nan=-35.0)
        vh_c = np.nan_to_num(vh, nan=-40.0)
        vv_norm = np.clip((vv_c - (-25.0)) / 20.0, 0.0, 1.0)
        vh_norm = np.clip((vh_c - (-32.0)) / 22.0, 0.0, 1.0)
        sar = np.clip((0.65 * vv_norm + 0.35 * vh_norm) ** 1.15, 0.0, 1.0)
        rgb_radar = np.stack([sar * 0.88, sar * 0.96, sar], axis=-1)
        rgb_radar[~val_sar] = [0.05, 0.07, 0.10]
        return Image.fromarray((rgb_radar * 255).astype(np.uint8))

    if kind == "label":
        lab_p = d / "LabelHand" / f"{chip}_LabelHand.tif"
        jrc_p = d / "JRCWaterHand" / f"{chip}_JRCWaterHand.tif"
        if not lab_p.exists() or not jrc_p.exists():
            return None
        lab = tifffile.imread(lab_p)
        jrc = (tifffile.imread(jrc_p) == 1)

        out = (rgb_clear * 255).astype(np.float32)
        is_flood = (lab == 1) & ~jrc
        is_perm = (lab == 1) & jrc
        nodata = lab == -1

        # Luminous semi-transparent flood water overlay over satellite terrain
        out[is_flood] = 0.25 * out[is_flood] + 0.75 * np.array([0, 255, 120], dtype=np.float32)
        out[is_perm] = 0.30 * out[is_perm] + 0.70 * np.array([40, 140, 255], dtype=np.float32)
        yy, xx = np.indices(lab.shape)
        out[nodata] = [18, 22, 28]
        out[nodata & (((yy + xx) // 6) % 2 == 0)] = [32, 38, 48]
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    return None


@functools.lru_cache(maxsize=128)
def _get_chip_image(root_str: str, chip: str, kind: str) -> Optional[Image.Image]:
    """Load chip image from pre-rendered cache or render from raw datasets."""
    root = Path(root_str)
    cache_file = _cache_dir(root) / f"{chip}_{kind}.webp"
    if cache_file.exists():
        try:
            return Image.open(cache_file).convert("RGB")
        except Exception:
            pass

    # Render from raw GeoTIFFs if available on disk
    img = _render_chip_from_raw(root, chip, kind)
    if img is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            img.save(cache_file, format="WEBP", quality=82)
        except Exception:
            pass
    return img


def tile_view(root: Path, chip: str, row: int, col: int, kind: str, scale: int) -> Optional[Image.Image]:
    """The high-definition picture of one tile (128*scale x 128*scale), or None when not available."""
    img = _get_chip_image(str(root), chip, kind)
    if img is None:
        return None

    # Crop the 128x128 tile from the 512x512 chip: (left, upper, right, lower)
    tile = img.crop((col * TILE, row * TILE, (col + 1) * TILE, (row + 1) * TILE))
    resample_filter = Image.NEAREST if kind == "label" else Image.BICUBIC
    return tile.resize((TILE * scale, TILE * scale), resample_filter)


@functools.lru_cache(maxsize=48)
def _s2(root: str, chip: str) -> np.ndarray:
    return np.nan_to_num(tifffile.imread(Path(root) / "data/raw/data/flood_events/HandLabeled/S2Hand" / f"{chip}_S2Hand.tif"))


@functools.lru_cache(maxsize=256)
def _clean_optical(root: str, chip: str) -> np.ndarray:
    return preprocess_optical_chip(_s2(root, chip))


def _crop(arr: np.ndarray, row: int, col: int) -> np.ndarray:
    return arr[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE]


def _cloud_alpha(chip: str, seed: int = HEADLINE_CLOUD_SEED) -> np.ndarray:
    return simulate_chip_cloud_mask(chip, seed=derive_seed_from_chip_id(chip, seed))


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
            p_opt.append((idx, p))
            e_opt.append((idx, e))
            ndwi.append((idx, float(nd)))
    out["cloud_frac"] = pd.Series(dict(cloud))
    out["p_optical"], out["extent_optical"] = pd.Series(dict(p_opt)), pd.Series(dict(e_opt))
    out["p_ndwi"] = pd.Series(dict(ndwi))
    radar = out["cloud_frac"] >= CLOUD_SWITCH
    out["sensor_used"] = np.where(radar, "radar", "optical")
    out["p_flood"] = out["p_radar"].where(radar, out["p_optical"])
    out["extent"] = out["extent_radar"].where(radar, out["extent_optical"])
    return out
