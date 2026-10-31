"""Synthetic cloud simulation for optical satellite imagery.

# SYNTHETIC CLOUDS: These clouds are simulated for experimentation and are NOT real cloud observations.

Generates reproducible soft cloud masks using Gaussian noise and Gaussian smoothing,
computes tile-level cloud fractions, and simulates optical occlusion by blending
towards bright white while leaving radar imagery unchanged.
"""

import hashlib
import zlib
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import gaussian_filter

from src.data.tiling import TILE_SIZE, GRID_SIZE, generate_tile_id

# Cloud switch threshold defined in CONTEXT.md and constants.py
CLOUD_SWITCH_THRESHOLD = 0.5
BRIGHT_WHITE_DN = 255.0


def derive_seed_from_chip_id(chip_id: str, base_salt: int = 42) -> int:
    """Derive a deterministic 32-bit unsigned integer seed from a chip ID string."""
    h = hashlib.sha256(f"{chip_id}_{base_salt}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def simulate_chip_cloud_mask(
    chip_id: str,
    target_cover: Optional[float] = None,
    sigma: float = 24.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a 512x512 continuous alpha mask in [0, 1] representing cloud opacity.

    # SYNTHETIC CLOUDS: These clouds are simulated for experimentation and are NOT real cloud observations.

    Args:
        chip_id: Chip identifier for reproducible seeding.
        target_cover: Desired fraction in [0, 0.9]. If None, drawn uniformly from [0, 0.9].
        sigma: Standard deviation for Gaussian smoothing kernel.
        seed: Optional explicit random seed.

    Returns:
        alpha: np.ndarray of shape (512, 512), dtype float32, in range [0, 1].
    """
    if seed is None:
        seed = derive_seed_from_chip_id(chip_id)

    rng = np.random.default_rng(seed)

    if target_cover is None:
        target_cover = float(rng.uniform(0.0, 0.9))
    else:
        target_cover = float(np.clip(target_cover, 0.0, 0.99))

    if target_cover <= 0.001:
        return np.zeros((512, 512), dtype=np.float32)

    # 1. Generate random Gaussian noise
    noise = rng.standard_normal((512, 512)).astype(np.float32)

    # 2. Apply Gaussian smoothing
    smooth = gaussian_filter(noise, sigma=sigma)

    # Standardize smooth field to zero mean, unit variance
    s_std = smooth.std()
    if s_std > 1e-6:
        smooth = (smooth - smooth.mean()) / s_std

    # Determine threshold so that fraction of pixels >= threshold equals target_cover
    # When smooth is above threshold, alpha ramps smoothly to 1.
    perc = 100.0 * (1.0 - target_cover)
    thresh = float(np.percentile(smooth, perc))

    # 4 & 5. Generate soft cloud boundaries with alpha in [0, 1]
    # Use sigmoid-like smooth transition centered around threshold
    transition_width = 0.5
    raw_alpha = 1.0 / (1.0 + np.exp(-(smooth - thresh) / transition_width))
    alpha = np.clip(raw_alpha, 0.0, 1.0).astype(np.float32)

    return alpha


def blend_clouds_optical(
    optical: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Blend optical image towards bright white based on cloud alpha mask.

    # SYNTHETIC CLOUDS: These clouds are simulated for experimentation and are NOT real cloud observations.

    Args:
        optical: uint8 array of shape (..., H, W, C) or (H, W, C), values in [0, 255].
        alpha: float array of shape (..., H, W) or (..., H, W, 1), values in [0, 1].

    Returns:
        blended: uint8 array of identical shape to optical.
    """
    if alpha.ndim == optical.ndim - 1:
        alpha = np.expand_dims(alpha, axis=-1)

    alpha = alpha.astype(np.float32)
    opt_float = optical.astype(np.float32)

    # Blend optical pixels towards bright white (255)
    blended = (1.0 - alpha) * opt_float + alpha * BRIGHT_WHITE_DN
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def compute_tile_cloud_metrics(
    tile_alpha: np.ndarray,
) -> Tuple[float, str]:
    """Compute cloud fraction and determined sensor for a single tile alpha mask.

    cloud_frac = fraction of tile pixels where alpha >= 0.5
    sensor_used = 'radar' if cloud_frac >= 0.5 else 'optical'
    """
    cloud_pixels = (tile_alpha >= 0.5).sum()
    total_pixels = tile_alpha.size
    cloud_frac = float(cloud_pixels / total_pixels) if total_pixels > 0 else 0.0
    sensor_used = "radar" if cloud_frac >= CLOUD_SWITCH_THRESHOLD else "optical"
    return cloud_frac, sensor_used


def opaque_mask(alpha: np.ndarray) -> np.ndarray:
    """Binary version of the soft alpha mask: 1.0 where alpha >= 0.5, else 0.0.

    The set of pixels is exactly the one cloud_frac counts, so a tile's cloud_frac and routing are the
    same whether clouds are drawn soft or opaque; only the pixel values inside the cloud change.
    """
    return (alpha >= 0.5).astype(np.float32)


def generate_chip_clouds(
    chip_id: str,
    optical_chip_hwc: np.ndarray,
    target_cover: Optional[float] = None,
    opaque: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, str]]]:
    """Simulate clouds for an entire 512x512 chip.

    # SYNTHETIC CLOUDS: These clouds are simulated for experimentation and are NOT real cloud observations.

    opaque=True (the project default, decided 2026-10-31): pixels inside the cloud mask (alpha >= 0.5)
    become pure white, so optical sees nothing there, as under dense real cloud. opaque=False is the
    earlier semi-transparent haze, kept as a sensitivity setting: it let optical models read through
    clouds (AUROC 0.93 on 75-100% cloud tiles against 0.79 when opaque).

    Returns:
        blended_chip: (512, 512, 4) uint8 optical chip with clouds blended.
        chip_alpha: (512, 512) float32 alpha mask actually used for blending.
        tile_stats: dict mapping tile_id to (cloud_frac, sensor_used); identical for both settings.
    """
    soft_alpha = simulate_chip_cloud_mask(chip_id, target_cover=target_cover)
    alpha = opaque_mask(soft_alpha) if opaque else soft_alpha
    blended_chip = blend_clouds_optical(optical_chip_hwc, alpha)

    tile_stats = {}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            tile_id = generate_tile_id(chip_id, r, c)
            sl = (slice(r * TILE_SIZE, (r + 1) * TILE_SIZE), slice(c * TILE_SIZE, (c + 1) * TILE_SIZE))
            cf, sensor = compute_tile_cloud_metrics(soft_alpha[sl])   # same for opaque: both count alpha >= 0.5
            tile_stats[tile_id] = (cf, sensor)

    return blended_chip, alpha, tile_stats
