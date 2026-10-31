"""Satellite chip tiling module for ORBIT-PRISM.

Splits 512x512 satellite chips into a 4x4 grid of sixteen 128x128 pixel tiles.
Maintains event and chip identity in reproducible tile identifiers: {chip}_{row}_{col}.
"""

from typing import Dict, Generator, List, Tuple, Union
import numpy as np

CHIP_SIZE = 512
TILE_SIZE = 128
GRID_SIZE = 4


def generate_tile_id(chip_id: str, row: int, col: int) -> str:
    """Generate reproducible tile identifier from chip ID and grid coordinates."""
    return f"{chip_id}_{row}_{col}"


def parse_tile_id(tile_id: str) -> Tuple[str, int, int]:
    """Parse a tile ID into (chip_id, row, col)."""
    parts = tile_id.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid tile_id format: '{tile_id}'. Expected '<chip>_<row>_<col>'")
    return parts[0], int(parts[1]), int(parts[2])


def tile_chip_2d(
    chip: np.ndarray,
    tile_size: int = TILE_SIZE,
    grid_size: int = GRID_SIZE,
) -> List[Tuple[int, int, np.ndarray]]:
    """Split a 2D (512, 512) array into 16 (128, 128) tiles.

    Returns:
        List of (row, col, tile_array) tuples for row, col in 0..3.
    """
    if chip.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {chip.shape}")
    h, w = chip.shape
    if h != CHIP_SIZE or w != CHIP_SIZE:
        raise ValueError(f"Expected ({CHIP_SIZE}, {CHIP_SIZE}) chip, got {chip.shape}")

    tiles = []
    for r in range(grid_size):
        r_start = r * tile_size
        r_end = (r + 1) * tile_size
        for c in range(grid_size):
            c_start = c * tile_size
            c_end = (c + 1) * tile_size
            tiles.append((r, c, chip[r_start:r_end, c_start:c_end]))
    return tiles


def tile_chip_hwc(
    chip: np.ndarray,
    tile_size: int = TILE_SIZE,
    grid_size: int = GRID_SIZE,
) -> List[Tuple[int, int, np.ndarray]]:
    """Split a 3D HWC (512, 512, C) array into 16 (128, 128, C) tiles.

    Returns:
        List of (row, col, tile_array) tuples for row, col in 0..3.
    """
    if chip.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {chip.shape}")
    h, w, c = chip.shape
    if h != CHIP_SIZE or w != CHIP_SIZE:
        raise ValueError(f"Expected ({CHIP_SIZE}, {CHIP_SIZE}, C) chip, got {chip.shape}")

    tiles = []
    for r in range(grid_size):
        r_start = r * tile_size
        r_end = (r + 1) * tile_size
        for c in range(grid_size):
            c_start = c * tile_size
            c_end = (c + 1) * tile_size
            tiles.append((r, c, chip[r_start:r_end, c_start:c_end, :]))
    return tiles


def tile_chip_dict(
    chip: np.ndarray,
    chip_id: str,
    is_chw: bool = False,
) -> Dict[str, np.ndarray]:
    """Tile a chip and return dictionary mapping tile_id to tile array.

    Args:
        chip: Input array (2D or 3D).
        chip_id: Base chip identifier.
        is_chw: If True and 3D, input is CHW and will be transposed to HWC.
    """
    if chip.ndim == 3 and is_chw:
        chip = np.transpose(chip, (1, 2, 0))

    if chip.ndim == 2:
        tiles = tile_chip_2d(chip)
    elif chip.ndim == 3:
        tiles = tile_chip_hwc(chip)
    else:
        raise ValueError(f"Unsupported chip dimension: {chip.ndim}")

    return {generate_tile_id(chip_id, r, c): arr for r, c, arr in tiles}
