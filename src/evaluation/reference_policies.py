"""Reference implementation of the five systems and two ablations, as send-log producers.

STAND-IN: the scheduler owner's src/scheduler/sim.py is the real implementation. This module
exists so the evaluation (metrics, plots, tables) can be built and tested now. Once
sim.run(tiles, policy, budget_bytes) exists, scripts/evaluate.py uses it with --scheduler sim
and this file only remains as a cross-check.

It is deliberately simple: only L0 and L2 packets (no L1 previews), and every pass of the
link is treated as one continuous byte stream; `pass_index` records which real 19.2 MB pass
a packet falls in.

tiles columns used:
    chip, row, col, cloud_frac, sensor_used, p_flood, extent      (always)
    p_optical   optical-model score for ALL tiles, including cloudy ones   (optical_only, no_radar)
    p_ndwi      NDWI colour-index score for all tiles                      (colour_index)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.constants import CLOUD_SWITCH, L0_B, OPTICAL_B, PASS_BYTES, RADAR_B

POLICIES = ["capture_order", "discard_cloudy", "optical_only", "colour_index",
            "orbit_prism", "orbit_no_radar", "orbit_no_dial"]
KEY = ["chip", "row", "col"]
FIXED_THRESHOLD = 0.5  # the "no dial" ablation: an uncalibrated cut-off


class NotAvailable(Exception):
    """A policy cannot run because an input column is missing."""


def _tile_bytes(sensor: pd.Series) -> np.ndarray:
    return np.where(sensor == "radar", RADAR_B, OPTICAL_B).astype(float)


def _build_log(tiles: pd.DataFrame, order: np.ndarray, layer: str, nbytes: np.ndarray,
               start_bytes: float = 0.0, start_order: int = 0) -> pd.DataFrame:
    sel = tiles.iloc[order]
    b = nbytes[order]
    cum = start_bytes + np.cumsum(b)
    return pd.DataFrame({
        "order": np.arange(start_order, start_order + len(sel)),
        "chip": sel["chip"].to_numpy(), "row": sel["row"].to_numpy(), "col": sel["col"].to_numpy(),
        "layer": layer, "bytes": b, "cum_bytes": cum,
        "pass_index": (cum // PASS_BYTES).astype(int),
    })


def _all_full_tiles(tiles: pd.DataFrame, order: np.ndarray, nbytes: np.ndarray) -> pd.DataFrame:
    return _build_log(tiles, order, "L2", nbytes)


def _orbit(tiles: pd.DataFrame, score: np.ndarray, bytes_per_tile: np.ndarray,
           keep: np.ndarray) -> pd.DataFrame:
    """Alerts first, then full tiles by priority per byte. Tiles that are not kept are never sent."""
    extent = tiles["extent"].to_numpy(dtype=float)
    priority = score * extent                       # calibrated-probability-style priority
    idx = np.flatnonzero(keep)
    alerts = idx[np.argsort(-priority[idx], kind="stable")]
    full = idx[np.argsort(-(priority[idx] / bytes_per_tile[idx]), kind="stable")]
    l0 = _build_log(tiles, alerts, "L0", np.full(len(tiles), float(L0_B)))
    l2 = _build_log(tiles, full, "L2", bytes_per_tile,
                    start_bytes=float(l0["cum_bytes"].max()) if len(l0) else 0.0, start_order=len(l0))
    return pd.concat([l0, l2], ignore_index=True)


def run_policy(tiles: pd.DataFrame, policy: str, thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Send log for one policy over `tiles` (already restricted to the evaluation split)."""
    tiles = tiles.reset_index(drop=True)
    n = len(tiles)
    sensor = tiles["sensor_used"]
    cloudy = (tiles["cloud_frac"] >= CLOUD_SWITCH).to_numpy()
    opt_bytes = np.full(n, float(OPTICAL_B))
    mix_bytes = _tile_bytes(sensor)

    if policy == "capture_order":
        return _all_full_tiles(tiles, np.arange(n), opt_bytes)
    if policy == "discard_cloudy":
        return _all_full_tiles(tiles, np.flatnonzero(~cloudy), opt_bytes)
    if policy == "optical_only":
        if "p_optical" not in tiles:
            raise NotAvailable("needs p_optical (optical model score on every tile)")
        score = tiles["p_optical"].to_numpy(dtype=float)
        return _all_full_tiles(tiles, np.argsort(-score, kind="stable"), opt_bytes)
    if policy == "colour_index":
        if "p_ndwi" not in tiles:
            raise NotAvailable("needs p_ndwi (results/scores_ndwi.csv, from real band imagery)")
        score = tiles["p_ndwi"].to_numpy(dtype=float)
        return _all_full_tiles(tiles, np.argsort(-score, kind="stable"), opt_bytes)

    if policy in ("orbit_prism", "orbit_no_dial", "orbit_no_radar"):
        if policy == "orbit_no_radar":
            if "p_optical" not in tiles:
                raise NotAvailable("needs p_optical on cloudy tiles")
            score, per_tile = tiles["p_optical"].to_numpy(dtype=float), opt_bytes
            path = np.full(n, "optical")
        else:
            score, per_tile = tiles["p_flood"].to_numpy(dtype=float), mix_bytes
            path = sensor.to_numpy()
        if policy == "orbit_no_dial":
            thr = np.full(n, FIXED_THRESHOLD)
        else:
            if not thresholds:
                raise NotAvailable("needs calibrated thresholds for the miss-rate dial")
            thr = np.where(path == "radar", thresholds["radar"], thresholds["optical"]).astype(float)
        return _orbit(tiles, score, per_tile, score >= thr)

    raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")


def kept_mask(tiles: pd.DataFrame, log: pd.DataFrame) -> pd.Series:
    """Tiles that appear in the log in any layer (so a dropped critical tile is a miss)."""
    tiles = tiles.reset_index(drop=True)
    sent = log[KEY].drop_duplicates().assign(_k=True)
    return tiles[KEY].merge(sent, on=KEY, how="left")["_k"].fillna(False).astype(bool)
