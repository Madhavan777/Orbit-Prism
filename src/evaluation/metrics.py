"""Evaluation metrics computed from a send log (the format in CONTEXT.md).

A send log has one row per packet, in transmission order:
    order, chip, row, col, layer, bytes, cum_bytes, pass_index
with layer in {"L0", "L1", "L2"}: L0 is the 32 B alert, L1 a preview, L2 the full tile.

`tiles` is a DataFrame in the 11-column tile schema (at least chip, row, col, critical,
cloud_frac), restricted to the tiles being evaluated (for example split == "holdout").

Headline recall (CONTEXT.md decision 3) is FULL-TILE recall: a critical tile counts as
delivered when its L2 packet has arrived. Alert recall (an L0 packet arrived) is a separate metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["chip", "row", "col"]
LINK_BPS = 256_000  # 256 kbps S-band link


def _critical_keys(tiles: pd.DataFrame) -> pd.DataFrame:
    return tiles.loc[tiles["critical"] == 1, KEY]


def recall_curve(log: pd.DataFrame, tiles: pd.DataFrame, layer: str = "L2") -> pd.DataFrame:
    """Critical-tile recall after each packet of `layer` (default: full tiles).

    Returns columns cum_bytes, recall. The curve starts at (0, 0) and is a step function:
    recall only changes when a critical tile's packet of that layer arrives.
    """
    crit = _critical_keys(tiles)
    n_crit = len(crit)
    sent = log[log["layer"] == layer].merge(crit, on=KEY, how="inner")
    sent = sent.sort_values("cum_bytes").drop_duplicates(KEY, keep="first")
    cum = sent["cum_bytes"].to_numpy(dtype=float)
    rec = (np.arange(1, len(sent) + 1) / n_crit) if n_crit else np.zeros(len(sent))
    curve = pd.DataFrame({"cum_bytes": np.r_[0.0, cum], "recall": np.r_[0.0, rec]})
    # Carry the last recall forward to the total bytes the system sent, so a plateau is visible
    # and a system that keeps sending after its last critical tile is not drawn as having stopped.
    total = float(log["cum_bytes"].max()) if len(log) else 0.0
    if total > curve["cum_bytes"].iloc[-1]:
        curve.loc[len(curve)] = [total, curve["recall"].iloc[-1]]
    return curve


def recall_at_bytes(curve: pd.DataFrame, budget_bytes: float) -> float:
    """Recall once at most `budget_bytes` have been sent."""
    done = curve[curve["cum_bytes"] <= budget_bytes]
    return float(done["recall"].iloc[-1]) if len(done) else 0.0


def final_recall(curve: pd.DataFrame) -> float:
    return float(curve["recall"].iloc[-1])


def bytes_to_recall(curve: pd.DataFrame, target: float = 0.95) -> float:
    """Smallest cumulative bytes at which recall >= target, or NaN if it is never reached."""
    hit = curve[curve["recall"] >= target - 1e-12]
    return float(hit["cum_bytes"].iloc[0]) if len(hit) else float("nan")


def seconds(nbytes: float, link_bps: int = LINK_BPS) -> float:
    return float(nbytes) * 8.0 / link_bps


def time_to_first_critical_alert(log: pd.DataFrame, tiles: pd.DataFrame, link_bps: int = LINK_BPS) -> float:
    """Seconds until the first packet of any layer about a critical tile has been received.

    Systems that send an L0 alert first reach this after 32 B; systems that send only full
    tiles wait for the first critical tile to arrive. NaN if no critical tile is ever sent.
    """
    sent = log.merge(_critical_keys(tiles), on=KEY, how="inner")
    if sent.empty:
        return float("nan")
    return seconds(sent["cum_bytes"].min(), link_bps)


def cloudy_recall_at_bytes(log: pd.DataFrame, tiles: pd.DataFrame, budget_bytes: float,
                           cloud_switch: float = 0.5) -> float:
    """Full-tile recall at `budget_bytes`, over critical tiles with cloud_frac >= cloud_switch only."""
    cloudy = tiles[(tiles["critical"] == 1) & (tiles["cloud_frac"] >= cloud_switch)]
    if cloudy.empty:
        return float("nan")
    curve = recall_curve(log, cloudy.assign(critical=1))
    return recall_at_bytes(curve, budget_bytes)


def realised_miss_rate(tiles: pd.DataFrame, kept: pd.Series) -> float:
    """Share of critical tiles that the policy dropped (never sends in any layer)."""
    crit = tiles["critical"] == 1
    n = int(crit.sum())
    return float((crit & ~kept).sum() / n) if n else float("nan")


def summarise(name: str, log: pd.DataFrame, tiles: pd.DataFrame, scaled_budget: float) -> dict:
    """One summary row per system (see results/summary.csv)."""
    full = recall_curve(log, tiles, "L2")
    alert = recall_curve(log, tiles, "L0")
    return {
        "system": name,
        "bytes_to_95pct_recall_MB": bytes_to_recall(full, 0.95) / 1e6,
        "recall_at_scaled_pass": recall_at_bytes(full, scaled_budget),
        "cloudy_recall_at_scaled_pass": cloudy_recall_at_bytes(log, tiles, scaled_budget),
        "final_full_tile_recall": final_recall(full),
        "final_alert_recall": final_recall(alert) if (log["layer"] == "L0").any() else float("nan"),
        "seconds_to_first_critical_alert": time_to_first_critical_alert(log, tiles),
        "total_MB_sent": float(log["cum_bytes"].max() / 1e6) if len(log) else 0.0,
    }
