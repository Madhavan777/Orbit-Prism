"""Calibration evaluation metrics for ORBIT-PRISM.

Implements:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Reliability Diagram binning
- Achieved Miss Rate and Recall
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Compute standard Brier score: mean squared error between probabilities and binary outcomes.

    Brier = (1/N) * sum((p_i - y_i)^2)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute empirical accuracy, mean confidence, and sample count per probability bin.

    Returns:
        (bin_accuracies, bin_confidences, bin_counts)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= low) & (y_prob <= high)
        else:
            in_bin = (y_prob >= low) & (y_prob < high)

        count = int(in_bin.sum())
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(float(y_true[in_bin].mean()))
            bin_confs.append(float(y_prob[in_bin].mean()))
        else:
            bin_accs.append(0.0)
            bin_confs.append((low + high) / 2.0)

    return np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return 0.0

    accs, confs, counts = compute_calibration_curve(y_true, y_prob, n_bins=n_bins)
    ece = 0.0
    for acc, conf, count in zip(accs, confs, counts):
        if count > 0:
            ece += (count / n) * abs(acc - conf)
    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Maximum Calibration Error (MCE).

    MCE = max_m |acc(B_m) - conf(B_m)| for non-empty bins.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0

    accs, confs, counts = compute_calibration_curve(y_true, y_prob, n_bins=n_bins)
    errors = [abs(acc - conf) for acc, conf, count in zip(accs, confs, counts) if count > 0]
    return float(max(errors)) if errors else 0.0


def compute_miss_rate(
    y_critical: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[float, int, int]:
    """Compute achieved miss rate for critical flood tiles.

    A critical tile (y_critical == 1) is detected if score >= threshold.
    It is missed if score < threshold.

    Returns:
        (achieved_miss_rate, n_critical, n_missed)
    """
    y_crit = np.asarray(y_critical, dtype=int)
    sc = np.asarray(scores, dtype=np.float64)

    crit_mask = (y_crit == 1)
    n_crit = int(crit_mask.sum())
    if n_crit == 0:
        return 0.0, 0, 0

    crit_scores = sc[crit_mask]
    n_missed = int((crit_scores < threshold).sum())
    miss_rate = float(n_missed / n_crit)
    return miss_rate, n_crit, n_missed
