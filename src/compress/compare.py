"""Model Precision Comparison Pipeline for ORBIT-PRISM (Harish D).

Compares FLOAT32 vs INT8 models on CAL (valid split) data:
- AUROC
- Recall at fixed threshold
- Model weight size (flash)
- Peak activation memory (SRAM)
- Model size reduction
- Delta performance (INT8 vs FLOAT32)

Design Rules:
- Evaluates strictly on CAL data.
- Never touches or tunes on TEST (holdout) data.
- Uses identical samples and fixed threshold for fair evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.compress.predict_int8 import Int8ModelRunner
from src.compress.profile_memory import inspect_tflite_model_memory
from src.models.radar import evaluate_radar_model, select_optimal_threshold_on_cal


def compare_radar_float_vs_int8_on_cal(
    float_model_path: Path,
    int8_model_path: Path,
    cal_images: np.ndarray,
    cal_p_true: np.ndarray,
    cal_ext_true: np.ndarray,
    target_recall: float = 0.90,
) -> Dict[str, any]:
    """Perform fair comparison between Radar FLOAT32 and INT8 models on CAL data."""
    import tensorflow as tf

    # 1. Load FLOAT32 model
    float_model = tf.keras.models.load_model(float_model_path)
    cal_threshold = select_optimal_threshold_on_cal(float_model, cal_images, cal_p_true, target_recall=target_recall)
    float_metrics = evaluate_radar_model(float_model, cal_images, cal_p_true, cal_ext_true, threshold=cal_threshold)

    # 2. Evaluate INT8 model
    runner = Int8ModelRunner(int8_model_path)
    p_preds_int8, ext_preds_int8 = runner.predict_batch(cal_images)

    try:
        auroc_int8 = float(roc_auc_score(cal_p_true, p_preds_int8))
    except Exception:
        auroc_int8 = 0.5

    crit_mask = cal_p_true == 1
    recall_int8 = float(np.sum((p_preds_int8 >= cal_threshold) & crit_mask) / np.sum(crit_mask)) if np.sum(crit_mask) > 0 else 0.0
    extent_mae_int8 = float(np.mean(np.abs(cal_ext_true - ext_preds_int8)))

    # 3. Profile memory
    float_prof = inspect_tflite_model_memory(float_model_path)
    int8_prof = inspect_tflite_model_memory(int8_model_path)

    weight_reduction = (1.0 - (int8_prof["weights_KB"] / max(float_prof["weights_KB"], 1e-6))) * 100.0
    act_reduction = (1.0 - (int8_prof["peak_activation_KB"] / max(float_prof["peak_activation_KB"], 1e-6))) * 100.0

    comparison_results = {
        "float32": {
            "auroc": float_metrics["auroc"],
            "recall": float_metrics["recall"],
            "extent_mae": float_metrics["extent_mae"],
            "weights_KB": float_prof["weights_KB"],
            "peak_activation_KB": float_prof["peak_activation_KB"],
        },
        "int8": {
            "auroc": auroc_int8,
            "recall": recall_int8,
            "extent_mae": extent_mae_int8,
            "weights_KB": int8_prof["weights_KB"],
            "peak_activation_KB": int8_prof["peak_activation_KB"],
        },
        "delta": {
            "delta_auroc": auroc_int8 - float_metrics["auroc"],
            "delta_recall": recall_int8 - float_metrics["recall"],
            "weight_reduction_pct": weight_reduction,
            "activation_reduction_pct": act_reduction,
        },
        "threshold": cal_threshold,
    }
    return comparison_results
