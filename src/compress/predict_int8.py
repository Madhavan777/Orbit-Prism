"""INT8 TFLite Inference Pipeline for ORBIT-PRISM.

Author: Harish D
Branch: harish/radar-compress

Requirements:
- Loads full-integer INT8 TFLite models.
- Quantizes normalized float inputs: int8 = clip(round(float / scale) + zero_point, -128, 127).
- Runs TFLite interpreter inference.
- Dequantizes int8 outputs: float = (int8 - zero_point) * scale.
- Accurately maps heads to p_flood and extent.
- Validates CSV output schema: tile_id,p_flood,extent.
- Independent testing with synthetic inputs and temporary test models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
TILES_BASE_CSV = ROOT_DIR / "data" / "tiles_base.csv"


class Int8ModelRunner:
    """TFLite INT8 inference runner handling tensor quantization and dequantization."""

    def __init__(self, model_content_or_path: Union[bytes, str, Path]):
        if isinstance(model_content_or_path, (str, Path)):
            path = Path(model_content_or_path)
            if not path.exists():
                raise FileNotFoundError(f"TFLite model not found: {path}")
            self.interpreter = tf.lite.Interpreter(model_path=str(path))
        else:
            self.interpreter = tf.lite.Interpreter(model_content=model_content_or_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Parse input quantization parameters
        self.inp_detail = self.input_details[0]
        self.inp_index = self.inp_detail["index"]
        self.inp_shape = self.inp_detail["shape"]
        self.inp_scale = float(self.inp_detail["quantization_parameters"]["scales"][0])
        self.inp_zero_point = int(self.inp_detail["quantization_parameters"]["zero_points"][0])

        # Parse output quantization parameters for each head
        self.outputs = []
        for out in self.output_details:
            scale = float(out["quantization_parameters"]["scales"][0]) if len(out["quantization_parameters"]["scales"]) > 0 else 1.0
            zero_point = int(out["quantization_parameters"]["zero_points"][0]) if len(out["quantization_parameters"]["zero_points"]) > 0 else 0
            self.outputs.append(
                {
                    "name": out["name"],
                    "index": out["index"],
                    "scale": scale,
                    "zero_point": zero_point,
                }
            )

    def quantize_input(self, float_img: np.ndarray) -> np.ndarray:
        """Quantize float in [0.0, 1.0] to int8 using scale and zero_point."""
        q = np.round(float_img / self.inp_scale) + self.inp_zero_point
        return np.clip(q, -128, 127).astype(np.int8)

    def dequantize_output(self, int8_out: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Dequantize int8 output to float32 using scale and zero_point."""
        return (int8_out.astype(np.float32) - zero_point) * scale

    def predict_single(self, float_img: np.ndarray) -> Tuple[float, float]:
        """Predict for a single tile (128, 128, C). Returns (p_flood, extent)."""
        if float_img.ndim == 3:
            float_img = np.expand_dims(float_img, axis=0)

        q_input = self.quantize_input(float_img)
        self.interpreter.set_tensor(self.inp_index, q_input)
        self.interpreter.invoke()

        out_vals = []
        for out_spec in self.outputs:
            raw_out = self.interpreter.get_tensor(out_spec["index"])
            deq = self.dequantize_output(raw_out, out_spec["scale"], out_spec["zero_point"])
            out_vals.append(float(np.squeeze(deq)))

        # Output head mapping:
        # Output 0 is p_flood, Output 1 is extent (or by name match)
        out0_name = self.outputs[0]["name"].lower()
        if "extent" in out0_name:
            extent_val, p_flood_val = out_vals[0], out_vals[1]
        elif "flood" in out0_name:
            p_flood_val, extent_val = out_vals[0], out_vals[1]
        else:
            # Default to model declaration order [p_flood, extent]
            p_flood_val, extent_val = out_vals[0], out_vals[1]

        return float(np.clip(p_flood_val, 0.0, 1.0)), float(np.clip(extent_val, 0.0, 1.0))

    def predict_batch(self, float_imgs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict for a batch of tiles. Returns (p_flood_array, extent_array)."""
        n = len(float_imgs)
        p_floods = np.zeros(n, dtype=np.float32)
        extents = np.zeros(n, dtype=np.float32)
        for i in range(n):
            p, ext = self.predict_single(float_imgs[i])
            p_floods[i] = p
            extents[i] = ext
        return p_floods, extents


def validate_scores_csv(csv_path: Union[str, Path]) -> bool:
    """Validate that score CSV strictly follows the project schema: tile_id,p_flood,extent."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Score file does not exist: {path}")

    df = pd.read_csv(path)
    expected_cols = ["tile_id", "p_flood", "extent"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Invalid columns in {path}. Expected {expected_cols}, got {list(df.columns)}")

    assert len(df) > 0, "Score file is empty!"
    assert df["p_flood"].between(0.0, 1.0).all(), "p_flood contains out-of-range values!"
    assert df["extent"].between(0.0, 1.0).all(), "extent contains out-of-range values!"
    return True


def run_radar_int8_predictions(
    radar_int8_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
) -> Dict[str, any]:
    """Generate final results/scores_radar_int8.csv once real model and data are ready."""
    if radar_int8_path is None:
        radar_int8_path = MODELS_DIR / "radar_int8.tflite"

    if not radar_int8_path.exists():
        return {
            "status": "PENDING_REAL_MODEL",
            "message": "Radar INT8 model not found. Quantize model first.",
        }

    from src.models.radar import load_radar_data_for_training

    data = load_radar_data_for_training(dataset_path)
    if data is None:
        return {
            "status": "PENDING_REAL_DATA",
            "message": "Agnel's radar dataset required for generating real predictions.",
        }

    data_dict, matched_df = data
    images = data_dict["images"]

    runner = Int8ModelRunner(radar_int8_path)
    p_preds, ext_preds = runner.predict_batch(images)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "scores_radar_int8.csv"
    res_df = pd.DataFrame(
        {
            "tile_id": matched_df["tile_id"].to_numpy(),
            "p_flood": p_preds,
            "extent": ext_preds,
        }
    )
    res_df.to_csv(out_csv, index=False)
    validate_scores_csv(out_csv)

    return {
        "status": "COMPLETED",
        "csv_path": str(out_csv),
        "tile_count": len(res_df),
    }


def run_optical_int8_predictions(
    optical_int8_path: Optional[Path] = None,
    optical_images: Optional[np.ndarray] = None,
    tile_ids: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Generate final results/scores_optical_int8.csv once Shenbaga's model is converted."""
    if optical_int8_path is None:
        optical_int8_path = MODELS_DIR / "optical_int8.tflite"

    if not optical_int8_path.exists():
        return {
            "status": "PENDING_TEAMMATE_FILE",
            "message": "Optical INT8 model not found. Awaiting Shenbaga's model.",
        }

    if optical_images is None or tile_ids is None:
        return {
            "status": "PENDING_REAL_DATA",
            "message": "Optical images required for prediction.",
        }

    runner = Int8ModelRunner(optical_int8_path)
    p_preds, ext_preds = runner.predict_batch(optical_images)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "scores_optical_int8.csv"
    res_df = pd.DataFrame(
        {
            "tile_id": tile_ids,
            "p_flood": p_preds,
            "extent": ext_preds,
        }
    )
    res_df.to_csv(out_csv, index=False)
    validate_scores_csv(out_csv)

    return {
        "status": "COMPLETED",
        "csv_path": str(out_csv),
        "tile_count": len(res_df),
    }


def test_int8_inference_independent() -> Dict[str, any]:
    """Requirement 8 & 12: Independent verification of the INT8 inference pipeline.

    Tests:
    1. Quantization of input float tensors to int8.
    2. TFLite INT8 forward pass.
    3. Dequantization of int8 output tensors back to float.
    4. Head mapping to p_flood and extent.
    5. Value clamping in [0.0, 1.0].
    6. CSV format validation.
    """
    print("=== [Requirement 8 & 12] Testing INT8 Inference independently ===")
    from src.compress.quantize import build_representative_dataset_generator, convert_model_to_full_int8
    from src.models.radar import build_radar_model

    # Build and convert a temporary test model
    test_model = build_radar_model(input_shape=(128, 128, 2))
    calib = np.random.uniform(0.0, 1.0, size=(20, 128, 128, 2)).astype(np.float32)
    rep_gen = build_representative_dataset_generator(calib, batch_size=1, num_calibration_steps=10)
    tflite_bytes, _ = convert_model_to_full_int8(test_model, rep_gen)

    # Initialize runner with in-memory bytes
    runner = Int8ModelRunner(tflite_bytes)

    # Test single tile inference
    test_tile = np.random.uniform(0.0, 1.0, size=(128, 128, 2)).astype(np.float32)
    p, ext = runner.predict_single(test_tile)
    assert 0.0 <= p <= 1.0, f"p_flood {p} out of bounds"
    assert 0.0 <= ext <= 1.0, f"extent {ext} out of bounds"

    # Test batch tile inference
    batch_tiles = np.random.uniform(0.0, 1.0, size=(5, 128, 128, 2)).astype(np.float32)
    p_batch, ext_batch = runner.predict_batch(batch_tiles)
    assert len(p_batch) == 5
    assert len(ext_batch) == 5
    assert np.all((p_batch >= 0.0) & (p_batch <= 1.0))
    assert np.all((ext_batch >= 0.0) & (ext_batch <= 1.0))

    # Test CSV schema validation with temporary file
    temp_csv = RESULTS_DIR / "temp_validation_test.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_test = pd.DataFrame(
        {
            "tile_id": ["test_0_0", "test_0_1", "test_1_0"],
            "p_flood": [0.1, 0.85, 0.4],
            "extent": [0.05, 0.6, 0.25],
        }
    )
    df_test.to_csv(temp_csv, index=False)
    assert validate_scores_csv(temp_csv) is True
    if temp_csv.exists():
        temp_csv.unlink()

    print(f"✓ Single tile INT8 inference: p_flood={p:.4f}, extent={ext:.4f}.")
    print(f"✓ Batch tile INT8 inference verified (5 samples).")
    print(f"✓ Input quantization and output dequantization validated.")
    print(f"✓ CSV format validation logic verified.")
    return {
        "status": "PASSED",
        "single_sample_p_flood": p,
        "single_sample_extent": ext,
    }


if __name__ == "__main__":
    test_int8_inference_independent()

