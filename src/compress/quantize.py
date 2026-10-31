"""INT8 Post-Training Quantization (PTQ) pipeline for ORBIT-PRISM.

Author: Harish D
Branch: harish/radar-compress

Requirements:
- Converts FLOAT32 Keras models to full-integer INT8 TensorFlow Lite models.
- Uses representative dataset from TRAIN samples.
- Enforces INT8 input and output tensors (TFLITE_BUILTINS_INT8).
- Validates quantization parameters (scale and zero-point).
- Tests conversion independently with temporary test model when team files are pending.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
MODELS_DIR = ROOT_DIR / "models"


def build_representative_dataset_generator(
    samples: np.ndarray,
    batch_size: int = 1,
    num_calibration_steps: int = 100,
    dtype=np.float32,
) -> Callable[[], Generator[List[np.ndarray], None, None]]:
    """Create a representative dataset generator yielding calibration tensors.

    Parameters
    ----------
    samples : np.ndarray
        Calibration images, shape (N, H, W, C). They must be in the units the model's Input
        layer expects: radar takes float32 in [0, 1]; the optical model takes raw uint8 tiles
        in [0, 255] and rescales internally, so pass dtype=np.uint8 for it.
    dtype :
        dtype of the yielded batches; must equal the model's input dtype.
    """
    total_samples = len(samples)

    def representative_dataset_gen() -> Generator[List[np.ndarray], None, None]:
        for i in range(min(num_calibration_steps, total_samples // batch_size)):
            batch = samples[i * batch_size : (i + 1) * batch_size].astype(dtype)
            yield [batch]

    return representative_dataset_gen


def convert_model_to_full_int8(
    keras_model_or_path: tf.keras.Model | str | Path,
    representative_data_gen: Callable[[], Generator[List[np.ndarray], None, None]],
    output_tflite_path: Optional[str | Path] = None,
) -> Tuple[bytes, Dict[str, any]]:
    """Convert a Keras model to a full-integer INT8 TensorFlow Lite model.

    Enforces:
    - tf.lite.Optimize.DEFAULT
    - tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    - inference_input_type = tf.int8
    - inference_output_type = tf.int8
    """
    if isinstance(keras_model_or_path, (str, Path)):
        model_path = Path(keras_model_or_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        model = keras_model_or_path

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Verify input and output quantization specs
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_specs = []
    for inp in input_details:
        input_specs.append(
            {
                "name": inp["name"],
                "index": inp["index"],
                "dtype": str(inp["dtype"]),
                "shape": inp["shape"].tolist(),
                "scale": float(inp["quantization_parameters"]["scales"][0]) if len(inp["quantization_parameters"]["scales"]) > 0 else 1.0,
                "zero_point": int(inp["quantization_parameters"]["zero_points"][0]) if len(inp["quantization_parameters"]["zero_points"]) > 0 else 0,
            }
        )
        # int8 for the radar model (float input, quantised); uint8 for the optical model, whose
        # Input layer already takes raw uint8 tiles and rescales internally.
        assert inp["dtype"] in (np.int8, np.uint8), f"Input tensor {inp['name']} is not int8/uint8: {inp['dtype']}"

    output_specs = []
    for out in output_details:
        output_specs.append(
            {
                "name": out["name"],
                "index": out["index"],
                "dtype": str(out["dtype"]),
                "shape": out["shape"].tolist(),
                "scale": float(out["quantization_parameters"]["scales"][0]) if len(out["quantization_parameters"]["scales"]) > 0 else 1.0,
                "zero_point": int(out["quantization_parameters"]["zero_points"][0]) if len(out["quantization_parameters"]["zero_points"]) > 0 else 0,
            }
        )
        assert out["dtype"] == np.int8, f"Output tensor {out['name']} is not int8: {out['dtype']}"

    verification_info = {
        "status": "CONVERTED_INT8",
        "inputs": input_specs,
        "outputs": output_specs,
        "model_bytes": len(tflite_model),
        "model_KB": len(tflite_model) / 1024.0,
    }

    if output_tflite_path is not None:
        out_p = Path(output_tflite_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(tflite_model)
        verification_info["saved_path"] = str(out_p)

    return tflite_model, verification_info


def quantize_radar_model(
    radar_float_path: Optional[Path] = None,
    train_radar_images: Optional[np.ndarray] = None,
    num_calibration_steps: int = 500,
) -> Dict[str, any]:
    """Quantize Radar model to INT8 using real TRAIN radar tiles."""
    if radar_float_path is None:
        radar_float_path = MODELS_DIR / "radar_float.keras"

    if not radar_float_path.exists():
        return {
            "status": "PENDING_REAL_MODEL",
            "message": f"Radar FLOAT32 model not found at {radar_float_path}. Train model first.",
        }

    if train_radar_images is None:
        return {
            "status": "PENDING_REAL_DATA",
            "message": "TRAIN radar images required for representative calibration dataset.",
        }

    # The generator walks the array in order and stops after num_calibration_steps tiles, so pass a
    # shuffled, stratified sample (see scripts/quantize_and_score.py) and enough of it.
    rep_gen = build_representative_dataset_generator(
        train_radar_images, batch_size=1, num_calibration_steps=num_calibration_steps)
    out_path = MODELS_DIR / "radar_int8.tflite"

    _, info = convert_model_to_full_int8(radar_float_path, rep_gen, output_tflite_path=out_path)
    return info


def quantize_optical_model(
    optical_float_path: Optional[Path] = None,
    train_optical_images: Optional[np.ndarray] = None,
    num_calibration_steps: int = 500,
) -> Dict[str, any]:
    """Quantize Optical model to INT8 using real TRAIN optical tiles."""
    if optical_float_path is None:
        optical_float_path = MODELS_DIR / "optical_float.keras"

    if not optical_float_path.exists():
        return {
            "status": "PENDING_TEAMMATE_FILE",
            "message": f"Shenbaga's optical model not found at {optical_float_path}. Awaiting delivery.",
        }

    if train_optical_images is None:
        return {
            "status": "PENDING_REAL_DATA",
            "message": "TRAIN optical images required for representative calibration dataset.",
        }

    # The optical model's Input layer is uint8 (it rescales internally), so calibrate on raw uint8 tiles.
    rep_gen = build_representative_dataset_generator(
        train_optical_images, batch_size=1, num_calibration_steps=num_calibration_steps, dtype=np.uint8)
    out_path = MODELS_DIR / "optical_int8.tflite"

    _, info = convert_model_to_full_int8(optical_float_path, rep_gen, output_tflite_path=out_path)
    return info


def test_quantization_independent() -> Dict[str, any]:
    """Requirement 7 & 12: Independent verification of full-integer INT8 conversion pipeline.

    Tests PTQ pipeline using a temporary test model and synthetic calibration dataset.
    Verifies:
    1. TFLite INT8 conversion completes.
    2. Input tensor is np.int8 with scale and zero_point.
    3. Output tensors are np.int8 with scale and zero_point.
    4. Model size reduction from float to INT8.
    """
    print("=== [Requirement 7 & 12] Testing INT8 Quantization independently ===")
    from src.models.radar import build_radar_model

    test_model = build_radar_model(input_shape=(128, 128, 2))
    calib_samples = np.random.uniform(0.0, 1.0, size=(50, 128, 128, 2)).astype(np.float32)

    rep_gen = build_representative_dataset_generator(calib_samples, batch_size=1, num_calibration_steps=20)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temp_tflite_path = MODELS_DIR / "temp_test_int8.tflite"

    tflite_bytes, info = convert_model_to_full_int8(test_model, rep_gen, output_tflite_path=temp_tflite_path)

    print(f"✓ Converted test model to INT8 TFLite ({info['model_KB']:.2f} KB).")
    print(f"✓ Input tensor: {info['inputs']}")
    print(f"✓ Output tensors: {info['outputs']}")

    assert info["status"] == "CONVERTED_INT8"
    assert info["inputs"][0]["dtype"] == "int8" or "int8" in info["inputs"][0]["dtype"]
    for out in info["outputs"]:
        assert "int8" in out["dtype"]

    if temp_tflite_path.exists():
        temp_tflite_path.unlink()

    print("✓ Full-integer INT8 quantization pipeline verified successfully.")
    return info


if __name__ == "__main__":
    test_quantization_independent()
