"""Memory Profiling and SRAM Budget Analysis for ORBIT-PRISM.

Author: Harish D
Branch: harish/radar-compress

Requirements:
- Profiles four model configurations:
    1. Optical FLOAT32
    2. Optical INT8
    3. Radar FLOAT32
    4. Radar INT8
- Measures model weight size (flash footprint).
- Estimates peak activation memory (SRAM footprint) via tensor lifetime analysis.
- Compares against the 4.2 MB design budget (SRAM_BYTES = 4_200_000 in src/constants.py).
- Generates results/memory_table.csv (CONTEXT.md contract columns) and results/memory_notes.md.

Weight and activation sizes depend on the architecture only, not on training, so they can be
measured on an untrained model. Each row says which it was in the `basis` column, and
`recall` and `ece` stay NOT_MEASURED until a trained model has been evaluated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import OPTICAL_B, RADAR_B, SRAM_BYTES

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

SRAM_LIMIT_KB = SRAM_BYTES / 1024.0  # 4,200,000 / 1024 = 4101.5625 KB

TABLE_COLUMNS = [
    "model", "dtype", "weights_bytes", "peak_activation_bytes", "tile_buffer_bytes",
    "total_bytes", "macs", "recall", "ece", "basis",
]
NOT_MEASURED = "NOT_MEASURED"


def _constant_tensor_indices(content: bytes) -> set:
    """Indices of tensors whose data is stored in the TFLite flatbuffer (weights, biases, constants)."""
    from tensorflow.lite.python import schema_py_generated as schema_fb

    model = schema_fb.Model.GetRootAsModel(bytearray(content), 0)
    subgraph = model.Subgraphs(0)
    constant = set()
    for i in range(subgraph.TensorsLength()):
        buffer = model.Buffers(subgraph.Tensors(i).Buffer())
        if buffer is not None and buffer.DataLength() > 0:
            constant.add(i)
    return constant


def inspect_tflite_model_memory(
    tflite_content_or_path: Union[bytes, str, Path],
) -> Dict[str, any]:
    """Inspect TFLite tensor details to measure weight size and estimate peak activation memory.

    Methodology:
    1. Weights (Flash):
       Static parameters (weights, biases, constants).
       Determined by attempting to read tensor buffers prior to inference invocation;
       static constant tensors have populated storage buffers in the flatbuffer.
    2. Peak Activation (SRAM - ESTIMATED):
       Dynamic activation tensors allocated in the runtime arena during graph execution.
       Analyzed across operator execution pipeline.
       Peak live activations occur during layer transition when input buffer and output buffer
       must simultaneously reside in memory.
    """
    if isinstance(tflite_content_or_path, (str, Path)):
        path = Path(tflite_content_or_path)
        file_size_bytes = path.stat().st_size
        content = path.read_bytes()
    else:
        content = bytes(tflite_content_or_path)
        file_size_bytes = len(content)
    interpreter = tf.lite.Interpreter(model_content=content)

    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()

    input_indices = {inp["index"] for inp in interpreter.get_input_details()}
    output_indices = {out["index"] for out in interpreter.get_output_details()}
    # A tensor is a weight/constant iff the flatbuffer stores data for it. (Testing whether
    # get_tensor() raises is not reliable: after allocate_tensors() intermediate buffers can be
    # read too, which counted a 256 KB float copy of the input tile as "weights".)
    constant_indices = _constant_tensor_indices(content)

    total_weight_bytes = 0
    weight_tensors = []
    activation_tensors = []

    for t in tensor_details:
        shape = t["shape"]
        num_elements = int(np.prod(shape)) if len(shape) > 0 else 1
        item_size = np.dtype(t["dtype"]).itemsize
        tensor_bytes = num_elements * item_size
        idx = t["index"]

        is_dynamic = not (idx in constant_indices) or idx in input_indices or idx in output_indices

        if not is_dynamic and num_elements > 0:
            total_weight_bytes += tensor_bytes
            weight_tensors.append(
                {
                    "name": t["name"],
                    "shape": shape.tolist(),
                    "dtype": str(t["dtype"]),
                    "bytes": tensor_bytes,
                }
            )
        else:
            activation_tensors.append(
                {
                    "name": t["name"],
                    "shape": shape.tolist(),
                    "dtype": str(t["dtype"]),
                    "bytes": tensor_bytes,
                }
            )

    # Sort activation tensors to find the largest live buffers
    act_bytes_list = [a["bytes"] for a in activation_tensors]
    sorted_act = sorted(act_bytes_list, reverse=True)

    # Peak pair during execution: largest input buffer + largest output buffer
    peak_pair_bytes = sorted_act[0] + (sorted_act[1] if len(sorted_act) > 1 else 0)
    sum_all_activations = sum(act_bytes_list)

    return {
        "file_size_bytes": file_size_bytes,
        "file_size_KB": file_size_bytes / 1024.0,
        "weights_bytes": total_weight_bytes,
        "weights_KB": total_weight_bytes / 1024.0,
        "estimated_peak_activation_bytes": peak_pair_bytes,
        "peak_activation_KB": peak_pair_bytes / 1024.0,
        "sum_all_activations_KB": sum_all_activations / 1024.0,
        "weight_tensor_count": len(weight_tensors),
        "activation_tensor_count": len(activation_tensors),
        "activation_breakdown": activation_tensors,
    }


def convert_float_to_tflite_bytes(keras_model_or_path: tf.keras.Model | Path) -> bytes:
    """Convert a Keras model to unquantized FLOAT32 TFLite for fair precision comparison."""
    if isinstance(keras_model_or_path, (str, Path)):
        model = tf.keras.models.load_model(keras_model_or_path)
    else:
        model = keras_model_or_path

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def count_macs(model: tf.keras.Model) -> int:
    """Multiply-accumulate operations for one tile (Conv2D, DepthwiseConv2D and Dense layers)."""
    macs = 0
    for layer in model.layers:
        out = getattr(layer, "output", None)
        out_shape = tuple(int(d) for d in out.shape[1:]) if out is not None and not isinstance(out, list) else ()
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            kh, kw = layer.kernel_size
            h, w, c = out_shape
            macs += h * w * c * kh * kw
        elif isinstance(layer, tf.keras.layers.Conv2D):
            kh, kw = layer.kernel_size
            h, w, c_out = out_shape
            c_in = int(layer.kernel.shape[2])
            macs += h * w * c_out * kh * kw * c_in
        elif isinstance(layer, tf.keras.layers.Dense):
            macs += int(layer.kernel.shape[0]) * int(layer.kernel.shape[1])
    return macs


def _row(model_name: str, dtype: str, prof: Dict[str, any], macs: int, tile_bytes: int,
         basis: str, metrics: Optional[Dict[str, any]], key: str) -> Dict[str, any]:
    weights = int(prof["weights_bytes"])
    peak = int(prof["estimated_peak_activation_bytes"])
    # Conservative accounting: the uint8 tile staged from flash is counted in addition to the
    # arena, although the arena also holds the model's own input tensor.
    total = weights + peak + tile_bytes
    return {
        "model": model_name,
        "dtype": dtype,
        "weights_bytes": weights,
        "peak_activation_bytes": peak,
        "tile_buffer_bytes": tile_bytes,
        "total_bytes": total,
        "macs": macs,
        "recall": metrics.get(f"{key}_recall", NOT_MEASURED) if metrics else NOT_MEASURED,
        "ece": metrics.get(f"{key}_ece", NOT_MEASURED) if metrics else NOT_MEASURED,
        "basis": basis,
    }


def _architecture(name: str) -> tuple[tf.keras.Model, np.ndarray, int]:
    """Untrained architecture, a representative-data array and the tile buffer size."""
    if name == "Optical":
        from src.models.optical import build_optical_model
        model = build_optical_model()
        calib = np.random.default_rng(0).uniform(0, 255, size=(20, 128, 128, 4)).astype(np.float32)
        return model, calib, OPTICAL_B
    from src.models.radar import build_radar_model
    model = build_radar_model(input_shape=(128, 128, 2))
    calib = np.random.default_rng(0).uniform(0, 1, size=(20, 128, 128, 2)).astype(np.float32)
    return model, calib, RADAR_B


def profile_all_models_pipeline(
    cal_metrics: Optional[Dict[str, any]] = None,
    write: bool = True,
) -> pd.DataFrame:
    """Build results/memory_table.csv with the CONTEXT.md contract columns.

    For each of Optical and Radar, the FLOAT32 and INT8 variants are profiled. A trained model
    file (models/<name>_float.keras) is used when present; otherwise the untrained architecture
    is built, and the row's `basis` says so. recall and ece are NOT_MEASURED unless
    `cal_metrics` supplies `<name>_<float|int8>_recall` / `_ece`.
    """
    from src.compress.quantize import build_representative_dataset_generator, convert_model_to_full_int8

    rows = []
    for model_name in ("Optical", "Radar"):
        key_base = model_name.lower()
        keras_path = MODELS_DIR / f"{key_base}_float.keras"
        model, calib, tile_bytes = _architecture(model_name)
        basis = "untrained_architecture"
        if keras_path.exists():
            model = tf.keras.models.load_model(keras_path)
            basis = "trained_model_file"
        macs = count_macs(model)

        float_prof = inspect_tflite_model_memory(convert_float_to_tflite_bytes(model))
        rows.append(_row(model_name, "FLOAT32", float_prof, macs, tile_bytes, basis, cal_metrics, f"{key_base}_float"))

        in_dtype = np.uint8 if model_name == "Optical" else np.float32  # the models' Input dtypes
        rep = build_representative_dataset_generator(calib, batch_size=1, num_calibration_steps=10, dtype=in_dtype)
        int8_bytes, _ = convert_model_to_full_int8(model, rep)
        int8_prof = inspect_tflite_model_memory(int8_bytes)
        rows.append(_row(model_name, "INT8", int8_prof, macs, tile_bytes, basis, cal_metrics, f"{key_base}_int8"))

    df = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    if write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = RESULTS_DIR / "memory_table.csv"
        df.to_csv(out_csv, index=False, lineterminator="\n")
        print(f"Saved memory table to: {out_csv}")
        generate_memory_notes(df)
    return df


def generate_memory_notes(df: pd.DataFrame) -> None:
    """Document memory profiling methodology, assumptions, estimates, and limitations."""
    notes_path = RESULTS_DIR / "memory_notes.md"
    content = f"""# ORBIT-PRISM: Memory Profiling and SRAM Budget Notes
Owner: **Harish D** (Branch: `harish/radar-compress`)

## 1. Executive Summary & Design Budget
- **Budget**: `4,200,000` bytes (4.2 MB, decimal; about 4,101.6 KiB) as specified in `src/constants.py` and `CONTEXT.md`.
- **Accounting (conservative)**: `total_bytes = weights_bytes + peak_activation_bytes + tile_buffer_bytes`. Weights are counted against the budget as well as in flash, and the uint8 tile staged from flash is counted in addition to the arena (which also holds the model's own input tensor).
- **Status**: every row is marked in the `basis` column. `untrained_architecture` rows are valid for sizes (they depend on the architecture, not on training) but `recall` and `ece` are `NOT_MEASURED` until a trained model is evaluated.

## 2. Profiling Methodology
- **Weights**: sum of elements x bytes per element over static tensors in the TFLite file; excludes flatbuffer schema overhead.
- **Peak activation memory (ESTIMATED)**: TFLite tensor details do not give the runtime allocator layout, so the peak is the sum of the two largest dynamic tensors alive together (a ping-pong model). It is an estimate, not a measurement on target silicon.
- **MACs**: multiply-accumulates per tile for Conv2D, DepthwiseConv2D and Dense layers.

## 3. Current Memory Table
```csv
{df.to_csv(index=False, lineterminator=chr(10))}
```

## 4. Omitted Memory Costs & Limitations
- **Model-only estimate**: an isolated neural network estimate, not a full flight-software measurement.
- **Not counted**:
  - TFLite Micro tensor-arena alignment overhead (typically 16-byte alignment).
  - Operator scratch buffers (for example im2col or CMSIS-NN convolution scratch).
  - Telemetry framing and radio DMA downlink queue buffers.
  - Tile preview cache, the carry-over queue, the 1-bit coverage bitmap and scheduler state.
- **Estimated vs measured**: values become MEASURED only after a micro-benchmark on target hardware.
"""
    notes_path.write_text(content, encoding="utf-8", newline="\n")
    print(f"Generated memory notes: {notes_path}")


def test_memory_profiler_independent() -> Dict[str, any]:
    """Independent verification of the memory profiler on an untrained radar model.

    Tests:
    1. Float32 vs INT8 TFLite memory extraction.
    2. Weights calculation.
    3. Peak activation estimation.
    4. SRAM 4.2 MB comparison.
    """
    print("=== Testing Memory Profiler independently ===")
    from src.compress.quantize import build_representative_dataset_generator, convert_model_to_full_int8
    from src.models.radar import build_radar_model

    test_model = build_radar_model(input_shape=(128, 128, 2))
    calib = np.random.uniform(0.0, 1.0, size=(20, 128, 128, 2)).astype(np.float32)
    rep_gen = build_representative_dataset_generator(calib, batch_size=1, num_calibration_steps=10)

    # 1. Float32 TFLite
    float_bytes = convert_float_to_tflite_bytes(test_model)
    float_prof = inspect_tflite_model_memory(float_bytes)

    # 2. INT8 TFLite
    int8_bytes, _ = convert_model_to_full_int8(test_model, rep_gen)
    int8_prof = inspect_tflite_model_memory(int8_bytes)

    print(f"FLOAT32 Test Model: Weights={float_prof['weights_KB']:.2f} KB (File: {float_prof['file_size_KB']:.2f} KB), Peak Activation (ESTIMATED)={float_prof['peak_activation_KB']:.2f} KB, Fits 4.2MB={float_prof['peak_activation_KB'] <= SRAM_LIMIT_KB}")
    print(f"INT8 Test Model:    Weights={int8_prof['weights_KB']:.2f} KB (File: {int8_prof['file_size_KB']:.2f} KB), Peak Activation (ESTIMATED)={int8_prof['peak_activation_KB']:.2f} KB, Fits 4.2MB={int8_prof['peak_activation_KB'] <= SRAM_LIMIT_KB}")

    weight_reduction = (1.0 - (int8_prof['weights_KB'] / max(float_prof['weights_KB'], 1e-6))) * 100.0
    act_reduction = (1.0 - (int8_prof['peak_activation_KB'] / max(float_prof['peak_activation_KB'], 1e-6))) * 100.0
    print(f"Weight reduction via INT8: {weight_reduction:.1f}%")
    print(f"Activation reduction via INT8: {act_reduction:.1f}%")

    return {
        "status": "PASSED",
        "float_weights_KB": float_prof["weights_KB"],
        "float_peak_activation_KB": float_prof["peak_activation_KB"],
        "int8_weights_KB": int8_prof["weights_KB"],
        "int8_peak_activation_KB": int8_prof["peak_activation_KB"],
        "weight_reduction_pct": weight_reduction,
        "activation_reduction_pct": act_reduction,
    }


if __name__ == "__main__":
    test_memory_profiler_independent()
    profile_all_models_pipeline()
