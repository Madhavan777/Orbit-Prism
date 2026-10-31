"""Optical Flood Classification and Water Extent Regression Model.

Module: Shenbaga (Optical Flood Lead)
Target: TinyML deployment (~100 KB or less weights), INT8-compatible operations.
"""

from typing import Tuple, Dict, Any, Optional
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def setup_gpu(memory_limit_mb: Optional[int] = None) -> Dict[str, Any]:
    """Configure TensorFlow GPU memory growth and device placement for RTX 3050 (4 GB).

    Avoids unnecessary full VRAM pre-allocation. Falls back smoothly to CPU if GPU
    is unavailable.

    Args:
        memory_limit_mb: Optional hard memory limit in MB (e.g., 2048 or 3072).

    Returns:
        Dict containing device info and status.
    """
    gpus = tf.config.list_physical_devices("GPU")
    status = {"gpu_available": False, "devices": [], "mode": "CPU"}

    if gpus:
        try:
            for gpu in gpus:
                if memory_limit_mb:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            status["gpu_available"] = True
            status["devices"] = [gpu.name for gpu in gpus]
            status["logical_devices"] = [gpu.name for gpu in logical_gpus]
            status["mode"] = f"GPU ({len(gpus)} detected, memory growth enabled)"
            print(f"[Optical] TensorFlow GPU ready: {status['devices']}")
        except RuntimeError as e:
            print(f"[Optical] GPU memory growth error: {e}. Falling back to default/CPU.")
            status["mode"] = f"Fallback error: {e}"
    else:
        print("[Optical] No GPU detected by TensorFlow. Running on CPU.")
        status["mode"] = "CPU fallback"

    return status


def build_optical_model(
    input_shape: Tuple[int, int, int] = (128, 128, 4),
    name: str = "optical_flood_net"
) -> keras.Model:
    """Build lightweight TinyML dual-output optical flood model.

    Architecture:
      - Input: 128x128x4 uint8 optical satellite image (e.g., R, G, B, NIR)
      - Normalization: Scale uint8 [0, 255] to float32 [0.0, 1.0]
      - Stage 1: Conv2D(8, 3x3, stride 2, padding='same') -> BatchNorm -> ReLU
      - Stage 2: Conv2D(16, 3x3, stride 2, padding='same') -> BatchNorm -> ReLU
      - Stage 3: Conv2D(32, 3x3, stride 2, padding='same') -> BatchNorm -> ReLU
      - Stage 4: Conv2D(32, 3x3, stride 2, padding='same') -> BatchNorm -> ReLU
      - GlobalAveragePooling2D
      - Head 1 (p_flood): Dense(1, sigmoid) -> binary flood classification
      - Head 2 (extent): Dense(1, sigmoid) -> water coverage fraction [0, 1]

    All ops are standard INT8-quantizable operations for TFLite / MCU deployment.
    Weight size is well under the 100 KB budget.

    Args:
        input_shape: Shape tuple (H, W, C), default (128, 128, 4).
        name: Name of the Keras model.

    Returns:
        Compiled or uncompiled keras.Model with outputs [p_flood, extent].
    """
    inputs = layers.Input(shape=input_shape, dtype="uint8", name="optical_input")

    # Rescaling uint8 [0, 255] to [0.0, 1.0]
    x = layers.Rescaling(scale=1.0 / 255.0, name="rescaling")(inputs)

    # Conv stage 1: 128x128x4 -> 64x64x8
    x = layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="conv1"
    )(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)

    # Conv stage 2: 64x64x8 -> 32x32x16
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)

    # Conv stage 3: 32x32x16 -> 16x16x32
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="conv3"
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)

    # Conv stage 4: 16x16x32 -> 8x8x32
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="conv4"
    )(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)

    # Global Average Pooling: 8x8x32 -> 32
    pool = layers.GlobalAveragePooling2D(name="gap")(x)

    # Head 1: Flood probability (binary classification: water_frac >= 0.30)
    p_flood = layers.Dense(1, activation="sigmoid", name="p_flood")(pool)

    # Head 2: Water extent regression (continuous fraction in [0, 1])
    extent = layers.Dense(1, activation="sigmoid", name="extent")(pool)

    model = keras.Model(inputs=inputs, outputs=[p_flood, extent], name=name)
    return model


def compile_optical_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    loss_weights: Optional[Dict[str, float]] = None
) -> keras.Model:
    """Compile optical model with BCE for p_flood and MSE for extent.

    Args:
        model: Model instance from build_optical_model.
        learning_rate: Initial learning rate for Adam optimizer.
        loss_weights: Loss weights dict, e.g. {'p_flood': 1.0, 'extent': 1.0}.

    Returns:
        Compiled model.
    """
    if loss_weights is None:
        loss_weights = {"p_flood": 1.0, "extent": 1.0}

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    losses = {
        "p_flood": keras.losses.BinaryCrossentropy(),
        "extent": keras.losses.MeanSquaredError()
    }

    metrics = {
        "p_flood": [
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.AUC(name="auroc")
        ],
        "extent": [
            keras.metrics.MeanAbsoluteError(name="mae")
        ]
    }

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    return model


def calculate_model_memory(model: keras.Model, batch_size: int = 1) -> Dict[str, Any]:
    """Calculate parameter counts, weight size, and peak activation footprint.

    Target: < 100 KB weights for TinyML deployment, compatible with Harish's INT8 pipeline.

    Args:
        model: Keras model instance.
        batch_size: Batch size for activation memory calculation (default 1 for edge inference).

    Returns:
        Dict containing parameter counts, estimated sizes, and activation memory footprint.
    """
    total_params = model.count_params()
    trainable_params = int(np.sum([keras.backend.count_params(p) for p in model.trainable_weights]))
    non_trainable_params = int(np.sum([keras.backend.count_params(p) for p in model.non_trainable_weights]))

    float32_kb = (total_params * 4) / 1024.0
    int8_kb = (total_params * 1) / 1024.0

    # Calculate activation tensor sizes for peak SRAM estimation on MCU
    activation_elements = 0
    peak_layer_elements = 0
    for layer in model.layers:
        output_shape = getattr(layer, "output_shape", getattr(layer, "shape", None))
        if output_shape is None:
            continue
        if isinstance(output_shape, list):
            for s in output_shape:
                if s is not None:
                    elements = int(np.prod([dim for dim in s if dim is not None]))
                    activation_elements += elements
                    peak_layer_elements = max(peak_layer_elements, elements)
        else:
            elements = int(np.prod([dim for dim in output_shape if dim is not None]))
            activation_elements += elements
            peak_layer_elements = max(peak_layer_elements, elements)

    peak_activation_float32_kb = (peak_layer_elements * 4 * batch_size) / 1024.0
    peak_activation_int8_kb = (peak_layer_elements * 1 * batch_size) / 1024.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "float32_size_kb": round(float32_kb, 2),
        "int8_size_kb": round(int8_kb, 2),
        "within_100kb_budget": float32_kb <= 100.0,
        "peak_activation_float32_kb": round(peak_activation_float32_kb, 2),
        "peak_activation_int8_kb": round(peak_activation_int8_kb, 2),
        "total_activation_elements": activation_elements
    }


def save_optical_float_model(model: keras.Model, filepath: str = "models/optical_float.keras") -> str:
    """Save the unquantized float32 optical model for evaluation and handoff to Harish.

    Args:
        model: Trained or compiled optical model.
        filepath: Destination file path.

    Returns:
        Canonical saved model path.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    actual_size_kb = out_path.stat().st_size / 1024.0
    print(f"[Optical] Model successfully saved to {out_path} ({actual_size_kb:.2f} KB)")
    return str(out_path)


def ndwi_baseline_predict(images: np.ndarray, green_idx: int = 1, nir_idx: int = 3, threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback NDWI baseline predictor.

    NDWI = (Green - NIR) / (Green + NIR + eps)
    Calculates water fraction per tile and triggers flood if water_frac >= 0.30.

    Args:
        images: Array of shape (N, H, W, C) in uint8 [0, 255].
        green_idx: Channel index for Green band (default 1).
        nir_idx: Channel index for NIR band (default 3).
        threshold: NDWI threshold where > threshold is considered water.

    Returns:
        p_flood: Flood probability array (N, 1).
        extent: Water extent fraction array (N, 1).
    """
    img_f = images.astype(np.float32)
    green = img_f[..., green_idx]
    nir = img_f[..., nir_idx]

    denom = green + nir + 1e-7
    ndwi = (green - nir) / denom

    water_mask = ndwi > threshold
    extent = np.mean(water_mask, axis=(1, 2)).astype(np.float32).reshape(-1, 1)

    # Critical flood definition: water_frac >= 0.30
    # Sigmoidal mapping centered at 0.30 with sharp transition
    p_flood = (1.0 / (1.0 + np.exp(-20.0 * (extent - 0.30)))).astype(np.float32)

    return p_flood, extent
