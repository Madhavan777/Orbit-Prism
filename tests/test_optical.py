"""Unit tests for Optical Flood Model (Shenbaga's Module).

Tests architecture, shapes, parameter budget, random tensor forward pass,
and NDWI baseline fallback.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import tensorflow as tf

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.optical import (
    build_optical_model,
    compile_optical_model,
    calculate_model_memory,
    ndwi_baseline_predict,
    setup_gpu
)


def test_model_build_and_shapes():
    """Verify input shape and dual output heads."""
    model = build_optical_model(input_shape=(128, 128, 4))
    assert model is not None

    # Check input shape: (None, 128, 128, 4)
    assert model.input_shape == (None, 128, 128, 4)

    # Check two outputs
    assert len(model.outputs) == 2
    assert model.output_names == ["p_flood", "extent"]

    # Check output shapes: both (None, 1)
    p_flood_shape, extent_shape = model.output_shape
    assert p_flood_shape == (None, 1)
    assert extent_shape == (None, 1)


def test_model_parameter_budget():
    """Verify model parameter count and TinyML weight budget (< 100 KB)."""
    model = build_optical_model()
    memory_info = calculate_model_memory(model)

    print("\nModel Memory Profile:")
    print(f"Total params: {memory_info['total_params']:,}")
    print(f"Trainable params: {memory_info['trainable_params']:,}")
    print(f"Estimated FP32 weight size: {memory_info['float32_size_kb']} KB")
    print(f"Estimated INT8 weight size: {memory_info['int8_size_kb']} KB")

    # Assert budget constraints
    assert memory_info["within_100kb_budget"], f"FP32 weights exceed 100 KB budget: {memory_info['float32_size_kb']} KB"
    assert memory_info["total_params"] < 25_000, f"Param count too high for TinyML: {memory_info['total_params']}"


def test_random_tensor_forward_pass():
    """Test model execution using random uint8 tensors."""
    model = build_optical_model()
    compile_optical_model(model)

    batch_size = 4
    # Satellite images are uint8 in [0, 255]
    random_inputs = np.random.randint(0, 256, size=(batch_size, 128, 128, 4), dtype=np.uint8)

    p_flood_preds, extent_preds = model.predict(random_inputs, verbose=0)

    # Assert output shapes
    assert p_flood_preds.shape == (batch_size, 1)
    assert extent_preds.shape == (batch_size, 1)

    # Assert sigmoid bounds [0, 1]
    assert np.all(p_flood_preds >= 0.0) and np.all(p_flood_preds <= 1.0)
    assert np.all(extent_preds >= 0.0) and np.all(extent_preds <= 1.0)
    assert not np.isnan(p_flood_preds).any()
    assert not np.isnan(extent_preds).any()


def test_train_step_random_data():
    """Test gradient computation and 1 training step on dummy data."""
    model = build_optical_model()
    compile_optical_model(model)

    batch_size = 8
    x_dummy = np.random.randint(0, 256, size=(batch_size, 128, 128, 4), dtype=np.uint8)
    y_p_flood = np.random.randint(0, 2, size=(batch_size, 1)).astype(np.float32)
    y_extent = np.random.uniform(0.0, 1.0, size=(batch_size, 1)).astype(np.float32)

    history = model.fit(
        x_dummy,
        {"p_flood": y_p_flood, "extent": y_extent},
        epochs=1,
        batch_size=batch_size,
        verbose=0
    )

    assert "loss" in history.history
    assert not np.isnan(history.history["loss"][0])


def test_ndwi_baseline_fallback():
    """Verify fallback NDWI baseline computation."""
    batch_size = 3
    # Dummy synthetic images: (B, G, R, NIR)
    synthetic_images = np.full((batch_size, 128, 128, 4), fill_value=50, dtype=np.uint8)
    # Set Green higher than NIR in image 0 (water-like)
    synthetic_images[0, :, :, 1] = 200
    synthetic_images[0, :, :, 3] = 20

    p_flood, extent = ndwi_baseline_predict(synthetic_images)

    assert p_flood.shape == (batch_size, 1)
    assert extent.shape == (batch_size, 1)
    assert extent[0, 0] > 0.9  # Should identify high water fraction
    assert p_flood[0, 0] > 0.9  # Should trigger flood probability


if __name__ == "__main__":
    pytest.main(["-v", __file__])
