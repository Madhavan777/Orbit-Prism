"""Specialized TinyML Wildfire Detection Model for ORBIT-PRISM.

Modality: Optical (Red/NIR) + Short-Wave Infrared (SWIR) / Thermal bands.
Outputs:
  1. p_fire: Probability of active wildfire / thermal hotspot.
  2. burn_extent: Fractional perimeter area affected by burn scar.

Footprint target: <= 60 KB weights for MCU deployment.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_wildfire_model(
    input_shape: Tuple[int, int, int] = (128, 128, 4),
    name: str = "wildfire_tinyml_net"
) -> keras.Model:
    """Build lightweight TinyML wildfire detector.

    Input channels:
      0: Red (B4)
      1: NIR (B8A)
      2: SWIR1 (B11 - 1.61 um)
      3: SWIR2 (B12 - 2.19 um, sensitive to high temperature thermal anomalies)

    Architecture:
      Conv2D(8) -> BN -> ReLU (stride 2) -> (64, 64, 8)
      Conv2D(16) -> BN -> ReLU (stride 2) -> (32, 32, 16)
      Conv2D(32) -> BN -> ReLU (stride 2) -> (16, 16, 32)
      GlobalAveragePooling2D -> (32,)
      p_fire (Dense 1, sigmoid)
      burn_extent (Dense 1, sigmoid)
    """
    inputs = layers.Input(shape=input_shape, dtype="uint8", name="wildfire_input")
    x = layers.Rescaling(scale=1.0 / 255.0, name="rescaling")(inputs)

    # Stage 1: (64, 64, 8)
    x = layers.Conv2D(8, (3, 3), strides=2, padding="same", use_bias=False, name="wf_conv1")(x)
    x = layers.BatchNormalization(name="wf_bn1")(x)
    x = layers.ReLU(name="wf_relu1")(x)

    # Stage 2: (32, 32, 16)
    x = layers.Conv2D(16, (3, 3), strides=2, padding="same", use_bias=False, name="wf_conv2")(x)
    x = layers.BatchNormalization(name="wf_bn2")(x)
    x = layers.ReLU(name="wf_relu2")(x)

    # Stage 3: (16, 16, 32)
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", use_bias=False, name="wf_conv3")(x)
    x = layers.BatchNormalization(name="wf_bn3")(x)
    x = layers.ReLU(name="wf_relu3")(x)

    gap = layers.GlobalAveragePooling2D(name="wf_gap")(x)

    p_fire = layers.Dense(1, activation="sigmoid", name="p_fire")(gap)
    burn_extent = layers.Dense(1, activation="sigmoid", name="burn_extent")(gap)

    return keras.Model(inputs=inputs, outputs=[p_fire, burn_extent], name=name)


def compute_nbr_index(images: np.ndarray, nir_idx: int = 1, swir_idx: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalized Burn Ratio (NBR) baseline.

    NBR = (NIR - SWIR2) / (NIR + SWIR2 + eps)
    Active fires and fresh burn scars have very low or negative NBR.
    """
    img_f = images.astype(np.float32)
    nir = img_f[..., nir_idx]
    swir = img_f[..., swir_idx]

    denom = nir + swir + 1e-7
    nbr = (nir - swir) / denom

    # Burned area mask: NBR < 0.1
    burn_mask = nbr < 0.1
    burn_extent = np.mean(burn_mask, axis=(1, 2)).astype(np.float32).reshape(-1, 1)

    # Probability of active fire based on SWIR intensity
    swir_norm = swir / 255.0
    hotspot_intensity = np.max(swir_norm, axis=(1, 2)).astype(np.float32).reshape(-1, 1)
    p_fire = (1.0 / (1.0 + np.exp(-15.0 * (hotspot_intensity - 0.40)))).astype(np.float32)

    return p_fire, burn_extent
