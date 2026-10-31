"""Specialized TinyML Tropical Cyclone Intensity Model for ORBIT-PRISM.

Modality: Satellite Cloud-top Infrared (10.8 um) + Visible band.
Outputs:
  1. p_cyclone: Probability of active cyclonic vortex / storm system.
  2. intensity_score: Normalized storm intensity [0, 1] (Saffir-Simpson / wind speed proxy).

Target footprint: <= 120 KB weights for MCU / edge deployment.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cyclone_model(
    input_shape: Tuple[int, int, int] = (128, 128, 2),
    name: str = "cyclone_tinyml_net"
) -> keras.Model:
    """Build compact depthwise-separable cyclone intensity detector.

    Input channels:
      0: Thermal IR (10.8 um) - detects cold convective cloud tops
      1: Visible / Water Vapor - detects spiral arm structure

    Architecture:
      Conv2D(16) -> DepthwiseSeparable(32) -> DepthwiseSeparable(48) -> GAP
    """
    inputs = layers.Input(shape=input_shape, dtype="uint8", name="cyclone_input")
    x = layers.Rescaling(scale=1.0 / 255.0, name="rescaling")(inputs)

    # Stem: 128x128x2 -> 64x64x16
    x = layers.Conv2D(16, (3, 3), strides=2, padding="same", use_bias=False, name="cyc_stem")(x)
    x = layers.BatchNormalization(name="cyc_bn_stem")(x)
    x = layers.ReLU(name="cyc_relu_stem")(x)

    # Stage 1: Depthwise Separable 64x64x16 -> 32x32x32
    x = layers.SeparableConv2D(32, (3, 3), strides=2, padding="same", use_bias=False, name="cyc_sep1")(x)
    x = layers.BatchNormalization(name="cyc_bn1")(x)
    x = layers.ReLU(name="cyc_relu1")(x)

    # Stage 2: Depthwise Separable 32x32x32 -> 16x16x48
    x = layers.SeparableConv2D(48, (3, 3), strides=2, padding="same", use_bias=False, name="cyc_sep2")(x)
    x = layers.BatchNormalization(name="cyc_bn2")(x)
    x = layers.ReLU(name="cyc_relu2")(x)

    gap = layers.GlobalAveragePooling2D(name="cyc_gap")(x)

    p_cyclone = layers.Dense(1, activation="sigmoid", name="p_cyclone")(gap)
    intensity_score = layers.Dense(1, activation="sigmoid", name="intensity_score")(gap)

    return keras.Model(inputs=inputs, outputs=[p_cyclone, intensity_score], name=name)
