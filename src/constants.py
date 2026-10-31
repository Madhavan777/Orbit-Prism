"""Shared constants. Keep in sync with CONTEXT.md."""

OPTICAL_B = 65_536          # 128x128 tile, 4 bands, 8-bit
RADAR_B = 32_768            # 128x128 tile, 2 bands, 8-bit
L0_B = 32                   # alert packet
L1_BYTES = None             # preview size: Akhilesh decides and sets it
PASS_BYTES = 19_200_000     # 256 kbps x 600 s
SRAM_BYTES = 4_200_000      # on-device memory ceiling (4.2 MB)

TILE = 128
GRID = 4
CRITICAL_THRESHOLD = 0.30
CLOUD_SWITCH = 0.5          # cloud_frac >= this -> radar
SCALED_PASS_FRACTION = 0.20 # of the event's all-optical volume; label "scaled"
TEST_SPLIT = "holdout"      # Bolivia
# Whole events whose `train` tiles are held back for early stopping / model selection by BOTH
# models, so the `valid` split stays untouched until calibration. (Radar default matches.)
INTERNAL_VAL_EVENTS = ("Spain", "Somalia")
