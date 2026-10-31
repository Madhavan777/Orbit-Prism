"""Shared constants. Keep in sync with CONTEXT.md."""

OPTICAL_B = 65_536          # 128x128 tile, 4 bands, 8-bit
RADAR_B = 32_768            # 128x128 tile, 2 bands, 8-bit
L0_B = 32                   # alert packet
# L1 preview: 2,048 B = the tile downsampled 4x to 32x32 pixels, 2 bands, 8 bits per sample
# (32*32*2 = 2,048). Optical keeps green (B3) and near-infrared (B8), the two bands of the water index;
# radar keeps VV and VH. It costs 3.1% of an optical tile (6.3% of a radar tile).
L1_BYTES = 2048
# Scheduler queue entry (an accounting figure, not a measured on-device structure): tile index uint32
# + priority float32 + layer 1 B + sensor flag 1 B + pass uint16 = 12 B. The queue holds references;
# the pixels stay in flash, so queue memory is entries x QUEUE_ENTRY_B, never the payload size.
QUEUE_ENTRY_B = 12
# Event order used in the 32-byte alert (1 byte); unknown events encode as 255. Alphabetical, fixed.
EVENT_INDEX = ("Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay", "Somalia",
               "Spain", "Sri-Lanka", "USA")
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
