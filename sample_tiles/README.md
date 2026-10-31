# ORBIT-PRISM Sample Tiles (.npz)

These standalone sample tile files are provided for testing the **"Triage your own tile"** interface on the web dashboard (or `/api/triage/upload`).

Each `.npz` file contains:
* `optical`: `(128, 128, 4)` array, `uint8` (Sentinel-2 bands B2, B3, B4, B8)
* `radar`: `(128, 128, 2)` array, `uint8` (Sentinel-1 SAR VV, VH)

## Included Sample Tiles

| Filename | Source Tile ID | Water Frac | Cloud Frac | Target Testing Scenario |
| :--- | :--- | :---: | :---: | :--- |
| `sample_optical_clear.npz` | `Bolivia_103757_3_2` | 0.37 (Critical) | 0.08 (Clear) | Set cloud slider < 50% to test **Optical INT8** detection of floods |
| `sample_radar_cloudy.npz` | `Bolivia_242570_0_1` | 0.39 (Critical) | 0.86 (Dense Cloud) | Set cloud slider $\ge$ 50% to test **Radar INT8** detection through clouds |
| `sample_non_flood.npz` | `Bolivia_195474_0_0` | 0.00 (Dry land) | 0.85 (Cloudy) | Test true negative / non-flood rejection (low score, skipped by dial) |

## How to Test on the Web Interface

1. Open [http://localhost:8000](http://localhost:8000).
2. Scroll to the **"Triage a tile"** section and expand **"Triage your own tile"**.
3. Click **Choose File** and select any file from `sample_tiles/`.
4. Adjust the **cloud fraction slider**:
   - Set to `< 50%` to test optical model routing.
   - Set to `≥ 50%` to test radar model routing.
5. Click **Run triage** to see live predictions, probability thresholds, and the 32-byte L0 packet.
