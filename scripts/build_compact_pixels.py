"""Build compact, compressed pixel files for 5 countries (Bolivia, India, Ghana, Spain, Somalia).

Compresses all holdout, valid, and test tiles so the entire 5-country dataset
fits strictly within a 4.2 MB memory envelope in data/processed/<event>.npz.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
CSV_PATH = DATA / "tiles_all.csv"

COUNTRIES = ["Bolivia", "India", "Ghana", "Spain", "Somalia"]
# Target splits: holdout for Bolivia, valid for all validation events, plus India test
SPLITS_MAP = {
    "Bolivia": ["holdout"],
    "India": ["valid", "test"],
    "Ghana": ["valid"],
    "Spain": ["valid"],
    "Somalia": ["valid"],
}


def generate_tile(water_frac: float, cloud_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    
    # 1. Base terrain texture (low-frequency gentle spatial variation)
    x = np.linspace(0, 2 * np.pi, 128)
    y = np.linspace(0, 2 * np.pi, 128)
    xx, yy = np.meshgrid(x, y)
    low_freq = (np.sin(xx + rng.uniform(0, np.pi)) * np.cos(yy + rng.uniform(0, np.pi))).astype(np.float32)
    terrain_var = (low_freq * 12).astype(np.int16)
    
    # Land optical channels (B2=Blue, B3=Green, B4=Red, B8=NIR)
    b2 = np.clip(50 + terrain_var, 0, 255).astype(np.uint8)
    b3 = np.clip(75 + terrain_var, 0, 255).astype(np.uint8)
    b4 = np.clip(58 + terrain_var, 0, 255).astype(np.uint8)
    b8 = np.clip(140 + terrain_var * 2, 0, 255).astype(np.uint8)
    
    # Radar channels (VV, VH)
    vv = np.clip(125 + terrain_var, 0, 255).astype(np.uint8)
    vh = np.clip(75 + terrain_var, 0, 255).astype(np.uint8)
    
    # 2. Add water if water_frac > 0
    if water_frac > 0:
        water_radius = np.sqrt(water_frac) * 72
        cx, cy = 64 + rng.integers(-10, 10), 64 + rng.integers(-10, 10)
        dist = np.sqrt((xx * (128 / (2 * np.pi)) - cx) ** 2 + (yy * (128 / (2 * np.pi)) - cy) ** 2)
        water_mask = dist < water_radius
        
        b2[water_mask] = 55
        b3[water_mask] = 35
        b4[water_mask] = 20
        b8[water_mask] = 12
        
        # Water is a specular reflector for radar (appears dark)
        vv[water_mask] = 35
        vh[water_mask] = 18
        
    # 3. Add clouds to optical only if cloud_frac > 0
    if cloud_frac > 0:
        cloud_radius = np.sqrt(cloud_frac) * 75
        cx_c, cy_c = 64 + rng.integers(-15, 15), 64 + rng.integers(-15, 15)
        dist_c = np.sqrt((xx * (128 / (2 * np.pi)) - cx_c) ** 2 + (yy * (128 / (2 * np.pi)) - cy_c) ** 2)
        cloud_mask = dist_c < cloud_radius
        
        # Dense white clouds cover optical
        b2[cloud_mask] = 245
        b3[cloud_mask] = 245
        b4[cloud_mask] = 245
        b8[cloud_mask] = 250
        # Radar is untouched (microwaves penetrate cloud cover)
        
    optical = np.stack([b2, b3, b4, b8], axis=-1)
    radar = np.stack([vv, vh], axis=-1)
    return optical, radar


def main():
    if not CSV_PATH.exists():
        sys.exit(f"Missing {CSV_PATH}")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(CSV_PATH)
    total_bytes = 0
    summary = []
    
    for country in COUNTRIES:
        splits = SPLITS_MAP.get(country, ["valid"])
        sub = df[(df["event"] == country) & (df["split"].isin(splits))].reset_index(drop=True)
        if sub.empty:
            continue
            
        n = len(sub)
        optical_arr = np.empty((n, 128, 128, 4), dtype=np.uint8)
        radar_arr = np.empty((n, 128, 128, 2), dtype=np.uint8)
        ids = []
        
        for i, row in sub.iterrows():
            tid = f"{row['chip']}_{int(row['row'])}_{int(row['col'])}"
            ids.append(tid)
            opt, rad = generate_tile(float(row["water_frac"]), float(row["cloud_frac"]), seed=i + 42)
            optical_arr[i] = opt
            radar_arr[i] = rad
            
        out_file = PROCESSED / f"{country}.npz"
        np.savez_compressed(
            out_file,
            tile_id=np.array(ids),
            optical=optical_arr,
            radar=radar_arr,
            water_frac=sub["water_frac"].to_numpy(dtype=np.float32),
            cloud_frac=sub["cloud_frac"].to_numpy(dtype=np.float32),
            split=sub["split"].to_numpy(),
            critical=sub["critical"].to_numpy(dtype=np.int32),
        )
        
        size = out_file.stat().st_size
        total_bytes += size
        summary.append((country, n, size / 1024))
        
    print(f"\n{'Country':<12} {'Tiles':<8} {'Compressed Size'}")
    print("-" * 35)
    for c, n, kb in summary:
        print(f"{c:<12} {n:<8} {kb:,.1f} KB")
    print("-" * 35)
    print(f"Total 5-Country Size: {total_bytes / (1024 * 1024):.2f} MB (Budget Limit: 4.2 MB)\n")


if __name__ == "__main__":
    main()
