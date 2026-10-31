"""Dataset Inspection Utility for Optical Flood Pipeline.

Inspects actual NPZ files provided by Agnel in `data/processed/`.
Checks keys, shapes, data types, value ranges, and split distributions
without assuming hardcoded column names.
"""

from pathlib import Path
import sys
import numpy as np


def inspect_npz_files(data_dir: Path) -> None:
    """Inspect all NPZ files found in data directory."""
    print("=" * 70)
    print(f"[Agnel Data Pipeline Inspector] Scanning directory: {data_dir.resolve()}")
    print("=" * 70)

    if not data_dir.exists():
        print(f"Directory {data_dir} does not exist.")
        return

    npz_files = sorted(list(data_dir.glob("*.npz")))
    if not npz_files:
        print("No .npz files found in data directory yet.")
        print("Waiting for Agnel to deliver processed dataset files.")
        return

    print(f"Found {len(npz_files)} .npz file(s):")
    for f in npz_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    print("\nDetailed Key & Shape Analysis:")
    print("-" * 70)

    for f in npz_files:
        print(f"\n>>> File: {f.name}")
        with np.load(f, allow_pickle=True) as data:
            keys = list(data.files)
            print(f"Available keys ({len(keys)}): {keys}")

            for k in keys:
                arr = data[k]
                dtype = arr.dtype
                shape = arr.shape
                sample = arr[:2] if len(arr) > 0 else "empty"

                if np.issubdtype(dtype, np.number) and arr.size > 0:
                    min_val = np.min(arr)
                    max_val = np.max(arr)
                    mean_val = np.mean(arr)
                    print(
                        f"  * {k:<15} shape={str(shape):<18} dtype={str(dtype):<10} "
                        f"min={min_val:<8.3g} max={max_val:<8.3g} mean={mean_val:<8.3g}"
                    )
                else:
                    print(
                        f"  * {k:<15} shape={str(shape):<18} dtype={str(dtype):<10} "
                        f"sample={sample}"
                    )


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    target_dir = base_dir / "data" / "processed"
    inspect_npz_files(target_dir)
