"""ORBIT-PRISM data pipeline package.

Modules:
    download: Acquisition of Sen1Floods11 hand-labelled subset.
    tiling: 512x512 chip to 128x128 tile partitioning.
    preprocess: Optical, radar, and label preprocessing.
    cloud_simulation: Reproducible synthetic cloud simulation on optical tiles.
    split_events: Strict event-based and official dataset splitting.
    generate_metadata: Generation and validation of tiles_base.csv.
    dataset: NPZ dataset container and loader.
    validate_data: Data verification and team handoff reporting.
"""
