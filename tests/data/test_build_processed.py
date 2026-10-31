"""process_event_chips / scripts/build_processed.py on a tiny synthetic raw dataset (random pixels)."""
import numpy as np
import pytest
import tifffile

from src.data.dataset import process_event_chips
from src.data.preprocess import preprocess_radar_chip


def _write_chip(raw, event, chip_no, seed):
    rng = np.random.default_rng(seed)
    hand = raw / "data" / "flood_events" / "HandLabeled"
    chip = f"{event}_{chip_no}"
    label = np.zeros((512, 512), dtype=np.int16)
    label[:256, :256] = 1                                  # a flooded quadrant
    label[500:, :] = -1                                    # a no-data border
    for folder, name, arr in [
        ("LabelHand", "LabelHand", label),
        ("JRCWaterHand", "JRCWaterHand", np.zeros((512, 512), dtype=np.uint8)),
        ("S1Hand", "S1Hand", rng.uniform(-25, -5, (2, 512, 512)).astype(np.float32)),
        ("S2Hand", "S2Hand", rng.integers(0, 4000, (13, 512, 512)).astype(np.uint16)),
    ]:
        (hand / folder).mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(hand / folder / f"{chip}_{name}.tif", arr)
    return chip


def test_a_chip_without_a_known_split_is_an_error(tmp_path):
    raw = tmp_path / "raw"
    _write_chip(raw, "Zed", 1, 0)
    with pytest.raises(ValueError, match="No split known"):
        process_event_chips("Zed", raw_dir=raw, processed_dir=tmp_path / "out")          # no chip_splits
    with pytest.raises(ValueError, match="No split known"):
        process_event_chips("Zed", raw_dir=raw, processed_dir=tmp_path / "out", chip_splits={"Other_1": "train"})


def test_npz_has_the_contract_layout_and_radar_ignores_clouds(tmp_path):
    raw = tmp_path / "raw"
    chip = _write_chip(raw, "Zed", 1, 1)
    npz, records, counts = process_event_chips("Zed", raw_dir=raw, processed_dir=tmp_path / "out",
                                               chip_splits={chip: "valid"})
    with np.load(npz) as z:
        n = len(z["tile_id"])
        assert n == counts["valid_tiles"] == len(records) == 16        # all 16 tiles have valid pixels
        assert z["optical"].shape == (n, 128, 128, 4) and z["optical"].dtype == np.uint8
        assert z["radar"].shape == (n, 128, 128, 2) and z["radar"].dtype == np.uint8
        assert set(z["split"]) == {"valid"}
        assert z["tile_id"][0] == f"{chip}_0_0"
        # tile (0,0) lies inside the flooded quadrant, tile (3,3) does not
        i00, i33 = list(z["tile_id"]).index(f"{chip}_0_0"), list(z["tile_id"]).index(f"{chip}_3_3")
        assert z["critical"][i00] == 1 and z["critical"][i33] == 0
        # Radar is never touched by the simulated clouds: it equals the plain preprocessed chip.
        s1 = tifffile.imread(raw / "data" / "flood_events" / "HandLabeled" / "S1Hand" / f"{chip}_S1Hand.tif")
        expected = preprocess_radar_chip(s1)[:128, :128]
        assert np.array_equal(z["radar"][i00], expected)


def test_given_cloud_fractions_are_used(tmp_path):
    raw = tmp_path / "raw"
    chip = _write_chip(raw, "Zed", 2, 2)
    lookup = {f"{chip}_{r}_{c}": 0.77 for r in range(4) for c in range(4)}
    npz, _, _ = process_event_chips("Zed", raw_dir=raw, processed_dir=tmp_path / "out",
                                    chip_splits={chip: "train"}, target_cloud_lookup=lookup)
    with np.load(npz) as z:
        assert np.allclose(z["cloud_frac"], 0.77)
