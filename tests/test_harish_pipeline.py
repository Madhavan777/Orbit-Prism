"""Comprehensive independent test suite for Harish D's deliverables in ORBIT-PRISM.

Verifies:
1. TensorFlow installation
2. GPU detection
3. Radar CNN initialization
4. Radar CNN forward pass
5. Two output heads
6. Output shapes
7. Model saving and loading
8. Quantization conversion with a temporary test model
9. INT8 input and output handling
10. INT8 inference
11. Memory profiling (SRAM comparison)
12. CSV validation logic
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.compress.predict_int8 import Int8ModelRunner, validate_scores_csv
from src.compress.profile_memory import inspect_tflite_model_memory
from src.compress.quantize import build_representative_dataset_generator, convert_model_to_full_int8
from src.models.radar import build_radar_model, run_otsu_baseline


class TestHarishIndependentPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("STARTING INDEPENDENT TEST SUITE (HARISH D / ORBIT-PRISM)")
        print("=======================================================")

    def test_01_tensorflow_installation(self):
        """1. TensorFlow installation check."""
        print("\n[Test 1] Verifying TensorFlow installation...")
        version = tf.__version__
        self.assertIsNotNone(version)
        print(f"✓ TensorFlow is installed (version: {version})")

    def test_02_gpu_detection(self):
        """2. GPU detection check."""
        print("\n[Test 2] Checking GPU devices...")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"✓ Physical GPUs detected by TensorFlow: {gpus} (CPU fallback active on Windows native)")

    def test_03_radar_cnn_initialization(self):
        """3. Radar CNN initialization check."""
        print("\n[Test 3] Verifying Radar CNN architecture initialization...")
        model = build_radar_model(input_shape=(128, 128, 2))
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 128, 128, 2))
        self.assertGreater(model.count_params(), 10000)
        print(f"✓ Radar CNN initialized with {model.count_params():,} parameters.")

    def test_04_05_06_forward_pass_and_output_heads(self):
        """4, 5, 6. Radar CNN forward pass, two output heads, and output shapes."""
        print("\n[Tests 4, 5, 6] Verifying forward pass, dual heads, and shapes...")
        model = build_radar_model(input_shape=(128, 128, 2))
        batch_size = 3
        dummy_input = np.random.uniform(0.0, 1.0, size=(batch_size, 128, 128, 2)).astype(np.float32)

        p_pred, ext_pred = model(dummy_input, training=False)
        p_pred_np = p_pred.numpy()
        ext_pred_np = ext_pred.numpy()

        # Check shapes
        self.assertEqual(p_pred_np.shape, (batch_size, 1))
        self.assertEqual(ext_pred_np.shape, (batch_size, 1))

        # Check ranges [0, 1]
        self.assertTrue(np.all((p_pred_np >= 0.0) & (p_pred_np <= 1.0)))
        self.assertTrue(np.all((ext_pred_np >= 0.0) & (ext_pred_np <= 1.0)))

        print("✓ Forward pass executed successfully.")
        print(f"✓ Output 1 (p_flood) shape: {p_pred_np.shape}, min: {p_pred_np.min():.4f}, max: {p_pred_np.max():.4f}")
        print(f"✓ Output 2 (extent) shape: {ext_pred_np.shape}, min: {ext_pred_np.min():.4f}, max: {ext_pred_np.max():.4f}")

    def test_07_model_saving_and_loading(self):
        """7. Model saving and reloading check."""
        print("\n[Test 7] Verifying model saving and reloading...")
        model = build_radar_model(input_shape=(128, 128, 2))
        dummy_input = np.random.uniform(0.0, 1.0, size=(2, 128, 128, 2)).astype(np.float32)
        orig_p, orig_ext = model.predict(dummy_input, verbose=0)

        temp_path = ROOT_DIR / "models" / "temp_unit_test.keras"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(temp_path)

        reloaded = tf.keras.models.load_model(temp_path)
        re_p, re_ext = reloaded.predict(dummy_input, verbose=0)

        self.assertTrue(np.allclose(orig_p, re_p, atol=1e-5))
        self.assertTrue(np.allclose(orig_ext, re_ext, atol=1e-5))

        if temp_path.exists():
            temp_path.unlink()
        print("✓ Model saved and reloaded with identical prediction tensors.")

    def test_08_09_quantization_and_int8_tensors(self):
        """8, 9. TFLite INT8 post-training quantization and tensor scale/zero_point verification."""
        print("\n[Tests 8, 9] Verifying TFLite INT8 quantization and tensor types...")
        model = build_radar_model(input_shape=(128, 128, 2))
        calib_data = np.random.uniform(0.0, 1.0, size=(10, 128, 128, 2)).astype(np.float32)
        rep_gen = build_representative_dataset_generator(calib_data, batch_size=1, num_calibration_steps=5)

        tflite_bytes, info = convert_model_to_full_int8(model, rep_gen)

        self.assertGreater(len(tflite_bytes), 0)
        self.assertEqual(info["status"], "CONVERTED_INT8")

        # Verify int8 input details
        inp = info["inputs"][0]
        self.assertIn("int8", inp["dtype"].lower())
        self.assertGreater(inp["scale"], 0.0)

        # Verify int8 output details
        for out in info["outputs"]:
            self.assertIn("int8", out["dtype"].lower())
            self.assertGreater(out["scale"], 0.0)

        print(f"✓ Converted to full INT8 model ({info['model_KB']:.1f} KB).")
        print(f"✓ Input tensor dtype: {inp['dtype']}, scale: {inp['scale']:.6f}, zero_point: {inp['zero_point']}")
        print(f"✓ Output tensor count: {len(info['outputs'])}, all confirmed int8.")

    def test_10_int8_inference(self):
        """10. INT8 inference pipeline execution and numerical bounds."""
        print("\n[Test 10] Verifying INT8 inference pipeline...")
        model = build_radar_model(input_shape=(128, 128, 2))
        calib_data = np.random.uniform(0.0, 1.0, size=(10, 128, 128, 2)).astype(np.float32)
        rep_gen = build_representative_dataset_generator(calib_data, batch_size=1, num_calibration_steps=5)
        tflite_bytes, _ = convert_model_to_full_int8(model, rep_gen)

        runner = Int8ModelRunner(tflite_bytes)
        test_tile = np.random.uniform(0.0, 1.0, size=(128, 128, 2)).astype(np.float32)

        p, ext = runner.predict_single(test_tile)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertGreaterEqual(ext, 0.0)
        self.assertLessEqual(ext, 1.0)
        print(f"✓ INT8 inference executed: p_flood={p:.4f}, extent={ext:.4f} (strictly bounded in [0, 1]).")

    def test_11_memory_profiling(self):
        """11. Memory profiling and 4.2 MB SRAM limit comparison."""
        print("\n[Test 11] Verifying memory profiler on INT8 model...")
        model = build_radar_model(input_shape=(128, 128, 2))
        calib_data = np.random.uniform(0.0, 1.0, size=(10, 128, 128, 2)).astype(np.float32)
        rep_gen = build_representative_dataset_generator(calib_data, batch_size=1, num_calibration_steps=5)
        tflite_bytes, _ = convert_model_to_full_int8(model, rep_gen)

        prof = inspect_tflite_model_memory(tflite_bytes)
        self.assertGreater(prof["weights_KB"], 0.0)
        self.assertGreater(prof["peak_activation_KB"], 0.0)
        self.assertLess(prof["peak_activation_KB"], 4101.56)  # 4.2 MB limit

        print(f"✓ Measured Weights: {prof['weights_KB']:.2f} KB")
        print(f"✓ Estimated Peak Activation: {prof['peak_activation_KB']:.2f} KB")
        print(f"✓ Fits 4.2 MB SRAM Budget: {prof['peak_activation_KB'] <= 4101.56}")

    def test_12_csv_validation_logic(self):
        """12. CSV validation logic for required schema (tile_id,p_flood,extent)."""
        print("\n[Test 12] Verifying CSV schema validation...")
        temp_csv = ROOT_DIR / "results" / "temp_test_scores.csv"
        temp_csv.parent.mkdir(parents=True, exist_ok=True)

        valid_df = pd.DataFrame(
            {
                "tile_id": ["chip1_0_0", "chip1_0_1"],
                "p_flood": [0.15, 0.92],
                "extent": [0.10, 0.88],
            }
        )
        valid_df.to_csv(temp_csv, index=False)
        self.assertTrue(validate_scores_csv(temp_csv))

        # Check invalid schema rejection
        bad_df = pd.DataFrame({"id": ["c1"], "score": [0.5]})
        bad_df.to_csv(temp_csv, index=False)
        with self.assertRaises(ValueError):
            validate_scores_csv(temp_csv)

        if temp_csv.exists():
            temp_csv.unlink()
        print("✓ CSV schema validation passed (strictly enforces tile_id,p_flood,extent).")

    def test_13_otsu_baseline(self):
        """13. Fallback Otsu thresholding on VV band."""
        print("\n[Test 13] Verifying fallback Otsu baseline...")
        test_images = np.random.uniform(0.0, 1.0, size=(2, 128, 128, 2)).astype(np.float32)
        p_sc, ext_sc = run_otsu_baseline(test_images)
        self.assertEqual(len(p_sc), 2)
        self.assertEqual(len(ext_sc), 2)
        self.assertTrue(np.all((p_sc >= 0.0) & (p_sc <= 1.0)))
        print(f"✓ Otsu baseline functional: p_scores={p_sc}, extents={ext_sc}")


if __name__ == "__main__":
    unittest.main()
