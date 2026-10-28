r"""
Unit and Integration Tests for Inference Engine
-----------------------------------------------
Tests:
1. Model loading & checkpoint verification
2. Image preprocessing & prediction (Mode A)
3. Audio preprocessing & prediction (Mode C)
4. Video extraction & prediction (Mode B)
5. Paired multimodal audio-video prediction (Mode D: fusion & rmsav)
6. Error handling: corrupt bytes, unsupported formats, missing checkpoints
7. Probability threshold consistency & uncertainty warnings
"""

import os
import tempfile
import unittest
import numpy as np
import cv2
import torch
from PIL import Image

from inference import InferenceEngine


class TestInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(checkpoint_dir="checkpoints", device="cpu")
        cls.has_checkpoints = len(cls.engine.available_modes) > 0

    def test_checkpoint_loading(self):
        """Verify that at least the trained checkpoints are loaded."""
        self.assertTrue(self.has_checkpoints, "No checkpoints loaded into inference engine!")
        for mode in ["audio", "video", "fusion", "rmsav"]:
            if mode in self.engine.models:
                info = self.engine.model_info[mode]
                self.assertGreater(info["total_params"], 0)
                self.assertIsNotNone(info["checkpoint_path"])

    def test_predict_image(self):
        """Test Mode A with a valid PIL image and image bytes."""
        if "video" not in self.engine.models:
            self.skipTest("Video checkpoint not available.")

        # Create a synthetic 224x224 RGB image
        img_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_arr)

        res = self.engine.predict_image(pil_img, filename="test_synthetic.png")
        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], ["Likely authentic", "Likely manipulated"])
        self.assertTrue(0.0 <= res["manipulation_probability"] <= 1.0)
        self.assertTrue(0.0 <= res["authenticity_probability"] <= 1.0)
        self.assertAlmostEqual(res["manipulation_probability"] + res["authenticity_probability"], 1.0, places=5)
        self.assertEqual(res["media_type"], "Still Image (Visual Only)")

        # Verify real image from teammates if present
        sample_jpg = "teammates/zip2/frames/video_000/frame_00.jpg"
        if os.path.exists(sample_jpg):
            with open(sample_jpg, "rb") as f:
                res_real = self.engine.predict_image(f.read(), filename="video_000_f0.jpg")
            self.assertIn(res_real["prediction"], ["Likely authentic", "Likely manipulated"])

    def test_predict_audio(self):
        """Test Mode C with existing pilot audio file."""
        if "audio" not in self.engine.models:
            self.skipTest("Audio checkpoint not available.")

        sample_wav = "teammates/zip2/audio/video_000.wav"
        if not os.path.exists(sample_wav):
            self.skipTest("Pilot audio sample not found.")

        with open(sample_wav, "rb") as f:
            audio_bytes = f.read()

        res = self.engine.predict_audio(audio_bytes, filename="video_000.wav")
        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], ["Likely authentic", "Likely manipulated"])
        self.assertTrue(0.0 <= res["manipulation_probability"] <= 1.0)
        self.assertGreater(res["duration_seconds"], 0.0)
        self.assertEqual(res["mode_used"], "audio")

    def test_predict_video(self):
        """Test Mode B with synthetic MP4 video."""
        if "video" not in self.engine.models:
            self.skipTest("Video checkpoint not available.")

        # Create a temporary synthetic 10-frame video
        temp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_vid.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_vid.name, fourcc, 10.0, (224, 224))
        for _ in range(10):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        try:
            with open(temp_vid.name, "rb") as f:
                vid_bytes = f.read()

            res = self.engine.predict_video(vid_bytes, filename="synthetic.mp4", num_frames=5)
            self.assertIn("prediction", res)
            self.assertEqual(res["frames_analyzed"], 5)
            self.assertEqual(res["total_video_frames"], 10)
            self.assertTrue(0.0 <= res["manipulation_probability"] <= 1.0)
        finally:
            if os.path.exists(temp_vid.name):
                os.remove(temp_vid.name)

    def test_predict_audio_video(self):
        """Test Mode D paired prediction with RMS-AV and Fusion."""
        if "rmsav" not in self.engine.models or "fusion" not in self.engine.models:
            self.skipTest("RMS-AV or Fusion checkpoint not available.")

        # Create synthetic video
        temp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_vid.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_vid.name, fourcc, 10.0, (224, 224))
        for _ in range(10):
            writer.write(np.zeros((224, 224, 3), dtype=np.uint8))
        writer.release()

        sample_wav = "teammates/zip2/audio/video_000.wav"
        if not os.path.exists(sample_wav):
            if os.path.exists(temp_vid.name):
                os.remove(temp_vid.name)
            self.skipTest("Sample wav not found.")

        try:
            with open(temp_vid.name, "rb") as f_v, open(sample_wav, "rb") as f_a:
                v_bytes = f_v.read()
                a_bytes = f_a.read()

            # Test RMS-AV
            res_rmsav = self.engine.predict_audio_video(v_bytes, a_bytes, mode="rmsav")
            self.assertIn("prediction", res_rmsav)
            self.assertIsNotNone(res_rmsav["gate_weights"])
            self.assertIn("audio_weight", res_rmsav["gate_weights"])
            self.assertIn("video_weight", res_rmsav["gate_weights"])
            sum_gates = res_rmsav["gate_weights"]["audio_weight"] + res_rmsav["gate_weights"]["video_weight"]
            self.assertAlmostEqual(sum_gates, 1.0, places=3)

            # Test Standard Fusion
            res_fusion = self.engine.predict_audio_video(v_bytes, a_bytes, mode="fusion")
            self.assertIn("prediction", res_fusion)
            self.assertIsNone(res_fusion["gate_weights"])
        finally:
            if os.path.exists(temp_vid.name):
                os.remove(temp_vid.name)

    def test_corrupt_inputs(self):
        """Verify that corrupt byte inputs raise clean ValueError without crashing."""
        corrupt_bytes = b"not_a_valid_file_at_all"

        # Corrupt image
        with self.assertRaises(ValueError):
            self.engine.preprocess_image(corrupt_bytes)

        # Corrupt audio
        with self.assertRaises(ValueError):
            self.engine.preprocess_audio(corrupt_bytes)

        # Corrupt video
        with self.assertRaises(ValueError):
            self.engine.extract_video_frames(corrupt_bytes)

    def test_threshold_uncertainty_flag(self):
        """Verify that is_uncertain flag correctly identifies borderline probabilities."""
        if "video" in self.engine.models:
            # Test synthetic image
            img = Image.new("RGB", (224, 224), color="blue")
            res = self.engine.predict_image(img)
            p = res["manipulation_probability"]
            expected_uncertain = (0.45 <= p <= 0.55)
            self.assertEqual(res["is_uncertain"], expected_uncertain)


if __name__ == "__main__":
    unittest.main()
