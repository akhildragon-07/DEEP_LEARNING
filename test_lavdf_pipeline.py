r"""
Unit Test Suite for LAV-DF Multimodal Pipeline
----------------------------------------------
Verifies:
1. Model forward passes across all 4 modes (audio, video, fusion, rmsav).
2. Reliability gate normalization (ga + gv = 1.0) and missing-stream validity masking.
3. Preprocessor output shapes and standardization.
4. Decision classification and uncertainty margin logic.
"""

import unittest
import numpy as np
import torch

from lavdf_pipeline.model import build_lavdf_model, count_parameters
from lavdf_pipeline.preprocessing import PreprocessingConfig, VideoPreprocessor, AudioPreprocessor
from lavdf_pipeline.inference import LAVDFInferenceEngine, INCONCLUSIVE_VERDICT


class TestLAVDFPipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.b_size = 2
        self.t_frames = 5
        self.img_size = 224
        self.n_mels = 80
        self.t_audio = 400
        self.dim = 128

        self.dummy_video = torch.randn(self.b_size, self.t_frames, 3, self.img_size, self.img_size)
        self.dummy_audio = torch.randn(self.b_size, 1, self.n_mels, self.t_audio)
        self.has_v = torch.tensor([True, True], dtype=torch.bool)
        self.has_a = torch.tensor([True, True], dtype=torch.bool)

    def test_audio_model_forward(self):
        model = build_lavdf_model(mode="audio", dim=self.dim, pretrained=False)
        batch = {
            "video": None,
            "audio": self.dummy_audio,
            "has_video": torch.tensor([False, False]),
            "has_audio": self.has_a,
        }
        out = model(batch)
        self.assertIn("logit_av", out)
        self.assertEqual(out["logit_av"].shape, (self.b_size,))

    def test_video_model_forward(self):
        model = build_lavdf_model(mode="video", dim=self.dim, pretrained=False)
        batch = {
            "video": self.dummy_video,
            "audio": None,
            "has_video": self.has_v,
            "has_audio": torch.tensor([False, False]),
        }
        out = model(batch)
        self.assertIn("logit_av", out)
        self.assertEqual(out["logit_av"].shape, (self.b_size,))

    def test_fusion_model_forward(self):
        model = build_lavdf_model(mode="fusion", dim=self.dim, pretrained=False)
        batch = {
            "video": self.dummy_video,
            "audio": self.dummy_audio,
            "has_video": self.has_v,
            "has_audio": self.has_a,
        }
        out = model(batch)
        self.assertIn("logit_av", out)
        self.assertEqual(out["logit_av"].shape, (self.b_size,))

    def test_rmsav_model_forward_and_gating(self):
        model = build_lavdf_model(mode="rmsav", dim=self.dim, pretrained=False)
        batch = {
            "video": self.dummy_video,
            "audio": self.dummy_audio,
            "has_video": self.has_v,
            "has_audio": self.has_a,
        }
        out = model(batch)
        self.assertIn("logit_av", out)
        self.assertIn("logit_v", out)
        self.assertIn("logit_a", out)
        self.assertIn("gate", out)

        self.assertEqual(out["logit_av"].shape, (self.b_size,))
        self.assertEqual(out["logit_v"].shape, (self.b_size,))
        self.assertEqual(out["logit_a"].shape, (self.b_size,))
        self.assertEqual(out["gate"].shape, (self.b_size, 2))

        # Check gate normalization: ga + gv = 1.0
        gates = out["gate"].detach().numpy()
        sum_gates = np.sum(gates, axis=-1)
        np.testing.assert_allclose(sum_gates, np.ones(self.b_size), atol=1e-5)

    def test_rmsav_validity_masking(self):
        """Verify that when a stream is missing, validity mask directs 100% of weight to the active stream."""
        model = build_lavdf_model(mode="rmsav", dim=self.dim, pretrained=False)

        # Video missing: has_video = False, has_audio = True
        batch_missing_video = {
            "video": torch.zeros_like(self.dummy_video),
            "audio": self.dummy_audio,
            "has_video": torch.tensor([False, False], dtype=torch.bool),
            "has_audio": torch.tensor([True, True], dtype=torch.bool),
        }
        out_mv = model(batch_missing_video)
        gates_mv = out_mv["gate"].detach().numpy()
        # [g_a, g_v] -> g_a should be 1.0, g_v should be 0.0
        np.testing.assert_allclose(gates_mv[:, 0], np.ones(self.b_size), atol=1e-4)
        np.testing.assert_allclose(gates_mv[:, 1], np.zeros(self.b_size), atol=1e-4)

        # Audio missing: has_video = True, has_audio = False
        batch_missing_audio = {
            "video": self.dummy_video,
            "audio": torch.zeros_like(self.dummy_audio),
            "has_video": torch.tensor([True, True], dtype=torch.bool),
            "has_audio": torch.tensor([False, False], dtype=torch.bool),
        }
        out_ma = model(batch_missing_audio)
        gates_ma = out_ma["gate"].detach().numpy()
        np.testing.assert_allclose(gates_ma[:, 0], np.zeros(self.b_size), atol=1e-4)
        np.testing.assert_allclose(gates_ma[:, 1], np.ones(self.b_size), atol=1e-4)

    def test_preprocessors(self):
        cfg = PreprocessingConfig(num_frames=5, img_size=224, n_mels=80, target_audio_frames=400)
        v_proc = VideoPreprocessor(cfg, is_train=False)
        a_proc = AudioPreprocessor(cfg, is_train=False)

        # Video frames
        raw_frames = np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8)
        v_t = v_proc.process_frames(raw_frames)
        self.assertEqual(v_t.shape, (5, 3, 224, 224))
        self.assertEqual(v_t.dtype, torch.float32)

        # Audio waveform
        raw_audio = torch.randn(1, 16000 * 2)  # 2 seconds at 16 kHz
        a_t = a_proc.process_waveform(raw_audio, sr=16000)
        self.assertEqual(a_t.shape, (1, 80, 400))
        self.assertEqual(a_t.dtype, torch.float32)

    def test_uncertainty_classification(self):
        # Near chance level prob 50.82%
        v, inc, (low, up) = LAVDFInferenceEngine.classify_with_uncertainty(0.5082, threshold=0.50, uncertainty_margin=0.05)
        self.assertEqual(v, INCONCLUSIVE_VERDICT)
        self.assertTrue(inc)
        self.assertEqual(low, 0.45)
        self.assertEqual(up, 0.55)


if __name__ == "__main__":
    unittest.main()
