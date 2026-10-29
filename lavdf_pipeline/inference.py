r"""
LAV-DF Model Inference Engine
-----------------------------
Provides safe, standalone model inference for video, audio, and multimodal inputs.
- Validates checkpoint existence (raises explicit errors; never silently uses random weights).
- Supports configurable decision thresholds and uncertainty margin bands.
- Accurately interprets raw logits, probabilities, and RMS-AV reliability gate weights.
"""

import os
import time
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from lavdf_pipeline.model import build_lavdf_model, count_parameters
from lavdf_pipeline.preprocessing import PreprocessingConfig, VideoPreprocessor, AudioPreprocessor

INCONCLUSIVE_VERDICT = "INCONCLUSIVE — MODEL SCORE NEAR THRESHOLD"


class LAVDFInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        mode: str = "rmsav",
        dim: int = 256,
        device: Optional[str] = None,
    ):
        self.mode = mode.lower()
        self.dim = dim
        self.checkpoint_path = os.path.abspath(checkpoint_path)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at: {self.checkpoint_path}. "
                "Refusing to initialize inference with random uncalibrated weights."
            )

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint weights
        self.model = build_lavdf_model(mode=self.mode, dim=self.dim, pretrained=False)
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)

        if "model_state_dict" not in ckpt:
            raise ValueError(f"Invalid checkpoint format in {self.checkpoint_path}. Missing 'model_state_dict'.")

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.total_params, _ = count_parameters(self.model)
        self.best_metric = ckpt.get("best_metric")
        self.epoch = ckpt.get("epoch")

        # Preprocessors
        self.cfg = PreprocessingConfig(use_augmentation=False)
        self.video_preprocessor = VideoPreprocessor(self.cfg, is_train=False)
        self.audio_preprocessor = AudioPreprocessor(self.cfg, is_train=False)

        print(f"[InferenceEngine] Ready: mode='{self.mode}' from {os.path.basename(self.checkpoint_path)} ({self.total_params:,} params)")

    @staticmethod
    def classify_with_uncertainty(
        prob: float,
        threshold: float = 0.50,
        uncertainty_margin: float = 0.05,
    ) -> Tuple[str, bool, Tuple[float, float]]:
        """Classifies probability into three states: Authentic, Manipulated, or Inconclusive."""
        low_b = max(0.0, float(threshold - uncertainty_margin))
        up_b = min(1.0, float(threshold + uncertainty_margin))

        if low_b <= prob <= up_b:
            return INCONCLUSIVE_VERDICT, True, (round(low_b, 4), round(up_b, 4))
        elif prob > up_b:
            return "Likely manipulated", False, (round(low_b, 4), round(up_b, 4))
        else:
            return "Likely authentic", False, (round(low_b, 4), round(up_b, 4))

    @torch.inference_mode()
    def predict_video(
        self,
        video_path: str,
        threshold: float = 0.50,
        uncertainty_margin: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Runs multimodal inference on an MP4 video clip (processing both visual frames and audio track).
        """
        t0 = time.time()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video file not found: {video_path}")

        v_tensor, num_frames, has_v = self.video_preprocessor.extract_from_video(video_path)
        a_tensor, dur_aud, has_a = self.audio_preprocessor.extract_from_audio(video_path)

        # Batch dimension [1, ...]
        batch = {
            "video": v_tensor.unsqueeze(0).to(self.device),
            "audio": a_tensor.unsqueeze(0).to(self.device),
            "has_video": torch.tensor([has_v], device=self.device),
            "has_audio": torch.tensor([has_a], device=self.device),
        }

        out = self.model(batch)
        logit = float(out["logit_av"].cpu().item())
        prob = float(torch.sigmoid(out["logit_av"]).cpu().item())

        elapsed = time.time() - t0
        verdict, is_inconclusive, (low_b, up_b) = self.classify_with_uncertainty(
            prob, threshold=threshold, uncertainty_margin=uncertainty_margin
        )

        gate_info = None
        if out.get("gate") is not None:
            g = out["gate"].cpu().numpy()[0]
            gate_info = {
                "audio_weight": round(float(g[0]), 4),
                "video_weight": round(float(g[1]), 4),
            }

        aux_info = {}
        if "logit_v" in out:
            p_v = float(torch.sigmoid(out["logit_v"]).cpu().item())
            aux_info["visual_manipulation_prob"] = round(p_v, 4)
        if "logit_a" in out:
            p_a = float(torch.sigmoid(out["logit_a"]).cpu().item())
            aux_info["audio_manipulation_prob"] = round(p_a, 4)

        return {
            "prediction": verdict,
            "manipulation_probability": round(prob, 4),
            "authenticity_probability": round(1.0 - prob, 4),
            "raw_logit": round(logit, 4),
            "threshold_applied": threshold,
            "uncertainty_margin": uncertainty_margin,
            "uncertainty_band": [low_b, up_b],
            "is_inconclusive": is_inconclusive,
            "mode_used": self.mode,
            "latency_ms": round(elapsed * 1000, 2),
            "gate_weights": gate_info,
            "auxiliary_predictions": aux_info,
            "file": os.path.basename(video_path),
            "has_video_stream": has_v,
            "has_audio_stream": has_a,
        }
