r"""
Inference Engine for RMS-AV Deepfake Detection Dashboard
--------------------------------------------------------
Provides safe, standalone, and cached model inference across:
- Mode A: Image upload (visual-only prediction)
- Mode B: Video upload (multi-frame visual prediction)
- Mode C: Audio upload (80-channel log-mel acoustic prediction)
- Mode D: Paired Audio + Video upload (Standard Fusion or RMS-AV)
"""

import os
import time
import tempfile
import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from model import build_model, count_params
from dataset import compute_log_mel_spectrogram

logger = logging.getLogger("DeepfakeInference")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

DEFAULT_CHECKPOINTS = {
    "audio": "checkpoints/pilot_audio_best.pth",
    "video": "checkpoints/pilot_video_best.pth",
    "fusion": "checkpoints/pilot_fusion_best.pth",
    "rmsav": "checkpoints/pilot_rmsav_best.pth",
}

# ImageNet normalization standard (matches dataset.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class InferenceEngine:
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        checkpoint_map: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_map = checkpoint_map or DEFAULT_CHECKPOINTS
        self.models: Dict[str, nn.Module] = {}
        self.available_modes: List[str] = []
        self.model_info: Dict[str, Dict] = {}

        self._load_all_models()

    def _load_all_models(self) -> None:
        """Attempts to load each available model mode from disk."""
        for mode, rel_path in self.checkpoint_map.items():
            full_path = rel_path if os.path.isabs(rel_path) else os.path.join(os.getcwd(), rel_path)
            if not os.path.exists(full_path):
                # Fallback to latest if best not found
                latest_path = full_path.replace("_best.pth", "_latest.pth")
                if os.path.exists(latest_path):
                    full_path = latest_path
                else:
                    logger.warning(f"Checkpoint for mode '{mode}' not found at: {full_path}")
                    continue

            try:
                ckpt = torch.load(full_path, map_location=self.device)
                cfg = ckpt.get("config", {})
                dim = cfg.get("dim", 256)

                model = build_model(mode=mode, cfg={"dim": dim, "pretrained": False})
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(self.device)
                model.eval()

                self.models[mode] = model
                self.available_modes.append(mode)

                tot_p, trn_p = count_params(model)
                self.model_info[mode] = {
                    "checkpoint_path": full_path,
                    "epoch": ckpt.get("epoch"),
                    "best_metric": ckpt.get("best_metric"),
                    "total_params": tot_p,
                    "trainable_params": trn_p,
                    "dim": dim,
                }
                logger.info(f"Loaded mode '{mode}' from {os.path.basename(full_path)} ({tot_p:,} params)")
            except Exception as e:
                logger.error(f"Failed to load checkpoint for '{mode}' from {full_path}: {e}")

    def is_mode_available(self, mode: str) -> bool:
        return mode in self.models

    # -------------------------------------------------------------------------
    # Preprocessing Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def preprocess_image(image_input: Union[str, bytes, Image.Image], img_size: int = 224) -> np.ndarray:
        """
        Preprocesses a single image into normalized float32 array [1, 3, 224, 224].
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image path not found: {image_input}")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError("Failed to decode image from disk.")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Failed to decode uploaded image bytes.")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        else:
            raise TypeError("Unsupported image input type.")

        if img_rgb.shape[0] != img_size or img_rgb.shape[1] != img_size:
            img_rgb = cv2.resize(img_rgb, (img_size, img_size))

        norm_img = ((img_rgb / 255.0) - IMAGENET_MEAN) / IMAGENET_STD  # [224, 224, 3]
        tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()   # [3, 224, 224]
        return tensor

    @staticmethod
    def extract_video_frames(
        video_input: Union[str, bytes],
        num_frames: int = 5,
        img_size: int = 224,
    ) -> Tuple[torch.Tensor, int]:
        """
        Extracts num_frames uniformly sampled normalized frames from video.
        Returns: [num_frames, 3, img_size, img_size], total_raw_frames
        """
        temp_file = None
        if isinstance(video_input, bytes):
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.write(video_input)
            temp_file.close()
            path_to_open = temp_file.name
        else:
            path_to_open = video_input

        cap = cv2.VideoCapture(path_to_open)
        if not cap.isOpened():
            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            raise ValueError("Could not open video file for reading. Ensure valid MP4/MOV/AVI encoding.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            raise ValueError("Video contains 0 readable frames.")

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames_dict = {}
        target_indices = set(indices)

        idx = 0
        while cap.isOpened() and len(frames_dict) < len(target_indices):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in target_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[0] != img_size or frame_rgb.shape[1] != img_size:
                    frame_rgb = cv2.resize(frame_rgb, (img_size, img_size))
                norm = ((frame_rgb / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
                frames_dict[idx] = norm
            idx += 1
        cap.release()

        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)

        if not frames_dict:
            raise ValueError("No frames could be extracted from video.")

        # Assemble in temporal order
        sorted_frames = []
        for i in indices:
            if i in frames_dict:
                sorted_frames.append(frames_dict[i])
            else:
                sorted_frames.append(list(frames_dict.values())[-1])

        arr = np.array(sorted_frames, dtype=np.float32)            # [T, H, W, 3]
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() # [T, 3, H, W]
        return tensor, total_frames

    @staticmethod
    def preprocess_audio(
        audio_input: Union[str, bytes],
        target_sr: int = 16000,
    ) -> Tuple[torch.Tensor, float]:
        """
        Extracts 80-channel log-mel spectrogram from audio file or bytes.
        Returns: [1, 80, 400], duration_seconds
        """
        temp_file = None
        if isinstance(audio_input, bytes):
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.write(audio_input)
            temp_file.close()
            path_to_open = temp_file.name
        else:
            path_to_open = audio_input

        try:
            import scipy.io.wavfile as wav
            sr, data = wav.read(path_to_open)
        except Exception as e:
            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            raise ValueError(f"Failed to read WAV audio: {e}. Please ensure standard PCM 16-bit WAV.")

        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)

        waveform = data.astype(np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        duration = len(waveform) / float(sr) if sr > 0 else 0.0

        # Resample if needed
        if sr != target_sr and sr > 0:
            num_samples = int(len(waveform) * target_sr / sr)
            waveform = np.interp(
                np.linspace(0, len(waveform), num_samples, endpoint=False),
                np.arange(len(waveform)),
                waveform,
            ).astype(np.float32)
            sr = target_sr

        mel_tensor = compute_log_mel_spectrogram(waveform, sr=target_sr, n_mels=80, target_frames=400)
        return mel_tensor, duration

    # -------------------------------------------------------------------------
    # Standalone Prediction Methods
    # -------------------------------------------------------------------------

    def predict_image(
        self,
        image_input: Union[str, bytes, Image.Image],
        filename: str = "uploaded_image.jpg",
        threshold: float = 0.50,
    ) -> Dict:
        """
        Mode A: Visual-only prediction on a single still image.
        Uses the visual encoder of the video baseline model.
        """
        t0 = time.time()
        if "video" not in self.models:
            raise RuntimeError("Video model checkpoint ('video') is not available for visual prediction.")

        frame_t = self.preprocess_image(image_input)  # [3, 224, 224]
        # Duplicate to T=5 frames to match temporal shape [1, 5, 3, 224, 224]
        video_t = frame_t.unsqueeze(0).repeat(5, 1, 1, 1).unsqueeze(0).to(self.device)

        model = self.models["video"]
        with torch.no_grad():
            batch = {
                "video": video_t,
                "audio": None,
                "has_video": torch.tensor([True], device=self.device),
                "has_audio": torch.tensor([False], device=self.device),
            }
            out = model(batch)
            logit = out["logit_av"]
            prob = float(torch.sigmoid(logit).cpu().item())
            raw_logit = float(logit.cpu().item())

        elapsed = time.time() - t0
        pred_label = "Likely manipulated" if prob >= threshold else "Likely authentic"
        is_uncertain = (0.45 <= prob <= 0.55)

        return {
            "prediction": pred_label,
            "manipulation_probability": prob,
            "authenticity_probability": 1.0 - prob,
            "raw_logit": round(raw_logit, 4),
            "threshold_applied": threshold,
            "is_uncertain": is_uncertain,
            "mode_used": "video (visual-only)",
            "media_type": "Still Image (Visual Only)",
            "filename": filename,
            "latency_ms": round(elapsed * 1000, 2),
            "gate_weights": None,
            "disclaimer": "Visual-Only Analysis: Evaluates visual facial artifact cues only. Does not analyze acoustic or temporal-lip synchronization.",
        }

    def predict_video(
        self,
        video_input: Union[str, bytes],
        filename: str = "uploaded_video.mp4",
        num_frames: int = 5,
        threshold: float = 0.50,
    ) -> Dict:
        """
        Mode B: Multi-frame visual prediction on a video stream.
        """
        t0 = time.time()
        if "video" not in self.models:
            raise RuntimeError("Video model checkpoint ('video') is not available.")

        frames_t, total_raw = self.extract_video_frames(video_input, num_frames=num_frames)
        video_t = frames_t.unsqueeze(0).to(self.device)  # [1, 5, 3, 224, 224]

        model = self.models["video"]
        with torch.no_grad():
            batch = {
                "video": video_t,
                "audio": None,
                "has_video": torch.tensor([True], device=self.device),
                "has_audio": torch.tensor([False], device=self.device),
            }
            out = model(batch)
            logit = out["logit_av"]
            prob = float(torch.sigmoid(logit).cpu().item())
            raw_logit = float(logit.cpu().item())

        elapsed = time.time() - t0
        pred_label = "Likely manipulated" if prob >= threshold else "Likely authentic"
        is_uncertain = (0.45 <= prob <= 0.55)

        return {
            "prediction": pred_label,
            "manipulation_probability": prob,
            "authenticity_probability": 1.0 - prob,
            "raw_logit": round(raw_logit, 4),
            "threshold_applied": threshold,
            "is_uncertain": is_uncertain,
            "mode_used": "video",
            "media_type": "Video Stream (Visual Frames)",
            "filename": filename,
            "frames_analyzed": num_frames,
            "total_video_frames": total_raw,
            "latency_ms": round(elapsed * 1000, 2),
            "gate_weights": None,
            "disclaimer": "Visual Stream Analysis: Evaluates 5 uniform temporal frames across the video. Audio stream was not evaluated in this mode.",
        }

    def predict_audio(
        self,
        audio_input: Union[str, bytes],
        filename: str = "uploaded_audio.wav",
        threshold: float = 0.50,
    ) -> Dict:
        """
        Mode C: Acoustic prediction on an audio track.
        """
        t0 = time.time()
        if "audio" not in self.models:
            raise RuntimeError("Audio model checkpoint ('audio') is not available.")

        mel_t, duration = self.preprocess_audio(audio_input)
        audio_t = mel_t.unsqueeze(0).to(self.device)  # [1, 1, 80, 400]

        model = self.models["audio"]
        with torch.no_grad():
            batch = {
                "video": None,
                "audio": audio_t,
                "has_video": torch.tensor([False], device=self.device),
                "has_audio": torch.tensor([True], device=self.device),
            }
            out = model(batch)
            logit = out["logit_av"]
            prob = float(torch.sigmoid(logit).cpu().item())
            raw_logit = float(logit.cpu().item())

        elapsed = time.time() - t0
        pred_label = "Likely manipulated" if prob >= threshold else "Likely authentic"
        is_uncertain = (0.45 <= prob <= 0.55)

        return {
            "prediction": pred_label,
            "manipulation_probability": prob,
            "authenticity_probability": 1.0 - prob,
            "raw_logit": round(raw_logit, 4),
            "threshold_applied": threshold,
            "is_uncertain": is_uncertain,
            "mode_used": "audio",
            "media_type": "Audio Track (Log-Mel Spectrogram)",
            "filename": filename,
            "duration_seconds": round(duration, 2),
            "latency_ms": round(elapsed * 1000, 2),
            "gate_weights": None,
            "disclaimer": "Acoustic Stream Analysis: Evaluates 80-channel log-mel spectral representations for synthetic voice or speech artifacts.",
        }

    def predict_audio_video(
        self,
        video_input: Union[str, bytes],
        audio_input: Union[str, bytes],
        video_filename: str = "video.mp4",
        audio_filename: str = "audio.wav",
        mode: str = "rmsav",
        threshold: float = 0.50,
    ) -> Dict:
        """
        Mode D: Paired Audio + Video prediction using Standard Fusion or RMS-AV.
        """
        t0 = time.time()
        if mode not in ["fusion", "rmsav"]:
            raise ValueError(f"Multimodal paired inference requires mode 'fusion' or 'rmsav', got: {mode}")

        if mode not in self.models:
            raise RuntimeError(f"Requested fusion model checkpoint ('{mode}') is not available.")

        frames_t, total_raw = self.extract_video_frames(video_input, num_frames=5)
        mel_t, duration = self.preprocess_audio(audio_input)

        video_t = frames_t.unsqueeze(0).to(self.device)  # [1, 5, 3, 224, 224]
        audio_t = mel_t.unsqueeze(0).to(self.device)     # [1, 1, 80, 400]

        model = self.models[mode]
        with torch.no_grad():
            batch = {
                "video": video_t,
                "audio": audio_t,
                "has_video": torch.tensor([True], device=self.device),
                "has_audio": torch.tensor([True], device=self.device),
            }
            out = model(batch)
            logit = out["logit_av"]
            prob = float(torch.sigmoid(logit).cpu().item())
            raw_logit = float(logit.cpu().item())

            gate_info = None
            if out.get("gate") is not None:
                g = out["gate"].cpu().numpy()[0]
                gate_info = {
                    "audio_weight": round(float(g[0]), 4),
                    "video_weight": round(float(g[1]), 4),
                }

        elapsed = time.time() - t0
        pred_label = "Likely manipulated" if prob >= threshold else "Likely authentic"
        is_uncertain = (0.45 <= prob <= 0.55)

        return {
            "prediction": pred_label,
            "manipulation_probability": prob,
            "authenticity_probability": 1.0 - prob,
            "raw_logit": round(raw_logit, 4),
            "threshold_applied": threshold,
            "is_uncertain": is_uncertain,
            "mode_used": mode,
            "media_type": "Paired Multimodal (Audio + Video)",
            "video_filename": video_filename,
            "audio_filename": audio_filename,
            "frames_analyzed": 5,
            "total_video_frames": total_raw,
            "duration_seconds": round(duration, 2),
            "latency_ms": round(elapsed * 1000, 2),
            "gate_weights": gate_info,
            "disclaimer": "Full Audio-Visual Analysis: Features from both video and audio streams are processed jointly through selective reliability gating.",
        }
