r"""
LAV-DF Multimodal Preprocessing Module
---------------------------------------
Provides reproducible, matched preprocessing pipelines for audio and video streams:
1. Video: Uniform temporal frame sampling, resizing (224x224), ImageNet normalization.
   - Training: RandomHorizontalFlip, slight ColorJitter.
   - Validation / Inference: Deterministic resize and normalization.
2. Audio: Resampling to 16 kHz, 80-channel log-mel spectrogram computation, fixed-length padding/truncation.
   - Training: SpecAugment (time and frequency masking).
   - Validation / Inference: Deterministic spectrogram extraction.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, Optional, Union
from dataclasses import dataclass


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class PreprocessingConfig:
    # Video settings (aligned with Untitled0.ipynb)
    num_frames: int = 8
    img_size: int = 224
    fps: int = 25
    # Audio settings (aligned with Untitled0.ipynb)
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    target_audio_frames: int = 401
    target_audio_seconds: float = 4.0
    # Augmentation controls
    use_augmentation: bool = False
    # Normalization mode: 'imagenet' or 'unit' ([0, 1])
    video_norm: str = "imagenet"
    # Audio crop policy: 'manipulation_aware' or 'start'
    audio_crop_policy: str = "manipulation_aware"


class VideoPreprocessor:
    def __init__(self, cfg: PreprocessingConfig, is_train: bool = False):
        self.cfg = cfg
        self.is_train = is_train and cfg.use_augmentation

    def process_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Input: list or array of BGR/RGB images [T, H, W, 3] in uint8 [0, 255].
        Output: Normalized float32 tensor [T, 3, img_size, img_size].
        """
        processed = []
        # Decide augmentation parameters per video clip for consistency across frames
        do_flip = self.is_train and (np.random.rand() > 0.5)

        for img in frames:
            # Ensure RGB
            if img.shape[-1] == 3:
                # If OpenCV BGR is passed, caller should convert, but we handle safe check
                h, w = img.shape[:2]
            else:
                raise ValueError(f"Expected 3-channel image, got shape: {img.shape}")

            if (h, w) != (self.cfg.img_size, self.cfg.img_size):
                img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_LINEAR)

            if do_flip:
                img = cv2.flip(img, 1)

            # Training jitter (slight brightness/contrast adjustment)
            if self.is_train and np.random.rand() > 0.7:
                alpha = np.random.uniform(0.9, 1.1)
                beta = np.random.uniform(-10, 10)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

            if self.cfg.video_norm == "imagenet":
                norm = ((img / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
            else:
                norm = (img / 255.0).astype(np.float32)
            processed.append(norm)

        arr = np.array(processed, dtype=np.float32)  # [T, H, W, 3]
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # [T, 3, H, W]
        return tensor

    def extract_from_video(self, video_path: str) -> Tuple[torch.Tensor, int, bool]:
        """
        Uniformly extracts num_frames from an MP4/AVI video file.
        Returns: (frames_tensor [T, 3, H, W], total_video_frames, success_flag)
        """
        if not os.path.exists(video_path):
            dummy = torch.zeros(self.cfg.num_frames, 3, self.cfg.img_size, self.cfg.img_size, dtype=torch.float32)
            return dummy, 0, False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            dummy = torch.zeros(self.cfg.num_frames, 3, self.cfg.img_size, self.cfg.img_size, dtype=torch.float32)
            return dummy, 0, False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            dummy = torch.zeros(self.cfg.num_frames, 3, self.cfg.img_size, self.cfg.img_size, dtype=torch.float32)
            return dummy, 0, False

        indices = np.linspace(0, total_frames - 1, self.cfg.num_frames, dtype=int)
        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        cap.release()

        if len(frames) == 0:
            dummy = torch.zeros(self.cfg.num_frames, 3, self.cfg.img_size, self.cfg.img_size, dtype=torch.float32)
            return dummy, total_frames, False

        # If any frames were dropped, pad with the last valid frame
        while len(frames) < self.cfg.num_frames:
            frames.append(frames[-1])

        tensor = self.process_frames(np.array(frames))
        return tensor, total_frames, True


class AudioPreprocessor:
    def __init__(self, cfg: PreprocessingConfig, is_train: bool = False):
        self.cfg = cfg
        self.is_train = is_train and cfg.use_augmentation

        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB()

        if self.is_train:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
            self.time_mask = T.TimeMasking(time_mask_param=35)
        else:
            self.freq_mask = None
            self.time_mask = None

    def process_waveform(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Input: waveform tensor [1, N] or [N] at sr Hz.
        Output: Log-mel spectrogram tensor [1, n_mels, target_audio_frames].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] > 1:
            # Multi-channel to mono
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.cfg.sample_rate:
            resampler = T.Resample(sr, self.cfg.sample_rate)
            waveform = resampler(waveform)

        # Compute log-mel spectrogram
        mel = self.mel_transform(waveform)      # [1, n_mels, time_steps]
        log_mel = self.amp_to_db(mel)

        # Standardize representation
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        # Apply SpecAugment during training
        if self.is_train and self.freq_mask is not None and self.time_mask is not None:
            log_mel = self.freq_mask(log_mel)
            log_mel = self.time_mask(log_mel)

        # Pad or truncate to target_audio_frames
        t_len = log_mel.shape[-1]
        target_len = self.cfg.target_audio_frames

        if t_len < target_len:
            pad_amount = target_len - t_len
            log_mel = torch.nn.functional.pad(log_mel, (0, pad_amount), mode="constant", value=0.0)
        elif t_len > target_len:
            log_mel = log_mel[:, :, :target_len]

        return log_mel

    def extract_from_audio(
        self,
        media_path: str,
        fake_periods: Optional[list] = None,
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Extracts 80-channel log-mel spectrogram from a media file (MP4 video or WAV audio).
        Applies manipulation-aware cropping when fake_periods is provided.
        Returns: (spec_tensor [1, n_mels, target_audio_frames], duration_seconds, success_flag)
        """
        if not os.path.exists(media_path):
            dummy = torch.zeros(1, self.cfg.n_mels, self.cfg.target_audio_frames, dtype=torch.float32)
            return dummy, 0.0, False

        waveform = None
        sr = self.cfg.sample_rate

        # Attempt fast ffmpeg stream decoding (optimal for MP4 video containers)
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-i", media_path,
                    "-vn", "-ac", "1", "-ar", str(self.cfg.sample_rate),
                    "-f", "f32le", "-"
                ],
                capture_output=True,
                check=True
            )
            raw_audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
            if len(raw_audio) > 0:
                waveform = torch.from_numpy(raw_audio).unsqueeze(0)
        except Exception:
            waveform = None

        # Fallback to torchaudio if ffmpeg pipe failed
        if waveform is None:
            try:
                w, loaded_sr = torchaudio.load(media_path)
                if loaded_sr != self.cfg.sample_rate:
                    resampler = T.Resample(loaded_sr, self.cfg.sample_rate)
                    w = resampler(w)
                if w.shape[0] > 1:
                    w = w.mean(dim=0, keepdim=True)
                waveform = w
            except Exception:
                dummy = torch.zeros(1, self.cfg.n_mels, self.cfg.target_audio_frames, dtype=torch.float32)
                return dummy, 0.0, False

        total_samples = waveform.shape[-1]
        duration = total_samples / float(self.cfg.sample_rate)
        target_samples = int(self.cfg.sample_rate * self.cfg.target_audio_seconds)

        # Audio Crop Policy: Manipulation-Aware vs Start Crop
        if self.cfg.audio_crop_policy == "manipulation_aware" and fake_periods and duration > self.cfg.target_audio_seconds:
            try:
                # Find midpoint of primary fake segment
                f_start, f_end = fake_periods[0][0], fake_periods[0][1]
                midpoint_sec = (float(f_start) + float(f_end)) / 2.0
                crop_start_sec = max(0.0, min(midpoint_sec - (self.cfg.target_audio_seconds / 2.0), duration - self.cfg.target_audio_seconds))
                start_sample = int(crop_start_sec * self.cfg.sample_rate)
                end_sample = start_sample + target_samples
                waveform = waveform[:, start_sample:end_sample]
            except Exception:
                waveform = waveform[:, :target_samples]
        elif total_samples > target_samples:
            waveform = waveform[:, :target_samples]
        elif total_samples < target_samples:
            pad_len = target_samples - total_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_len), value=0.0)

        spec = self.process_waveform(waveform, sr=self.cfg.sample_rate)
        return spec, duration, True
