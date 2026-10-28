r"""
==================================================================================================
Dataset Pipeline & Leakage-Safe Manifest Generator for RMS-AV (Member 1 Integration)
--------------------------------------------------------------------------------------------------
Integrates Member 1's dataset structure and Member 2's agreed model interface:
1. Batch Interface:
   - video:      [T, 3, H, W] (default T=5, 224x224, ImageNet normalized)
   - audio:      [1, 80, time] (80-channel Log-Mel Spectrogram)
   - has_video:  bool scalar
   - has_audio:  bool scalar
   - label:      float (0.0 = Real, 1.0 = Fake)
   - video_label: float (0.0 = Real Video, 1.0 = Fake Video)
   - audio_label: float (0.0 = Real Audio, 1.0 = Fake Audio)
2. Ingestion Support:
   - Pre-extracted frame directories (frames/video_XXX/*.jpg) & audio (audio/video_XXX.wav)
   - Direct raw video files (.mp4) with OpenCV uniform frame sampling
   - Manifests (processed_manifest.csv, processed_manifest_clean.csv, meta_data.csv)
3. Zero-Leakage Identity-Grouped Splitting (train_manifest.csv, val_manifest.csv, test_manifest.csv).
==================================================================================================
"""

import glob
import json
import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("DeepfakeDataset")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =========================================================================
# 1. Mel Spectrogram Audio Extraction (Robust & Dependency-Free)
# =========================================================================

def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    target_frames: int = 400,
) -> torch.Tensor:
    """
    Computes 80-channel log-mel spectrogram from a 1D audio waveform.
    Uses PyTorch STFT with a triangular mel filterbank.
    Returns: [1, n_mels, time] tensor.
    """
    waveform = torch.from_numpy(audio).float()
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)

    # Normalize amplitude
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / (max_val + 1e-6)

    # PyTorch STFT
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power_spec = stft.abs().pow(2)  # [n_fft//2 + 1, time]

    # Create Mel filterbank
    num_bins = n_fft // 2 + 1
    f_min = 0.0
    f_max = float(sr) / 2.0

    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = torch.floor((n_fft + 1) * hz_pts / sr).long()

    filters = torch.zeros(n_mels, num_bins)
    for i in range(1, n_mels + 1):
        left = bin_pts[i - 1].item()
        center = bin_pts[i].item()
        right = bin_pts[i + 1].item()

        if center > left:
            filters[i - 1, left:center] = (torch.arange(left, center) - left) / (center - left)
        if right > center:
            filters[i - 1, center:right] = (right - torch.arange(center, right)) / (right - center)

    mel_spec = torch.matmul(filters, power_spec)
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))

    # Pad or truncate to fixed temporal length
    t_dim = log_mel.size(1)
    if t_dim < target_frames:
        pad_size = target_frames - t_dim
        log_mel = torch.nn.functional.pad(log_mel, (0, pad_size))
    else:
        log_mel = log_mel[:, :target_frames]

    return log_mel.unsqueeze(0)  # [1, 80, target_frames]


# =========================================================================
# 2. Identity Extraction & Leakage-Safe Manifest Creation
# =========================================================================

def infer_identity_key(row: pd.Series, path_str: str) -> str:
    """Extracts subject identity to prevent train/val/test data leakage."""
    for col in ["identity", "speaker_id", "person_id", "subject_id"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            return str(row[col]).strip()

    # Match FakeAVCeleb directory structure: e.g. id00076 or idXXXXX
    norm = path_str.replace("\\", "/")
    match = re.search(r"(id\d{4,6})", norm, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Fallback to video ID prefix
    vid_id = str(row.get("video_id", ""))
    if vid_id.strip():
        return vid_id.strip()

    return "unknown_group"


def create_leakage_safe_splits(
    manifest_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[str, str, str]:
    """
    Creates identity-grouped split manifests with zero identity leakage.
    Guarantees that all manipulations of a subject remain in the same partition.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(manifest_path)

    col_map = {c.lower().strip(): c for c in df.columns}
    label_col = next((col_map[c] for c in ["type", "label", "category"] if c in col_map), None)
    path_col = next((col_map[c] for c in ["full_path", "path", "video_path"] if c in col_map), None)

    records = []
    for _, row in df.iterrows():
        raw_type = str(row[label_col]).lower().strip() if label_col else "real"
        p_str = str(row[path_col]) if path_col else ""

        is_real = raw_type in ["0", "0.0", "real", "realvideo-realaudio", "authentic"]
        is_fake_video = (not is_real) and ("fakevideo" in raw_type or "faceswap" in raw_type or "1" in raw_type)
        is_fake_audio = (not is_real) and ("fakeaudio" in raw_type or "wavtolip" in raw_type or "wav2lip" in raw_type or "fakevideo-fakeaudio" in raw_type)

        id_key = infer_identity_key(row, p_str)

        row_dict = row.to_dict()
        row_dict["label"] = 0.0 if is_real else 1.0
        row_dict["video_label"] = 1.0 if is_fake_video else 0.0
        row_dict["audio_label"] = 1.0 if is_fake_audio else 0.0
        row_dict["identity_group"] = id_key
        records.append(row_dict)

    parsed_df = pd.DataFrame(records)
    unique_groups = parsed_df["identity_group"].unique()

    # Deterministic grouped split
    np.random.seed(seed)
    shuffled_groups = np.random.permutation(unique_groups)

    n_groups = len(shuffled_groups)
    n_train = max(1, int(n_groups * train_ratio))
    n_val = max(1, int(n_groups * val_ratio))

    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train:n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val:])
    if len(test_groups) == 0 and len(val_groups) > 1:
        # Rebalance if small group count
        test_groups.add(val_groups.pop())

    train_df = parsed_df[parsed_df["identity_group"].isin(train_groups)].reset_index(drop=True)
    val_df = parsed_df[parsed_df["identity_group"].isin(val_groups)].reset_index(drop=True)
    test_df = parsed_df[parsed_df["identity_group"].isin(test_groups)].reset_index(drop=True)

    train_path = os.path.join(output_dir, "train_manifest.csv")
    val_path = os.path.join(output_dir, "val_manifest.csv")
    test_path = os.path.join(output_dir, "test_manifest.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_identities": len(train_groups),
        "val_identities": len(val_groups),
        "test_identities": len(test_groups),
        "leakage_overlap": {
            "train_val": len(train_groups.intersection(val_groups)),
            "train_test": len(train_groups.intersection(test_groups)),
            "val_test": len(val_groups.intersection(test_groups)),
        },
    }
    with open(os.path.join(output_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Generated leakage-safe splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}. 0% overlap."
    )
    return train_path, val_path, test_path


# =========================================================================
# 3. Multimodal Dataset Implementation
# =========================================================================

class FakeAVCelebDataset(Dataset):
    """
    Multimodal Dataset class producing Member 2's agreed input batch format:
      video:     [T, 3, 224, 224] float
      audio:     [1, 80, target_frames] float (Log-Mel Spectrogram)
      has_video: bool
      has_audio: bool
      label:     float (0.0=real, 1.0=fake)
      video_label: float (0.0=real video, 1.0=fake video)
      audio_label: float (0.0=real audio, 1.0=fake audio)
    """
    def __init__(
        self,
        manifest_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        num_frames: int = 5,
        target_frames_audio: int = 400,
        img_size: int = 224,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.target_frames_audio = target_frames_audio
        self.img_size = img_size
        self.data_dir = os.path.abspath(data_dir) if data_dir else None

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Ingest manifest
        if manifest_path and os.path.exists(manifest_path):
            self.df = pd.read_csv(manifest_path)
            self.records = self.df.to_dict(orient="records")
            if self.data_dir is None:
                self.data_dir = os.path.dirname(os.path.abspath(manifest_path))
        else:
            self.records = []

    def __len__(self) -> int:
        return len(self.records)

    def _load_frames(self, record: Dict) -> Tuple[torch.Tensor, bool]:
        """Loads video frames from pre-extracted frame folders or directly from video."""
        vid_id = str(record.get("video_id", "")).strip()
        frames = []

        # 1. Check pre-extracted frame folder: e.g. <data_dir>/frames/<video_id>/*.jpg
        if self.data_dir and vid_id:
            frame_dir = os.path.join(self.data_dir, "frames", vid_id)
            if os.path.isdir(frame_dir):
                jpg_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
                if jpg_paths:
                    indices = np.linspace(0, len(jpg_paths) - 1, self.num_frames, dtype=int)
                    for idx in indices:
                        img = cv2.imread(jpg_paths[idx])
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                                img = cv2.resize(img, (self.img_size, self.img_size))
                            norm_img = ((img / 255.0) - self.mean) / self.std
                            frames.append(norm_img)

        # 2. Check full_path or video_path
        if len(frames) == 0:
            v_path = record.get("full_path", record.get("video_path", ""))
            if self.data_dir and not os.path.isabs(v_path):
                v_path = os.path.join(self.data_dir, v_path)

            if os.path.exists(v_path) and v_path.endswith((".mp4", ".avi", ".mov")):
                cap = cv2.VideoCapture(v_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total > 0:
                    indices = set(np.linspace(0, total - 1, self.num_frames, dtype=int))
                    idx = 0
                    while cap.isOpened() and len(frames) < self.num_frames:
                        ret, f = cap.read()
                        if not ret:
                            break
                        if idx in indices:
                            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                            if f.shape[0] != self.img_size or f.shape[1] != self.img_size:
                                f = cv2.resize(f, (self.img_size, self.img_size))
                            norm_f = ((f / 255.0) - self.mean) / self.std
                            frames.append(norm_f)
                        idx += 1
                cap.release()

        # Handle fallback
        if len(frames) == 0:
            return torch.zeros((self.num_frames, 3, self.img_size, self.img_size), dtype=torch.float32), False

        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        # Convert to [T, 3, H, W]
        arr = np.array(frames[:self.num_frames], dtype=np.float32)  # [T, H, W, 3]
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)         # [T, 3, H, W]
        return tensor, True

    def _load_audio(self, record: Dict) -> Tuple[torch.Tensor, bool]:
        """Loads audio track from pre-extracted .wav or video."""
        vid_id = str(record.get("video_id", "")).strip()
        audio_data = None
        sr = 16000

        # 1. Check pre-extracted audio: e.g. <data_dir>/audio/<video_id>.wav
        if self.data_dir and vid_id:
            a_path = os.path.join(self.data_dir, "audio", f"{vid_id}.wav")
            if os.path.exists(a_path):
                try:
                    import scipy.io.wavfile as wav
                    sr, data = wav.read(a_path)
                    audio_data = data.astype(np.float32)
                except Exception:
                    pass

        # 2. Check full_path or audio_path
        if audio_data is None:
            a_path = record.get("audio_path", "")
            if not a_path and "full_path" in record:
                stem, _ = os.path.splitext(record["full_path"])
                a_path = f"{stem}.wav"

            if self.data_dir and not os.path.isabs(a_path):
                a_path = os.path.join(self.data_dir, a_path)

            if os.path.exists(a_path):
                try:
                    import scipy.io.wavfile as wav
                    sr, data = wav.read(a_path)
                    audio_data = data.astype(np.float32)
                except Exception:
                    pass

        if audio_data is None or len(audio_data) == 0:
            return torch.zeros((1, 80, self.target_frames_audio), dtype=torch.float32), False

        mel_spec = compute_log_mel_spectrogram(
            audio_data, sr=sr, target_frames=self.target_frames_audio
        )
        return mel_spec, True

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        record = self.records[idx]

        # Video & Audio
        video_tensor, has_video = self._load_frames(record)
        audio_tensor, has_audio = self._load_audio(record)

        # Labels
        label = float(record.get("label", 0.0))
        video_label = float(record.get("video_label", label))
        audio_label = float(record.get("audio_label", label))

        return {
            "video": video_tensor,
            "audio": audio_tensor,
            "has_video": torch.tensor(has_video, dtype=torch.bool),
            "has_audio": torch.tensor(has_audio, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.float32),
            "video_label": torch.tensor(video_label, dtype=torch.float32),
            "audio_label": torch.tensor(audio_label, dtype=torch.float32),
            "video_id": str(record.get("video_id", f"sample_{idx}")),
        }
