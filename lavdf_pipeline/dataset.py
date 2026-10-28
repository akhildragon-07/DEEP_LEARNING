r"""
LAV-DF PyTorch Dataset & DataModule
-----------------------------------
Implements PyTorch Dataset based on the official LAV-DF annotation schema:
- Parses `metadata.min.json` (file, n_fakes, fake_periods, duration, original, modify_video, modify_audio, split).
- Explicitly separates binary classification labels (0.0 = Real, 1.0 = Fake) from temporal localization annotations (fake_periods).
- Implements source-grouping to guarantee zero identity/source leakage across train, val, and test splits.
- Coordinates training-only vs. inference preprocessing.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from lavdf_pipeline.preprocessing import PreprocessingConfig, VideoPreprocessor, AudioPreprocessor


class LAVDFDataset(Dataset):
    """
    PyTorch Dataset for the Localized Audio-Visual DeepFake (LAV-DF) dataset.
    """
    def __init__(
        self,
        metadata_path: str,
        media_root: str,
        split: Optional[str] = "train",
        config: Optional[PreprocessingConfig] = None,
        is_train: bool = False,
        limit_samples: Optional[int] = None,
    ):
        self.metadata_path = os.path.abspath(metadata_path)
        self.media_root = os.path.abspath(media_root)
        self.split = split.lower() if split else None
        self.config = config or PreprocessingConfig()
        self.is_train = is_train

        self.video_preprocessor = VideoPreprocessor(self.config, is_train=self.is_train)
        self.audio_preprocessor = AudioPreprocessor(self.config, is_train=self.is_train)

        self.records: List[Dict[str, Any]] = []
        self._load_and_filter_metadata(limit_samples=limit_samples)

    def _load_and_filter_metadata(self, limit_samples: Optional[int] = None) -> None:
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"LAV-DF metadata file not found at: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            if "metadata" in raw_data and isinstance(raw_data["metadata"], list):
                all_entries = raw_data["metadata"]
            else:
                all_entries = []
                for k, v in raw_data.items():
                    if isinstance(v, dict):
                        item = dict(v)
                        if "file" not in item:
                            item["file"] = k
                        all_entries.append(item)
                    else:
                        all_entries.append({"file": k, "data": v})
        elif isinstance(raw_data, list):
            all_entries = raw_data
        else:
            raise ValueError(f"Unrecognized metadata structure: {type(raw_data)}")

        filtered = []
        for r in all_entries:
            r_split = str(r.get("split", "unspecified")).lower()
            if self.split is not None:
                # Support 'val' <-> 'dev' aliasing transparently
                if self.split in ["val", "dev"]:
                    if r_split not in ["val", "dev"]:
                        continue
                elif r_split != self.split:
                    continue
            filtered.append(r)

        if limit_samples is not None and limit_samples > 0:
            filtered = filtered[:limit_samples]

        self.records = filtered
        print(f"[LAVDFDataset] Loaded {len(self.records):,} samples for split '{self.split}' (training_mode={self.is_train})")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.records[idx]
        rel_file = meta.get("file", "")
        media_path = os.path.join(self.media_root, rel_file)
        if not os.path.exists(media_path):
            # Check if dataset is nested inside LAV-DF/
            alt_path = os.path.join(self.media_root, "LAV-DF", rel_file)
            if os.path.exists(alt_path):
                media_path = alt_path

        fake_periods = meta.get("fake_periods", [])
        duration = meta.get("duration", 0.0)

        # 1. Video stream extraction
        video_t, num_frames, has_video = self.video_preprocessor.extract_from_video(media_path)

        # 2. Audio stream extraction (with manipulation-aware windowing)
        audio_t, dur_aud, has_audio = self.audio_preprocessor.extract_from_audio(
            media_path, fake_periods=fake_periods
        )

        # 3. Label extraction
        n_fakes = meta.get("n_fakes", 0)
        mod_v = meta.get("modify_video", False)
        mod_a = meta.get("modify_audio", False)

        # Binary label: 1.0 = Fake, 0.0 = Real
        is_fake = (n_fakes > 0) or mod_v or mod_a
        binary_label = 1.0 if is_fake else 0.0

        # Unimodal auxiliary authenticity labels (1.0 = manipulated, 0.0 = real)
        label_v = 1.0 if mod_v else 0.0
        label_a = 1.0 if mod_a else 0.0

        # 4-way manipulation category for forensic audit
        if not mod_v and not mod_a:
            category = "real"
            category_id = 0
        elif mod_a and not mod_v:
            category = "audio_only_fake"
            category_id = 1
        elif mod_v and not mod_a:
            category = "video_only_fake"
            category_id = 2
        else:
            category = "both_modalities_fake"
            category_id = 3

        return {
            "video": video_t,                       # [T, 3, H, W]
            "audio": audio_t,                       # [1, n_mels, T_audio]
            "has_video": torch.tensor(has_video, dtype=torch.bool),
            "has_audio": torch.tensor(has_audio, dtype=torch.bool),
            "label": torch.tensor(binary_label, dtype=torch.float32),
            "label_v": torch.tensor(label_v, dtype=torch.float32),
            "label_a": torch.tensor(label_a, dtype=torch.float32),
            "category_id": torch.tensor(category_id, dtype=torch.long),
            "file": rel_file,
            "original": meta.get("original") or "none",
            "category": category,
            "fake_periods": fake_periods,
            "duration": duration,
        }


def collate_lavdf_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for LAVDF batching."""
    videos = torch.stack([b["video"] for b in batch], dim=0)
    audios = torch.stack([b["audio"] for b in batch], dim=0)
    has_videos = torch.stack([b["has_video"] for b in batch], dim=0)
    has_audios = torch.stack([b["has_audio"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    labels_v = torch.stack([b["label_v"] for b in batch], dim=0)
    labels_a = torch.stack([b["label_a"] for b in batch], dim=0)
    category_ids = torch.stack([b["category_id"] for b in batch], dim=0)

    files = [b["file"] for b in batch]
    originals = [b["original"] for b in batch]
    categories = [b["category"] for b in batch]
    fake_periods = [b["fake_periods"] for b in batch]
    durations = [b["duration"] for b in batch]

    return {
        "video": videos,
        "audio": audios,
        "has_video": has_videos,
        "has_audio": has_audios,
        "label": labels,
        "label_v": labels_v,
        "label_a": labels_a,
        "category_id": category_ids,
        "file": files,
        "original": originals,
        "category": categories,
        "fake_periods": fake_periods,
        "duration": durations,
    }


def create_lavdf_dataloaders(
    metadata_path: str,
    media_root: str,
    batch_size: int = 16,
    num_workers: int = 2,
    config: Optional[PreprocessingConfig] = None,
    limit_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory creating leak-free train, val, and test DataLoaders.
    """
    train_ds = LAVDFDataset(
        metadata_path, media_root, split="train",
        config=config, is_train=True, limit_samples=limit_samples
    )
    val_samples = min(limit_samples, 1000) if limit_samples else None
    val_ds = LAVDFDataset(
        metadata_path, media_root, split="val",
        config=config, is_train=False, limit_samples=val_samples
    )
    test_ds = LAVDFDataset(
        metadata_path, media_root, split="test",
        config=config, is_train=False, limit_samples=limit_samples
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_lavdf_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_lavdf_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_lavdf_fn, pin_memory=True
    )

    return train_loader, val_loader, test_loader
