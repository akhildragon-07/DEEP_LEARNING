r"""
Reproduction and Diagnostic Script for Video False Positive Investigation
--------------------------------------------------------------------------
Reproduces video inference outside the UI using:
1. Exact checkpoint: checkpoints/pilot_video_best.pth
2. Video frames extraction and preprocessing comparison
3. Per-frame feature logging and logit extraction
4. Analysis on known authentic and manipulated pilot samples
"""

import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from model import build_model, count_params
from dataset import FakeAVCelebDataset
from inference import InferenceEngine, IMAGENET_MEAN as INF_MEAN, IMAGENET_STD as INF_STD


def audit_preprocessing_consistency():
    print("=" * 80)
    print("1. PREPROCESSING CONSISTENCY AUDIT")
    print("=" * 80)

    ds_temp = FakeAVCelebDataset()
    DS_MEAN = ds_temp.mean
    DS_STD = ds_temp.std

    # Check Mean and Std
    mean_diff = np.max(np.abs(DS_MEAN - INF_MEAN))
    std_diff = np.max(np.abs(DS_STD - INF_STD))
    print(f"ImageNet Mean diff between dataset.py and inference.py: {mean_diff:.2e}")
    print(f"ImageNet Std diff between dataset.py and inference.py:  {std_diff:.2e}")
    assert mean_diff < 1e-6 and std_diff < 1e-6, "Normalization constants mismatch!"

    print("ImageNet normalization parameters: MATCH EXACTLY.")


def inspect_checkpoint_and_reproduce(sample_vid_id: str = "video_036"):
    print("\n" + "=" * 80)
    print("2. CHECKPOINT AND REPRODUCTION LOGS")
    print("=" * 80)

    ckpt_path = "checkpoints/pilot_video_best.pth"
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint Path:     {ckpt_path}")
    print(f"Saved Epoch:         {ckpt.get('epoch')}")
    print(f"Mode:                {ckpt.get('mode')}")
    print(f"Best Val Metric:     {ckpt.get('best_metric')}")
    print(f"Training Config:     {ckpt.get('config')}")

    model = build_model(mode="video", cfg={"dim": 256, "pretrained": False})
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tot_p, trn_p = count_params(model)
    print(f"Model Parameters:    Total={tot_p:,} | Trainable={trn_p:,}")

    # Load known authentic sample from dataset: video_036 (RealVideo-RealAudio)
    ds = FakeAVCelebDataset("splits_pilot/test_manifest.csv", data_dir="teammates/zip2")
    sample_idx = None
    for i, r in enumerate(ds.records):
        if r["video_id"] == sample_vid_id:
            sample_idx = i
            break

    if sample_idx is None:
        sample_idx = 3 # video_036
    
    sample_record = ds.records[sample_idx]
    vid_id = sample_record["video_id"]
    category = sample_record.get("type", "Unknown")
    ground_truth = sample_record.get("label", 0.0)

    print(f"\nEvaluating Verified Ground-Truth Sample: {vid_id} ({category})")
    print(f"Ground-Truth Label: {ground_truth} (0.0 = Real/Authentic, 1.0 = Fake/Manipulated)")

    # 1. Dataset preprocessing path
    item_ds = ds[sample_idx]
    v_ds = item_ds["video"].unsqueeze(0) # [1, 5, 3, 224, 224]

    # 2. Forward pass with per-frame inspection
    with torch.no_grad():
        # Check per-frame visual representations
        B, T = v_ds.shape[:2]
        frames_flat = v_ds.flatten(0, 1) # [5, 3, 224, 224]
        frame_feats = model.video_enc.backbone(frames_flat) # [5, 512]
        frame_proj = model.video_enc.proj(frame_feats)      # [5, 256]
        
        # Per-frame logits through video head
        per_frame_logits = model.video_head(frame_proj).squeeze(-1).numpy()
        per_frame_probs = torch.sigmoid(torch.from_numpy(per_frame_logits)).numpy() * 100.0

        # Full temporal average
        pooled_feat = frame_feats.view(B, T, -1).mean(dim=1)
        pooled_proj = model.video_enc.proj(pooled_feat)
        full_logit = model.video_head(pooled_proj).item()
        full_prob = torch.sigmoid(torch.tensor(full_logit)).item()

    print(f"\nExtracted Frames: {T} frames of shape {list(v_ds.shape[2:])}")
    print("Per-Frame Diagnostic Breakdown:")
    for f_idx in range(T):
        print(f"  Frame {f_idx}: Logit = {per_frame_logits[f_idx]:+.4f} | Prob = {per_frame_probs[f_idx]:.2f}%")

    print(f"\nTemporally Pooled Logit: {full_logit:+.4f}")
    print(f"Sigmoid Probability:     {full_prob*100:.2f}%")
    print(f"Default Cutoff (0.50):   {'LIKELY MANIPULATED (False Positive)' if full_prob >= 0.50 else 'LIKELY AUTHENTIC'}")

    return {
        "vid_id": vid_id,
        "category": category,
        "ground_truth": ground_truth,
        "logit": full_logit,
        "prob": full_prob,
        "per_frame_logits": per_frame_logits.tolist(),
        "per_frame_probs": per_frame_probs.tolist(),
    }


if __name__ == "__main__":
    audit_preprocessing_consistency()
    inspect_checkpoint_and_reproduce("video_036")
