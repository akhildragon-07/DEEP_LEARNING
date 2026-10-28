"""
Unit Tests for dataset.py:
1. Leakage-safe grouped splits with zero identity overlap.
2. Ingesting Member 1's pilot dataset (teammates/zip2).
3. DataLoader collation and tensor shape verification.
"""

import os
import shutil
import tempfile
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import create_leakage_safe_splits, FakeAVCelebDataset

def run_tests():
    print("=" * 70)
    print("RUNNING DATASET AND SPLIT MANIFEST UNIT TESTS")
    print("=" * 70)

    pilot_manifest = "teammates/zip2/processed_manifest.csv"
    pilot_dir = "teammates/zip2"

    temp_splits = tempfile.mkdtemp()
    try:
        # 1. Create leakage-safe splits from pilot manifest
        print("\n[Test 1] Generating Leakage-Safe Splits from Pilot Manifest...")
        train_csv, val_csv, test_csv = create_leakage_safe_splits(
            manifest_path=pilot_manifest,
            output_dir=temp_splits,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)

        train_ids = set(train_df["identity_group"])
        val_ids = set(val_df["identity_group"])
        test_ids = set(test_df["identity_group"])

        assert len(train_ids.intersection(val_ids)) == 0, "DATA LEAKAGE between Train and Val!"
        assert len(train_ids.intersection(test_ids)) == 0, "DATA LEAKAGE between Train and Test!"
        assert len(val_ids.intersection(test_ids)) == 0, "DATA LEAKAGE between Val and Test!"
        assert len(train_df) + len(val_df) + len(test_df) == 40, "Total sample count mismatch!"
        print(f"  -> Splits verified! Train={len(train_df)} ({len(train_ids)} IDs), "
              f"Val={len(val_df)} ({len(val_ids)} IDs), Test={len(test_df)} ({len(test_ids)} IDs).")
        print("  -> Zero identity overlap confirmed (100% leakage-free).")

        # 2. Test Real Pilot Data Ingestion
        print("\n[Test 2] Loading Pilot Samples via FakeAVCelebDataset...")
        dataset = FakeAVCelebDataset(manifest_path=train_csv, data_dir=pilot_dir, num_frames=5)
        assert len(dataset) == len(train_df)

        sample = dataset[0]
        assert "video" in sample
        assert "audio" in sample
        assert "has_video" in sample
        assert "has_audio" in sample
        assert "label" in sample

        assert sample["video"].shape == (5, 3, 224, 224), f"Unexpected video shape {sample['video'].shape}"
        assert sample["audio"].shape == (1, 80, 400), f"Unexpected audio shape {sample['audio'].shape}"
        assert sample["has_video"].item() is True
        assert sample["has_audio"].item() is True
        print("  -> Pilot sample successfully loaded! Video: [5, 3, 224, 224], Audio: [1, 80, 400]")

        # 3. Test Batch Collation
        print("\n[Test 3] Testing DataLoader Collation...")
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["video"].shape == (4, 5, 3, 224, 224)
        assert batch["audio"].shape == (4, 1, 80, 400)
        assert batch["label"].shape == (4,)
        print(f"  -> DataLoader collation passed! Batch labels: {batch['label'].shape}")

        print("\n" + "=" * 70)
        print("ALL DATASET & SPLIT TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_splits, ignore_errors=True)

if __name__ == "__main__":
    run_tests()
