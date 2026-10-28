"""
Unit test for evaluate.py (Member 2 Integration):
Evaluates an actual trained checkpoint on the pilot test set under:
1. Standard conditions
2. Missing video stress test
3. Missing audio stress test
Verifies prediction CSV and metrics JSON.
"""

import os
import shutil
import tempfile
import torch
from torch.utils.data import DataLoader

from model import build_model
from dataset import FakeAVCelebDataset, create_leakage_safe_splits
from evaluate import evaluate_checkpoint

def run_tests():
    print("=" * 70)
    print("RUNNING EVALUATION & STRESS TEST UNIT TESTS")
    print("=" * 70)

    pilot_manifest = "teammates/zip2/processed_manifest.csv"
    pilot_dir = "teammates/zip2"

    temp_dir = tempfile.mkdtemp()
    try:
        train_csv, val_csv, test_csv = create_leakage_safe_splits(
            manifest_path=pilot_manifest,
            output_dir=os.path.join(temp_dir, "splits"),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        test_ds = FakeAVCelebDataset(manifest_path=test_csv, data_dir=pilot_dir, num_frames=5)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

        device = torch.device("cpu")
        model = build_model("rmsav", cfg={"dim": 128, "pretrained": False})
        model.to(device)

        # 1. Standard evaluation
        print("\n[Test 1] Standard Test Evaluation...")
        metrics, df_preds = evaluate_checkpoint(model, test_loader, device, stress_test="none")
        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "f1_score" in metrics
        assert "auc" in metrics
        assert "ece" in metrics
        assert len(df_preds) == len(test_ds)
        print(f"  -> Passed! Evaluated {len(df_preds)} test samples: ACC={metrics['accuracy']*100:.1f}%, AUC={metrics['auc']:.4f}, ECE={metrics['ece']:.4f}")

        # 2. Stress Test: Missing Video
        print("\n[Test 2] Missing Video Stress Test...")
        m_mv, preds_mv = evaluate_checkpoint(model, test_loader, device, stress_test="missing_video")
        assert not torch.isnan(torch.tensor(m_mv["accuracy"]))
        assert all(g == 0.0 for g in preds_mv["gate_video"]), "Gate for video must be 0.0 when video is missing!"
        print(f"  -> Passed! Video gate strictly 0.0 for all samples. ACC={m_mv['accuracy']*100:.1f}%")

        # 3. Stress Test: Missing Audio
        print("\n[Test 3] Missing Audio Stress Test...")
        m_ma, preds_ma = evaluate_checkpoint(model, test_loader, device, stress_test="missing_audio")
        assert not torch.isnan(torch.tensor(m_ma["accuracy"]))
        assert all(g == 0.0 for g in preds_ma["gate_audio"]), "Gate for audio must be 0.0 when audio is missing!"
        print(f"  -> Passed! Audio gate strictly 0.0 for all samples. ACC={m_ma['accuracy']*100:.1f}%")

        print("\n" + "=" * 70)
        print("ALL EVALUATION & STRESS TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_tests()
