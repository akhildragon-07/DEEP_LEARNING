"""
End-to-End Training Pipeline Integration Test:
Runs 1 epoch of training and validation for all 4 modes ('audio', 'video', 'fusion', 'rmsav')
using Member 1's real pilot samples (teammates/zip2).
Verifies:
- Loss computation
- Backprop and optimizer steps
- Checkpoint saving (.pth)
- State dict reloading
"""

import os
import shutil
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_model
from dataset import FakeAVCelebDataset, create_leakage_safe_splits
from train import train_one_epoch, evaluate

def run_tests():
    print("=" * 70)
    print("RUNNING END-TO-END TRAINING INTEGRATION TEST")
    print("=" * 70)

    pilot_manifest = "teammates/zip2/processed_manifest.csv"
    pilot_dir = "teammates/zip2"

    temp_dir = tempfile.mkdtemp()
    try:
        # Generate persistent split manifests
        train_csv, val_csv, test_csv = create_leakage_safe_splits(
            manifest_path=pilot_manifest,
            output_dir=os.path.join(temp_dir, "splits"),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        train_ds = FakeAVCelebDataset(manifest_path=train_csv, data_dir=pilot_dir, num_frames=5)
        val_ds = FakeAVCelebDataset(manifest_path=val_csv, data_dir=pilot_dir, num_frames=5)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        device = torch.device("cpu")
        criterion = nn.BCEWithLogitsLoss()

        for mode in ["audio", "video", "fusion", "rmsav"]:
            print(f"\nTesting 1-epoch training & validation for mode '{mode}'...")
            net = build_model(mode=mode, cfg={"dim": 128, "pretrained": False})
            net.to(device)

            optimizer = torch.optim.Adam(
                [p for p in net.parameters() if p.requires_grad], lr=1e-3
            )

            # Train 1 epoch
            train_loss = train_one_epoch(
                model=net,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=1,
                mode=mode,
                aux_weight=0.2,
                use_amp=False,
            )
            assert not torch.isnan(torch.tensor(train_loss)), f"Train loss is NaN for {mode}!"
            print(f"  -> Train Loss ({mode}): {train_loss:.4f}")

            # Validate 1 epoch
            metrics = evaluate(
                model=net,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch=1,
                use_amp=False,
            )
            assert not torch.isnan(torch.tensor(metrics["val_loss"])), f"Val loss is NaN for {mode}!"
            print(f"  -> Val Metrics ({mode}): Loss={metrics['val_loss']:.4f}, ACC={metrics['accuracy']*100:.1f}%, AUC={metrics['auc']:.4f}")

            # Test checkpoint saving and reloading
            ckpt_path = os.path.join(temp_dir, f"{mode}_test.pth")
            torch.save({
                "epoch": 1,
                "mode": mode,
                "model_state_dict": net.state_dict(),
                "metrics": metrics,
            }, ckpt_path)

            loaded = torch.load(ckpt_path, map_location="cpu")
            net.load_state_dict(loaded["model_state_dict"])
            print(f"  -> Checkpoint successfully saved and reloaded ({mode})")

        print("\n" + "=" * 70)
        print("ALL TRAINING PIPELINE INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_tests()
