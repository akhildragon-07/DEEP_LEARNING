r"""
LAV-DF Model Training Pipeline
------------------------------
Implements GPU-accelerated training, validation, checkpointing, and early stopping
across Audio, Video, Standard Fusion, and RMS-AV architectures on LAV-DF.
"""

import os
import sys
import time
import json
import random
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from lavdf_pipeline.preprocessing import PreprocessingConfig
from lavdf_pipeline.model import build_lavdf_model, count_parameters
from lavdf_pipeline.dataset import create_lavdf_dataloaders


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LAVDFTrainer:
    def __init__(
        self,
        mode: str = "rmsav",
        metadata_path: str = "metadata.min.json",
        media_root: str = "data",
        dim: int = 256,
        pretrained_visual: bool = False,
        batch_size: int = 16,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        aux_weight: float = 0.20,
        patience: int = 5,
        num_workers: int = 4,
        use_amp: bool = True,
        save_dir: str = "checkpoints_lavdf",
        log_dir: str = "results_lavdf",
        checkpoint_prefix: str = "lavdf_",
        drive_backup: Optional[str] = None,
        limit_samples: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.mode = mode.lower()
        self.metadata_path = metadata_path
        self.media_root = media_root
        self.dim = dim
        self.pretrained_visual = pretrained_visual
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.aux_weight = aux_weight
        self.patience = patience
        self.num_workers = num_workers
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.limit_samples = limit_samples

        # Drive backup directory auto-detection or explicit configuration
        if drive_backup:
            self.drive_backup_dir = drive_backup
        elif os.path.exists("/content/drive/MyDrive"):
            self.drive_backup_dir = "/content/drive/MyDrive/RMS_AV_Checkpoints"
        else:
            self.drive_backup_dir = None

        if self.drive_backup_dir:
            os.makedirs(self.drive_backup_dir, exist_ok=True)
            print(f"[Persistence] Google Drive auto-sync target: {self.drive_backup_dir}")

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[{self.mode.upper()} Trainer] Initializing on device: {self.device} (AMP={self.use_amp})")

        # Build model
        self.model = build_lavdf_model(mode=self.mode, dim=self.dim, pretrained=self.pretrained_visual)
        self.model.to(self.device)
        tot_p, trn_p = count_parameters(self.model)
        print(f"[{self.mode.upper()} Model] Total params: {tot_p:,} | Trainable: {trn_p:,}")

        # Loss function & Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.scaler = torch.amp.GradScaler(self.device.type, init_scale=1024.0, enabled=self.use_amp)
            except Exception:
                self.scaler = torch.cuda.amp.GradScaler(init_scale=1024.0, enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(init_scale=1024.0, enabled=self.use_amp)

        # Preprocessing & DataLoaders
        self.prep_cfg = PreprocessingConfig(use_augmentation=True)
        self.train_loader, self.val_loader, self.test_loader = create_lavdf_dataloaders(
            metadata_path=self.metadata_path,
            media_root=self.media_root,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            config=self.prep_cfg,
            limit_samples=self.limit_samples,
        )

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "val_auc": [],
            "lr": [],
        }

    def train_epoch(self, epoch: int = 1) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        t_start = time.time()

        for step, batch in enumerate(self.train_loader, 1):
            # Send tensors to device
            batch_dev = {
                "video": batch["video"].to(self.device, non_blocking=True),
                "audio": batch["audio"].to(self.device, non_blocking=True),
                "has_video": batch["has_video"].to(self.device, non_blocking=True),
                "has_audio": batch["has_audio"].to(self.device, non_blocking=True),
            }
            targets = batch["label"].to(self.device, non_blocking=True)
            targets_v = batch["label_v"].to(self.device, non_blocking=True)
            targets_a = batch["label_a"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=self.use_amp)
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
                else torch.cuda.amp.autocast(enabled=self.use_amp)
            )
            with autocast_ctx:
                out = self.model(batch_dev)
                logits_joint = out["logit_av"]

                if self.mode == "rmsav":
                    loss_joint = self.bce_loss(logits_joint, targets)
                    loss_v = self.bce_loss(out["logit_v"], targets_v)
                    loss_a = self.bce_loss(out["logit_a"], targets_a)
                    loss = loss_joint + self.aux_weight * (loss_v + loss_a)
                else:
                    loss = self.bce_loss(logits_joint, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if step % 50 == 0 or step == num_batches:
                elapsed = time.time() - t_start
                speed = step / max(elapsed, 1e-4)
                remaining = (num_batches - step) / max(speed, 1e-4)
                rem_m, rem_s = divmod(int(remaining), 60)
                running_loss = total_loss / step
                print(
                    f"  [Epoch {epoch}/{self.epochs} | Batch {step}/{num_batches} "
                    f"({step/num_batches*100:.1f}%)] Loss: {running_loss:.4f} | "
                    f"Speed: {speed:.1f} b/s | ETA: {rem_m}m {rem_s:02d}s"
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_probs = []
        num_batches = len(self.val_loader)

        for step, batch in enumerate(self.val_loader, 1):
            batch_dev = {
                "video": batch["video"].to(self.device, non_blocking=True),
                "audio": batch["audio"].to(self.device, non_blocking=True),
                "has_video": batch["has_video"].to(self.device, non_blocking=True),
                "has_audio": batch["has_audio"].to(self.device, non_blocking=True),
            }
            targets = batch["label"].to(self.device, non_blocking=True)

            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=self.use_amp)
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
                else torch.cuda.amp.autocast(enabled=self.use_amp)
            )
            with autocast_ctx:
                out = self.model(batch_dev)
                logits = out["logit_av"]
                loss = self.bce_loss(logits, targets)

            probs = torch.sigmoid(logits).cpu().numpy()
            total_loss += loss.item()
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

            if step % 50 == 0 or step == num_batches:
                print(f"  [Validating] Batch {step}/{num_batches} ({step/num_batches*100:.1f}%)")

        val_loss = total_loss / max(len(self.val_loader), 1)
        y_true = np.array(all_targets)
        y_prob = np.array(all_probs)
        y_pred = (y_prob >= 0.50).astype(int)

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Robust AUROC (requires both classes present)
        try:
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = 0.5
        except Exception:
            auc = 0.5

        return val_loss, acc, macro_f1, auc

    def _atomic_save(self, state: Dict[str, Any], target_path: str) -> None:
        """Saves checkpoint atomically using a temp file and syncs to Drive if available."""
        tmp_path = f"{target_path}.tmp"
        torch.save(state, tmp_path)
        if os.path.exists(target_path):
            os.replace(tmp_path, target_path)
        else:
            os.rename(tmp_path, target_path)

        if self.drive_backup_dir:
            drive_dest = os.path.join(self.drive_backup_dir, os.path.basename(target_path))
            try:
                import shutil
                shutil.copy2(target_path, drive_dest)
            except Exception:
                pass

    def fit(self, resume: bool = True) -> Dict[str, Any]:
        """Executes full training loop with early stopping and automatic resume."""
        best_val_metric = -1.0
        patience_counter = 0
        best_epoch = 0
        start_epoch = 1

        best_ckpt_path = os.path.join(self.save_dir, f"{self.checkpoint_prefix}{self.mode}_best.pth")
        latest_ckpt_path = os.path.join(self.save_dir, f"{self.checkpoint_prefix}{self.mode}_latest.pth")

        # Fallback: Restore from Google Drive backup if local dir was reset
        if resume and not os.path.exists(latest_ckpt_path) and self.drive_backup_dir:
            drive_latest = os.path.join(self.drive_backup_dir, os.path.basename(latest_ckpt_path))
            if os.path.exists(drive_latest):
                import shutil
                print(f"[{self.mode.upper()} Resume] Restoring latest checkpoint from Google Drive: {drive_latest}")
                shutil.copy2(drive_latest, latest_ckpt_path)
            drive_best = os.path.join(self.drive_backup_dir, os.path.basename(best_ckpt_path))
            if os.path.exists(drive_best) and not os.path.exists(best_ckpt_path):
                import shutil
                shutil.copy2(drive_best, best_ckpt_path)

        if resume and os.path.exists(latest_ckpt_path):
            try:
                print(f"[{self.mode.upper()} Resume] Resuming from checkpoint: {latest_ckpt_path}")
                ckpt = torch.load(latest_ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scaler_state_dict" in ckpt and self.use_amp:
                    self.scaler.load_state_dict(ckpt["scaler_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_metric = ckpt.get("best_metric", -1.0)
                best_epoch = ckpt.get("epoch", 0)
                self.history = ckpt.get("history", self.history)
                print(f"[{self.mode.upper()} Resume] Successfully restored epoch {start_epoch - 1} (best metric: {best_val_metric:.4f})")
            except Exception as e:
                print(f"[!] Warning: Failed to resume from {latest_ckpt_path}: {e}. Starting fresh.")
                start_epoch = 1

        print(f"\n{'='*75}")
        print(f"STARTING TRAINING: {self.mode.upper()} (Epochs {start_epoch} to {self.epochs})")
        print(f"{'='*75}")

        for ep in range(start_epoch, self.epochs + 1):
            t0 = time.time()
            trn_loss = self.train_epoch(ep)

            # Instantly persist latest checkpoint so training progress is never lost
            interim_state = {
                "epoch": ep,
                "mode": self.mode,
                "best_metric": best_val_metric,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                "history": self.history,
                "config": {"dim": self.dim, "mode": self.mode},
            }
            self._atomic_save(interim_state, latest_ckpt_path)

            print(f"  [Epoch {ep}] Running validation to calculate Macro-F1 & AUROC...")
            val_loss, val_acc, val_f1, val_auc = self.validate()
            elapsed = time.time() - t0

            # Primary validation selection metric: Macro-F1 (or AUC)
            val_metric = val_f1

            self.history["epoch"].append(ep)
            self.history["train_loss"].append(round(trn_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_acc"].append(round(val_acc, 4))
            self.history["val_macro_f1"].append(round(val_f1, 4))
            self.history["val_auc"].append(round(val_auc, 4))
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            print(
                f"Epoch [{ep:02d}/{self.epochs:02d}] "
                f"Train Loss: {trn_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"Val Macro-F1: {val_f1*100:.2f}% | "
                f"Val AUC: {val_auc:.4f} "
                f"({elapsed:.1f}s)"
            )

            # Checkpoint saving (selection strictly on validation)
            ckpt_state = {
                "epoch": ep,
                "mode": self.mode,
                "best_metric": val_metric,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                "config": {
                    "mode": self.mode,
                    "dim": self.dim,
                    "lr": self.lr,
                    "batch_size": self.batch_size,
                    "aux_weight": self.aux_weight,
                },
                "history": self.history,
            }
            self._atomic_save(ckpt_state, latest_ckpt_path)

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_epoch = ep
                patience_counter = 0
                self._atomic_save(ckpt_state, best_ckpt_path)
                print(f"  --> Saved new best checkpoint to: {os.path.basename(best_ckpt_path)} (Macro-F1: {val_f1*100:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n[Early Stopping Triggered] No validation improvement in {self.patience} epochs. Halting.")
                    break

        # Save training history files
        csv_history_path = os.path.join(self.log_dir, f"{self.checkpoint_prefix}{self.mode}_history.csv")
        pd.DataFrame(self.history).to_csv(csv_history_path, index=False)

        json_summary_path = os.path.join(self.log_dir, f"{self.checkpoint_prefix}{self.mode}_summary.json")
        summary_data = {
            "mode": self.mode,
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_val_metric, 4),
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else 0.0,
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else 0.0,
            "best_checkpoint": best_ckpt_path,
        }
        with open(json_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        print(f"\n[Training Finished] Best validation Macro-F1: {best_val_metric*100:.2f}% (Epoch {best_epoch})")
        return summary_data


def main():
    parser = argparse.ArgumentParser(description="Train Audio-Visual Deepfake Models on LAV-DF.")
    parser.add_argument("--mode", type=str, default="rmsav", choices=["audio", "video", "fusion", "rmsav"])
    parser.add_argument("--metadata_path", type=str, default="metadata.min.json")
    parser.add_argument("--media_root", type=str, default="data")
    parser.add_argument("--batch_size", "--bs", type=int, default=16, dest="batch_size")
    parser.add_argument("--epochs", "--num_epochs", type=int, default=20, dest="epochs")
    parser.add_argument("--lr", "--learning_rate", type=float, default=1e-4, dest="lr")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--aux_weight", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints_lavdf")
    parser.add_argument("--log_dir", type=str, default="results_lavdf")
    parser.add_argument("--drive_backup", type=str, default=None, help="Directory on Google Drive to sync checkpoints")
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True, help="Automatically resume from latest checkpoint if available.")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    args, unknown = parser.parse_known_args()

    set_seed(args.seed)

    meta_path = args.metadata_path
    media_root = args.media_root

    # Auto-resolve metadata and media paths across Kaggle and Colab
    if not os.path.exists(meta_path):
        found = False
        # 1. Search dynamically under /kaggle/input if on Kaggle
        if os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                for target_meta in ["metadata.min.json", "metadata.json"]:
                    if target_meta in files:
                        meta_path = os.path.join(root, target_meta)
                        media_root = root
                        print(f"[Auto-Path] Dynamically found Kaggle metadata: {meta_path}")
                        print(f"[Auto-Path] Dynamically set Kaggle media root: {media_root}")
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f"[!] Warning: /kaggle/input exists but LAV-DF metadata was not found.")
                print(f"    Attached in /kaggle/input: {os.listdir('/kaggle/input')}")
                print(f"    Please attach the dataset: elin75/localized-audio-visual-deepfake-dataset-lav-df")

        # 2. Search dynamically under /root/.cache/kagglehub if on Colab
        if not found and os.path.exists("/root/.cache/kagglehub"):
            for root, dirs, files in os.walk("/root/.cache/kagglehub"):
                for target_meta in ["metadata.min.json", "metadata.json"]:
                    if target_meta in files:
                        meta_path = os.path.join(root, target_meta)
                        media_root = root
                        print(f"[Auto-Path] Dynamically found Colab metadata: {meta_path}")
                        print(f"[Auto-Path] Dynamically set Colab media root: {media_root}")
                        found = True
                        break
                if found:
                    break

    trainer = LAVDFTrainer(
        mode=args.mode,
        metadata_path=meta_path,
        media_root=media_root,
        dim=args.dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        aux_weight=args.aux_weight,
        patience=args.patience,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        drive_backup=args.drive_backup,
        limit_samples=args.limit_samples,
    )
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
