r"""
==================================================================================================
Configurable Training Pipeline for RMS-AV and Baselines (Member 1 & Member 2 Integration)
--------------------------------------------------------------------------------------------------
Supports:
1. Model modes: 'audio', 'video', 'fusion', 'rmsav'.
2. Leakage-safe split manifests (train_manifest.csv, val_manifest.csv).
3. Multi-task loss supervising unimodal heads (logit_video, logit_audio) and fused heads (logit_av).
4. Automatic Mixed Precision (AMP) support for GPU efficiency.
5. Checkpoint saving (best & latest), resume capability, and CSV history logging.
==================================================================================================
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model, count_params
from dataset import FakeAVCelebDataset, create_leakage_safe_splits

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeepfakeTrainer")


def set_seed(seed: int = 42) -> None:
    """Enforces deterministic reproducibility across random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    mode: str = "rmsav",
    aux_weight: float = 0.2,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """Executes a single training epoch across batches."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for batch in pbar:
        # Move batch tensors to device
        batch_dev = {
            "video": batch["video"].to(device),
            "audio": batch["audio"].to(device),
            "has_video": batch["has_video"].to(device),
            "has_audio": batch["has_audio"].to(device),
        }
        labels = batch["label"].to(device).float()
        video_labels = batch["video_label"].to(device).float()
        audio_labels = batch["audio_label"].to(device).float()

        b_size = labels.size(0)
        optimizer.zero_grad()

        # Forward pass
        with torch.amp.autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
            outputs = model(batch_dev)
            logit_av = outputs["logit_av"]
            loss = criterion(logit_av, labels)

            # Unimodal auxiliary supervision for RMS-AV
            if mode == "rmsav" and aux_weight > 0:
                has_v = batch_dev["has_video"].bool()
                has_a = batch_dev["has_audio"].bool()

                if has_v.sum() > 0:
                    loss_v = criterion(outputs["logit_video"][has_v], video_labels[has_v])
                    loss = loss + (aux_weight * loss_v)

                if has_a.sum() > 0:
                    loss_a = criterion(outputs["logit_audio"][has_a], audio_labels[has_a])
                    loss = loss + (aux_weight * loss_a)

        # Backward propagation
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * b_size
        total_samples += b_size
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Evaluates validation performance without modifying weights."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_targets: List[float] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Val]", leave=False)
    for batch in pbar:
        batch_dev = {
            "video": batch["video"].to(device),
            "audio": batch["audio"].to(device),
            "has_video": batch["has_video"].to(device),
            "has_audio": batch["has_audio"].to(device),
        }
        labels = batch["label"].to(device).float()
        b_size = labels.size(0)

        with torch.amp.autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
            outputs = model(batch_dev)
            logit_av = outputs["logit_av"]
            loss = criterion(logit_av, labels)

        running_loss += loss.item() * b_size
        total_samples += b_size

        probs = torch.sigmoid(logit_av).cpu().view(-1).numpy()
        preds = (probs >= 0.5).astype(int)
        targets = labels.cpu().view(-1).numpy().astype(int)

        all_targets.extend(targets.tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

    val_loss = running_loss / max(total_samples, 1)

    if len(all_targets) > 0:
        acc = float(accuracy_score(all_targets, all_preds))
        bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
        f1 = float(f1_score(all_targets, all_preds, zero_division=0))
        if len(set(all_targets)) > 1:
            auc = float(roc_auc_score(all_targets, all_probs))
        else:
            auc = 0.5
    else:
        acc, bal_acc, f1, auc = 0.0, 0.0, 0.0, 0.5

    return {
        "val_loss": val_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "auc": auc,
    }


def print_paper_table(epoch: int, total_epochs: int, train_loss: float, metrics: Dict[str, float]) -> None:
    """Renders results cleanly in an academic evaluation table."""
    acc_pct = metrics["accuracy"] * 100.0
    bal_acc_pct = metrics["balanced_accuracy"] * 100.0
    f1_val = metrics["f1_score"]
    auc_val = metrics["auc"]
    val_loss = metrics["val_loss"]

    table = f"""
+===============================================================================================+
|            EPOCH [{epoch:02d} / {total_epochs:02d}] EVALUATION SUMMARY (RESEARCH PAPER METRICS)             |
+===============================================================================================+
| Metric Name                     | Value       | Academic Notation / Description               |
+---------------------------------+-------------+-----------------------------------------------+
| Training Loss                   | {train_loss:11.4f} | Multi-Task Training Objective                 |
| Validation Loss                 | {val_loss:11.4f} | Validation Cross-Entropy Loss                 |
| Overall Accuracy (ACC)          | {acc_pct:10.2f}% | Overall Correct Rate (Real & Fake)            |
| Balanced Accuracy (B-ACC)       | {bal_acc_pct:10.2f}% | Unweighted Average Recall                     |
| F1-Score (Forensic Detection)   | {f1_val:11.4f} | Harmonic Mean of Precision & Recall (Fake)    |
| Area Under ROC Curve (AUC)      | {auc_val:11.4f} | Threshold-Independent Discriminability        |
+===============================================================================================+
[LaTeX Table Row]:
Epoch {epoch} & {acc_pct:.2f}\\% & {bal_acc_pct:.2f}\\% & {f1_val:.4f} & {auc_val:.4f} & {val_loss:.4f} \\\\
-------------------------------------------------------------------------------------------------"""
    print(table)


def main():
    parser = argparse.ArgumentParser(description="RMS-AV Training Pipeline")
    parser.add_argument("--mode", type=str, default="rmsav", choices=["audio", "video", "fusion", "rmsav"], help="Model mode")
    parser.add_argument("--manifest_dir", type=str, default=None, help="Directory containing train_manifest.csv and val_manifest.csv")
    parser.add_argument("--train_manifest", type=str, default=None, help="Direct path to train_manifest.csv")
    parser.add_argument("--val_manifest", type=str, default=None, help="Direct path to val_manifest.csv")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory (contains frames/ and audio/ or raw videos)")
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of video frames per sample")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--aux_weight", type=float, default=0.2, help="Loss weight for unimodal heads in RMS-AV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (AMP)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./results", help="Directory to save training logs and CSV history")
    parser.add_argument("--checkpoint_prefix", type=str, default="", help="Optional prefix for saved checkpoints and history files (e.g. 'pilot_')")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # Determinism and Device Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initialized training execution on device: {device} (AMP: {args.amp})")

    # Resolve Manifests
    train_manifest = args.train_manifest
    val_manifest = args.val_manifest

    if train_manifest is None and args.manifest_dir is not None:
        train_manifest = os.path.join(args.manifest_dir, "train_manifest.csv")
        val_manifest = os.path.join(args.manifest_dir, "val_manifest.csv")

    if (train_manifest is None or not os.path.exists(train_manifest)) and args.data_dir is not None:
        # Check if processed_manifest.csv exists in data_dir
        base_manifest = os.path.join(args.data_dir, "processed_manifest.csv")
        if not os.path.exists(base_manifest):
            base_manifest = os.path.join(args.data_dir, "meta_data.csv")

        if os.path.exists(base_manifest):
            logger.info(f"Generating leakage-safe split manifests from {base_manifest}...")
            splits_out = os.path.join(args.save_dir, "splits")
            train_manifest, val_manifest, _ = create_leakage_safe_splits(
                manifest_path=base_manifest,
                output_dir=splits_out,
                seed=args.seed,
            )

    if train_manifest is None or not os.path.exists(train_manifest):
        logger.error("No valid train manifest found! Please specify --manifest_dir, --train_manifest, or --data_dir.")
        sys.exit(1)

    logger.info(f"Loading train manifest: {train_manifest}")
    logger.info(f"Loading val manifest: {val_manifest}")

    train_dataset = FakeAVCelebDataset(manifest_path=train_manifest, data_dir=args.data_dir, num_frames=args.num_frames)
    val_dataset = FakeAVCelebDataset(manifest_path=val_manifest, data_dir=args.data_dir, num_frames=args.num_frames)

    logger.info(f"Dataset ready: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Build Model
    logger.info(f"Instantiating model mode: '{args.mode}' (dim={args.dim})...")
    model = build_model(mode=args.mode, cfg={"dim": args.dim, "pretrained": False})
    model.to(device)

    total_params, trainable_params = count_params(model)
    logger.info(
        f"Model Parameters: Total = {total_params:,} | Trainable = {trainable_params:,}"
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp) if torch.cuda.is_available() else None

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    start_epoch = 1
    best_metric = -1.0
    history: List[Dict[str, float]] = []

    # Optional Resume
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", -1.0)

    # Training Loop
    logger.info(f"Beginning training protocol for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            mode=args.mode,
            aux_weight=args.aux_weight,
            use_amp=args.amp,
            scaler=scaler,
        )

        metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=args.amp,
        )

        print_paper_table(
            epoch=epoch,
            total_epochs=args.epochs,
            train_loss=train_loss,
            metrics=metrics,
        )

        record = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(record)

        prefix = args.checkpoint_prefix
        # Save latest checkpoint
        latest_path = os.path.join(args.save_dir, f"{prefix}{args.mode}_latest.pth")
        torch.save(
            {
                "epoch": epoch,
                "mode": args.mode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": vars(args),
            },
            latest_path,
        )

        # Save best checkpoint (based on validation AUC)
        if metrics["auc"] > best_metric:
            best_metric = metrics["auc"]
            best_path = os.path.join(args.save_dir, f"{prefix}{args.mode}_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "mode": args.mode,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "metrics": metrics,
                    "config": vars(args),
                },
                best_path,
            )
            logger.info(f"--> Saved optimal checkpoint to: {best_path} (AUC: {best_metric:.4f})")

    # Save History CSV
    history_csv = os.path.join(args.log_dir, f"{prefix}{args.mode}_history.csv")
    pd.DataFrame(history).to_csv(history_csv, index=False)
    logger.info(f"Training history saved to: {history_csv}")
    logger.info(f"Training complete. Best AUC: {best_metric:.4f}")


if __name__ == "__main__":
    main()
