r"""
==================================================================================================
Training and Evaluation Pipeline for PG-ABN Deepfake Detection
--------------------------------------------------------------------------------------------------
IEEE-Style Documentation & Module Specification:
Author: Multimodal Deepfake Detection Research Team
Target Venue: IEEE Transactions on Information Forensics and Security (T-IFS) / IEEE Access

Theoretical Formulation:
1. Multi-Task Objective Function:
   The total optimization objective \mathcal{L}_{total} balances binary manipulation classification
   and temporal manipulation interval localization:
       \mathcal{L}_{total} = \mathcal{L}_{cls}(\hat{y}_{cls}, y_{cls}) + \lambda_{loc} \cdot \mathbb{I}(y_{cls}=1) \cdot \mathcal{L}_{loc}(\hat{\tau}, \tau)
   where:
       - \mathcal{L}_{cls} is Binary Cross-Entropy with Logits (BCEWithLogitsLoss).
       - \mathcal{L}_{loc} is Smooth L1 / Mean Squared Error regression loss on normalized interval coordinates.
       - \mathbb{I}(y_{cls}=1) is an indicator masking localization loss on pristine (real) samples.
       - \lambda_{loc} is the temporal localization regularization coefficient (default: 0.5).

2. Evaluation Protocol & Research Paper Metrics:
   At the culmination of each training epoch, comprehensive validation is executed to derive:
       - Classification Accuracy (ACC): Ratio of true predictions over total samples.
       - F1-Score: Harmonic mean of precision and recall for deepfake detection.
       - Area Under Receiver Operating Characteristic Curve (ROC-AUC): Discrimination threshold metric.
   Results are formatted into publication-ready ASCII and LaTeX table rows for direct insertion
   into academic manuscripts.
==================================================================================================
"""

import argparse
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import local PG-ABN modular architecture and dataset
from model import PGABN, build_pgabn_model
from dataset import FakeAVCelebDataset

# Setup publication-grade logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PG-ABN-Trainer")


def set_seed(seed: int = 42) -> None:
    """
    Enforces deterministic reproducibility across random generators.

    Args:
        seed (int): Seed integer value. Default: 42.
    """
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
    criterion_cls: nn.Module,
    criterion_loc: nn.Module,
    device: torch.device,
    epoch: int,
    loc_weight: float = 0.5,
) -> float:
    """
    Executes a single training epoch across all training batches.

    Args:
        model (nn.Module): The PG-ABN network.
        dataloader (DataLoader): PyTorch training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer (Adam).
        criterion_cls (nn.Module): Binary classification loss (BCEWithLogitsLoss).
        criterion_loc (nn.Module): Temporal localization loss (MSELoss / SmoothL1).
        device (torch.device): Execution device (CPU or CUDA).
        epoch (int): Current epoch counter.
        loc_weight (float): Multi-task weight for localization.

    Returns:
        float: Mean training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)       # (B, 16, 3, 224, 224)
        input_features = batch["input_features"].to(device)   # (B, 80, 3000)
        labels = batch["label"].to(device)                   # (B, 1)
        targets_loc = batch["localization"].to(device)       # (B, 2)

        optimizer.zero_grad()

        # Forward propagation
        outputs = model(pixel_values, input_features)
        cls_logits = outputs["cls_logits"]
        pred_loc = outputs["localization"]

        # Classification loss
        loss_cls = criterion_cls(cls_logits, labels)

        # Localization loss (active primarily for fake samples where ground truth is annotated)
        fake_mask = (labels == 1.0).squeeze(-1)
        if fake_mask.sum() > 0:
            loss_loc = criterion_loc(pred_loc[fake_mask], targets_loc[fake_mask])
        else:
            loss_loc = torch.tensor(0.0, device=device)

        # Joint multi-task objective
        loss = loss_cls + (loc_weight * loss_loc)

        # Backward propagation
        loss.backward()

        # Gradient clipping to stabilize prompt and fusion weights
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        running_loss += loss.item() * pixel_values.size(0)
        total_samples += pixel_values.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "ClsLoss": f"{loss_cls.item():.4f}"})

    epoch_loss = running_loss / max(total_samples, 1)
    return epoch_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_cls: nn.Module,
    criterion_loc: nn.Module,
    device: torch.device,
    epoch: int,
    loc_weight: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluates the model and computes research paper metrics: Accuracy, F1-Score, and AUC.

    Args:
        model (nn.Module): The PG-ABN network.
        dataloader (DataLoader): Validation dataloader.
        criterion_cls (nn.Module): BCE classification criterion.
        criterion_loc (nn.Module): Localization criterion.
        device (torch.device): Execution device.
        epoch (int): Current epoch number.
        loc_weight (float): Localization loss scale.

    Returns:
        Dict[str, float]: Computed metric values (Loss, Accuracy, F1, AUC).
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_targets: List[float] = []
    all_pred_probs: List[float] = []
    all_pred_binary: List[int] = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Val]", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        input_features = batch["input_features"].to(device)
        labels = batch["label"].to(device)
        targets_loc = batch["localization"].to(device)

        outputs = model(pixel_values, input_features)
        cls_logits = outputs["cls_logits"]
        pred_loc = outputs["localization"]

        loss_cls = criterion_cls(cls_logits, labels)

        fake_mask = (labels == 1.0).squeeze(-1)
        if fake_mask.sum() > 0:
            loss_loc = criterion_loc(pred_loc[fake_mask], targets_loc[fake_mask])
        else:
            loss_loc = torch.tensor(0.0, device=device)

        loss = loss_cls + (loc_weight * loss_loc)
        running_loss += loss.item() * pixel_values.size(0)
        total_samples += pixel_values.size(0)

        # Collect probabilities and binary thresholded predictions (threshold = 0.5)
        probs = torch.sigmoid(cls_logits).cpu().view(-1).numpy()
        binary_preds = (probs >= 0.5).astype(int)
        targets = labels.cpu().view(-1).numpy().astype(int)

        all_targets.extend(targets.tolist())
        all_pred_probs.extend(probs.tolist())
        all_pred_binary.extend(binary_preds.tolist())

    val_loss = running_loss / max(total_samples, 1)

    # Calculate standard evaluation metrics via scikit-learn
    if len(all_targets) > 0:
        acc = accuracy_score(all_targets, all_pred_binary)
        f1 = f1_score(all_targets, all_pred_binary, zero_division=0)

        # Compute ROC-AUC (guard against single-class edge case in small batches)
        if len(set(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_pred_probs)
        else:
            auc = 0.5
            logger.warning("Validation set contains only one class. ROC-AUC defaulted to 0.5.")
    else:
        acc, f1, auc = 0.0, 0.0, 0.0

    return {
        "val_loss": val_loss,
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc,
    }


def print_paper_table(epoch: int, total_epochs: int, train_loss: float, metrics: Dict[str, float]) -> None:
    """
    Renders research paper results cleanly in an IEEE-style table and a LaTeX row
    ready to be copied directly into academic manuscript tables.
    """
    acc_pct = metrics["accuracy"] * 100.0
    f1_val = metrics["f1_score"]
    auc_val = metrics["auc"]
    val_loss = metrics["val_loss"]

    table = f"""
+===============================================================================================+
|            EPOCH [{epoch:02d} / {total_epochs:02d}] EVALUATION SUMMARY (RESEARCH PAPER METRICS)             |
+===============================================================================================+
| Metric Name                     | Value       | Academic Notation / Description               |
+---------------------------------+-------------+-----------------------------------------------+
| Training Loss                   | {train_loss:11.4f} | Total Multi-Task Training Objective           |
| Validation Loss                 | {val_loss:11.4f} | Total Multi-Task Validation Objective         |
| Classification Accuracy (ACC)   | {acc_pct:10.2f}% | Overall Correct Rate (Real & Fake)            |
| F1-Score (Forensic Detection)   | {f1_val:11.4f} | Harmonic Mean of Precision & Recall (Fake)    |
| Area Under ROC Curve (AUC)      | {auc_val:11.4f} | Threshold-Independent Discriminability        |
+===============================================================================================+
[Copy-Paste LaTeX Table Row for Manuscript]:
Epoch {epoch} & {acc_pct:.2f}\\% & {f1_val:.4f} & {auc_val:.4f} & {val_loss:.4f} \\\\
-------------------------------------------------------------------------------------------------"""
    print(table)


def main():
    """
    Command-line entry point for training and evaluating PG-ABN.
    """
    parser = argparse.ArgumentParser(
        description="Prompt-Guided Asymmetric Bridge Network (PG-ABN) Training Protocol"
    )
    # Required and standard CLI arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/content/FakeAVCeleb_Dataset",
        help="Path to directory containing meta_data.csv and multimedia files (default: /content/FakeAVCeleb_Dataset)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Mini-batch size for training and validation (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Total number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization (default: 1e-4)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of dataset allocated for validation (default: 0.2)",
    )
    parser.add_argument(
        "--bridge_dim",
        type=int,
        default=512,
        help="Dimension of Asymmetric Bridge latent space (default: 512)",
    )
    parser.add_argument(
        "--loc_weight",
        type=float,
        default=0.5,
        help="Loss weight multiplier for temporal localization head (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training runs (default: 42)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default: 0 for compatibility)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save the best model weights (default: ./checkpoints)",
    )
    args = parser.parse_args()

    # 1. Determinism and Device Configuration
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initialized training execution on device: {device}")

    # 2. Dataset Initialization
    logger.info(f"Loading FakeAVCeleb dataset from: {args.data_dir}")
    full_dataset = FakeAVCelebDataset(data_dir=args.data_dir)

    if len(full_dataset) == 0:
        logger.error(
            f"No records found in dataset at '{args.data_dir}'! Please verify meta_data.csv path. Exiting."
        )
        sys.exit(1)

    # Split into Train and Validation Partitions
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Dataset partitioned: {train_size} training samples, {val_size} validation samples.")

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 3. Model Architecture Instantiation
    logger.info("Instantiating PG-ABN architecture with frozen foundation backbones...")
    model = build_pgabn_model(bridge_dim=args.bridge_dim)
    model.to(device)

    # Verification: Only Prompts, Fusion, and Heads should require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model Parameters: Total = {total_params:,} | Trainable = {total_trainable:,} ({(total_trainable/total_params)*100:.2f}%)"
    )

    # 4. Optimization & Loss Criteria
    optimizer = torch.optim.Adam(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_loc = nn.MSELoss()

    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    best_auc = -1.0
    best_epoch = -1

    # 5. Training & Evaluation Epoch Loop
    logger.info(f"Beginning training protocol for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Training step
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion_cls=criterion_cls,
            criterion_loc=criterion_loc,
            device=device,
            epoch=epoch,
            loc_weight=args.loc_weight,
        )

        # Validation step
        metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion_cls=criterion_cls,
            criterion_loc=criterion_loc,
            device=device,
            epoch=epoch,
            loc_weight=args.loc_weight,
        )

        # Print cleanly formatted table for research paper
        print_paper_table(
            epoch=epoch,
            total_epochs=args.epochs,
            train_loss=train_loss,
            metrics=metrics,
        )

        # Checkpoint best model based on ROC-AUC metric
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_epoch = epoch
            save_path = os.path.join(args.save_dir, "pg_abn_best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                },
                save_path,
            )
            logger.info(f"--> Saved new optimal checkpoint to: {save_path} (AUC: {best_auc:.4f})")

    logger.info(
        f"Training protocol completed! Optimal AUC: {best_auc:.4f} achieved at Epoch {best_epoch}."
    )


if __name__ == "__main__":
    main()
