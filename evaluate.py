r"""
==================================================================================================
Standalone Evaluation and Stress Testing Protocol for RMS-AV (Member 2 Integration)
--------------------------------------------------------------------------------------------------
Capabilities:
1. Reloads saved checkpoints (.pth) for modes: 'audio', 'video', 'fusion', 'rmsav'.
2. Evaluates on holdout test manifests (test_manifest.csv).
3. Computes research metrics:
   - Overall Accuracy & Balanced Accuracy
   - Precision, Recall, Macro-F1
   - Area Under ROC Curve (AUC)
   - False Positive Rate (FPR) & False Negative Rate (FNR)
   - Expected Calibration Error (ECE)
   - Confusion Matrix (TN, FP, FN, TP)
4. Robustness Stress Testing:
   - Evaluates under missing video (--stress_test missing_video)
   - Evaluates under missing audio (--stress_test missing_audio)
   - Evaluates under audio noise or video blur
5. Exports predictions to CSV and summary report to JSON in results/eval/.
==================================================================================================
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model
from dataset import FakeAVCelebDataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeepfakeEvaluator")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


@torch.no_grad()
def evaluate_checkpoint(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    stress_test: str = "none",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Runs evaluation across dataloader with optional stress-test modality masking."""
    model.eval()
    all_targets: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []
    all_gate_a: List[float] = []
    all_gate_v: List[float] = []

    pbar = tqdm(dataloader, desc=f"Evaluating [Stress: {stress_test}]", leave=False)
    for batch in pbar:
        v_tensor = batch["video"].to(device)
        a_tensor = batch["audio"].to(device)
        has_v = batch["has_video"].to(device)
        has_a = batch["has_audio"].to(device)
        labels = batch["label"].to(device)

        # Apply stress testing perturbations
        if stress_test == "missing_video":
            has_v = torch.zeros_like(has_v)
        elif stress_test == "missing_audio":
            has_a = torch.zeros_like(has_a)
        elif stress_test == "audio_noise":
            a_tensor = a_tensor + 0.5 * torch.randn_like(a_tensor)
        elif stress_test == "video_blur":
            v_tensor = v_tensor * 0.1

        batch_dev = {
            "video": v_tensor,
            "audio": a_tensor,
            "has_video": has_v,
            "has_audio": has_a,
        }

        outputs = model(batch_dev)
        logit_av = outputs["logit_av"]

        probs = torch.sigmoid(logit_av).cpu().view(-1).numpy()
        preds = (probs >= 0.5).astype(int)
        targets = labels.cpu().view(-1).numpy().astype(int)

        gate = outputs.get("gate")
        if gate is not None:
            g_a = gate[:, 0].cpu().numpy().tolist()
            g_v = gate[:, 1].cpu().numpy().tolist()
        else:
            g_a = [float("nan")] * len(targets)
            g_v = [float("nan")] * len(targets)

        all_targets.extend(targets.tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_gate_a.extend(g_a)
        all_gate_v.extend(g_v)

    targets_arr = np.array(all_targets)
    probs_arr = np.array(all_probs)
    preds_arr = np.array(all_preds)

    acc = float(accuracy_score(targets_arr, preds_arr)) if len(targets_arr) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(targets_arr, preds_arr)) if len(targets_arr) > 0 else 0.0
    prec = float(precision_score(targets_arr, preds_arr, zero_division=0)) if len(targets_arr) > 0 else 0.0
    rec = float(recall_score(targets_arr, preds_arr, zero_division=0)) if len(targets_arr) > 0 else 0.0
    f1 = float(f1_score(targets_arr, preds_arr, zero_division=0)) if len(targets_arr) > 0 else 0.0

    if len(set(targets_arr)) > 1:
        auc = float(roc_auc_score(targets_arr, probs_arr))
    else:
        auc = 0.5

    ece = compute_ece(probs_arr, targets_arr)

    if len(targets_arr) > 0 and len(set(targets_arr)) > 1:
        tn, fp, fn, tp = confusion_matrix(targets_arr, preds_arr, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        fpr, fnr = 0.0, 0.0

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": auc,
        "fpr": fpr,
        "fnr": fnr,
        "ece": ece,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_evaluated": len(targets_arr),
    }

    df_preds = pd.DataFrame({
        "target": all_targets,
        "probability": all_probs,
        "prediction": all_preds,
        "gate_audio": all_gate_a,
        "gate_video": all_gate_v,
    })

    return metrics, df_preds


def main():
    parser = argparse.ArgumentParser(description="RMS-AV Evaluation Protocol")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint (.pth)")
    parser.add_argument("--test_manifest", type=str, required=True, help="Path to test_manifest.csv")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument(
        "--stress_test",
        type=str,
        default="none",
        choices=["none", "missing_video", "missing_audio", "audio_noise", "video_blur"],
        help="Robustness stress testing condition",
    )
    parser.add_argument("--output_dir", type=str, default="./results/eval", help="Directory to save evaluation reports")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running evaluation on device: {device} | Stress condition: {args.stress_test}")

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file '{args.checkpoint}' not found!")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})
    mode = ckpt.get("mode", config.get("mode", "rmsav"))
    dim = config.get("dim", 256)

    logger.info(f"Loaded checkpoint from: {args.checkpoint} (Mode: {mode}, Dim: {dim})")

    model = build_model(mode=mode, cfg={"dim": dim, "pretrained": False})
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if not os.path.exists(args.test_manifest):
        logger.error(f"Test manifest '{args.test_manifest}' not found!")
        sys.exit(1)

    dataset = FakeAVCelebDataset(manifest_path=args.test_manifest, data_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics, df_preds = evaluate_checkpoint(
        model=model,
        dataloader=dataloader,
        device=device,
        stress_test=args.stress_test,
    )

    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {mode.upper()} [Stress Condition: {args.stress_test}]")
    print("=" * 80)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:10.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_json = os.path.join(args.output_dir, f"{base_name}_{args.stress_test}_metrics.json")
    out_csv = os.path.join(args.output_dir, f"{base_name}_{args.stress_test}_preds.csv")

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    df_preds.to_csv(out_csv, index=False)

    logger.info(f"Evaluation report saved to: {out_json}")
    logger.info(f"Detailed predictions saved to: {out_csv}")


if __name__ == "__main__":
    main()
