r"""
LAV-DF Model Evaluation & Forensic Stress Testing Module
--------------------------------------------------------
Evaluates trained checkpoints on the held-out test split of LAV-DF.
Calculates:
1. Classification Metrics: Accuracy, Balanced Accuracy, Precision, Recall, Real-F1, Fake-F1, True Macro-F1.
2. Ranking & Discrimination: Continuous ROC-AUC, Average Precision (PR-AUC).
3. Probability Calibration: Brier Score, Expected Calibration Error (ECE).
4. Modality-Drop Stress Testing: Clean vs. Missing Audio vs. Missing Video.
5. Dynamic Reliability Gate Statistics for RMS-AV (g_a, g_v).
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

from lavdf_pipeline.model import build_lavdf_model, LAVDFDetector
from lavdf_pipeline.dataset import LAVDFDataset, collate_lavdf_fn
from lavdf_pipeline.preprocessing import PreprocessingConfig


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) with equal-width probability bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= low) & (probs < high) if i < n_bins - 1 else (probs >= low) & (probs <= high)
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


class LAVDFEvaluator:
    def __init__(
        self,
        metadata_path: str,
        media_root: str,
        checkpoint_dir: str = "checkpoints_lavdf",
        output_dir: str = "results_lavdf",
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        self.metadata_path = metadata_path
        self.media_root = media_root
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        os.makedirs(self.output_dir, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load held-out test split
        self.test_dataset = LAVDFDataset(
            metadata_path=self.metadata_path,
            media_root=self.media_root,
            split="test",
            config=PreprocessingConfig(use_augmentation=False),
            is_train=False,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_lavdf_fn,
        )
        print(f"[LAVDFEvaluator] Initialized on {self.device} with {len(self.test_dataset):,} test samples.")

    def load_checkpoint(self, checkpoint_path: str, mode: str) -> LAVDFDetector:
        """Loads model strictly from checkpoint. Raises error if checkpoint is missing or corrupt."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt.get("config", {})
        dim = cfg.get("dim", 256)

        model = build_lavdf_model(mode=mode, dim=dim, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def evaluate_model(
        self,
        model: LAVDFDetector,
        mode: str,
        condition: str = "clean",
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Runs inference across the test split under specified condition:
        - 'clean'         : Uncorrupted audio and video streams
        - 'missing_video' : Video stream zeroed / masked out
        - 'missing_audio' : Audio stream zeroed / masked out
        """
        all_targets = []
        all_probs = []
        all_logits = []
        all_categories = []
        gate_a_list = []
        gate_v_list = []

        for batch in self.test_loader:
            v_tensor = batch["video"].to(self.device)
            a_tensor = batch["audio"].to(self.device)
            has_v = batch["has_video"].to(self.device)
            has_a = batch["has_audio"].to(self.device)
            targets = batch["label"].numpy()

            # Apply stress condition
            if condition == "missing_video":
                v_tensor = torch.zeros_like(v_tensor)
                has_v = torch.zeros_like(has_v)
            elif condition == "missing_audio":
                a_tensor = torch.zeros_like(a_tensor)
                has_a = torch.zeros_like(has_a)

            batch_dev = {
                "video": v_tensor,
                "audio": a_tensor,
                "has_video": has_v,
                "has_audio": has_a,
            }

            out = model(batch_dev)
            logits = out["logit_av"].cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))

            all_targets.extend(targets.tolist())
            all_probs.extend(probs.tolist())
            all_logits.extend(logits.tolist())
            all_categories.extend(batch.get("category", []))

            if out.get("gate") is not None:
                gates = out["gate"].cpu().numpy()
                gate_a_list.extend(gates[:, 0].tolist())
                gate_v_list.extend(gates[:, 1].tolist())

            if out.get("gate") is not None:
                gates = out["gate"].cpu().numpy()
                gate_a_list.extend(gates[:, 0].tolist())
                gate_v_list.extend(gates[:, 1].tolist())

        y_true = np.array(all_targets)
        y_prob = np.array(all_probs)
        y_pred = (y_prob >= threshold).astype(int)

        # Basic confusion counts
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        # Class-specific F1
        f1_fake = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_real = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        macro_f1 = (f1_fake + f1_real) / 2.0

        # Continuous Discrimination Metrics
        try:
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob)
                ap = average_precision_score(y_true, y_prob)
            else:
                auc = 0.5
                ap = 0.5
        except Exception:
            auc = 0.5
            ap = 0.5

        # Calibration
        brier = brier_score_loss(y_true, y_prob)
        ece = compute_ece(y_prob, y_true)

        # Per-Category Diagnostic Breakdown
        cats_arr = np.array(all_categories)
        cat_metrics = {}
        for c_name in ["real", "audio_only_fake", "video_only_fake", "both_modalities_fake"]:
            mask = (cats_arr == c_name)
            if np.sum(mask) > 0:
                sub_true = y_true[mask]
                sub_pred = y_pred[mask]
                sub_acc = float(np.mean(sub_pred == sub_true))
                cat_metrics[f"acc_{c_name}"] = round(sub_acc, 4)
                cat_metrics[f"count_{c_name}"] = int(np.sum(mask))
            else:
                cat_metrics[f"acc_{c_name}"] = 0.0
                cat_metrics[f"count_{c_name}"] = 0

        results = {
            "mode": mode,
            "condition": condition,
            "threshold": threshold,
            "total_samples": len(y_true),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_fake": round(float(f1_fake), 4),
            "f1_real": round(float(f1_real), 4),
            "macro_f1": round(float(macro_f1), 4),
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(ap), 4),
            "brier_score": round(float(brier), 4),
            "ece": round(float(ece), 4),
            **cat_metrics,
        }

        if gate_a_list:
            results["mean_gate_audio"] = round(float(np.mean(gate_a_list)), 4)
            results["mean_gate_video"] = round(float(np.mean(gate_v_list)), 4)

        return results

    def run_benchmark(self, checkpoint_map: Dict[str, str]) -> pd.DataFrame:
        """Runs full comparative benchmark across all models and stress conditions."""
        all_results = []
        conditions = ["clean", "missing_video", "missing_audio"]

        print(f"\n{'='*75}")
        print("RUNNING BENCHMARK EVALUATION ACROSS ALL ARCHITECTURES")
        print(f"{'='*75}")

        for mode, ckpt_path in checkpoint_map.items():
            if not os.path.exists(ckpt_path):
                print(f"[!] Checkpoint for mode '{mode}' not found at {ckpt_path}. Skipping.")
                continue

            print(f"Loading {mode.upper()} from {os.path.basename(ckpt_path)}...")
            model = self.load_checkpoint(ckpt_path, mode=mode)

            for cond in conditions:
                res = self.evaluate_model(model, mode=mode, condition=cond)
                all_results.append(res)
                print(
                    f"  [{mode.upper():<7} | {cond:<14}] "
                    f"Acc: {res['accuracy']*100:.2f}% | "
                    f"Macro-F1: {res['macro_f1']*100:.2f}% | "
                    f"AUC: {res['roc_auc']:.4f} | "
                    f"ECE: {res['ece']:.4f}"
                )

        df = pd.DataFrame(all_results)
        csv_path = os.path.join(self.output_dir, "lavdf_benchmark_comparison.csv")
        df.to_csv(csv_path, index=False)

        json_path = os.path.join(self.output_dir, "lavdf_benchmark_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSaved benchmark outputs to:")
        print(f"  - CSV: {csv_path}")
        print(f"  - JSON: {json_path}")
        return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate LAV-DF Deepfake Detection Models.")
    parser.add_argument("--mode", type=str, default=None, choices=["audio", "video", "fusion", "rmsav"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific model checkpoint")
    parser.add_argument("--metadata_path", type=str, default="metadata.min.json")
    parser.add_argument("--media_root", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_lavdf")
    parser.add_argument("--output_dir", type=str, default="results_lavdf")
    parser.add_argument("--prefix", type=str, default="lavdf_")
    parser.add_argument("--batch_size", "--bs", type=int, default=16, dest="batch_size")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--stress_test", action="store_true", default=False)
    args, unknown = parser.parse_known_args()

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

    evaluator = LAVDFEvaluator(
        metadata_path=meta_path,
        media_root=media_root,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    if args.checkpoint and args.mode:
        if not os.path.exists(args.checkpoint):
            print(f"[!] Specified checkpoint not found: {args.checkpoint}")
            return
        model = evaluator.load_checkpoint(args.checkpoint, mode=args.mode)
        conditions = ["clean", "missing_video", "missing_audio"] if args.stress_test else ["clean"]
        for cond in conditions:
            res = evaluator.evaluate_model(model, mode=args.mode, condition=cond)
            print(
                f"  [{args.mode.upper():<7} | {cond:<14}] "
                f"Acc: {res['accuracy']*100:.2f}% | "
                f"Macro-F1: {res['macro_f1']*100:.2f}% | "
                f"AUC: {res['roc_auc']:.4f} | "
                f"ECE: {res['ece']:.4f}"
            )
            for cat, acc in res.get("per_category_accuracy", {}).items():
                print(f"    - {cat:<24}: {acc*100:.2f}%")
        return

    ckpt_map = {
        "audio": os.path.join(args.checkpoint_dir, f"{args.prefix}audio_best.pth"),
        "video": os.path.join(args.checkpoint_dir, f"{args.prefix}video_best.pth"),
        "fusion": os.path.join(args.checkpoint_dir, f"{args.prefix}fusion_best.pth"),
        "rmsav": os.path.join(args.checkpoint_dir, f"{args.prefix}rmsav_best.pth"),
    }
    evaluator.run_benchmark(ckpt_map)


if __name__ == "__main__":
    main()
