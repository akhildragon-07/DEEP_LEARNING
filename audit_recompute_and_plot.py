r"""
Audit and Metric Recomputation Script for RMS-AV Pilot Experiment
-----------------------------------------------------------------
Strictly recalculates all metrics from actual saved predictions and checkpoints.
Ensures macro-F1 != positive-class F1, correct confusion matrix arithmetic,
and generates verified publication plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "results", ".matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs("results", exist_ok=True)

MODES = ["audio", "video", "fusion", "rmsav"]
CONDITIONS = ["none", "missing_video", "missing_audio"]

CHECKPOINT_MAP = {
    "audio": "checkpoints/pilot_audio_best.pth",
    "video": "checkpoints/pilot_video_best.pth",
    "fusion": "checkpoints/pilot_fusion_best.pth",
    "rmsav": "checkpoints/pilot_rmsav_best.pth",
}


def compute_ece_prob(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Class-wise probability calibration error (|p - y| binned)."""
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
            avg_prob_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_prob_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_ece_conf(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Standard binary ECE based on confidence max(p, 1-p) and classification correctness."""
    preds = (probs >= 0.5).astype(int)
    correct = (preds == labels).astype(float)
    confs = np.maximum(probs, 1.0 - probs)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs >= bin_lower) & (confs < bin_upper) if i < n_bins - 1 else (confs >= bin_lower) & (confs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            bin_acc = np.mean(correct[in_bin])
            bin_conf = np.mean(confs[in_bin])
            ece += np.abs(bin_conf - bin_acc) * prop_in_bin
    return float(ece)


def main():
    test_manifest_path = "splits_pilot/test_manifest.csv"
    test_df = pd.read_csv(test_manifest_path)
    test_ids = test_df["video_id"].tolist()
    gt_labels = test_df["label"].astype(int).tolist()

    all_predictions_rows = []
    all_metrics_records = []

    # 1. Ingest predictions and compute exact metrics
    for mode in MODES:
        ckpt_name = CHECKPOINT_MAP[mode]

        for cond in CONDITIONS:
            preds_file = f"results/eval/pilot_{mode}_best_{cond}_preds.csv"
            if not os.path.exists(preds_file):
                raise FileNotFoundError(f"Missing predictions file: {preds_file}")

            df_pred = pd.read_csv(preds_file)
            assert len(df_pred) == len(test_ids), f"Mismatch in sample count for {mode} {cond}"

            # Ensure ground truth matches test manifest exactly
            assert (df_pred["target"].astype(int).values == np.array(gt_labels)).all(), (
                f"Target labels in {preds_file} do not match test manifest!"
            )

            targets = df_pred["target"].astype(int).values
            probs = df_pred["probability"].astype(float).values
            preds = df_pred["prediction"].astype(int).values

            gate_a = df_pred["gate_audio"].values if "gate_audio" in df_pred.columns else [np.nan] * len(targets)
            gate_v = df_pred["gate_video"].values if "gate_video" in df_pred.columns else [np.nan] * len(targets)

            # Record per-sample predictions
            for idx in range(len(targets)):
                all_predictions_rows.append({
                    "sample_id": test_ids[idx],
                    "ground_truth": int(targets[idx]),
                    "predicted_label": int(preds[idx]),
                    "predicted_probability": float(probs[idx]),
                    "model_mode": mode,
                    "checkpoint_used": ckpt_name,
                    "condition": cond,
                    "gate_audio": float(gate_a[idx]) if not np.isnan(gate_a[idx]) else None,
                    "gate_video": float(gate_v[idx]) if not np.isnan(gate_v[idx]) else None,
                })

            # Calculate Confusion Matrix
            # Classes: [0 (Real), 1 (Fake)]
            cm = confusion_matrix(targets, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            total = int(tn + fp + fn + tp)
            assert total == len(targets), "Confusion matrix total does not equal sample count!"

            acc = float(accuracy_score(targets, preds))
            bal_acc = float(balanced_accuracy_score(targets, preds))

            # Precision and Recall for Positive Class (1 = Fake)
            prec_pos = float(precision_score(targets, preds, pos_label=1, zero_division=0))
            rec_pos = float(recall_score(targets, preds, pos_label=1, zero_division=0))
            f1_pos = float(f1_score(targets, preds, pos_label=1, zero_division=0))

            # Precision and Recall for Negative Class (0 = Real)
            prec_neg = float(precision_score(targets, preds, pos_label=0, zero_division=0))
            rec_neg = float(recall_score(targets, preds, pos_label=0, zero_division=0))
            f1_neg = float(f1_score(targets, preds, pos_label=0, zero_division=0))

            # True Macro-F1 (unweighted average of class 0 and class 1 F1)
            f1_macro = float(f1_score(targets, preds, average="macro", zero_division=0))
            # Verify macro-F1 equals average of f1_pos and f1_neg
            expected_macro = 0.5 * (f1_pos + f1_neg)
            assert abs(f1_macro - expected_macro) < 1e-6, f"Macro F1 calculation mismatch: {f1_macro} vs {expected_macro}"

            # ROC-AUC (using continuous positive class probabilities)
            if len(set(targets)) > 1 and len(set(probs)) > 1:
                auc_val = float(roc_auc_score(targets, probs))
            elif len(set(probs)) == 1:
                # Constant predictions -> uninformative ranking AUC = 0.5
                auc_val = 0.5
            else:
                auc_val = float("nan")

            # Average Precision (PR-AUC)
            if len(set(targets)) > 1 and len(set(probs)) > 1:
                ap_val = float(average_precision_score(targets, probs))
            else:
                ap_val = float(np.mean(targets))

            fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr_val = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

            ece_p = compute_ece_prob(probs, targets, n_bins=10)
            ece_c = compute_ece_conf(probs, targets, n_bins=10)

            avg_g_a = float(np.nanmean(gate_a)) if not np.isnan(gate_a).all() else None
            avg_g_v = float(np.nanmean(gate_v)) if not np.isnan(gate_v).all() else None

            all_metrics_records.append({
                "mode": mode,
                "condition": cond,
                "checkpoint": ckpt_name,
                "total_samples": total,
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "precision_positive": prec_pos,
                "recall_positive": rec_pos,
                "f1_positive": f1_pos,
                "precision_negative": prec_neg,
                "recall_negative": rec_neg,
                "f1_negative": f1_neg,
                "f1_macro": f1_macro,
                "roc_auc": auc_val,
                "pr_auc": ap_val,
                "fpr": fpr_val,
                "fnr": fnr_val,
                "ece_prob": ece_p,
                "ece_conf": ece_c,
                "avg_gate_audio": avg_g_a,
                "avg_gate_video": avg_g_v,
            })

    # 2. Save pilot_predictions.csv
    df_all_preds = pd.DataFrame(all_predictions_rows)
    pred_path = "results/pilot_predictions.csv"
    df_all_preds.to_csv(pred_path, index=False)
    print(f"[OK] Saved per-sample predictions to: {pred_path} ({len(df_all_preds)} records)")

    # 3. Save pilot_comparison.csv and pilot_results.json
    df_metrics = pd.DataFrame(all_metrics_records)
    comp_path = "results/pilot_comparison.csv"
    json_path = "results/pilot_results.json"
    df_metrics.to_csv(comp_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_metrics_records, f, indent=2)
    print(f"[OK] Saved corrected comparison table to: {comp_path}")
    print(f"[OK] Saved corrected results JSON to: {json_path}")

    # Display clean condition summary
    print("\n" + "=" * 105)
    print("VERIFIED CLEAN TEST EVALUATION METRICS (CONDITION: 'none', N=8):")
    print("=" * 105)
    clean_cols = [
        "mode", "accuracy", "balanced_accuracy", "precision_positive",
        "recall_positive", "f1_positive", "f1_negative", "f1_macro", "roc_auc", "ece_conf"
    ]
    df_clean = df_metrics[df_metrics["condition"] == "none"][clean_cols]
    print(df_clean.to_string(index=False))

    # 4. Generate Figures
    generate_figures(df_metrics, df_all_preds)


def generate_figures(df_metrics: pd.DataFrame, df_all_preds: pd.DataFrame):
    df_clean_preds = df_all_preds[df_all_preds["condition"] == "none"]
    df_clean_metrics = df_metrics[df_metrics["condition"] == "none"].set_index("mode")

    # Figure 1: Corrected Pilot Metrics Bar Chart
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    modes = MODES
    x = np.arange(len(modes))
    width = 0.20

    accs = [df_clean_metrics.loc[m, "accuracy"] * 100 for m in modes]
    bal_accs = [df_clean_metrics.loc[m, "balanced_accuracy"] * 100 for m in modes]
    f1_macros = [df_clean_metrics.loc[m, "f1_macro"] * 100 for m in modes]
    aucs = [df_clean_metrics.loc[m, "roc_auc"] * 100 for m in modes]

    plt.bar(x - 1.5 * width, accs, width, label="Accuracy (%)", color="#4C72B0")
    plt.bar(x - 0.5 * width, bal_accs, width, label="Balanced Acc (%)", color="#55A868")
    plt.bar(x + 0.5 * width, f1_macros, width, label="Macro-F1 (%)", color="#DD8452")
    plt.bar(x + 1.5 * width, aucs, width, label="ROC-AUC (%)", color="#C44E52")

    plt.xlabel("Model Mode", fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Corrected Test Metrics (Clean Condition, N=8)", fontsize=12, fontweight="bold")
    plt.xticks(x, [m.upper() for m in modes])
    plt.ylim(0, 110)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)

    plt.subplot(1, 2, 2)
    # Modality Drop Robustness Comparison (Balanced Accuracy)
    conditions = ["none", "missing_video", "missing_audio"]
    cond_labels = ["Clean (AV)", "Missing Video", "Missing Audio"]
    colors = ["#4C72B0", "#E1812C", "#55A868"]

    width_cond = 0.25
    for i, cond in enumerate(conditions):
        cond_df = df_metrics[df_metrics["condition"] == cond].set_index("mode").reindex(modes)
        vals = cond_df["balanced_accuracy"].values * 100
        offset = (i - 1) * width_cond
        plt.bar(x + offset, vals, width_cond, label=cond_labels[i], color=colors[i])

    plt.xlabel("Model Mode", fontsize=11, fontweight="bold")
    plt.ylabel("Balanced Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Modality Drop Robustness (Balanced Accuracy)", fontsize=12, fontweight="bold")
    plt.xticks(x, [m.upper() for m in modes])
    plt.ylim(0, 110)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("results/pilot_metrics.png", dpi=300)
    plt.close()
    print("[OK] Generated results/pilot_metrics.png")

    # Figure 2: Confusion Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    mode_names = {"audio": "Audio-Only Baseline", "video": "Video-Only Baseline", "fusion": "Standard AV Fusion", "rmsav": "Proposed RMS-AV"}

    for idx, mode in enumerate(MODES):
        ax = axes[idx // 2, idx % 2]
        m_row = df_clean_metrics.loc[mode]
        cm = np.array([
            [int(m_row["true_negatives"]), int(m_row["false_positives"])],
            [int(m_row["false_negatives"]), int(m_row["true_positives"])],
        ])
        cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > 3 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=18, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real (0)", "Fake (1)"], fontsize=10)
        ax.set_yticklabels(["Real (0)", "Fake (1)"], fontsize=10)
        ax.set_title(f"{mode_names[mode]}\n(Acc: {m_row['accuracy']*100:.1f}%, Bal-Acc: {m_row['balanced_accuracy']*100:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Ground-Truth", fontsize=10)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=300)
    plt.close()
    print("[OK] Generated results/confusion_matrices.png")

    # Figure 3: ROC Curves
    plt.figure(figsize=(8, 7))
    palette = {"audio": "#4C72B0", "video": "#55A868", "fusion": "#DD8452", "rmsav": "#C44E52"}

    for mode in MODES:
        sub = df_clean_preds[df_clean_preds["model_mode"] == mode]
        y_true = sub["ground_truth"].values
        y_score = sub["predicted_probability"].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = df_clean_metrics.loc[mode, "roc_auc"]
        plt.step(fpr, tpr, where="post", label=f"{mode_names[mode]} (AUC = {auc_val:.3f})", color=palette[mode], lw=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Line (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (TPR)", fontsize=11, fontweight="bold")
    plt.title("ROC Curves Comparison (Pilot Test Set, N=8)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig("results/roc_curves.png", dpi=300)
    plt.close()
    print("[OK] Generated results/roc_curves.png")

    # Figure 4: Precision-Recall Curves
    plt.figure(figsize=(8, 7))
    for mode in MODES:
        sub = df_clean_preds[df_clean_preds["model_mode"] == mode]
        y_true = sub["ground_truth"].values
        y_score = sub["predicted_probability"].values
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap_val = df_clean_metrics.loc[mode, "pr_auc"]
        plt.step(rec, prec, where="post", label=f"{mode_names[mode]} (AP = {ap_val:.3f})", color=palette[mode], lw=2)

    prevalence = np.mean(df_clean_preds[df_clean_preds["model_mode"] == "rmsav"]["ground_truth"].values)
    plt.axhline(prevalence, color="k", linestyle="--", alpha=0.6, label=f"Prevalence Baseline ({prevalence*100:.1f}%)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curves Comparison (Fake Class, N=8)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig("results/precision_recall_curves.png", dpi=300)
    plt.close()
    print("[OK] Generated results/precision_recall_curves.png")


if __name__ == "__main__":
    main()
