r"""
Runner script for pilot experiments on the 40-sample dataset (teammates/zip2).
Executes:
1. Training: audio, video, fusion, rmsav (3 epochs each)
2. Evaluation across clean test set, missing_video, missing_audio
3. Metric aggregation & visualization
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("PilotRunner")

MODES = [
    {"mode": "audio", "lr": 1e-3, "aux_weight": 0.0},
    {"mode": "video", "lr": 1e-4, "aux_weight": 0.0},
    {"mode": "fusion", "lr": 1e-4, "aux_weight": 0.0},
    {"mode": "rmsav", "lr": 1e-4, "aux_weight": 0.2},
]

STRESS_CONDITIONS = ["none", "missing_video", "missing_audio"]


def run_command(cmd_list):
    logger.info(f"Running: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Command failed with code {res.returncode}:\n{res.stderr}\n{res.stdout}")
        raise RuntimeError(f"Command failed: {' '.join(cmd_list)}")
    print(res.stdout)
    return res.stdout


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results/eval", exist_ok=True)

    # 1. Train all modes
    for m_cfg in MODES:
        mode = m_cfg["mode"]
        logger.info(f"=== Starting Training for Mode: {mode} ===")
        cmd = [
            sys.executable,
            "train.py",
            "--mode", mode,
            "--manifest_dir", "splits_pilot",
            "--data_dir", "teammates/zip2",
            "--epochs", "3",
            "--batch_size", "4",
            "--lr", str(m_cfg["lr"]),
            "--aux_weight", str(m_cfg["aux_weight"]),
            "--seed", "42",
            "--checkpoint_prefix", "pilot_",
            "--save_dir", "checkpoints",
            "--log_dir", "results",
        ]
        run_command(cmd)

    # 2. Evaluate all modes under all stress conditions
    all_eval_records = []
    for m_cfg in MODES:
        mode = m_cfg["mode"]
        ckpt_path = os.path.join("checkpoints", f"pilot_{mode}_best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join("checkpoints", f"pilot_{mode}_latest.pth")

        for cond in STRESS_CONDITIONS:
            logger.info(f"=== Evaluating Mode: {mode} | Condition: {cond} ===")
            cmd = [
                sys.executable,
                "evaluate.py",
                "--checkpoint", ckpt_path,
                "--test_manifest", "splits_pilot/test_manifest.csv",
                "--data_dir", "teammates/zip2",
                "--batch_size", "4",
                "--stress_test", cond,
                "--output_dir", "results/eval",
            ]
            run_command(cmd)

            # Load the exported json
            base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            json_file = os.path.join("results/eval", f"{base_name}_{cond}_metrics.json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    metrics = json.load(f)
                metrics["mode"] = mode
                metrics["condition"] = cond
                all_eval_records.append(metrics)

    # 3. Create aggregated summary DataFrame and JSON
    df_results = pd.DataFrame(all_eval_records)
    summary_csv = "results/pilot_comparison.csv"
    summary_json = "results/pilot_results.json"
    df_results.to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(all_eval_records, f, indent=2)

    logger.info(f"Aggregated results saved to {summary_csv} and {summary_json}")
    print("\n" + "=" * 90)
    print("PILOT EVALUATION SUMMARY ACROSS ALL MODES AND CONDITIONS:")
    print("=" * 90)
    display_cols = ["mode", "condition", "accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "avg_gate_v", "avg_gate_a"]
    available_cols = [c for c in display_cols if c in df_results.columns]
    print(df_results[available_cols].to_string(index=False))

    # 4. Generate comparison plots
    plot_path = "results/pilot_metrics.png"
    plt.figure(figsize=(14, 6))

    # Subplot 1: Clean performance across models
    plt.subplot(1, 2, 1)
    df_clean = df_results[df_results["condition"] == "none"]
    modes = df_clean["mode"].tolist()
    x = np.arange(len(modes))
    width = 0.25

    plt.bar(x - width, df_clean["accuracy"] * 100, width, label="Accuracy (%)", color="#4C72B0")
    plt.bar(x, df_clean["balanced_accuracy"] * 100, width, label="Balanced Acc (%)", color="#55A868")
    plt.bar(x + width, df_clean["roc_auc"] * 100, width, label="ROC-AUC (%)", color="#C44E52")

    plt.xlabel("Model Mode", fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Clean Test Performance (Pilot Dataset, N=8)", fontsize=12, fontweight="bold")
    plt.xticks(x, [m.upper() for m in modes])
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    # Subplot 2: Robustness Under Modality Drop
    plt.subplot(1, 2, 2)
    # Compare Balanced Accuracy across conditions for each mode
    conditions = ["none", "missing_video", "missing_audio"]
    cond_labels = ["Clean (AV)", "Missing Video", "Missing Audio"]
    x = np.arange(len(modes))
    width = 0.25

    for i, cond in enumerate(conditions):
        cond_df = df_results[df_results["condition"] == cond].set_index("mode").reindex(modes)
        vals = cond_df["balanced_accuracy"].fillna(0).values * 100
        offset = (i - 1) * width
        plt.bar(x + offset, vals, width, label=cond_labels[i])

    plt.xlabel("Model Mode", fontsize=11, fontweight="bold")
    plt.ylabel("Balanced Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Modality Drop Robustness Comparison", fontsize=12, fontweight="bold")
    plt.xticks(x, [m.upper() for m in modes])
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Visual metrics chart saved to {plot_path}")


if __name__ == "__main__":
    main()
