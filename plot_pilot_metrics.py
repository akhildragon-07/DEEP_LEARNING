import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "results", ".matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

df_results = pd.read_csv("results/pilot_comparison.csv")
modes = ["audio", "video", "fusion", "rmsav"]

plot_path = "results/pilot_metrics.png"
plt.figure(figsize=(15, 6))

# Subplot 1: Clean performance across models
plt.subplot(1, 2, 1)
df_clean = df_results[df_results["condition"] == "none"].set_index("mode").reindex(modes).reset_index()
x = np.arange(len(modes))
width = 0.25

plt.bar(x - width, df_clean["accuracy"] * 100, width, label="Accuracy (%)", color="#4C72B0")
plt.bar(x, df_clean["balanced_accuracy"] * 100, width, label="Balanced Acc (%)", color="#55A868")
plt.bar(x + width, df_clean["auc"] * 100, width, label="ROC-AUC (%)", color="#C44E52")

plt.xlabel("Model Mode", fontsize=11, fontweight="bold")
plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
plt.title("Clean Test Performance (Pilot Dataset, N=8)", fontsize=12, fontweight="bold")
plt.xticks(x, [m.upper() for m in modes])
plt.ylim(0, 105)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

# Subplot 2: Modality Drop Robustness Comparison (Balanced Accuracy)
plt.subplot(1, 2, 2)
conditions = ["none", "missing_video", "missing_audio"]
cond_labels = ["Clean (AV)", "Missing Video", "Missing Audio"]
colors = ["#4C72B0", "#E1812C", "#55A868"]

for i, cond in enumerate(conditions):
    cond_df = df_results[df_results["condition"] == cond].set_index("mode").reindex(modes)
    vals = cond_df["balanced_accuracy"].fillna(0).values * 100
    offset = (i - 1) * width
    plt.bar(x + offset, vals, width, label=cond_labels[i], color=colors[i])

plt.xlabel("Model Mode", fontsize=11, fontweight="bold")
plt.ylabel("Balanced Accuracy (%)", fontsize=11, fontweight="bold")
plt.title("Modality Drop Robustness (Balanced Accuracy)", fontsize=12, fontweight="bold")
plt.xticks(x, [m.upper() for m in modes])
plt.ylim(0, 105)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Successfully generated {plot_path}")
