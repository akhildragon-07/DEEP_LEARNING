"""
Script to generate the complete production-grade Colab / Kaggle Training Notebook:
RMS_AV_Full_Scale_Training.ipynb
"""

import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 RMS-AV: Complete Research Training & Publication Pipeline\n",
            "### Reliability- and Manipulation-Aware Selective Audio-Visual Fusion with Cross-Modal Contrastive Incoherence (CM-FCI) & Temporal Saliency Aggregation (TSA)\n",
            "\n",
            "**Benchmark Dataset:** LAV-DF (Localized Audio-Visual Deepfake - 136,304 media samples)\n",
            "\n",
            "**Paper Target:** IEEE Conference / Journal Submission\n",
            "\n",
            "**Authors:** Akhil Rajan P, P T Sahasra, Aboli, Famie (`akhilvit28@gmail.com`)\n",
            "\n",
            "---\n",
            "### 📋 What This Notebook Does:\n",
            "1. **Environment & GPU Verification:** Detects Tesla T4/P100/A100 and configures Automatic Mixed Precision (AMP).\n",
            "2. **Dataset Setup:** Downloads/attaches official LAV-DF dataset (136,304 samples).\n",
            "3. **Codebase Synchronization:** Pulls the complete `DEEP_LEARNING` repository containing RMS-AV 2.0.\n",
            "4. **Automated Smoke Test:** Runs 10/10 automated mathematical and architectural integrity checks.\n",
            "5. **Complete Training:** Trains RMS-AV with Cosine Annealing, Gradient Accumulation, and Automatic Google Drive Checkpoint Sync.\n",
            "6. **Full Held-Out Evaluation:** Evaluates on the official 26,100 test split under Clean, Missing-Video, and Missing-Audio conditions.\n",
            "7. **Publication Artifacts:** Generates 300 DPI figures and LaTeX tables ready for Overleaf."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. System & GPU Health Check"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!nvidia-smi\n",
            "\n",
            "import torch\n",
            "print(f\"PyTorch Version: {torch.__version__}\")\n",
            "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"Active GPU: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n",
            "else:\n",
            "    print(\"[!] Warning: No GPU detected. Please go to Runtime -> Change runtime type -> Select T4 GPU.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Mount Google Drive for Persistent Auto-Checkpointing\n",
            "*Allows training to save checkpoints directly to Google Drive after every epoch so progress is never lost if session disconnects.*"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    DRIVE_BACKUP_DIR = '/content/drive/MyDrive/RMS_AV_Checkpoints'\n",
            "    os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)\n",
            "    print(f\"[Drive Mounted] Checkpoints will auto-sync to: {DRIVE_BACKUP_DIR}\")\n",
            "except Exception as e:\n",
            "    print(\"Running on Kaggle or local environment (Google Drive skipped).\")\n",
            "    DRIVE_BACKUP_DIR = None"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Clone / Update Repository"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "\n",
            "REPO_URL = \"https://github.com/akhildragon-07/DEEP_LEARNING.git\"\n",
            "REPO_DIR = \"DEEP_LEARNING\"\n",
            "\n",
            "if not os.path.exists(REPO_DIR):\n",
            "    !git clone $REPO_URL\n",
            "else:\n",
            "    %cd $REPO_DIR\n",
            "    !git pull origin main\n",
            "    %cd ..\n",
            "\n",
            "%cd $REPO_DIR\n",
            "!pip install -q -r requirements.txt\n",
            "print(\"Repository synchronized and dependencies verified!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Download & Locate Official LAV-DF Dataset (136,304 clips)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q kagglehub\n",
            "import kagglehub\n",
            "import os\n",
            "\n",
            "# Downloads/locates the official LAV-DF benchmark dataset\n",
            "DATASET_PATH = kagglehub.dataset_download(\"elin75/localized-audio-visual-deepfake-dataset-lav-df\")\n",
            "print(f\"LAV-DF Dataset root path: {DATASET_PATH}\")\n",
            "\n",
            "# Verify metadata.min.json exists\n",
            "meta_found = None\n",
            "for root, dirs, files in os.walk(DATASET_PATH):\n",
            "    for target in [\"metadata.min.json\", \"metadata.json\"]:\n",
            "        if target in files:\n",
            "            meta_found = os.path.join(root, target)\n",
            "            media_root = root\n",
            "            break\n",
            "    if meta_found:\n",
            "        break\n",
            "\n",
            "print(f\"Metadata located at: {meta_found}\")\n",
            "print(f\"Media root located at: {media_root}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Execute Architectural Smoke Test (10/10 Checks)\n",
            "*Verifies Temporal Saliency Aggregator (TSA), Cross-Modal Forensic Contrastive Incoherence (CM-FCI), and Validity Masking.*"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python -m lavdf_pipeline.smoke_test"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Launch Complete RMS-AV Training Pipeline\n",
            "Select your training scale:\n",
            "- **Full Dataset Scale:** `LIMIT_SAMPLES = None` (All 78,703 training clips for maximum publication strength)\n",
            "- **Accelerated Scale:** `LIMIT_SAMPLES = 5000` or `10000` (Fast convergence for rapid experimentation)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Configuration\n",
            "EPOCHS = 10\n",
            "BATCH_SIZE = 16\n",
            "GRAD_ACCUM = 2          # Effective batch size = 32\n",
            "LEARNING_RATE = 1e-4\n",
            "LIMIT_SAMPLES = None    # Set to None for full dataset, or e.g. 5000 for rapid run\n",
            "\n",
            "cmd = [\n",
            "    \"python -m lavdf_pipeline.train\",\n",
            "    \"--mode rmsav\",\n",
            "    f\"--epochs {EPOCHS}\",\n",
            "    f\"--batch_size {BATCH_SIZE}\",\n",
            "    f\"--grad_accum_steps {GRAD_ACCUM}\",\n",
            "    f\"--lr {LEARNING_RATE}\",\n",
            "    \"--aux_weight 0.20\",\n",
            "    \"--fci_weight 0.15\",\n",
            "    \"--save_dir checkpoints_lavdf\",\n",
            "    \"--log_dir results_lavdf\",\n",
            "]\n",
            "\n",
            "if meta_found:\n",
            "    cmd.append(f'--metadata_path \"{meta_found}\"')\n",
            "    cmd.append(f'--media_root \"{media_root}\"')\n",
            "\n",
            "if DRIVE_BACKUP_DIR:\n",
            "    cmd.append(f'--drive_backup \"{DRIVE_BACKUP_DIR}\"')\n",
            "\n",
            "if LIMIT_SAMPLES:\n",
            "    cmd.append(f'--limit_samples {LIMIT_SAMPLES}')\n",
            "\n",
            "full_cmd = \" \".join(cmd)\n",
            "print(\"Executing:\", full_cmd)\n",
            "!{full_cmd}"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Evaluate on Held-Out Test Split (Clean & Stress Testing)\n",
            "*Runs comprehensive evaluation across Clean, Missing-Video, and Missing-Audio scenarios with Per-Category forensic breakdown.*"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python -m lavdf_pipeline.evaluate \\\n",
            "    --mode rmsav \\\n",
            "    --checkpoint checkpoints_lavdf/lavdf_rmsav_best.pth \\\n",
            "    --stress_test \\\n",
            "    --save_dir results_lavdf"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Generate Publication Figures & LaTeX Tables\n",
            "*Outputs 300 DPI figures and ready-to-insert LaTeX tables directly into `paper/artifacts/`.*"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python generate_paper_artifacts.py \\\n",
            "    --results_csv results_lavdf/lavdf_benchmark_comparison.csv \\\n",
            "    --output_dir paper/artifacts\n",
            "\n",
            "# Display generated figures in notebook\n",
            "from IPython.display import Image, display\n",
            "import glob\n",
            "\n",
            "for p in glob.glob(\"paper/artifacts/*.png\") + glob.glob(\"paper/figures/*.png\"):\n",
            "    print(f\"\\nFigure: {p}\")\n",
            "    display(Image(filename=p, width=650))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Model Packaging for Instant Demonstration\n",
            "*Optimizes checkpoints by stripping optimizer states so they can be pushed or transferred anywhere (< 50MB) for 1-click live demos.*"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python prepare_deployable_checkpoints.py\n",
            "print(\"Deployable compact checkpoints ready in checkpoints/\")"
        ]
    }
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

with open("RMS_AV_Full_Scale_Training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("RMS_AV_Full_Scale_Training.ipynb generated successfully!")
