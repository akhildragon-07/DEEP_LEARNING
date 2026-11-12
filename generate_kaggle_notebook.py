"""
Script to generate the complete Kaggle Training Notebook:
RMS_AV_Kaggle_Training.ipynb
"""

import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🛡️ RMS-AV: Kaggle Full-Scale Deepfake Detection Training Pipeline\n",
            "### Fail-Safe & Auto-Resuming Multimodal Training on the LAV-DF Benchmark (136,304 Media Samples)\n",
            "\n",
            "> ⚠️ **CRITICAL: READ BEFORE CLOSING YOUR BROWSER ON KAGGLE!**\n",
            "> If you click the interactive ▶ **Play** button and close your browser, Kaggle terminates idle interactive VMs and wipes `/kaggle/working/` UNLESS you do these 2 things:\n",
            "> 1. **Enable Persistence (Right Sidebar):** Under **Notebook options $\\to$ Persistence**, select **`Files only`** (never leave it on *No persistence*!).\n",
            "> 2. **Run in Background (`Save & Run All`):** To train while your laptop/browser is closed, click **Save Version** (top-right) $\\to$ select **Save & Run All (Commit)** $\\to$ **Save**. Kaggle will train on cloud servers for up to **12 hours** in the background and permanently save all checkpoints to the **Output** tab!\n",
            "\n",
            "**Fail-Safe Features Built Into This Pipeline:**\n",
            "- 💾 **Mid-Epoch Step Checkpointing:** Saves interim weights every **250 steps (~5-8 mins)** to `/kaggle/working/checkpoints_lavdf/lavdf_rmsav_latest.pth`.\n",
            "- 🔄 **Exact Mid-Epoch Step Resume:** If interrupted at e.g. Batch 1500/4919, resuming fast-forwards and starts directly from Batch 1501 (not 0%).\n",
            "- ⏱️ **10-Hour Clean Time Budget (`--max_hours 10.0`):** Gracefully stops & saves before Kaggle's 12-hour limit so your Commit always finishes with **Succeeded** status.\n",
            "- 📂 **Multi-Version Auto-Resume:** Automatically scans `/kaggle/input/` if you attach a previous Commit's output to continue training."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Verify Kaggle GPU Environment (Tesla T4 / P100)"]
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
            "print(f\"PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"Active GPU: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n",
            "else:\n",
            "    print(\"[!] Warning: No GPU detected. In Kaggle Notebook Settings (right sidebar), set Accelerator to 'GPU T4 x2' or 'GPU P100'.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Clone Latest Research Repository & Install Requirements"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "%cd /kaggle/working\n",
            "REPO_URL = \"https://github.com/akhildragon-07/DEEP_LEARNING.git\"\n",
            "REPO_DIR = \"/kaggle/working/DEEP_LEARNING\"\n",
            "\n",
            "if not os.path.exists(REPO_DIR):\n",
            "    !git clone $REPO_URL\n",
            "else:\n",
            "    %cd $REPO_DIR\n",
            "    !git pull origin main\n",
            "\n",
            "%cd $REPO_DIR\n",
            "!pip install -q -r requirements.txt\n",
            "print(\"Repository synchronized and ready at:\", os.getcwd())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Verify & Locate Official LAV-DF Dataset on Kaggle"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "\n",
            "# Check /kaggle/input for attached LAV-DF dataset\n",
            "meta_found = None\n",
            "media_root = None\n",
            "\n",
            "if os.path.exists(\"/kaggle/input\"):\n",
            "    for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
            "        for target in [\"metadata.min.json\", \"metadata.json\"]:\n",
            "            if target in files:\n",
            "                meta_found = os.path.join(root, target)\n",
            "                media_root = root\n",
            "                break\n",
            "        if meta_found:\n",
            "            break\n",
            "\n",
            "if meta_found:\n",
            "    print(f\"[SUCCESS] LAV-DF Metadata located: {meta_found}\")\n",
            "    print(f\"[SUCCESS] LAV-DF Media root located: {media_root}\")\n",
            "else:\n",
            "    print(\"[!] Dataset not found in /kaggle/input. Downloading via kagglehub...\")\n",
            "    !pip install -q kagglehub\n",
            "    import kagglehub\n",
            "    path = kagglehub.dataset_download(\"elin75/localized-audio-visual-deepfake-dataset-lav-df\")\n",
            "    print(\"Downloaded to:\", path)\n",
            "    for root, dirs, files in os.walk(path):\n",
            "        for target in [\"metadata.min.json\", \"metadata.json\"]:\n",
            "            if target in files:\n",
            "                meta_found = os.path.join(root, target)\n",
            "                media_root = root\n",
            "                break\n",
            "        if meta_found:\n",
            "            break\n",
            "    print(f\"Metadata located: {meta_found}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Run Architectural Smoke Test (10/10 Checks)\n",
            "*Validates Temporal Saliency Aggregator (TSA), CM-FCI, Dynamic Reliability Gating, and Backprop on 122 parameter tensors.*"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%cd /kaggle/working/DEEP_LEARNING\n",
            "!python -m lavdf_pipeline.smoke_test"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Launch Training with Fail-Safe Auto-Resume & Periodic Step Checkpointing\n",
            "\n",
            "**How Crash-Proofing Works:**\n",
            "1. **Every 500 steps (~15 min):** An interim checkpoint is saved to `/kaggle/working/checkpoints_lavdf/lavdf_rmsav_latest.pth`.\n",
            "2. **If an error or timeout happens:** An emergency crash checkpoint (`lavdf_rmsav_emergency_crash.pth`) is automatically captured.\n",
            "3. **Auto-Resume:** Setting `--resume` automatically detects previous checkpoints (including from previous Kaggle notebook versions attached as input) and resumes without starting from scratch!\n",
            "\n",
            "> **Tip for 100% Autonomous Training:** Click **Save Version** (top-right) -> Select **Save & Run All (Commit)** -> Kaggle will train in the background for up to 9-12 hours even if you close your computer."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Training Hyperparameters\n",
            "EPOCHS = 10\n",
            "BATCH_SIZE = 16\n",
            "GRAD_ACCUM = 2              # Effective batch size = 32\n",
            "LEARNING_RATE = 1e-4\n",
            "SAVE_INTERVAL_STEPS = 250   # Fail-safe save every 250 batches (~5-8 mins)\n",
            "MAX_HOURS = 10.0            # Stops cleanly before Kaggle 12h timeout so Outputs ALWAYS save\n",
            "LIMIT_SAMPLES = None        # Set to None for full 78,703 clips; or e.g. 5000 for rapid test\n",
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
            "    f\"--save_interval_steps {SAVE_INTERVAL_STEPS}\",\n",
            "    f\"--max_hours {MAX_HOURS}\",\n",
            "    \"--save_dir /kaggle/working/checkpoints_lavdf\",\n",
            "    \"--log_dir /kaggle/working/results_lavdf\",\n",
            "    \"--resume\",\n",
            "]\n",
            "\n",
            "if meta_found:\n",
            "    cmd.append(f'--metadata_path \"{meta_found}\"')\n",
            "    cmd.append(f'--media_root \"{media_root}\"')\n",
            "\n",
            "if LIMIT_SAMPLES:\n",
            "    cmd.append(f\"--limit_samples {LIMIT_SAMPLES}\")\n",
            "\n",
            "full_cmd = \" \".join(cmd)\n",
            "print(\"Running Command:\", full_cmd)\n",
            "!{full_cmd}"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Full Held-Out Evaluation & Modality-Drop Stress Testing\n",
            "*Evaluates the best checkpoint on the official 26,100 held-out test split under Clean, Missing-Video, and Missing-Audio conditions.*"
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
            "    --checkpoint /kaggle/working/checkpoints_lavdf/lavdf_rmsav_best.pth \\\n",
            "    --stress_test \\\n",
            "    --save_dir /kaggle/working/results_lavdf"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 7. Generate Publication Figures (ROC Curves, Gate Distribution, Stress Retention)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python generate_paper_artifacts.py \\\n",
            "    --results_csv /kaggle/working/results_lavdf/lavdf_benchmark_comparison.csv \\\n",
            "    --output_dir /kaggle/working/paper/artifacts\n",
            "\n",
            "from IPython.display import Image, display\n",
            "import glob\n",
            "for fig in sorted(glob.glob(\"/kaggle/working/paper/artifacts/*.png\") + glob.glob(\"paper/figures/*.png\")):\n",
            "    print(f\"Displaying: {fig}\")\n",
            "    display(Image(filename=fig, width=650))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Package Checkpoints & Results for 1-Click Download\n",
            "*Produces a clean, compact zip file in `/kaggle/working/RMSAV_Trained_Models_and_Results.zip`.*"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python prepare_deployable_checkpoints.py\n",
            "!zip -r /kaggle/working/RMSAV_Trained_Models_and_Results.zip \\\n",
            "    /kaggle/working/checkpoints_lavdf \\\n",
            "    /kaggle/working/results_lavdf \\\n",
            "    /kaggle/working/paper/artifacts\n",
            "print(\"\\n[DONE] Package ready for 1-click download at: /kaggle/working/RMSAV_Trained_Models_and_Results.zip\")"
        ]
    }
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"}
    },
    "cells": cells
}

with open("RMS_AV_Kaggle_Training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("RMS_AV_Kaggle_Training.ipynb generated successfully!")
