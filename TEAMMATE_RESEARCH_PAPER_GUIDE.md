# Teammate Handover Guide: Completing the RMS-AV Research Paper

**Project**: RMS-AV — Reliability- and Manipulation-Aware Selective Audio-Visual Fusion  
**Repository**: [github.com/akhildragon-07/DEEP_LEARNING](https://github.com/akhildragon-07/DEEP_LEARNING)  
**Dataset**: LAV-DF (Localized Audio-Visual Deepfake, 136,304 clips)  
**Target Paper Template**: IEEE Conference Double-Column ([`paper/manuscript.tex`](paper/manuscript.tex))

---

## Overview for Teammates

This guide provides clear, chronological instructions to take our trained deep learning models from Google Colab, evaluate them on the 26,100 held-out test clips, auto-generate the publication tables and figures, and finalize our IEEE research paper manuscript for conference or journal submission.

```
                      WORKFLOW AT A GLANCE
                      
┌───────────────────────┐      ┌─────────────────────────┐
│ 1. Checkpoint Check   │ ───► │ 2. Evaluate on Test Set │
│ (Google Drive Check)  │      │ (26,100 LAV-DF Clips)   │
└───────────────────────┘      └─────────────────────────┘
                                            │
                                            ▼
┌───────────────────────┐      ┌─────────────────────────┐
│ 4. Compile in Overleaf│ ◄─── │ 3. Auto-Generate Tables │
│ (Download Final PDF)  │      │ (Run Python Generator)  │
└───────────────────────┘      └─────────────────────────┘
```

---

## Step 1: Environment & Repository Setup

On your local machine or Google Colab:

```bash
# Clone the latest codebase
git clone https://github.com/akhildragon-07/DEEP_LEARNING.git
cd DEEP_LEARNING
git pull origin main

# Install local dependencies for plotting and artifact generation
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

---

## Step 2: Verify Checkpoints on Google Drive

All models automatically save and mirror to Google Drive during training under:
`/content/drive/MyDrive/RMS_AV_Checkpoints/`

Ensure the following 4 model checkpoints are present before running the benchmark:
1. `lavdf_audio_best.pth` (Acoustic Unimodal Baseline)
2. `lavdf_video_best.pth` (Visual Unimodal Baseline)
3. `lavdf_fusion_best.pth` (Standard Concatenation Fusion Baseline)
4. `lavdf_rmsav_best.pth` (Our Proposed Method: RMS-AV)

> **Note**: If any model is still training in Colab, let it finish or check its current epoch in Google Drive. If disconnected, re-running the training command automatically resumes from `latest.pth`.

---

## Step 3: Run Full Benchmark Evaluation in Google Colab

Run this cell in Google Colab to evaluate all 4 models on the **26,100 held-out test clips** under clean conditions and modality-drop stress testing (missing video, missing audio):

```python
# ===========================================================================
# EVALUATE ALL MODELS ON HELD-OUT TEST SPLIT (26,100 SAMPLES)
# ===========================================================================
%cd /content/DEEP_LEARNING
!git pull origin main

!python -m lavdf_pipeline.evaluate \
    --metadata_path /root/.cache/kagglehub/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/versions/1/LAV-DF/metadata.min.json \
    --media_root /root/.cache/kagglehub/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/versions/1/LAV-DF \
    --checkpoint_dir /content/drive/MyDrive/RMS_AV_Checkpoints \
    --output_dir results_lavdf \
    --prefix lavdf_ \
    --batch_size 32
```

### What this produces:
* `results_lavdf/lavdf_benchmark_comparison.csv` (Spreadsheet of all metrics)
* `results_lavdf/lavdf_benchmark_results.json` (Structured JSON benchmark data)

Download the `results_lavdf/` folder from Colab to your local machine inside the repository directory.

---

## Step 4: Auto-Generate Paper Tables & Plots (1 Command)

On your local machine, run the artifact generator script:

```bash
python generate_paper_artifacts.py --results_csv results_lavdf/lavdf_benchmark_comparison.csv --output_dir paper/artifacts
```

### What gets generated automatically:
1. `paper/artifacts/table1_main_benchmark.tex`:
   * LaTeX table comparing Accuracy, Balanced Accuracy, Precision, Recall, Macro-F1, AUROC, and ECE for all 4 models.
2. `paper/artifacts/table2_stress_test.tex`:
   * LaTeX table showing robustness and Macro-F1 retention when audio or video is corrupted/missing.
3. `paper/artifacts/fig_modality_stress_comparison.png`:
   * 300 DPI publication bar chart comparing modality-drop resilience.

---

## Step 5: Finalize and Compile Paper in Overleaf

1. Go to [Overleaf](https://www.overleaf.com) and create a **New Project** $\rightarrow$ **Blank Project** (e.g. name it `RMS-AV-Deepfake-Paper`).
2. Upload the entire `paper/` folder from the repository:
   * `manuscript.tex` (Main LaTeX document)
   * `artifacts/` folder (containing the `.tex` tables and `.png` figures generated in Step 4)
3. In `manuscript.tex`, make sure your name and affiliation are set in the `\author{}` block.
4. Click **Recompile**.
5. Your IEEE double-column research paper PDF is ready!

---

## Step 6: Key Points to Write in the Paper Sections

When reviewing or expanding text in [`paper/manuscript.tex`](paper/manuscript.tex), emphasize these four core technical contributions:

### 1. The Core Problem: Asymmetric Manipulation
* **Explain**: In real deepfakes, attackers often swap faces while keeping real voices, or clone voices over real broadcast footage.
* **Why baselines fail**: Vanilla fusion ($[\mathbf{h}_v; \mathbf{h}_a]$) allows the authentic modality to suppress the fake signals, causing false negatives.

### 2. The Solution: RMS-AV Reliability Gating
* **Dynamic Gate**:
  $$\mathbf{s} = \mathbf{W}_2 \operatorname{ReLU}(\mathbf{W}_1 [\mathbf{h}_a; \mathbf{h}_v] + \mathbf{b}_1) + \mathbf{b}_2 \quad \in \mathbb{R}^2$$
* **FP16 Bounded Validity Masking**: Uses $-10^4$ (not $-10^9$) before softmax so float16 does not overflow:
  $$s_i' = \begin{cases} s_i & \text{if } m_i = 1 \\ -10^4 & \text{if } m_i = 0 \end{cases} \quad \implies \quad [g_a, g_v] = \operatorname{softmax}([s_a', s_v'])$$
  *Strictly guarantees $g_i = 0$ if a stream is missing or corrupted.*
* **Selective Representation**: $\mathbf{h}_{\text{fused}} = [g_v \mathbf{h}_v; \; g_a \mathbf{h}_a]$.

### 3. Auxiliary Supervision ($\lambda_{\text{aux}} = 0.20$)
* Multi-task loss trains auxiliary unimodal heads alongside the joint head:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{av}}, y) + 0.20 \left( \mathcal{L}_{\text{BCE}}(\hat{y}_a, y_a) + \mathcal{L}_{\text{BCE}}(\hat{y}_v, y_v) \right)$$
* Prevents gradient starvation and keeps unimodal representations sharp.

### 4. Zero Cross-Split Identity Leakage
* In LAV-DF, every manipulated sample originates from a source clip (`original`). All deepfakes derived from an original clip reside in the exact same split as the original. There is **zero source identity overlap** between Train, Val, and Test.

---

## Step 7: How to Answer Faculty / Reviewer Defense Questions

| Reviewer / Faculty Question | Verified Mathematical / Empirical Answer |
| :--- | :--- |
| **"Why not just use raw Accuracy?"** | LAV-DF has class imbalance (75% fakes across 3 forged categories vs 25% real). A naive model predicting all fakes gets 75% accuracy but 0% real recall. **Macro-F1** and **Balanced Accuracy** evaluate both classes equally without bias. |
| **"Why doesn't the gate collapse to one modality?"** | We initialized $\mathbf{W}_2 = \mathbf{0}$ and $\mathbf{b}_2 = \mathbf{0}$, giving an exact neutral prior $[0.50, 0.50]$. Coupled with auxiliary supervision ($\lambda_{\text{aux}} = 0.20$), both encoders receive consistent gradient flow. |
| **"How is missing modality handled in deployment?"** | Binary masks $m_a, m_v \in \{0, 1\}$ set unobserved stream logits to $-10^4$, giving them an exact softmax weight of 0. The network safely switches to pure unimodal operation without crashing. |
| **"Is there identity leakage between train and test?"** | No. LAV-DF partitions clips by root recording identity. Deepfakes derived from subject $X$ in train never appear in dev or test. |

---

## Final Submission Checklist

- [ ] All 4 models trained (Audio, Video, Concat Fusion, RMS-AV).
- [ ] Evaluation executed on full 26,100 test clips.
- [ ] `generate_paper_artifacts.py` executed to update `table1_main_benchmark.tex` and `table2_stress_test.tex`.
- [ ] Author names, email addresses, and institution filled in `paper/manuscript.tex`.
- [ ] Overleaf compiles cleanly with 0 errors.
- [ ] Check page limit (typical IEEE conferences: 4 to 6 pages including references).
- [ ] Export PDF and submit to conference portal!
