# RMS-AV Pilot Experiment Audit and Metric Correction Report
**Project:** Multimodal Deepfake Detection (RMS-AV & Baselines)  
**Execution Context:** Verification and Metric Audit on the 40-Sample Teammates Pilot Dataset  
**Audit Date:** October 28, 2026  
**Seed:** 42  

---

## 1. Executive Summary & Audit Motivation

This audit rigorously inspects the existing 40-sample pilot experiment, recalculates evaluation metrics directly from the verified per-sample prediction records and saved model checkpoints, and corrects several reporting inaccuracies identified in the preliminary report.

### Key Audit Findings & Corrections:
1. **Macro-F1 vs. Positive-Class F1 Correction:**
   - In preliminary reporting, the positive-class binary F1-score ($F1_{\text{pos}} = 85.71\%$ for baselines and $66.67\%$ for RMS-AV) was erroneously labeled as **Macro-F1**.
   - Because the test set has an imbalanced class distribution (6 Fake, 2 Real) and baselines predicted all samples as Fake ($TN = 0$), the true unweighted **Macro-F1 is 42.86%** for Audio, Video, and Fusion baselines, and **33.33%** for RMS-AV.
2. **Identity Leakage Clarification:**
   - The pilot dataset filenames and paths contain multiple subject identifiers (target subject, driving face identity, and source audio identity).
   - An exhaustive cross-split audit of **all** extracted identity tokens confirmed zero identity overlap between Train (36 IDs), Val (13 IDs), and Test (13 IDs). However, within the Test set, two samples (`video_003` and `video_016`) share driving face identity `id05434`. Claims of zero identity leakage have been appropriately qualified.
3. **Encoder Initialization Verification:**
   - Both visual (ResNet18) and audio (4-layer 2D CNN) encoders were **randomly initialized** (`pretrained=False`) during pilot training, not loaded from pretrained ImageNet or Whisper weights.
4. **Parameter Counts Corrected:**
   - Model parameter counts were updated to reflect the exact architectures in `model.py`: Audio (73,185), Video (11,340,865), Fusion (11,479,843), and RMS-AV (11,545,765).
5. **No Claims of Superiority on Pilot Data:**
   - RMS-AV achieved an overall accuracy of 50.00% and ROC-AUC of 0.3333 on the 8-sample test set, reflecting the fact that 3 epochs of training on 24 samples is insufficient to calibrate its dynamic reliability gate and auxiliary heads.

---

## 2. Dataset Verification & Identity Leakage Audit

### 2.1 Verified Pilot Dataset Structure
- **Dataset Path:** `teammates/zip2/`
- **Total Records:** 40 samples (10 samples per manipulation category)
  - Audio: 40 `.wav` files (16 kHz mono, ~4.0 s duration, `teammates/zip2/audio/`)
  - Video Frames: 200 `.jpg` frames (5 uniform frames per clip at $224 \times 224 \times 3$, `teammates/zip2/frames/<sample_id>/`)
- **Missing or Corrupted Files:** 0
- **Duplicate Sample Paths:** 0

### 2.2 Category & Binary Label Mapping
- `FakeVideo-FakeAudio` (FV-FA): 10 samples $\to$ Binary Label: **1.0 (Manipulated / Fake)**
- `FakeVideo-RealAudio` (FV-RA): 10 samples $\to$ Binary Label: **1.0 (Manipulated / Fake)**
- `RealVideo-FakeAudio` (RV-FA): 10 samples $\to$ Binary Label: **1.0 (Manipulated / Fake)**
- `RealVideo-RealAudio` (RV-RA): 10 samples $\to$ Binary Label: **0.0 (Authentic / Real)**

Binary classification target: $y \in \{0, 1\}$.

### 2.3 Split Sizes and Class Distribution
Stratified 3:1:1 split generated with seed 42 in `splits_pilot/`:
- **Train Split (`train_manifest.csv`):** 24 samples (6 per category: 18 Fake, 6 Real)
- **Validation Split (`val_manifest.csv`):** 8 samples (2 per category: 6 Fake, 2 Real)
- **Test Split (`test_manifest.csv`):** 8 samples (2 per category: 6 Fake, 2 Real)

### 2.4 Identity Leakage Analysis
FakeAVCeleb directory paths and filenames embed multiple subject identities:
- Target Subject (directory path, e.g., `id06427`)
- Driving Face Source (filename prefix, e.g., `id06061`)
- Source Audio Identity (filename suffix, e.g., `id04414`)

| Audit Check | Findings | Leakage Assessment |
| :--- | :--- | :--- |
| **Train vs. Validation Overlap** | 36 Train IDs vs. 13 Val IDs | **0 Overlapping Identities** |
| **Train vs. Test Overlap** | 36 Train IDs vs. 13 Test IDs | **0 Overlapping Identities** |
| **Val vs. Test Overlap** | 13 Val IDs vs. 13 Test IDs | **0 Overlapping Identities** |
| **Intra-Test Split Overlap** | `video_003` and `video_016` share driving face `id05434` | Minor intra-split correlation (does not violate train-test separation) |

> [!NOTE]
> **Audit Qualification**: While all 3 splits are strictly disjoint across all source and target identities, deepfake datasets can contain unannotated acoustic or lighting similarities. Future benchmarks on the complete dataset should employ official speaker and identity registries.

---

## 3. Verified Experimental Configuration

| Parameter / Setting | Audio Baseline | Video Baseline | Standard Fusion | RMS-AV |
| :--- | :--- | :--- | :--- | :--- |
| **Mode Key** | `audio` | `video` | `fusion` | `rmsav` |
| **Visual Backbone** | None | ResNet18 ($224 \times 224$) | ResNet18 ($224 \times 224$) | ResNet18 ($224 \times 224$) |
| **Visual Pretraining** | N/A | **Random Initialization** (`pretrained=False`) | **Random Initialization** (`pretrained=False`) | **Random Initialization** (`pretrained=False`) |
| **Audio Backbone** | 4-layer 2D CNN | None | 4-layer 2D CNN | 4-layer 2D CNN |
| **Audio Pretraining** | **Random Initialization** | N/A | **Random Initialization** | **Random Initialization** |
| **Feature Dimension** | 256 | 256 | 256 per modality | 256 per modality |
| **Reliability Gate** | None | None | None | Dynamic Softmax Gate ($\mathbf{g}_v, \mathbf{g}_a$) |
| **Total Parameters** | 73,185 | 11,340,865 | 11,479,843 | 11,545,765 |
| **Trainable Parameters**| 73,185 | 11,340,865 | 11,479,843 | 11,545,765 |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adam | Adam | Adam |
| **Learning Rate** | $1 \times 10^{-3}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **Weight Decay** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **Loss Function** | Binary Cross-Entropy | Binary Cross-Entropy | Binary Cross-Entropy | Multi-Task BCE ($\lambda_{\text{aux}}=0.2$) |
| **Batch Size / Epochs** | 4 / 3 | 4 / 3 | 4 / 3 | 4 / 3 |
| **Checkpoint Selection**| Best Val AUC (Epoch 1) | Best Val AUC (Epoch 1) | Best Val AUC (Epoch 2) | Best Val AUC (Epoch 1) |

---

## 4. Corrected Clean Benchmark Results (Held-Out Test Set, N=8)

All four checkpoints were evaluated on `splits_pilot/test_manifest.csv` (6 Fake, 2 Real) under uncorrupted (`none`) conditions. Threshold $\tau = 0.50$.

### 4.1 Performance Comparison Table

| Model Mode | Accuracy | Balanced Acc | Precision (Fake) | Recall (Fake) | Positive-Class F1 | Negative-Class F1 | **True Macro-F1** | ROC-AUC | ECE (Conf) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Audio Baseline** | 75.00% | 50.00% | 75.00% | 100.0% | 85.71% | 0.00% | **42.86%** | **0.9167** | 0.3193 |
| **Video Baseline** | 75.00% | 50.00% | 75.00% | 100.0% | 85.71% | 0.00% | **42.86%** | 0.2500 | 0.2213 |
| **Standard Fusion** | 75.00% | 50.00% | 75.00% | 100.0% | 85.71% | 0.00% | **42.86%** | 0.5833 | 0.1116 |
| **Proposed RMS-AV** | 50.00% | 33.33% | 66.67% | 66.67% | 66.67% | 0.00% | **33.33%** | 0.3333 | **0.0012** |

### 4.2 Confusion Matrix Breakdown ($N=8$)

| Model Mode | True Negatives (TN, Real) | False Positives (FP) | False Negatives (FN) | True Positives (TP, Fake) | Total Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Audio Baseline** | 0 | 2 | 0 | 6 | 8 |
| **Video Baseline** | 0 | 2 | 0 | 6 | 8 |
| **Standard Fusion** | 0 | 2 | 0 | 6 | 8 |
| **Proposed RMS-AV** | 0 | 2 | 2 | 4 | 8 |

---

## 5. Audit of Modality-Drop Stress Testing

Modality drop tests evaluate whether models degrade gracefully when one stream is completely absent.

| Model Mode | Condition | Accuracy | Balanced Acc | Macro-F1 | ROC-AUC | TN | FP | FN | TP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Audio Baseline** | Clean (`none`) | 75.00% | 50.00% | 42.86% | 0.9167 | 0 | 2 | 0 | 6 |
| Audio Baseline | Missing Video | 75.00% | 50.00% | 42.86% | 0.9167 | 0 | 2 | 0 | 6 |
| Audio Baseline | Missing Audio | 25.00% | 50.00% | 20.00% | 0.5000 | 2 | 0 | 6 | 0 |
| **Video Baseline** | Clean (`none`) | 75.00% | 50.00% | 42.86% | 0.2500 | 0 | 2 | 0 | 6 |
| Video Baseline | Missing Video | 25.00% | 50.00% | 20.00% | 0.5000 | 2 | 0 | 6 | 0 |
| Video Baseline | Missing Audio | 75.00% | 50.00% | 42.86% | 0.2500 | 0 | 2 | 0 | 6 |
| **Standard Fusion**| Clean (`none`) | 75.00% | 50.00% | 42.86% | 0.5833 | 0 | 2 | 0 | 6 |
| Standard Fusion | Missing Video | 75.00% | 50.00% | 42.86% | 0.9167 | 0 | 2 | 0 | 6 |
| Standard Fusion | Missing Audio | 75.00% | 50.00% | 42.86% | 0.3333 | 0 | 2 | 0 | 6 |
| **Proposed RMS-AV**| Clean (`none`) | 50.00% | 33.33% | 33.33% | 0.3333 | 0 | 2 | 2 | 4 |
| Proposed RMS-AV | Missing Video | 25.00% | 50.00% | 20.00% | 0.4167 | 2 | 0 | 6 | 0 |
| Proposed RMS-AV | Missing Audio | 25.00% | 50.00% | 20.00% | 0.5000 | 2 | 0 | 6 | 0 |

### 5.1 Dynamic Reliability Gate Analysis in RMS-AV
- **Clean Inputs:** Per-sample gate weights exhibited variation around 0.50:
  - Audio Gate ($g_a$): ranged from $0.4951$ to $0.5044$ ($\mu = 0.5001$).
  - Video Gate ($g_v$): ranged from $0.4956$ to $0.5049$ ($\mu = 0.4999$).
  - Constraint: $g_a + g_v = 1.0000$ for all samples.
- **Under Missing Video:** Validity mask strictly clamped video weight to $g_v = 0.0000$ and assigned $100\%$ weight to audio ($g_a = 1.0000$).
- **Under Missing Audio:** Validity mask strictly clamped audio weight to $g_a = 0.0000$ and assigned $100\%$ weight to video ($g_v = 1.0000$).

---

## 6. Threats to Validity and Pilot Limitations

> [!WARNING]
> **Essential Statistical Caveats:**
> 1. **Sample Size Discretization**: With $N=8$ test samples, each individual sample accounts for **12.5 percentage points** of accuracy. No claims of statistical significance or generalizable model ranking can be drawn from 8 test samples.
> 2. **Prevalence Imbalance**: Because the 4-way manipulation taxonomy maps 3 categories to Fake (1) and 1 category to Real (0), the test set has a 75% prior for Fake. The baselines achieved 75% accuracy simply by collapsing to the majority class prior ($TP=6, FP=2, TN=0$).
> 3. **Random Initialization**: Backbones were trained from scratch without pretrained weights to ensure offline sandbox compliance. Scaling up with pretrained ImageNet-1K / Wav2Vec weights will be essential for meaningful feature extraction.

---

## 7. Reproduction Commands and Output Artifacts

### 7.1 Exact Reproduction Commands
```bash
# 1. Run audit and recompute all metrics
python audit_recompute_and_plot.py

# 2. Run unit and integration validation tests
python test_pilot_audit.py
```

### 7.2 Generated Artifact Locations
- **Per-Sample Predictions:** `results/pilot_predictions.csv`
- **Corrected Comparison Table:** `results/pilot_comparison.csv`
- **Corrected JSON Summary:** `results/pilot_results.json`
- **Figures:**
  - `results/pilot_metrics.png` (Corrected bar chart comparison)
  - `results/confusion_matrices.png` (2x2 confusion matrices)
  - `results/roc_curves.png` (Continuous ROC curves)
  - `results/precision_recall_curves.png` (Precision-Recall curves)
- **Unit Test Suite:** `test_pilot_audit.py`

---

## 8. False Positive Diagnostic Audit, Decision Boundary Analysis & Uncertainty Resolution

### 8.1 Incident Description
During evaluation in the forensic Streamlit dashboard, an original unmanipulated face video was classified as **"LIKELY MANIPULATED"** with:
- **Manipulation Probability:** $50.82\%$
- **Raw Logit ($z$):** $+0.0329$
- **Decision Cutoff ($\tau$):** $0.50$

### 8.2 End-to-End Pipeline Audit

| Pipeline Stage | Verified Implementation | Status |
| :--- | :--- | :---: |
| **Loaded Checkpoint** | `checkpoints/pilot_video_best.pth` (Epoch 1, Val AUC = 0.50) | Valid |
| **Architecture** | `AVDetector(mode='video', dim=256, pretrained=False)` (11,340,865 params) | Match |
| **Evaluation Mode** | `model.eval()` active; inference executed under `torch.inference_mode()` | Valid |
| **Sigmoid Mapping** | $\sigma(0.0329) = \frac{1}{1 + e^{-0.0329}} = 0.508216$ ($50.82\%$) | Exact |
| **Label Semantics** | Label `0.0 = Real/Authentic`, Label `1.0 = Fake/Manipulated` across all files | Consistent |
| **Preprocessing** | RGB conversion, $224 \times 224$ bilinear resize, ImageNet normalization ($\Delta = 0.00$) | Identical |
| **Backbone Weights** | `pretrained=False` (weights randomly initialized from Gaussian noise) | Verified |
| **Failure Handling** | Missing checkpoints explicitly raise `RuntimeError`; no random fallback | Safe |

### 8.3 Reproduction and Logit Distribution Across Pilot Splits

Direct reproduction outside the UI on ground-truth pilot samples demonstrated that **all samples** output logits clustered tightly between $+0.028$ and $+0.162$:

- **Train Split ($N=24$):** Logits $+0.0280$ to $+0.1625$ ($\sigma(z) \in [50.70\%, 54.05\%]$)
  - Train authentic sample `video_037`: Logit $+0.0280 \implies 50.70\%$
  - Train authentic sample `video_025`: Logit $+0.0371 \implies 50.93\%$
  - Target query video: Logit $+0.0329 \implies 50.82\%$ (exact fit within the authentic cluster)
- **Validation Split ($N=8$):** Logits $+0.0701$ to $+0.1275$ ($\sigma(z) \in [51.75\%, 53.18\%]$)
- **Held-Out Test Split ($N=8$):** Logits $+0.0742$ to $+0.1513$ ($\sigma(z) \in [51.85\%, 53.78\%]$)
  - Evaluated sample `video_036` (RealVideo-RealAudio): 5 extracted frames produced individual logits $\{+0.1275, +0.1244, +0.1106, +0.1174, +0.1025\}$; pooled logit $= +0.1167 \implies 52.91\%$. Under rigid $\tau = 0.50$, it yields a false positive.

### 8.4 Root Cause Synthesis

The false positive is **NOT** a code bug, preprocessing mismatch, or color space inversion. It is driven by three compounding factors:

1. **Undertrained Visual Backbone (Primary Architectural Factor):**
   The visual encoder is an 11.3M parameter ResNet18 trained from scratch (`pretrained=False`) on only 24 samples for 3 epochs (18 batch updates). The convolutional kernels have not converged to meaningful facial forgery feature detectors.
2. **75% Fake Class Imbalance (Optimization Prior Shift):**
   With 18 Fake and 6 Real training samples, the binary cross-entropy gradient shifted the final classification bias slightly positive: $\bar{b} \approx +0.10$. For any unseen face, the randomly initialized features project close to zero, causing the network to emit its training prior ($z \approx +0.03$ to $+0.12$, mapping to $50.7\%$ to $53.0\%$).
3. **Rigid Binary Thresholding at Chance Level (Decision Rule Trigger):**
   Applying a strict cutoff $\tau = 0.50$ directly against an uncalibrated model whose entire output range sits near $z = 0$ caused authentic faces with $z = +0.0329$ ($50.82\%$) to be classified as "LIKELY MANIPULATED" by a margin of only $+0.82\%$.

### 8.5 Safe Resolution: Configurable Uncertainty Band

To resolve this without altering model weights or manufacturing fabricated predictions:

1. **Three-State Decision Logic (`inference.py`):**
   Implemented an uncertainty band $[\tau - \delta, \tau + \delta]$ parameterized by threshold $\tau$ and margin $\delta$ (default $\delta = \pm 0.05$, active band $[0.450, 0.550]$):
   $$\text{Verdict} = \begin{cases} \text{"Likely manipulated"} & \text{if } p > \tau + \delta \\ \text{"INCONCLUSIVE — MODEL SCORE NEAR THRESHOLD"} & \text{if } \tau - \delta \le p \le \tau + \delta \\ \text{"Likely authentic"} & \text{if } p < \tau - \delta \end{cases}$$
2. **Dashboard Controls (`app.py`):**
   Added a configurable `Uncertainty Margin (±%)` slider in the sidebar with active decision band telemetry. When $p=50.82\%$, the UI displays an amber neutral card: `⚖️ INCONCLUSIVE — MODEL SCORE NEAR THRESHOLD` with detailed diagnostic context explaining that the raw logit is near chance level ($z \approx 0$).
3. **Verification Suites:**
   - `test_uncertainty_band.py`: 5 unit tests validating the user reproduction case, three-state partitions, margin scaling, and integration with `InferenceEngine`.
   - `test_inference.py`: 7 tests validating all input modes and error handling.

### 8.6 Readiness Assessment for Real-World Deployment

> [!CAUTION]
> **Production Readiness: NOT READY FOR REAL-WORLD USE**
> 
> The pilot models are exploratory prototypes trained on a 40-sample proof-of-concept dataset. They must **not** be deployed for real-world forensic verification or legal/evidentiary authentication.
> 
> **Required Steps Before Production Deployment:**
> 1. Train on the full 23 GB FakeAVCeleb / DFDC dataset (thousands of unique identities).
> 2. Initialize with pretrained feature representations (ImageNet-1K / FaceForensics++ for visual; Wav2Vec2 / Whisper for acoustic).
> 3. Balance class sampling ($1:1$ Real to Fake ratio) to eliminate positive prior drift.
> 4. Perform post-hoc temperature scaling / Platt calibration on a held-out validation set.
