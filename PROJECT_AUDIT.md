# PROJECT AUDIT: Multimodal Deepfake Detection Pipeline

**Audit Date:** October 28, 2026  
**Auditor:** Antigravity Research Assistant  
**Repository Working Directory:** `c:\Users\akhil\OneDrive\Documents\DL--PROJECT`  
**Current Status:** Phase 1 Complete (Audit & Diagnostic Inspection)

---

## Executive Summary

The project repository was initialized from a prompt that generated a preliminary codebase titled **PG-ABN (Prompt-Guided Asymmetric Bridge Network)**. While some fundamental building blocks (CLIP visual extraction, Whisper audio extraction, and basic gated fusion) are present, the current codebase does **not** match the user's research objective of **RMS-AV (Reliability- and Manipulation-Aware Selective Audio-Visual Fusion)**. 

Furthermore, the existing codebase contains critical bugs (e.g., `BatchNorm1d` failures on batch size 1, file corruption in `README.md`, unhandled zero-tensor propagation from corrupted samples), severe methodological risks (uncontrolled sample-level splitting with high identity data leakage), and lacks essential components (unimodal authenticity heads, reliability estimation, missing-modality handling, baselines, and standalone evaluation).

This audit provides a comprehensive breakdown of the existing files, current capabilities, identified flaws, and a prioritized upgrade path.

---

## 1. Existing Files and Their Purposes

| File Name | Size (Bytes) | Intended / Current Purpose | Verification Status |
| :--- | :--- | :--- | :--- |
| `README.md` | 6,673 | Project documentation, architecture overview, and usage instructions. | **Corrupted & Outdated**: Appended UTF-16LE text caused trailing null bytes (`\x00`), making text readers detect it as binary. Documents PG-ABN rather than RMS-AV and contains placeholder metric tables. |
| `requirements.txt` | 170 | Python package dependency specifications. | **Incomplete**: Lists 10 packages (`torch`, `torchvision`, `torchaudio`, `transformers`, `librosa`, `opencv-python`, `pandas`, `scikit-learn`, `numpy`, `tqdm`). Missing plotting and configuration utilities (`matplotlib`, `pyyaml`, `scipy`). `torchaudio` and `librosa` are absent in local environment. |
| `model.py` | 20,757 | Defines model architecture: `AntigravityPrompt`, `AsymmetricGatedFusion`, `BinaryClassificationHead`, `TemporalLocalizationHead`, and `PGABN`. | **Architectural Mismatch & Fragile**: Implements PG-ABN with temporal localization instead of RMS-AV. Lacks unimodal heads, reliability estimation, and missing modality support. `BatchNorm1d` crashes on batch size 1. |
| `dataset.py` | 13,467 | PyTorch `Dataset` loader for FakeAVCeleb: extracts 16 uniform frames via OpenCV and 80-channel log-mel spectrograms via Librosa + Whisper feature extractor. | **High I/O Latency & Leakage Risk**: Hard import of `librosa` breaks on environments without it. Does heavy video/audio decoding on the fly. Returns zero-tensors on failure without proper model handling. Does not prevent cross-split identity leakage. |
| `train.py` | 16,931 | Training loop with binary BCE loss, temporal localization regression loss, evaluation metrics, and checkpointing. | **Misaligned & Leaky**: Random sample splitting (`random_split`) causes data leakage between modified clips of the same source. Trains only PG-ABN with temporal regression. Ignores dataset `is_valid` flags. |
| `verify_pipeline.py` | 6,269 | Standalone smoke test verifying sub-module tensor shapes and backpropagation with synthetic tensors. | **Partially Verified**: Sub-module unit test passes on synthetic tensors. However, it does not instantiate or test full models (`PGABN` or `RMS-AV`), does not test missing modalities, and mocks evaluation tables. |

---

## 2. Implemented Functionality

The following components are currently implemented and verified:

1. **Vision-Language and Audio Encoders**:
   - Visual backbone wrapper using frozen `openai/clip-vit-base-patch32` with temporal mean pooling across sampled frames ($D_v = 768$).
   - Acoustic backbone wrapper using frozen `openai/whisper-tiny` encoder with temporal mean pooling over mel spectrograms ($D_a = 384$).
   - Backbone parameter freezing (`param.requires_grad = False`) functions as intended.
2. **Sub-module Mechanics**:
   - `AntigravityPrompt`: Implements learnable residual parameter injection ($\tilde{F} = F + \alpha \odot P$). Passes gradient checks.
   - `AsymmetricGatedFusion`: Projects visual and audio features to a shared bridge dimension ($D_{\text{bridge}} = 512$), computes sigmoid gating weights $g_v, g_a \in [0, 1]$, and produces fused representation.
3. **Training Engine Foundations**:
   - Standard PyTorch optimization loop using Adam with gradient clipping (`clip_grad_norm_`).
   - Checkpoint saving based on validation ROC-AUC.
   - Metric computation via `scikit-learn` (Accuracy, F1-Score, ROC-AUC) and formatted console output.
4. **Fault-Tolerant Dataset Fallback**:
   - Catches OpenCV/Librosa file loading exceptions and returns zero-filled tensors with an `is_valid=0` tensor indicator.

---

## 3. Missing or Incomplete Functionality

### A. Architectural Deficiencies (RMS-AV Requirements)
- **Proposed RMS-AV Architecture**: The requested model is **RMS-AV** (*Reliability- and Manipulation-Aware Selective Audio-Visual Fusion*). The existing model is **PG-ABN**.
- **Unimodal Authenticity Prediction Heads**: Missing dedicated Video Authenticity Head and Audio Authenticity Head. Unimodal heads are essential to detect unilateral deepfakes (where only one stream is manipulated) and to derive unimodal confidence/authenticity scores.
- **Modality Reliability Estimation**: No module exists to assess modality reliability (e.g., audio SNR/silence, visual blur/corruption, feature entropy, or learned reliability logits).
- **Reliability-Aware Gating**: The current gating mechanism only performs cross-modal MLP projection without explicitly conditioning on modality reliability or missingness indicators.
- **Missing Modality Support**: The existing forward pass requires both `pixel_values` and `input_features`. If either modality is omitted or set to `None`, the model raises an immediate exception.
- **Baselines Missing**:
  1. Video-only model.
  2. Audio-only model.
  3. Standard audio-visual fusion baseline (e.g., concatenation / vanilla cross-attention).

### B. Dataset and Preprocessing Pipeline
- **Dataset Structure & Label Auditor**: No tool exists to inspect the 23 GB dataset, count classes, verify paths, check corruptions, or inspect metadata.
- **Identity / Group Leakage Prevention**: No grouping logic by speaker identity or source video ID. A random split allows synthetic variations of a person's video to exist in both training and test sets.
- **Split Manifests**: Train, validation, and test splits are generated dynamically on every execution without saving a persistent manifest (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`).
- **Offline Preprocessing Script (`preprocess.py`)**: Decoding full MP4 videos and computing spectrograms inside PyTorch's `__getitem__` on Google Drive causes severe runtime bottlenecks. Pre-extracting frame sequences or spectrograms is currently unsupported.

### C. Evaluation and Reproducibility
- **Dedicated Evaluation Script (`evaluate.py`)**: Evaluation is only embedded within `train.py`. There is no CLI script to load an arbitrary checkpoint and evaluate it on a separate test set.
- **Missing Metrics**: Balanced accuracy, precision, recall, macro-F1, false-positive rate (FPR), false-negative rate (FNR), per-class metrics, confusion matrices, and calibration error (ECE).
- **Stress & Robustness Tests**: No scripts to test performance under missing modality conditions (video-only, audio-only) or corruption conditions (Gaussian noise, temporal masking, blur).
- **Configuration Management (`configs/`)**: No structured YAML configuration files for experiment management.
- **Results Output Directory (`results/`)**: No standardized folder structure for saving figures, CSV prediction dumps, or training logs.

---

## 4. Bugs and Integration Issues

1. **`README.md` Binary Corruption**:
   - *Cause*: A PowerShell `echo` command appended UTF-16LE characters with null bytes (`\x00`) to the UTF-8 file.
   - *Impact*: Standard text readers fail or report the file as binary.
2. **`BatchNorm1d` Crash on Single-Sample Batches**:
   - *Location*: `model.py` (`BinaryClassificationHead`).
   - *Cause*: `nn.BatchNorm1d` fails when `batch_size = 1` during training mode with `ValueError: Expected more than 1 value per channel when training`.
   - *Impact*: If the dataset size is not divisible by batch size and `drop_last=False`, training crashes at the final step of an epoch. Single-instance inference also requires careful eval mode toggling. `LayerNorm` is required.
3. **Silent Error Propagation from Corrupted Inputs**:
   - *Location*: `dataset.py` & `train.py`.
   - *Cause*: `dataset.py` returns zero-tensors and `is_valid = 0` upon file decoding errors. However, `train.py` completely ignores `is_valid` and passes the zero-tensors into the model.
   - *Impact*: Non-linear foundation backbones (CLIP/Whisper) with biases and layer normalization transform zero-tensors into non-zero arbitrary embedding vectors. The model computes loss and gradients against corrupted zero-fallbacks as if they were genuine inputs.
4. **Environment Fragility (Hard `librosa` Import)**:
   - *Location*: `dataset.py`.
   - *Cause*: Unconditional `import librosa` at top level.
   - *Impact*: Fails immediately with `ModuleNotFoundError: No module named 'librosa'` in local Windows or lightweight environments where librosa is not installed, blocking any offline verification or testing.
5. **Data Leakage via Naive Splitting**:
   - *Location*: `train.py`.
   - *Cause*: `random_split(full_dataset, [train_size, val_size])` splits at the clip level.
   - *Impact*: Source videos and manipulated derivatives sharing the same subject identity appear in both train and validation sets, producing artificially high validation metrics that will fail to generalize to novel identities.
6. **Temporal Localization Objective Contamination**:
   - *Location*: `train.py` & `dataset.py`.
   - *Cause*: `dataset.py` assigns dummy coordinates `[0.0, 1.0]` for fake samples and `[0.0, 0.0]` for real samples, and `train.py` optimizes an MSE regression loss $\mathcal{L}_{\text{loc}}$ scaled by $\lambda_{\text{loc}} = 0.5$.
   - *Impact*: Gradients from a synthetic, ungrounded regression target distort feature learning for deepfake classification.
7. **Potential Out-of-Memory (OOM) on Google Colab T4**:
   - *Location*: `model.py` (`extract_visual_features`).
   - *Cause*: Batch of 8 with 16 frames flattens into 128 images of $224 \times 224$ passed through CLIP ViT-Base. On a 15 GB Colab T4 GPU, this can trigger CUDA OOM if intermediate activation caching is unmanaged.

---

## 5. Recommended Fixes in Priority Order

| Priority | Category | Action Item |
| :---: | :--- | :--- |
| **P1** | Stability & Syntax | Clean `README.md` null bytes; replace `BatchNorm1d` with `LayerNorm` in classification heads; make `librosa` optional/lazy to allow seamless local testing. |
| **P2** | Research Architecture | Implement **RMS-AV** model: visual encoder, audio encoder, unimodal authenticity heads (video head, audio head), reliability estimator, reliability-aware gating mechanism, and deepfake classification head with explicit missing-modality support. |
| **P3** | Baselines | Implement clean baseline models sharing the same backbone interface: Video-Only, Audio-Only, and Standard Audio-Visual Concat/Fusion. |
| **P4** | Dataset & Leakage | Build dataset audit script; implement identity-/subject-grouped train/val/test splitting to prevent data leakage; export permanent CSV split manifests. |
| **P5** | Training Pipeline | Update `train.py` to support `--model_type [rms_av, video_only, audio_only, standard_fusion]`, respect `is_valid` flags, remove spurious temporal localization loss, and log structured metrics to JSON/CSV. |
| **P6** | Evaluation & Stress Testing | Implement standalone `evaluate.py` to evaluate checkpoints, test under missing audio, missing video, audio noise, and video blur conditions, and compute comprehensive metrics (Balanced ACC, Macro F1, AUC, ECE, confusion matrices). |
| **P7** | Colab Workflow & Docs | Provide configuration files (`configs/`), structured results directories (`results/`), update `README.md`, and create a step-by-step Google Colab notebook/script. |

---

## 6. Exact Next Steps

1. **Submit Audit & Request User Decisions (Phase 2)**:
   - Present this audit report.
   - Propose the detailed design for RMS-AV reliability estimation and missing-modality handling.
   - Align on whether to preserve temporal localization as an optional head or eliminate it entirely to focus cleanly on deepfake detection under modality unreliability.
2. **Phase 3 (Architecture Implementation)**:
   - Create the RMS-AV architecture with unimodal heads and reliability-aware gating.
   - Build Video-Only, Audio-Only, and Standard Fusion baselines.
   - Add unit tests with synthetic inputs covering missing audio, missing video, and batch sizes $\ge 1$.
3. **Phase 4 (Dataset Pipeline & Audit)**:
   - Build dataset auditor and leakage-safe manifest generator.
   - Provide offline preprocessing support for Colab / Google Drive execution.
