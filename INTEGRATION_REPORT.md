# INTEGRATION REPORT: Integrating Member 1 (Dataset) & Member 2 (Model) into RMS-AV

**Date:** October 28, 2026  
**Integration Lead:** Member 3 (Cloud Execution, Training, and Integration)  
**Status:** Step 1 Complete (Conflict Identification & Proposed Resolution)

---

## 1. Executive Summary

Member 1 and Member 2 have completed their respective development modules for the **RMS-AV** multimodal deepfake detection project. 
- **Member 1** provided the dataset processing artifacts:
  - 400-sample cleaned manifest (`processed_manifest_clean.csv`) and extracted feature CSVs (ResNet50 visual embeddings, audio stats, lip-sync landmarks).
  - 40-sample real multimodal pilot dataset (`teammates/zip2`) with extracted frames (5 frames per video, $224 \times 224$ RGB) and synchronized 16kHz audio tracks (`.wav`).
- **Member 2** provided the core model architecture:
  - `reliability_gate.py`: Bounded softmax-based modality reliability gating with missing-modality masking and NaN-safe zeroing.
  - `model.py` (`AVDetector`): ResNet18 temporal-pooled video encoder, 2D CNN log-mel spectrogram audio encoder, unimodal authenticity heads, and unified modes (`audio`, `video`, `fusion`, `rmsav`).

This report outlines all identified integration gaps, tensor shape discrepancies, interface conflicts, and the approved unified design.

---

## 2. Identified Conflicts & Discrepancies

### A. Model Architecture & Backbones
| Dimension | Existing Project Code | Member 2 Code | Conflict / Gap | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Visual Encoder** | Frozen CLIP ViT-Base ($D_v=768$) | ResNet18 ($D_v=512 \to \text{dim}$) | CLIP ViT has 86M params; ResNet18 has 11M params. ResNet18 is much faster and eliminates Colab T4 OOM risks. | **Adopt Member 2's ResNet18 architecture** as the primary default video backbone for `AVDetector`. Preserve CLIP/foundation options in backup. |
| **Audio Encoder** | Frozen Whisper-Tiny ($D_a=384$) | 3-Block 2D CNN ($D_a=64 \to \text{dim}$) | Whisper requires downloading 39M weights; 2D CNN is 25k params and operates directly on mel spectrograms `[B, 1, n_mels, time]`. | **Adopt Member 2's 2D CNN audio encoder** operating on standard log-mel spectrograms. |
| **Reliability Gate** | MLP with linear projection + scaling | Softmax with masked fill ($-\infty$) and `nan_to_num` | Member 2's `ReliabilityGate` guarantees weights $w_a + w_v = 1.0$ and strictly assigns $0.0$ to missing modalities. | **Integrate Member 2's `ReliabilityGate`** as the canonical RMS-AV gating module. |
| **Modes** | Separate class per model type | Single `AVDetector(mode=...)` | Member 2's `mode in ("audio", "video", "fusion", "rmsav")` cleanly instantiates only necessary parameters. | **Adopt Member 2's `AVDetector`** with `build_model(mode=...)`. |

### B. Tensor Shapes & Batch Interfaces
| Key | Existing Dataset Output | Member 2 Expected Input | Resolution |
| :--- | :--- | :--- | :--- |
| `video` | `[B, 16, 3, 224, 224]` | `[B, T, 3, H, W]` | Configurable $T$ (default $T=5$ matching Member 1's pilot frames). Normalization to standard ImageNet mean/std. |
| `audio` | `[B, 80, 3000]` (Whisper mel) | `[B, 1, n_mels, time]` (2D Mel Spectrogram) | Compute standard 80-channel log-mel spectrogram `[1, 80, time]` using `torchaudio` or fallback `scipy/numpy` STFT filterbank. |
| `has_video` | `video_mask` `[B, 1]` float | `has_video` `[B]` bool | Dataset returns boolean `has_video` (and float mask for multi-task loss). |
| `has_audio` | `audio_mask` `[B, 1]` float | `has_audio` `[B]` bool | Dataset returns boolean `has_audio` (and float mask for multi-task loss). |
| `label` | `label` `[B, 1]` | `label` `[B]` float | Squeezed to 1D `[B]` to match Member 2's raw logit outputs. |

### C. Output Keys & Loss Requirements
| Output Key | Member 2 Output | Supervision Target | Loss Function |
| :--- | :--- | :--- | :--- |
| `logit_av` | `[B]` raw logits | `batch["label"]` `[B]` (Overall Real/Fake) | `BCEWithLogitsLoss()` |
| `logit_video` | `[B]` raw logits | `batch["video_label"]` `[B]` (Video Fake) | `BCEWithLogitsLoss()` masked by `has_video` |
| `logit_audio` | `[B]` raw logits | `batch["audio_label"]` `[B]` (Audio Fake) | `BCEWithLogitsLoss()` masked by `has_audio` |
| `gate` | `[B, 2]` | Modality contributions $(w_a, w_v)$ | Logged for interpretability & calibration |

### D. Dataset Paths & Split Manifests
- **Member 1 Pilot Dataset (`teammates/zip2`)**:
  - Contains 40 balanced pilot clips (10 Real-Real, 10 Fake-Real, 10 Real-Fake, 10 Fake-Fake).
  - Pre-extracted frames in `frames/video_XXX/frame_00.jpg` to `frame_04.jpg` ($224 \times 224$).
  - Audio in `audio/video_XXX.wav` (16 kHz mono).
- **Leakage Prevention**:
  - `full_path` in `processed_manifest.csv` contains identity patterns (`id00076`, `id00021`, etc.).
  - The split generator will group by identity so related manipulations never cross train/val/test splits.

---

## 3. Integration Plan & Action Items

1. **Step 2 — Integrate Dataset & Model**:
   - Save `reliability_gate.py` with Member 2's `ReliabilityGate`.
   - Update `model.py` with Member 2's `AVDetector`, `VideoEncoder`, `AudioEncoder`, and `build_model`.
   - Update `dataset.py` to support loading from Member 1's pre-extracted frame folders (`frames/video_XXX/*.jpg`) and audio files (`audio/video_XXX.wav`), as well as raw MP4 video files. Add Mel Spectrogram extraction to produce `[1, n_mels, time]`.
2. **Step 3 — Create Training Pipeline**:
   - Update `train.py` to support Member 2's `AVDetector` with `--mode [audio, video, fusion, rmsav]`.
   - Use multi-task loss supervising unimodal and fused logits.
   - Save checkpoints, training history, and configuration.
3. **Step 4 — Update Evaluation Protocol**:
   - Update `evaluate.py` to evaluate checkpoints and run missing-modality stress tests.
4. **Step 5 — Create `colab_execution.ipynb`**:
   - Assemble ordered Colab notebook connecting Drive, dependencies, smoke tests, sanity run, and baseline/RMS-AV training.
5. **Step 6 — Verify & Test**:
   - Run synthetic smoke tests and a real-data sanity run using Member 1's 40 pilot samples.
