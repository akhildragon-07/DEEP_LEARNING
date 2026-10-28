# DATASET STATUS REPORT: Teammates Directory Audit

**Audit Date:** October 28, 2026  
**Directory Inspected:** `c:\Users\akhil\OneDrive\Documents\DL--PROJECT\teammates`  
**Purpose:** Determine whether the local `teammates/` directory contains the complete deepfake detection dataset for full training.

---

## Executive Summary

**Verdict: The `teammates/` directory does NOT contain the complete dataset for full training.**
- **Actual Size on Disk:** **22.55 MB** across all files (far below the expected ~23 GB full FakeAVCeleb benchmark).
- **`teammates/zip1` (13.00 MB):** Contains **only tabular feature CSVs** for 400 samples (ResNet50 visual embeddings, acoustic statistics, lip landmark coordinates, and synchronization metrics). It contains **zero** raw audio (`.wav`) or video (`.mp4`/`.jpg`) files.
- **`teammates/zip2` (9.55 MB):** Contains a **40-clip pilot subset** (10 Real-Real, 10 Fake-Real, 10 Real-Fake, 10 Fake-Fake) with 40 audio files and 200 frame images.
- **Full Dataset Location:** The full ~23 GB FakeAVCeleb dataset resides on Google Drive (or external storage) and must be accessed in Google Colab via Google Drive mount.

The local 40-clip pilot subset in `teammates/zip2` is fully functional and verified for local sanity checks and unit testing, but is insufficient for final research publication models.

---

## 1. Directory Structure & Physical File Counts

| Subfolder | Purpose / Content | Video Files (`.mp4`) | Audio Files (`.wav`) | Frame Images (`.jpg`) | CSV Files | Total Files | Size on Disk |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `teammates/zip1/` | Pre-extracted feature tables | 0 | 0 | 0 | 7 | 7 | 13.00 MB |
| `teammates/zip2/audio/` | Extracted pilot audio tracks | 0 | 40 | 0 | 0 | 40 | 7.15 MB |
| `teammates/zip2/frames/` | Extracted pilot video frames (40 folders) | 0 | 0 | 200 | 0 | 200 | 2.39 MB |
| `teammates/zip2/` (root) | Pilot manifest | 0 | 0 | 0 | 1 | 1 | 0.01 MB |
| **Total** | | **0** | **40** | **200** | **8** | **248** | **22.55 MB (0.022 GB)** |

### Media Specifications:
- **Audio Files (`.wav`):** 40 files (`video_000.wav` to `video_039.wav`), 16,000 Hz sample rate, 16-bit mono PCM, average duration ~4.0 seconds.
- **Frame Images (`.jpg`):** 200 images distributed across 40 folders (`frames/video_000/` to `frames/video_039/`), exactly 5 frames per video (`frame_00.jpg` to `frame_04.jpg`), resolution $224 \times 224$ pixels, RGB.
- **Raw Video Files (`.mp4`):** **0** raw `.mp4` video files present in `teammates/`.

---

## 2. Manifest Inspection & Label Distribution

Eight manifest and feature tables were inspected:

### A. Dataset Manifests
1. **`teammates/zip2/processed_manifest.csv`** (Pilot Manifest):
   - **Records:** 40 samples (`video_000` to `video_039`).
   - **Label Distribution:** Perfectly balanced across 4 forensic categories:
     - `FakeVideo-FakeAudio`: 10 (25.0%)
     - `FakeVideo-RealAudio`: 10 (25.0%)
     - `RealVideo-FakeAudio`: 10 (25.0%)
     - `RealVideo-RealAudio`: 10 (25.0%)
   - **Duplicates:** 0 duplicate paths or video IDs.
   - **Physical File Presence:** 40 / 40 audio files exist; 40 / 40 frame directories exist.
2. **`teammates/zip1/processed_manifest_clean.csv`** (Extended Feature Manifest):
   - **Records:** 400 samples (`video_0000` to `video_0399`).
   - **Label Distribution:**
     - `FakeVideo-FakeAudio`: 100 (25.0%)
     - `FakeVideo-RealAudio`: 100 (25.0%)
     - `RealVideo-FakeAudio`: 100 (25.0%)
     - `RealVideo-RealAudio`: 100 (25.0%)
   - **Duplicates:** 0 duplicate paths or video IDs.
   - **Physical File Presence:** **0 / 400** raw audio/video files are physically present on disk in `teammates/zip1`.

### B. Pre-extracted Feature Tables in `teammates/zip1`
| File Name | Shape (Rows x Cols) | Content Description |
| :--- | :---: | :--- |
| `resnet50_visual_features.csv` | 400 x 2,049 | 2048-dimensional visual feature vectors extracted via ResNet50 for 400 samples. |
| `visual_features.csv` | 400 x 11 | Color channel statistics (mean, std of RGB and brightness). |
| `audio_features.csv` | 400 x 33 | Acoustic descriptors (duration, ZCR, RMS energy, 13 MFCC means and stds). |
| `sync_aligned_pilot.csv` | 1,194 x 15 | Frame-by-frame time-aligned lip opening width and acoustic RMS/ZCR. |
| `sync_audio_features_10fps.csv` | 1,195 x 5 | Timestamped audio energy at 10 fps. |
| `sync_lip_landmarks_10fps.csv` | 1,207 x 6 | Timestamped mouth opening/width landmark coordinates at 10 fps. |

---

## 3. Analysis: `zip1` vs `zip2` Role in the Research Project

- **`zip1` is a Feature & Metadata Package:**
  - Prepared by Member 1 during preliminary feature engineering and exploratory data analysis (EDA).
  - Contains tabular representations for 400 clips.
  - Does **not** contain the media files necessary for end-to-end video/audio neural network training.
- **`zip2` is a Functional Pilot Dataset:**
  - Prepared by Member 1 as a lightweight sanity and verification benchmark.
  - Contains true perceptual media (200 RGB frames, 40 WAV audio tracks).
  - Specifically designed to verify pipeline compatibility before deploying expensive training on the full 23 GB corpus.
- **Relationship:**
  - The 40 samples in `zip2` (`video_000` through `video_039`) correspond exactly to the first 40 entries (`video_0000` through `video_0039`) in `zip1`.

---

## 4. Compatibility with `dataset.py`

The existing [dataset.py](file:///c:/Users/akhil/OneDrive/Documents/DL--PROJECT/dataset.py) has been specifically verified and adapted to:
1. **Load `zip2` Directly:** Automatically discovers `frames/video_XXX/*.jpg` and `audio/video_XXX.wav` from `processed_manifest.csv`.
2. **Handle Full Datasets:** Supports raw `.mp4` video uniform frame sampling and `.wav` audio extraction when pointed to the full FakeAVCeleb directory.
3. **Tensor Shapes Produced:**
   - `video`: `[5, 3, 224, 224]` (ImageNet normalized RGB).
   - `audio`: `[1, 80, 400]` (80-channel log-mel filterbank).
   - `has_video`: `bool` scalar (`True`).
   - `has_audio`: `bool` scalar (`True`).
   - `label`: `float` scalar (`0.0` or `1.0`).
4. **Leakage-Safe Splitting:**
   - Both manifests include `full_path`, which encodes the subject identity (e.g., `id00076`, `id00021`).
   - `create_leakage_safe_splits()` groups related manipulated variants by identity, guaranteeing 0% identity leakage between train, validation, and test sets.

---

## 5. Next Steps & Recommendations

1. **For Local Verification & Pipeline Testing:**
   - Continue using `teammates/zip2` (22.55 MB). It runs quickly on CPU and verifies data loading, forward propagation, multi-task loss calculation, and checkpoint saving without any network overhead.
2. **For Full Deepfake Detection Training:**
   - Full training requires the ~23 GB dataset.
   - Use Google Colab with Google Drive mounted: in [colab_execution.ipynb](file:///c:/Users/akhil/OneDrive/Documents/DL--PROJECT/colab_execution.ipynb), set `DATA_DIR` in **Cell 5** to the Google Drive folder where your 23 GB dataset is stored (`/content/drive/MyDrive/FakeAVCeleb_Dataset`).
   - The pipeline will audit the full dataset in **Cell 6** and train on the complete corpus using Colab GPU acceleration.
