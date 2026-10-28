# RMS-AV Web Dashboard: User Guide & Instructions

Interactive demonstration interface for **RMS-AV (Reliability- and Manipulation-Aware Selective Audio-Visual Fusion)** and baseline deepfake detection models.

---

## 1. Installation & Environment Setup

Ensure Python 3.9+ is installed with PyTorch and dependencies.

### Windows (PowerShell / Command Prompt):
```powershell
pip install -r requirements.txt
```

### Linux / Google Colab:
```bash
pip install -r requirements.txt
```

---

## 2. Checkpoint Locations

The dashboard automatically discovers and caches checkpoints from the `checkpoints/` directory.

Expected checkpoint files:
- `checkpoints/pilot_audio_best.pth` (Audio baseline)
- `checkpoints/pilot_video_best.pth` (Video baseline / visual encoder)
- `checkpoints/pilot_fusion_best.pth` (Standard Audio-Visual fusion)
- `checkpoints/pilot_rmsav_best.pth` (Proposed RMS-AV model)

If a checkpoint is missing, the dashboard will display a warning in the sidebar and disable modes that rely on that architecture.

---

## 3. How to Launch the Application

### Standard Local Execution (Windows / Mac / Linux):
```bash
streamlit run app.py
```
By default, the server will launch locally at `http://localhost:8501`.

### Google Colab Execution:
If running inside a Google Colab notebook, run:
```bash
streamlit run app.py --server.headless true &
npx localtunnel --port 8501
```
Or use the pre-built notebook cells in [colab_execution.ipynb](colab_execution.ipynb).

---

## 4. Supported Upload & Analysis Modes

### Mode A: Still Image (Visual-Only)
- **Supported Formats:** `.jpg`, `.jpeg`, `.png`
- **Model Used:** Visual encoder (ResNet18) and visual authenticity head from `pilot_video_best.pth`.
- **How it Works:** The image is resized to $224 \times 224$, normalized using ImageNet statistics, and processed through the visual backbone.
- **Important Note:** Image inference evaluates visual facial artifacts only. It is explicitly labeled as **Visual-Only** and does not perform audio-video synchronization checks.

### Mode B: Video Stream (Multi-Frame Visual)
- **Supported Formats:** `.mp4`, `.mov`, `.avi`
- **Model Used:** Video baseline model (`pilot_video_best.pth`).
- **How it Works:** Samples uniformly across the clip (default: 5 frames), normalizes frames, and passes them to the spatial-temporal visual backbone. Shows total video frame count and processed frame count.

### Mode C: Audio Track (Acoustic Spectral Analysis)
- **Supported Formats:** `.wav` (16-bit PCM mono or stereo)
- **Model Used:** 2D CNN acoustic model (`pilot_audio_best.pth`).
- **How it Works:** Ingests audio, resamples to 16 kHz if needed, computes an 80-channel log-mel spectrogram ($[1, 80, 400]$), and detects voice synthesis or manipulation artifacts.

### Mode D: Paired Audio + Video (RMS-AV & Fusion)
- **Supported Formats:** Video (`.mp4`, `.mov`, `.avi`) and Audio (`.wav`).
- **Model Options:**
  - `RMS-AV`: Dynamic reliability gate dynamically computes $g_a, g_v \in [0, 1]$ conditioned on modality features and validity masks. The UI renders the exact gate percentage distribution!
  - `Standard Fusion`: Concatenates features without reliability estimation.

---

## 5. Interpreting Predictions

- **Labels:**
  - 🛡️ **Likely authentic**: Model manipulation probability $< 50\%$.
  - ⚠️ **Likely manipulated**: Model manipulation probability $\ge 50\%$.
- **Uncertainty Warning:** When predicted probabilities fall within $45\% - 55\%$, the dashboard flags an **Uncertainty Warning**, alerting the researcher that the sample lies close to the decision boundary.
- **Probabilities vs. Confidence:** Predicted probabilities represent model decision scores and are not calibrated real-world Bayesian confidence values.
- **Reliability Gate Weights:** For RMS-AV, the interface displays the exact stream weighting ($g_a$ audio, $g_v$ video).

---

## 6. Testing & Validation

Run the dedicated automated inference test suite to verify all upload modes and error handling:
```bash
python test_inference.py
```
This tests:
1. Checkpoint discovery & parameter counting
2. Image preprocessing & visual inference
3. Audio extraction & acoustic CNN inference
4. Video decoding & uniform frame sampling
5. Paired multimodal inference & reliability gate weight extraction
6. Graceful error handling on empty or corrupt byte streams

---

## 7. Known Limitations & Troubleshooting

1. **Audio Format**: Only standard WAV format is natively decoded without external system ffmpeg binaries. Ensure audio uploads are uncompressed `.wav` files.
2. **Pilot Dataset Scope**: Checkpoints were trained on the 40-sample pilot subset with random encoder initialization. While the inference pipeline is fully functional, predictions reflect pilot-stage representations.
3. **Port In Use**: If port 8501 is occupied, launch on an alternative port:
   ```bash
   streamlit run app.py --server.port 8502
   ```
