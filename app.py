r"""
RMS-AV: Reliability- and Manipulation-Aware Selective Audio-Visual Fusion
========================================================================
Interactive Research Demonstration & Deepfake Detection Dashboard
"""

import io
import os
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference import InferenceEngine

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RMS-AV | Multimodal Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling: Charcoal / Dark Navy with Blue/Purple Accents
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global Background & Palette */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, #161b22 0%, #1f242c 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        color: #58a6ff;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .header-sub {
        color: #8b949e;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Result Cards */
    .result-card-fake {
        background: rgba(248, 81, 73, 0.1);
        border: 1px solid #f85149;
        border-radius: 8px;
        padding: 18px;
        margin-top: 15px;
    }
    .result-card-real {
        background: rgba(46, 160, 67, 0.1);
        border: 1px solid #2ea043;
        border-radius: 8px;
        padding: 18px;
        margin-top: 15px;
    }
    .result-title-fake {
        color: #f85149;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .result-title-real {
        color: #3fb950;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    /* Uncertainty Warning Box */
    .uncertainty-box {
        background: rgba(210, 153, 34, 0.15);
        border: 1px solid #d29922;
        border-radius: 6px;
        padding: 14px;
        color: #e3b341;
        font-size: 13px;
        margin-top: 12px;
        line-height: 1.5;
    }

    /* Reliability Gate Meter */
    .gate-meter {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        margin-top: 12px;
    }
    
    /* Disclaimers */
    .disclaimer-text {
        font-size: 12px;
        color: #8b949e;
        font-style: italic;
        margin-top: 8px;
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        background: #21262d;
        border: 1px solid #30363d;
        color: #c9d1d9;
        margin-right: 6px;
    }

    /* Custom HTML Table */
    table.custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 13px;
    }
    table.custom-table th, table.custom-table td {
        border: 1px solid #30363d;
        padding: 8px 12px;
        text-align: left;
    }
    table.custom-table th {
        background-color: #161b22;
        color: #58a6ff;
    }
    table.custom-table tr:nth-child(even) {
        background-color: #0d1117;
    }
    table.custom-table tr:nth-child(odd) {
        background-color: #161b22;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Caching via Streamlit Resource Cache
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing RMS-AV neural backbones & checkpoints...")
def get_inference_engine() -> InferenceEngine:
    return InferenceEngine(checkpoint_dir="checkpoints")


# -----------------------------------------------------------------------------
# Session State Initialization (History & Settings)
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def add_to_history(result: Dict):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": result.get("filename", result.get("video_filename", "media")),
        "media_type": result.get("media_type", "Unknown"),
        "mode_used": result.get("mode_used", "Unknown"),
        "prediction": result.get("prediction", "Unknown"),
        "manipulation_prob": f"{result.get('manipulation_probability', 0.0)*100:.2f}%",
        "raw_logit": result.get("raw_logit", 0.0),
        "threshold": result.get("threshold_applied", 0.50),
        "latency_ms": f"{result.get('latency_ms', 0.0):.1f} ms",
        "is_uncertain": result.get("is_uncertain", False),
    }
    st.session_state.history.append(entry)


def render_dataframe(df: pd.DataFrame, format_dict: Optional[Dict] = None):
    """Safely renders a dataframe with formatting, with automatic HTML fallback."""
    try:
        if format_dict:
            st.dataframe(df.style.format(format_dict))
        else:
            st.dataframe(df)
    except Exception:
        html_str = df.to_html(classes="custom-table", index=False)
        st.markdown(html_str, unsafe_allow_html=True)


def display_results(res: Dict):
    """Renders the forensic prediction card, probability gauges, and gate distribution."""
    is_fake = (res["prediction"] == "Likely manipulated")
    prob_fake = res["manipulation_probability"]
    prob_real = res["authenticity_probability"]

    card_class = "result-card-fake" if is_fake else "result-card-real"
    title_class = "result-title-fake" if is_fake else "result-title-real"
    icon = "⚠️" if is_fake else "🛡️"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="{title_class}">{icon} {res['prediction'].upper()}</div>
            <div style="font-size: 14px; color: #c9d1d9;">
                <b>Model Mode:</b> {res['mode_used'].upper()} &nbsp;|&nbsp; 
                <b>Raw Logit:</b> {res.get('raw_logit', 'N/A')} &nbsp;|&nbsp;
                <b>Latency:</b> {res['latency_ms']} ms
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("**Prediction Score Distribution:**")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("Manipulation Probability", f"{prob_fake * 100:.2f}%")
        st.progress(min(max(prob_fake, 0.0), 1.0))
    with col_p2:
        st.metric("Authenticity Probability", f"{prob_real * 100:.2f}%")
        st.progress(min(max(prob_real, 0.0), 1.0))

    # Prominent Uncertainty & Calibration Notice
    if res.get("is_uncertain"):
        st.markdown(
            f"""
            <div class="uncertainty-box">
                ⚠️ <b>High Model Uncertainty (Score: {prob_fake*100:.1f}%):</b><br>
                The predicted probability is near the 50% chance line (logit = {res.get('raw_logit', 0.0):.3f}).<br>
                <b>Why this happens on genuine images:</b> This pilot model was trained on only <b>24 pilot samples</b> with 
                <b>randomly initialized weights</b> (without ImageNet pretraining) and a <b>75% fake training prior</b>. 
                As verified in the pilot audit, the uncalibrated visual backbone outputs baseline scores around <b>51%–53%</b> for virtually all unseen faces.
                You can adjust the <i>Decision Cutoff</i> in the sidebar to > 0.54 to account for this pilot baseline bias.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # RMS-AV Dynamic Gate Breakdown
    if res.get("gate_weights") is not None:
        gates = res["gate_weights"]
        gw_a = gates["audio_weight"]
        gw_v = gates["video_weight"]
        st.markdown(
            f"""
            <div class="gate-meter">
                <b>Dynamic Reliability Gate Distribution (g<sub>a</sub>, g<sub>v</sub>):</b><br>
                <div style="margin-top: 6px; font-size: 13px;">
                    🎙️ <b>Audio Stream Weight:</b> {gw_a * 100:.2f}% &nbsp;|&nbsp; 
                    🎬 <b>Video Stream Weight:</b> {gw_v * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Technical Details Expander
    with st.expander("Inspection Metadata & Model Settings"):
        meta = {
            "Filename": res.get("filename", res.get("video_filename")),
            "Media Type": res.get("media_type"),
            "Inference Mode": res.get("mode_used"),
            "Raw Logit": res.get("raw_logit"),
            "Decision Threshold Applied": res.get("threshold_applied", 0.50),
            "Latency": f"{res.get('latency_ms')} ms",
            "Frames Analyzed": res.get("frames_analyzed", "N/A"),
            "Audio Duration": f"{res.get('duration_seconds')} s" if "duration_seconds" in res else "N/A",
        }
        st.json(meta)

    st.markdown(f"<div class='disclaimer-text'>ℹ️ {res.get('disclaimer', '')}</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ RMS-AV Research")
    st.caption("Multimodal Forensic Verification")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "🔬 Inference Dashboard",
            "📊 Benchmark & Metrics",
            "🧠 Architecture & RMS-AV",
            "📜 Prediction History",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Inference Controls")
    threshold = st.slider(
        "Decision Cutoff (Threshold)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="Cutoff above which a clip is classified as Manipulated. Pilot baseline prior hovers around 0.52.",
    )

    st.markdown("---")
    st.subheader("Checkpoint Status")
    engine = get_inference_engine()

    for m in ["audio", "video", "fusion", "rmsav"]:
        if engine.is_mode_available(m):
            info = engine.model_info[m]
            st.markdown(
                f"✅ **{m.upper()}**: Ready  \n"
                f"<span class='metric-badge'>{info['total_params']:,} params</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"❌ **{m.upper()}**: *Missing checkpoint*", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Deep Learning Research Lab • Pilot Release 2026")


# -----------------------------------------------------------------------------
# TAB 1: Inference Dashboard
# -----------------------------------------------------------------------------
if nav_selection == "🔬 Inference Dashboard":
    st.markdown(
        """
        <div class="header-box">
            <div class="header-title">RMS-AV Deepfake Detection Dashboard</div>
            <div class="header-sub">
                Reliability- and Manipulation-Aware Selective Audio-Visual Fusion.<br>
                Upload an image, video, audio, or paired stream to perform forensic inference using actual trained neural weights.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode_tab = st.selectbox(
        "Select Upload & Inference Mode:",
        [
            "Mode A: Still Image (Visual-Only)",
            "Mode B: Video Clip (Multi-Frame Visual)",
            "Mode C: Audio Track (Acoustic Log-Mel)",
            "Mode D: Paired Audio + Video (RMS-AV / Fusion)",
        ],
    )

    st.markdown("---")
    col_input, col_output = st.columns([1, 1], gap="large")

    # Mode A: Image Upload
    if "Mode A" in mode_tab:
        with col_input:
            st.subheader("1. Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a portrait or face image",
                type=["jpg", "jpeg", "png"],
                help="Accepts standard JPG, JPEG, and PNG images.",
            )

            if uploaded_file is not None:
                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    st.image(img, caption=f"Uploaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error opening image: {e}")
                    img = None
            else:
                img = None

            run_btn = st.button("Run Visual-Only Analysis", disabled=(uploaded_file is None), type="primary")

        with col_output:
            st.subheader("2. Forensic Analysis")
            if run_btn and uploaded_file is not None:
                with st.spinner("Processing image through visual backbone (ResNet18)..."):
                    try:
                        bytes_data = uploaded_file.getvalue()
                        res = engine.predict_image(bytes_data, filename=uploaded_file.name, threshold=threshold)
                        st.session_state.last_result = res
                        add_to_history(res)
                    except Exception as e:
                        st.error(f"Inference failure: {e}")

            if st.session_state.last_result and st.session_state.last_result.get("media_type") == "Still Image (Visual Only)":
                res = st.session_state.last_result
                display_results(res)

    # Mode B: Video Upload
    elif "Mode B" in mode_tab:
        with col_input:
            st.subheader("1. Upload Video")
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=["mp4", "mov", "avi"],
                help="Accepts MP4, MOV, or AVI video formats.",
            )

            num_frames = st.slider("Frames to extract & sample uniformly:", min_value=3, max_value=10, value=5)

            if uploaded_file is not None:
                st.video(uploaded_file)
            run_btn = st.button("Run Multi-Frame Video Analysis", disabled=(uploaded_file is None), type="primary")

        with col_output:
            st.subheader("2. Forensic Analysis")
            if run_btn and uploaded_file is not None:
                with st.spinner(f"Extracting {num_frames} frames & running video model..."):
                    try:
                        bytes_data = uploaded_file.getvalue()
                        res = engine.predict_video(bytes_data, filename=uploaded_file.name, num_frames=num_frames, threshold=threshold)
                        st.session_state.last_result = res
                        add_to_history(res)
                    except Exception as e:
                        st.error(f"Video inference failure: {e}")

            if st.session_state.last_result and st.session_state.last_result.get("media_type") == "Video Stream (Visual Frames)":
                res = st.session_state.last_result
                display_results(res)

    # Mode C: Audio Upload
    elif "Mode C" in mode_tab:
        with col_input:
            st.subheader("1. Upload Audio Track")
            uploaded_file = st.file_uploader(
                "Choose a WAV speech recording",
                type=["wav"],
                help="Accepts standard 16-bit PCM WAV audio files.",
            )

            if uploaded_file is not None:
                st.audio(uploaded_file)
            run_btn = st.button("Run Acoustic Spectral Analysis", disabled=(uploaded_file is None), type="primary")

        with col_output:
            st.subheader("2. Forensic Analysis")
            if run_btn and uploaded_file is not None:
                with st.spinner("Extracting 80-channel log-mel spectrogram & evaluating 2D CNN..."):
                    try:
                        bytes_data = uploaded_file.getvalue()
                        res = engine.predict_audio(bytes_data, filename=uploaded_file.name, threshold=threshold)
                        st.session_state.last_result = res
                        add_to_history(res)
                    except Exception as e:
                        st.error(f"Audio inference failure: {e}")

            if st.session_state.last_result and st.session_state.last_result.get("media_type") == "Audio Track (Log-Mel Spectrogram)":
                res = st.session_state.last_result
                display_results(res)

    # Mode D: Paired Audio + Video Upload
    elif "Mode D" in mode_tab:
        with col_input:
            st.subheader("1. Upload Paired Streams")
            up_vid = st.file_uploader("Video Track (MP4 / MOV)", type=["mp4", "mov", "avi"], key="pair_vid")
            up_aud = st.file_uploader("Audio Track (WAV)", type=["wav"], key="pair_aud")

            fusion_arch = st.radio(
                "Fusion Architecture:",
                ["rmsav", "fusion"],
                format_func=lambda x: "RMS-AV (Dynamic Reliability Gate)" if x == "rmsav" else "Standard AV Fusion (Concatenation)",
            )

            ready = (up_vid is not None and up_aud is not None)
            run_btn = st.button("Run Multimodal Fusion Analysis", disabled=(not ready), type="primary")

        with col_output:
            st.subheader("2. Multimodal Fusion Results")
            if run_btn and ready:
                with st.spinner(f"Extracting visual & acoustic streams, running {fusion_arch.upper()}..."):
                    try:
                        v_bytes = up_vid.getvalue()
                        a_bytes = up_aud.getvalue()
                        res = engine.predict_audio_video(
                            v_bytes,
                            a_bytes,
                            video_filename=up_vid.name,
                            audio_filename=up_aud.name,
                            mode=fusion_arch,
                            threshold=threshold,
                        )
                        st.session_state.last_result = res
                        add_to_history(res)
                    except Exception as e:
                        st.error(f"Multimodal inference failure: {e}")

            if st.session_state.last_result and st.session_state.last_result.get("media_type") == "Paired Multimodal (Audio + Video)":
                res = st.session_state.last_result
                display_results(res)


# -----------------------------------------------------------------------------
# TAB 2: Benchmark & Research Metrics
# -----------------------------------------------------------------------------
elif nav_selection == "📊 Benchmark & Metrics":
    st.markdown(
        """
        <div class="header-box">
            <div class="header-title">Pilot Benchmark & Metric Audit</div>
            <div class="header-sub">
                Empirical evaluation across all four architectures evaluated on the verified held-out test split (N=8).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comp_csv = "results/pilot_comparison.csv"
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        df_clean = df_comp[df_comp["condition"] == "none"].copy()

        st.subheader("1. Clean Test Performance Comparison (Condition: 'none')")
        display_cols = [
            "mode", "accuracy", "balanced_accuracy", "precision_positive",
            "recall_positive", "f1_positive", "f1_macro", "roc_auc", "true_positives", "true_negatives"
        ]
        available_cols = [c for c in display_cols if c in df_clean.columns]
        df_display = df_clean[available_cols].rename(columns={
            "mode": "Mode",
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced Acc",
            "precision_positive": "Precision (Fake)",
            "recall_positive": "Recall (Fake)",
            "f1_positive": "F1 (Fake)",
            "f1_macro": "True Macro-F1",
            "roc_auc": "ROC-AUC",
            "true_positives": "TP",
            "true_negatives": "TN",
        })
        render_dataframe(df_display, {
            "Accuracy": "{:.2%}",
            "Balanced Acc": "{:.2%}",
            "Precision (Fake)": "{:.2%}",
            "Recall (Fake)": "{:.2%}",
            "F1 (Fake)": "{:.4f}",
            "True Macro-F1": "{:.4f}",
            "ROC-AUC": "{:.4f}",
        })

        st.markdown(
            """
            > **Statistical Note on Pilot Evaluation**:
            > - **Sample Size**: Evaluating on $N=8$ test samples yields a discrete granularity of **12.5% per sample**.
            > - **Majority Prior**: Due to the 3:1 mapping of deepfake variants to pristine videos, standard baselines collapsed to majority-class prediction ($TP=6, FP=2, TN=0$), yielding 75% raw accuracy but 50% Balanced Accuracy.
            > - **Macro-F1 vs Positive F1**: The true Macro-F1 accounts for both classes ($F1_{\text{real}} = 0.0$), yielding 42.86% for baselines and 33.33% for RMS-AV.
            """
        )

        st.markdown("---")
        st.subheader("2. Publication Figures & Evaluation Visualizations")
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            if os.path.exists("results/confusion_matrices.png"):
                st.image("results/confusion_matrices.png", caption="Confusion Matrices Across Architectures")
            if os.path.exists("results/pilot_metrics.png"):
                st.image("results/pilot_metrics.png", caption="Benchmark Metrics & Modality-Drop Robustness")

        with col_f2:
            if os.path.exists("results/roc_curves.png"):
                st.image("results/roc_curves.png", caption="Receiver Operating Characteristic (ROC) Curves")
            if os.path.exists("results/precision_recall_curves.png"):
                st.image("results/precision_recall_curves.png", caption="Precision-Recall Curves (Fake Class)")
    else:
        st.warning("No benchmark results file found at `results/pilot_comparison.csv`.")


# -----------------------------------------------------------------------------
# TAB 3: Architecture & RMS-AV
# -----------------------------------------------------------------------------
elif nav_selection == "🧠 Architecture & RMS-AV":
    st.markdown(
        """
        <div class="header-box">
            <div class="header-title">RMS-AV Architectural Formulation</div>
            <div class="header-sub">
                Reliability- and Manipulation-Aware Selective Audio-Visual Fusion
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. Core Concept & Motivation")
    st.markdown(
        r"""
        Multimodal deepfakes frequently manipulate only a single modality (e.g., visual face swap with genuine voice, or audio voice clone with genuine face). 
        Standard fusion mechanisms ($[\mathbf{v}; \mathbf{a}]$) allow clean features from the unmanipulated stream to mask subtle forensic cues in the forged stream.

        **RMS-AV introduces three specialized innovations:**
        1. **Unimodal Authenticity Heads:** Direct auxiliary supervision on visual ($\hat{y}_v$) and acoustic ($\hat{y}_a$) heads prevents forensic sensitivity decay.
        2. **Dynamic Reliability Gate:** Evaluates modality quality confidence and validity masks, allocating dynamic weights $\mathbf{g}_a, \mathbf{g}_v \in [0, 1]$ where $\mathbf{g}_a + \mathbf{g}_v = 1$.
        3. **Validity Masking:** Missing or corrupted streams ($mask=False$) are strictly suppressed ($-\infty$ prior to softmax), directing 100% of weight to the active modality without zero-padded corruption.
        """
    )

    st.subheader("2. Mathematical Formulation")
    st.latex(r"\mathbf{s} = \mathbf{W}_2 \operatorname{ReLU}(\mathbf{W}_1 [\mathbf{v}; \mathbf{a}] + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^2")
    st.latex(r"s_i' = \begin{cases} s_i & \text{if } m_i = 1 \\ -\infty & \text{if } m_i = 0 \end{cases} \quad \implies \quad [g_a, g_v] = \operatorname{softmax}([s_a', s_v'])")
    st.latex(r"\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{av}}, y) + \lambda_{\text{aux}} \left( \mathcal{L}_{\text{BCE}}(\hat{y}_a, y_a) + \mathcal{L}_{\text{BCE}}(\hat{y}_v, y_v) \right)")

    st.subheader("3. Model Configurations & Parameters")
    engine = get_inference_engine()
    param_data = []
    for m in ["audio", "video", "fusion", "rmsav"]:
        if m in engine.model_info:
            info = engine.model_info[m]
            param_data.append({
                "Architecture": m.upper(),
                "Total Parameters": f"{info['total_params']:,}",
                "Trainable Parameters": f"{info['trainable_params']:,}",
                "Latent Dimension": info["dim"],
                "Checkpoint Path": os.path.basename(info["checkpoint_path"]),
            })
    render_dataframe(pd.DataFrame(param_data))


# -----------------------------------------------------------------------------
# TAB 4: Prediction History & Privacy
# -----------------------------------------------------------------------------
elif nav_selection == "📜 Prediction History":
    st.markdown(
        """
        <div class="header-box">
            <div class="header-title">Session Prediction History</div>
            <div class="header-sub">
                In-memory log of inferences performed during this session.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        render_dataframe(df_hist)

        col_h1, col_h2 = st.columns([1, 4])
        with col_h1:
            if st.button("🗑️ Clear Session History"):
                st.session_state.history = []
                st.rerun()
        with col_h2:
            csv_buf = io.StringIO()
            df_hist.to_csv(csv_buf, index=False)
            st.download_button(
                "📥 Export History (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"rms_av_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    else:
        st.info("No inferences have been executed in this session yet.")

    st.markdown("---")
    st.caption("🔒 **Privacy Guarantee:** Uploaded media is processed temporarily in memory / local temp storage and immediately unlinked. No data is stored permanently or sent to external servers.")
