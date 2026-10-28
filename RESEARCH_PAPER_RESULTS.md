# Empirical Evaluation and Research Paper Results: RMS-AV Deepfake Detection

**Document Purpose:** Formal LaTeX-Ready Results, Tables, and Discussion Section for Research Paper Submission  
**Method Investigated:** Reliability- and Manipulation-Aware Selective Audio-Visual Fusion (RMS-AV)  
**Evaluation Scope:** 40-Sample Verified Benchmark Audit (Teammates Pilot Dataset)  
**Audit Date:** October 28, 2026  

---

## 1. Methodology & Formal Formulation

Conventional multimodal audio-visual architectures fuse representations via unconstrained concatenation or linear projection:
$$\mathbf{z}_{\text{fused}} = \mathbf{W}[\mathbf{v}; \mathbf{a}] + \mathbf{b}$$

Under asymmetric forgeries (e.g., visual-only deepfakes with pristine audio) or modality corruption (e.g., missing streams, acoustic noise), this vanilla fusion permits uninformative or corrupted features to pollute the joint classification latent space.

### 1.1 Dynamic Reliability Gating
RMS-AV explicitly conditions fusion on modality validity and reliability. Given visual features $\mathbf{v} \in \mathbb{R}^D$ and acoustic features $\mathbf{a} \in \mathbb{R}^D$, with binary validity indicators $m_v, m_a \in \{0, 1\}$:

1. **Gating Representation:**
   $$\mathbf{s} = \mathbf{W}_2 \operatorname{ReLU}(\mathbf{W}_1 [\mathbf{v}; \mathbf{a}] + \mathbf{b}_1) + \mathbf{b}_2 \quad \in \mathbb{R}^2$$

2. **Validity-Masked Softmax Normalization:**
   $$s_i' = \begin{cases} s_i & \text{if } m_i = 1 \\ -\infty & \text{if } m_i = 0 \end{cases} \quad \text{for } i \in \{a, v\}$$
   $$[g_a, g_v] = \operatorname{softmax}([s_a', s_v'])$$
   If both streams are missing ($m_a = m_v = 0$), $g_a = g_v = 0$.

3. **Modality-Weighted Representation:**
   $$\mathbf{z}_{\text{RMS-AV}} = [g_a \mathbf{a}; g_v \mathbf{v}]$$

4. **Multi-Task Auxiliary Supervision:**
   To maintain unimodal forensic sensitivity, RMS-AV introduces auxiliary authenticity heads:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{av}}, y) + \lambda_{\text{aux}} \left( \mathcal{L}_{\text{BCE}}(\hat{y}_a, y_a) + \mathcal{L}_{\text{BCE}}(\hat{y}_v, y_v) \right)$$
   where $\lambda_{\text{aux}} = 0.2$, $y \in \{0, 1\}$ denotes overall manipulation, and $y_a, y_v \in \{0, 1\}$ denote stream-specific manipulation.

---

## 2. Experimental Setup & Verification

- **Dataset:** 40 verified samples from `teammates/zip2/` across 4 categories: FakeVideo-FakeAudio (10), FakeVideo-RealAudio (10), RealVideo-FakeAudio (10), and RealVideo-RealAudio (10).
- **Target Mapping:** RealVideo-RealAudio is authentic ($y = 0.0$); all three manipulated categories are fake ($y = 1.0$).
- **Stratified Partition:** Train ($N=24$), Validation ($N=8$), Test ($N=8$).
- **Identity Leakage Audit:** Cross-split analysis of all embedded subject IDs (target face, driving face, source audio) confirmed zero identity overlap between Train (36 IDs), Val (13 IDs), and Test (13 IDs).
- **Model Checkpoints:** Evaluated from optimal validation checkpoints trained with seed 42.

---

## 3. Benchmark Results and Analysis

### 3.1 Clean Condition Performance ($N=8$)

The held-out test set contains 6 Manipulated samples and 2 Authentic samples (75% manipulation prevalence).

| Model Architecture | Accuracy (ACC) | Balanced Acc (B-ACC) | Precision (Fake) | Recall (Fake) | Positive-Class F1 | Negative-Class F1 | **Macro-F1** | ROC-AUC | ECE (Conf) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Audio-Only Baseline** | 75.00% | 50.00% | 75.00% | 100.0% | 85.71% | 0.00% | **42.86%** | **0.9167** | 0.3193 |
| **Video-Only Baseline** | 75.00% | 50.00% | 75.00% | 100.0% | 85.71% | 0.00% | **42.86%** | 0.2500 | 0.2213 |
| **Standard AV Fusion** | 75.00% | 50.00% | 75.00% | 100.0% | 85.71% | 0.00% | **42.86%** | 0.5833 | 0.1116 |
| **RMS-AV (Proposed)** | 50.00% | 33.33% | 66.67% | 66.67% | 66.67% | 0.00% | **33.33%** | 0.3333 | **0.0012** |

#### LaTeX Table Source (Clean Performance)
```latex
\begin{table}[ht]
\centering
\caption{Performance comparison on held-out test set ($N=8$, 6 Fake, 2 Real) under clean conditions. All metrics computed with decision threshold $\tau=0.50$.}
\label{tab:clean_results}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model Mode} & \textbf{ACC (\%)} & \textbf{B-ACC (\%)} & \textbf{Prec$_{\text{pos}}$ (\%)} & \textbf{Rec$_{\text{pos}}$ (\%)} & \textbf{F1$_{\text{pos}}$} & \textbf{Macro-F1} & \textbf{ROC-AUC} & \textbf{ECE} \\
\midrule
Audio Baseline     & 75.00 & 50.00 & 75.00 & 100.00 & 0.8571 & 0.4286 & \textbf{0.9167} & 0.3193 \\
Video Baseline     & 75.00 & 50.00 & 75.00 & 100.00 & 0.8571 & 0.4286 & 0.2500          & 0.2213 \\
Standard AV Fusion & 75.00 & 50.00 & 75.00 & 100.00 & 0.8571 & 0.4286 & 0.5833          & 0.1116 \\
RMS-AV (Proposed)  & 50.00 & 33.33 & 66.67 & 66.67  & 0.6667 & 0.3333 & 0.3333          & \textbf{0.0012} \\
\bottomrule
\end{tabular}%
}
\end{table}
```

### 3.2 Confusion Matrix Breakdown

```
Audio Baseline:     Video Baseline:     Standard Fusion:    RMS-AV (Proposed):
Pred:   0   1       Pred:   0   1       Pred:   0   1       Pred:   0   1
True 0 [0,  2]      True 0 [0,  2]      True 0 [0,  2]      True 0 [0,  2]
True 1 [0,  6]      True 1 [0,  6]      True 1 [0,  6]      True 1 [2,  4]
```

### 3.3 Critical Metric Clarification: Macro-F1 vs. Positive-Class F1
Because 3 out of 4 manipulation categories map to the Fake class, the test partition exhibits a 75% positive class prior.
- All three baseline models converged to predict the majority class ($\hat{y} = 1$ for all 8 test samples).
- Consequently, their positive-class recall is 100% and precision is 75%, yielding a misleadingly high positive-class F1 of **85.71%**.
- However, their negative-class recall and precision are 0% ($TN = 0$). The true unweighted **Macro-F1** is:
  $$\text{Macro-F1} = \frac{F1_{\text{pos}} + F1_{\text{neg}}}{2} = \frac{0.8571 + 0.0000}{2} = 42.86\%$$
- RMS-AV avoided unconditional majority-class collapse, producing 2 negative predictions and 6 positive predictions, yielding $F1_{\text{pos}} = 66.67\%$ and $\text{Macro-F1} = 33.33\%$.

---

## 4. Modality-Drop Robustness Evaluation

To test hypothesis H1—whether selective reliability gating prevents catastrophic degradation when a stream is missing—we evaluate models under complete stream ablation.

| Model Mode | Condition | ACC (%) | B-ACC (%) | Macro-F1 | ROC-AUC | TN | FP | FN | TP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Audio Baseline** | Clean (`none`) | 75.00 | 50.00 | 0.4286 | 0.9167 | 0 | 2 | 0 | 6 |
| Audio Baseline | Missing Video | 75.00 | 50.00 | 0.4286 | 0.9167 | 0 | 2 | 0 | 6 |
| Audio Baseline | Missing Audio | 25.00 | 50.00 | 0.2000 | 0.5000 | 2 | 0 | 6 | 0 |
| **Video Baseline** | Clean (`none`) | 75.00 | 50.00 | 0.4286 | 0.2500 | 0 | 2 | 0 | 6 |
| Video Baseline | Missing Video | 25.00 | 50.00 | 0.2000 | 0.5000 | 2 | 0 | 6 | 0 |
| Video Baseline | Missing Audio | 75.00 | 50.00 | 0.4286 | 0.2500 | 0 | 2 | 0 | 6 |
| **Standard Fusion**| Clean (`none`) | 75.00 | 50.00 | 0.4286 | 0.5833 | 0 | 2 | 0 | 6 |
| Standard Fusion | Missing Video | 75.00 | 50.00 | 0.4286 | 0.9167 | 0 | 2 | 0 | 6 |
| Standard Fusion | Missing Audio | 75.00 | 50.00 | 0.4286 | 0.3333 | 0 | 2 | 0 | 6 |
| **RMS-AV (Proposed)**| Clean (`none`) | 50.00 | 33.33 | 0.3333 | 0.3333 | 0 | 2 | 2 | 4 |
| RMS-AV (Proposed) | Missing Video | 25.00 | 50.00 | 0.2000 | 0.4167 | 2 | 0 | 6 | 0 |
| RMS-AV (Proposed) | Missing Audio | 25.00 | 50.00 | 0.2000 | 0.5000 | 2 | 0 | 6 | 0 |

#### Dynamic Gate Adaptation:
In RMS-AV, the reliability gate adapts dynamically:
- Under `missing_video`: $g_v = 0.0000$ and $g_a = 1.0000$ for all samples.
- Under `missing_audio`: $g_a = 0.0000$ and $g_v = 1.0000$ for all samples.
- The validity masking effectively suppresses zero-padding artifacts, satisfying architectural design criteria.

---

## 5. Limitations & Threats to Validity

1. **Test Set Discretization Error:** An 8-sample test set has an inherent discretization step of $12.5\%$ per sample. These numbers serve as empirical proof-of-concept for pipeline compatibility and loss convergence, not as claims of statistical superiority.
2. **Untrained Foundation Weights:** Visual and audio encoders were trained from random initialization due to local sandbox constraints. On scaled runs (e.g., in Google Colab), initializing with pretrained ImageNet-1K / Wav2Vec weights is required.
3. **Pilot Conclusion:** RMS-AV's multi-task architecture requires sufficient training data and epochs to calibrate both its unimodal and fused classification heads. Claims of superiority over baselines are reserved for full-dataset training.

---

## 6. Reproduction Verification

All reported numbers are verified and reproducible:
- **Metrics Computation:** `python audit_recompute_and_plot.py`
- **Validation Test Suite:** `python test_pilot_audit.py` (7/7 tests passing)
- **Artifacts:** `results/pilot_predictions.csv`, `results/pilot_comparison.csv`, `results/pilot_results.json`, `results/confusion_matrices.png`, `results/roc_curves.png`, `results/precision_recall_curves.png`.
