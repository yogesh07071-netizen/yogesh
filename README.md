# Explainable Sleep Stage Classification from EEG Signals Using Attention-Based Deep Learning

An end-to-end, interpretable deep learning framework for automated sleep staging from multi-channel electroencephalogram (EEG) signals. The system combines multi-scale 1D Convolutional Neural Networks (1D-CNN), Bidirectional Long Short-Term Memory (BiLSTM) networks, and a Temporal Self-Attention mechanism to achieve high classification accuracy while providing neurophysiologically grounded explanations aligned with the American Academy of Sleep Medicine (AASM) standards.

---

## 🌟 Key Highlights

- **AASM Compliant Multi-Channel Signal Generation**: Synthesizes realistic 30-second multi-channel EEG/EOG epochs (`EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG Horizontal`) across 5 sleep stages: **Wake (W)**, **N1 (Light Sleep)**, **N2 (Intermediate Sleep)**, **N3 (Slow Wave Deep Sleep)**, and **REM (Rapid Eye Movement)**.
- **Deep Hybrid Architecture**:
  - *Multi-Scale 1D-CNN*: Dual receptive fields capturing both high-frequency micro-events (spindles, beta waves) and low-frequency slow oscillations (delta waves, K-complexes).
  - *BiLSTM*: Learns sequential temporal dynamics across sub-windows within the 30-second epoch.
  - *Temporal Self-Attention*: Highlights exactly which seconds of the epoch drove the stage prediction.
- **Explainability & Attribution Engine**:
  - Direct temporal attention weight extraction ($\alpha_t$).
  - Short-Time Fourier Transform (STFT) time-frequency decomposition.
  - Frequency band attribution quantifying relative contributions of **Delta (0.5–4 Hz)**, **Theta (4–8 Hz)**, **Alpha (8–12 Hz)**, and **Beta (12–30 Hz)**.
  - Automated natural language clinical narrative generation for sleep specialists.
- **Interactive CLI & Automated Pipeline**: Full menu-driven terminal interface to generate datasets, train models, evaluate with standard metrics, and export high-resolution diagnostic figures.

---

## 🧠 Neurophysiological Stage Criteria (AASM Manual)

| Sleep Stage | Dominant Frequency Bands | Characteristic EEG / EOG Micro-Events | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **Wake (W)** | Alpha (8–12 Hz) & Beta (13–30 Hz) | Posterior alpha rhythm with eyes closed, high-amplitude eye blinks, muscle activity. | Baseline alertness. |
| **N1** | Theta (4–7 Hz), attenuated Alpha | Low-amplitude mixed-frequency (LAMF), Vertex sharp waves ($<0.5$s), slow rolling eye movements. | Drowsiness & sleep onset. |
| **N2** | Theta (4–7 Hz) & Sigma/Beta | **Sleep Spindles** (12–14 Hz waxing/waning bursts) & **K-Complexes** ($>0.5$s biphasic waves $>100\mu\text{V}$). | Core non-REM sleep. |
| **N3 (SWS)** | Delta (0.5–2 Hz, $>75\mu\text{V}$) | Slow wave activity occupying $>20\%$ of the 30s epoch; synchronization across cortex. | Physical restoration & tissue recovery. |
| **REM** | Theta (4–8 Hz) & Beta | Sawtooth waves (2–6 Hz triangular waveforms), rapid eye movements (REM bursts), muscle atonia. | Memory consolidation & dreaming. |

---

## 🏗️ System Architecture

```
                      +---------------------------------------+
                      | Multi-Channel 30s EEG Epoch (B, 3, 3000)|
                      +---------------------------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
           [Fine Kernel Conv1D]                   [Coarse Kernel Conv1D]
             Kernel=7, Stride=2                     Kernel=31, Stride=2
          (Spindles, Beta, Spikes)                (Delta Waves, K-Complex)
                      |                                       |
                      +-------------------+-------------------+
                                          |
                                [Channel Concatenation]
                                          |
                             [Downsampling Conv Blocks]
                                          |
                       +--------------------------------------+
                       | Feature Sequence Tokens (B, 46, 128) |
                       +--------------------------------------+
                                          |
                            [Bidirectional LSTM Layer]
                                          |
                         [Temporal Self-Attention Layer]
                                 /                \
                                /                  \
                      [Context Vector]       [Attention Weights α_t]
                             |                         |
                    [Dense MLP Head]                   v
                             |              [Explainability Engine]
                             v              - STFT Spectrogram Overlay
                    [5-Stage Prediction]    - Delta/Theta/Alpha/Beta Power
                    (W, N1, N2, N3, REM)    - Clinical Narrative Report
```

---

## 📁 Repository Structure

```
.
├── app.py                 # Interactive Streamlit Web Application & Clinical Dashboard
├── data_generator.py      # Multi-channel synthetic EEG generator with AASM transients
├── model.py               # 1D-CNN + BiLSTM + Temporal Attention PyTorch implementation
├── explainability.py      # Attention extraction, STFT decomposition & clinical narrative
├── main.py                # Interactive CLI & automated training/evaluation pipeline
├── requirements.txt       # Python dependencies (includes streamlit)
├── README.md              # Project documentation & user guide
├── PROJECT_REPORT.md      # Full academic research paper and technical report
├── PROJECT_REPORT_YOGESH_25BTRCM058.pdf # Complete 65-page formatted university project report
└── outputs/               # Saved figures, heatmaps, and evaluation metrics (auto-generated)
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── spectral_band_importance.png
    ├── attention_analysis_stage_Wake.png
    ├── attention_analysis_stage_N1.png
    ├── attention_analysis_stage_N2.png
    ├── attention_analysis_stage_N3.png
    ├── attention_analysis_stage_REM.png
    └── evaluation_summary.json
```

---

## 🚀 Getting Started

### 1. Installation

Ensure Python 3.9+ is installed, then install project dependencies:

```bash
pip install -r requirements.txt
```

*(Note: PyTorch with CUDA support is automatically utilized if a compatible GPU is detected; otherwise, it executes efficiently on CPU).*

### 2. Streamlit Web Application (Interactive Clinical Dashboard)

Launch the modern browser-based web application:

```bash
streamlit run app.py
```

The web dashboard opens automatically in your browser (usually at `http://localhost:8501`) and features:
- **🔬 Single Epoch Live Diagnostic**: Simulate or upload 30s EEG epochs (Wake, N1, N2, N3, REM), execute real-time model inference, view synchronized multi-channel EEG leads, continuous STFT spectrograms, and temporal attention heatmap ribbons.
- **📈 Explainability & Attribution**: Inspect relative power contributions ($\delta, \theta, \alpha, \beta$), identify salient neurophysiological micro-events (Sleep Spindles, K-Complexes, Vertex Waves), and generate formatted AASM clinical narratives.
- **📊 Cohort Evaluation & Metrics**: Run multi-sample cohort tests to render normalized confusion matrices and precision/recall/F1-score tables.
- **🧠 Model Architecture & Math**: Review equations for self-attention, BiLSTM contextualization, and STFT-weighted band importance.
- **📥 Direct Download**: One-click download of the complete 65-page academic project report PDF (`PROJECT_REPORT_YOGESH_25BTRCM058.pdf`) and individual clinical text summaries.

### 3. Interactive CLI Mode

Launch the menu-driven command-line interface:

```bash
python main.py
```

You will be presented with interactive options:
```text
==================================================
   EXPLAINABLE SLEEP STAGE CLASSIFIER
==================================================
 [1] Generate Synthetic EEG Dataset (W, N1, N2, N3, REM)
 [2] Train Attention SleepNet Model (1D-CNN + BiLSTM + Attention)
 [3] Run Full Automated Pipeline (Train, Evaluate & Explain)
 [4] Run Explainability Analysis on Saved Test Epochs
 [5] Exit
==================================================
```

### 3. Automated Single-Command Execution

To execute the complete end-to-end pipeline non-interactively (ideal for evaluation scripts or headless servers):

```bash
python main.py --all --epochs 12 --samples_per_stage 80
```

---

## 🔍 Explainability & Clinical Interpretation

### 1. Attention Weight Formulation
Given the bidirectional LSTM hidden states $\mathbf{h}_1, \dots, \mathbf{h}_T$, the attention layer computes score $\alpha_t$:
$$\mathbf{u}_t = \tanh(\mathbf{W}_s \mathbf{h}_t + \mathbf{b}_s)$$
$$\alpha_t = \frac{\exp(\mathbf{u}_t^T \mathbf{v}_s)}{\sum_{k=1}^T \exp(\mathbf{u}_k^T \mathbf{v}_s)}$$

### 2. Frequency Band Attribution
To explain predictions beyond raw time stamps, the system computes the Short-Time Fourier Transform (STFT) $P(f, t)$ and correlates continuous attention $\alpha(t)$ with canonical frequency bands:
$$\text{Importance}(b) = \frac{\sum_t \alpha(t) \cdot \int_{f \in \text{band}_b} P(f, t) \, df}{\sum_{b'} \sum_t \alpha(t) \cdot \int_{f \in \text{band}_{b'}} P(f, t) \, df}$$

### 3. Sample Clinical Output
```text
================= CLINICAL EXPLAINABILITY REPORT =================
• Predicted Sleep Stage : N2 (Model Confidence: 95.82%)
• Ground Truth Reference: N2
• Key Temporal Focus    : Highest attention at t = 14.2s (Weight: 0.0892)
• Spectral Contributions: Beta: 42.1%, Theta: 31.8%, Delta: 16.4%, Alpha: 9.7%
• Diagnostic Rationale  : High temporal attention peak at t=14.2s, matching 
                          characteristic Sleep Spindle burst (12-14 Hz) and 
                          K-complex waveform. Co-dominant with Theta.
=================================================================
```

---

## 📊 Output Artifacts

Upon running the pipeline, the following artifacts are generated in `outputs/`:
- **`training_curves.png`**: Epoch-by-epoch loss reduction and accuracy improvement curves.
- **`confusion_matrix.png`**: Normalized 5x5 confusion matrix demonstrating inter-stage classification sensitivity.
- **`spectral_band_importance.png`**: Comparative bar graph demonstrating how the model prioritizes Delta in N3, Spindles/Beta in N2, and Alpha in Wake.
- **`attention_analysis_stage_{Stage}.png`**: Multi-panel diagnostic figures showing raw EEG leads with attention ribbons, STFT spectrogram, attention curves, and band importance charts.
- **`evaluation_summary.json`**: Machine-readable JSON summary of test accuracy and Cohen's Kappa score.

---

## 📜 Academic Project Report

For a complete in-depth academic study, theoretical derivations, related work comparison, and experimental discussion, refer to [PROJECT_REPORT.md](PROJECT_REPORT.md).
