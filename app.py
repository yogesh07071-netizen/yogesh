"""
app.py - NeuroSleep AI: Streamlit Web Application
=================================================
Explainable Sleep Stage Classification from EEG Signals Using Attention-Based Deep Learning

Student: YOGESH.L (USN: 25BTRCM058)
Institution: School of Computer Science & Engineering, Faculty of Engineering & Technology,
             JAIN (Deemed-to-be University)
Course: 25PCCS202 - Python Programming (III Semester)
Guide: Dr. B. Narmada, Assistant Professor
"""

import os
import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import streamlit as st

# Local project imports
from data_generator import (
    STAGE_NAMES,
    STAGE_TO_LABEL,
    LABEL_TO_STAGE,
    CHANNEL_NAMES,
    FREQUENCY_BANDS,
    generate_single_epoch,
    generate_dataset
)
from model import (
    TORCH_AVAILABLE,
    ExplainableSleepNet,
    FallbackSpectralSleepClassifier
)
from explainability import (
    correlate_attention_with_bands,
    generate_clinical_explanation,
    compute_stft_spectrogram,
    compute_band_power,
    BAND_COLORS
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

# ---------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="NeuroSleep AI | Explainable Sleep Staging",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1a365d;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #4a5568;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .badge-stage {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        color: white;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }
    .badge-wake { background-color: #d97706; }
    .badge-n1 { background-color: #0284c7; }
    .badge-n2 { background-color: #2563eb; }
    .badge-n3 { background-color: #4f46e5; }
    .badge-rem { background-color: #059669; }
    .report-box {
        background-color: #0f172a;
        color: #38bdf8;
        font-family: 'Consolas', 'Courier New', monospace;
        padding: 18px;
        border-radius: 8px;
        border-left: 5px solid #38bdf8;
        white-space: pre-wrap;
        line-height: 1.45;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Model Initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classification_model():
    """
    Initializes and caches the attention-based deep learning model
    or fallback spectral classifier.
    """
    if TORCH_AVAILABLE:
        model = ExplainableSleepNet(in_channels=3, num_classes=5, lstm_hidden=64)
        model.eval()
        return model, "PyTorch Deep Learning (1D-CNN + BiLSTM + Self-Attention)"
    else:
        model = FallbackSpectralSleepClassifier(n_classes=5)
        return model, "Spectral Rule Heuristic Baseline (NumPy/SciPy)"


def run_model_inference(model, epoch_signal: np.ndarray):
    """
    Performs inference on a single 3-channel epoch signal (3, 3000).
    Returns (pred_idx, probas, attention_weights).
    """
    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        tensor_x = torch.from_numpy(epoch_signal).unsqueeze(0).float()
        with torch.no_grad():
            logits, attn = model(tensor_x)
            probas = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            attention_weights = attn.squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probas))
        return pred_idx, probas, attention_weights
    else:
        input_batch = epoch_signal[np.newaxis, :, :]  # (1, 3, 3000)
        preds, probas, attentions = model.predict_with_attention(input_batch)
        return int(preds[0]), probas[0], attentions[0]


# ---------------------------------------------------------
# Sidebar: Project Identity & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=70)
    st.markdown("### **NeuroSleep AI**")
    st.caption("**Explainable Sleep Stage Classifier**")
    st.markdown("---")
    
    st.markdown("**🎓 Academic Details**")
    st.markdown("""
    - **Student**: `YOGESH.L`
    - **USN**: `25BTRCM058`
    - **Course**: `25PCCS202 - Python Programming`
    - **Faculty Guide**: `Dr. B. Narmada`
    - **Program Head**: `Dr. Rajapraveen`
    - **Institution**: `School of CS & Engineering (CTMA), FET, JAIN (Deemed-to-be University)`
    """)
    st.markdown("---")

    model, model_type_str = load_classification_model()
    st.info(f"**Engine Active:**\n{model_type_str}")

    st.markdown("---")
    st.markdown("**📄 Project Documentation**")
    pdf_path = "PROJECT_REPORT_YOGESH_25BTRCM058.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            st.download_button(
                label="📥 Download 65-Page Report (PDF)",
                data=pdf_bytes,
                file_name="PROJECT_REPORT_YOGESH_25BTRCM058.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.caption("Report PDF not found in root directory.")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🌙 NeuroSleep AI: Explainable Sleep Staging</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Channel EEG/EOG Polysomnography Analysis using 1D-CNN + BiLSTM + Temporal Self-Attention</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Single Epoch Live Diagnostic",
    "📊 Cohort Evaluation & Metrics",
    "🧠 Model Architecture & Math",
    "📑 Project Report & AASM Standards"
])


# =========================================================
# TAB 1: Single Epoch Live Diagnostic & Explainability
# =========================================================
with tab1:
    st.subheader("1. Epoch Signal Acquisition & Generation")
    
    col_input_1, col_input_2, col_input_3 = st.columns([1.2, 1.2, 1.6])
    
    with col_input_1:
        data_source = st.radio(
            "Input Mode:",
            ["Synthetic AASM Generator", "Custom File Upload (.npz, .npy)"],
            index=0
        )
        
    with col_input_2:
        if data_source == "Synthetic AASM Generator":
            selected_stage = st.selectbox(
                "Target Sleep Stage to Simulate:",
                STAGE_NAMES,
                index=2  # Default N2 (Sleep Spindles & K-Complexes)
            )
            random_seed = st.number_input("Random Generator Seed:", min_value=1, max_value=9999, value=42, step=1)
        else:
            uploaded_file = st.file_uploader("Upload 3-Channel EEG NumPy file", type=['npy', 'npz'])
            selected_stage = None
            random_seed = None

    with col_input_3:
        st.markdown("**AASM Clinical Characteristic Guide:**")
        if selected_stage == 'Wake':
            st.caption("🟡 **Wake**: Posterior Alpha (8-12 Hz) attenuation on eye closure, rapid beta, high ocular blink spikes.")
        elif selected_stage == 'N1':
            st.caption("🔵 **N1**: Light transition sleep, Alpha drops <50%, low-amplitude Theta (4-7 Hz), Vertex sharp waves.")
        elif selected_stage == 'N2':
            st.caption("🔷 **N2**: Baseline sleep, dominant Theta with classical **Sleep Spindles (12-14 Hz)** and biphasic **K-Complexes**.")
        elif selected_stage == 'N3':
            st.caption("🟣 **N3**: Slow Wave Sleep (SWS), high-amplitude Delta waves (0.5-2 Hz, >75 µV) across >20% of epoch.")
        elif selected_stage == 'REM':
            st.caption("🟢 **REM**: Paradoxical sleep, desynchronized LAMF waves, transient Sawtooth bursts (2-6 Hz), rapid eye kinetics.")

    # Generate or load signal
    if data_source == "Synthetic AASM Generator":
        rng = np.random.RandomState(random_seed)
        epoch_signal = generate_single_epoch(selected_stage, fs=100, duration=30.0, rng=rng)
        ground_truth_label = STAGE_TO_LABEL[selected_stage]
    else:
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.npy'):
                    epoch_signal = np.load(uploaded_file)
                else:
                    loaded = np.load(uploaded_file)
                    epoch_signal = loaded['X'] if 'X' in loaded else loaded[loaded.files[0]]
                if epoch_signal.ndim == 3:
                    epoch_signal = epoch_signal[0]
                ground_truth_label = None
                st.success(f"Loaded signal with shape: {epoch_signal.shape}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                epoch_signal = generate_single_epoch('N2', fs=100, duration=30.0)
                ground_truth_label = STAGE_TO_LABEL['N2']
        else:
            st.info("Awaiting file upload. Defaulting to synthetic N2 epoch.")
            epoch_signal = generate_single_epoch('N2', fs=100, duration=30.0)
            ground_truth_label = STAGE_TO_LABEL['N2']

    # -----------------------------------------------------
    # Execute Model Prediction
    # -----------------------------------------------------
    pred_idx, probas, attention_weights = run_model_inference(model, epoch_signal)
    pred_stage_name = STAGE_NAMES[pred_idx]
    confidence = float(probas[pred_idx])

    st.markdown("---")
    st.subheader("2. Model Inference & Diagnostic Classification")

    # Prediction Summary Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    
    stage_colors = {
        'Wake': 'badge-wake',
        'N1': 'badge-n1',
        'N2': 'badge-n2',
        'N3': 'badge-n3',
        'REM': 'badge-rem'
    }

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#64748b; font-weight:600;">PREDICTED STAGE</span><br>
            <div style="margin-top:8px;">
                <span class="badge-stage {stage_colors[pred_stage_name]}">{pred_stage_name}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#64748b; font-weight:600;">MODEL CONFIDENCE</span><br>
            <span style="font-size:1.6rem; font-weight:700; color:#1e293b;">{confidence*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        gt_text = STAGE_NAMES[ground_truth_label] if ground_truth_label is not None else "Unknown"
        is_match = (ground_truth_label == pred_idx) if ground_truth_label is not None else True
        match_color = "#16a34a" if is_match else "#dc2626"
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#64748b; font-weight:600;">GROUND TRUTH</span><br>
            <span style="font-size:1.6rem; font-weight:700; color:{match_color};">{gt_text}</span>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        peak_t = np.linspace(0, 30.0, len(attention_weights))[np.argmax(attention_weights)]
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#64748b; font-weight:600;">PEAK ATTENTION TIME</span><br>
            <span style="font-size:1.6rem; font-weight:700; color:#0284c7;">t = {peak_t:.1f}s</span>
        </div>
        """, unsafe_allow_html=True)

    # Probability Distribution Bar Chart
    st.write("")
    col_plot1, col_plot2 = st.columns([1.5, 1])
    
    with col_plot1:
        st.markdown("**Softmax Probability Distribution Across All Sleep Stages:**")
        df_probs = pd.DataFrame({
            "Sleep Stage": STAGE_NAMES,
            "Probability (%)": probas * 100.0
        })
        st.bar_chart(df_probs.set_index("Sleep Stage"), color="#2563eb", height=240)

    with col_plot2:
        st.markdown("**Relative Spectral Band Distribution (Channel: Fpz-Cz):**")
        band_powers = compute_band_power(epoch_signal[0], fs=100)
        df_bands = pd.DataFrame({
            "Frequency Band": list(band_powers.keys()),
            "Relative Power (%)": [v * 100.0 for v in band_powers.values()]
        })
        st.dataframe(df_bands.style.format({"Relative Power (%)": "{:.1f}%"}), use_container_width=True)

    # -----------------------------------------------------
    # Multi-Panel Comprehensive Diagnostic Figure
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("3. Comprehensive Multi-Channel Attention & Spectral Explainability")
    
    fs = 100
    n_channels, n_samples = epoch_signal.shape
    t_signal = np.linspace(0, 30.0, n_samples)
    t_attn = np.linspace(0, 30.0, len(attention_weights))
    
    band_importance = correlate_attention_with_bands(epoch_signal[0], attention_weights, fs=fs)

    fig = plt.figure(figsize=(14, 10), facecolor='#ffffff')
    gs = gridspec.GridSpec(4, 2, height_ratios=[2.2, 1.4, 0.9, 1.2], width_ratios=[2.5, 1.0], hspace=0.38, wspace=0.25)

    # Panel 1: Multi-Channel EEG Traces
    ax_raw = fig.add_subplot(gs[0, :])
    ax_raw.set_title(
        f"Multi-Channel Polysomnography (30s Epoch) | Pred: {pred_stage_name} ({confidence*100:.1f}%)" + 
        (f" | True: {STAGE_NAMES[ground_truth_label]}" if ground_truth_label is not None else ""),
        fontsize=12, fontweight='bold', pad=8
    )
    
    channel_colors = ['#1e3a8a', '#0369a1', '#9a3412']
    offsets = [3.5, 0.0, -3.5]
    for ch_idx in range(min(n_channels, 3)):
        offset = offsets[ch_idx]
        ch_sig = epoch_signal[ch_idx] + offset
        ax_raw.plot(t_signal, ch_sig, color=channel_colors[ch_idx], lw=0.85, label=CHANNEL_NAMES[ch_idx])
        ax_raw.text(0.2, offset + 0.9, CHANNEL_NAMES[ch_idx], fontsize=9, fontweight='bold', color=channel_colors[ch_idx])

    # Attention Heatmap Strip
    ax_raw.pcolormesh(
        t_attn, [-5.0, -4.2], attention_weights[np.newaxis, :],
        cmap='YlOrRd', alpha=0.85, shading='auto'
    )
    ax_raw.text(0.2, -4.7, "Attention Intensity (Salience)", fontsize=8, color='black', fontweight='bold')
    ax_raw.set_xlim(0, 30.0)
    ax_raw.set_ylim(-5.5, 5.5)
    ax_raw.set_xlabel("Time (seconds)", fontsize=9)
    ax_raw.grid(True, linestyle='--', alpha=0.3)

    # Panel 2: Continuous STFT Spectrogram
    ax_spec = fig.add_subplot(gs[1, 0])
    f_stft, t_stft, pxx = compute_stft_spectrogram(epoch_signal[0], fs=fs)
    mesh = ax_spec.pcolormesh(t_stft, f_stft, 10 * np.log10(pxx + 1e-8), cmap='magma', shading='gouraud')
    ax_spec.set_ylabel("Freq (Hz)", fontsize=9, fontweight='bold')
    ax_spec.set_ylim(0.5, 25.0)
    ax_spec.set_xlim(0, 30.0)
    ax_spec.set_title("Short-Time Fourier Transform (STFT) Spectrogram (EEG Fpz-Cz)", fontsize=10, fontweight='bold')
    plt.colorbar(mesh, ax=ax_spec, orientation='vertical', pad=0.02, aspect=10).set_label("PSD (dB)", fontsize=8)

    # Panel 3: Temporal Attention Weight Curve
    ax_att = fig.add_subplot(gs[2, 0], sharex=ax_spec)
    ax_att.plot(t_attn, attention_weights, color='#dc2626', lw=2.0, marker='o', markersize=3, label="Attention Weight alpha_t")
    ax_att.fill_between(t_attn, attention_weights, color='#fee2e2', alpha=0.7)
    ax_att.set_ylabel("Attention alpha", fontsize=9, fontweight='bold')
    ax_att.set_xlabel("Epoch Time (seconds)", fontsize=9, fontweight='bold')
    ax_att.set_xlim(0, 30.0)
    ax_att.grid(True, linestyle=':', alpha=0.4)

    # Panel 4: Frequency Band Importance Bar Chart
    ax_bands = fig.add_subplot(gs[1:3, 1])
    sorted_bands = sorted(band_importance.items(), key=lambda x: x[1], reverse=False)
    b_names = [b[0] for b in sorted_bands]
    b_vals = [b[1] * 100 for b in sorted_bands]
    bar_cols = [BAND_COLORS.get(b, '#4f46e5') for b in b_names]
    
    bars = ax_bands.barh(b_names, b_vals, color=bar_cols, height=0.55, edgecolor='#1e293b')
    ax_bands.set_xlabel("Band Importance (%)", fontsize=9, fontweight='bold')
    ax_bands.set_title("Attention-Weighted\nBand Attribution", fontsize=10, fontweight='bold')
    ax_bands.set_xlim(0, max(b_vals) * 1.25 if max(b_vals) > 0 else 100)
    for bar, val in zip(bars, b_vals):
        ax_bands.text(val + 1.0, bar.get_y() + bar.get_height()/2.0, f"{val:.1f}%", va='center', fontsize=8, fontweight='bold')
    ax_bands.grid(axis='x', linestyle='--', alpha=0.3)

    # Panel 5: Model Softmax Probability vs Ground Truth
    ax_prob = fig.add_subplot(gs[3, :])
    x_pos = np.arange(len(STAGE_NAMES))
    p_colors = ['#d97706' if s == 'Wake' else '#0284c7' if s == 'N1' else '#2563eb' if s == 'N2' else '#4f46e5' if s == 'N3' else '#059669' for s in STAGE_NAMES]
    bars_p = ax_prob.bar(x_pos, probas * 100, color=p_colors, width=0.45, edgecolor='#0f172a')
    ax_prob.set_xticks(x_pos)
    ax_prob.set_xticklabels(STAGE_NAMES, fontsize=9, fontweight='bold')
    ax_prob.set_ylabel("Softmax Conf (%)", fontsize=9, fontweight='bold')
    ax_prob.set_ylim(0, 115)
    ax_prob.set_title("Stage Predictive Confidence Distribution", fontsize=10, fontweight='bold')
    for bar, val in zip(bars_p, probas * 100):
        if val > 1.0:
            ax_prob.text(bar.get_x() + bar.get_width()/2.0, val + 2.5, f"{val:.1f}%", ha='center', fontsize=8, fontweight='bold')
    ax_prob.grid(axis='y', linestyle='--', alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------------------------
    # Automated Clinical Narrative Report Box
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("4. Automated AASM Clinical Explainability Report")
    
    clinical_report = generate_clinical_explanation(
        pred_idx=pred_idx,
        confidence=confidence,
        band_importance=band_importance,
        attention_weights=attention_weights,
        duration=30.0,
        true_idx=ground_truth_label
    )
    
    st.markdown(f'<div class="report-box">{clinical_report}</div>', unsafe_allow_html=True)
    
    st.download_button(
        label="📥 Download Clinical Diagnostic Report (.txt)",
        data=clinical_report,
        file_name=f"clinical_explanation_epoch_{pred_stage_name}.txt",
        mime="text/plain"
    )


# =========================================================
# TAB 2: Cohort Evaluation & Performance Metrics
# =========================================================
with tab2:
    st.subheader("📊 Cohort Evaluation & Performance Benchmark")
    st.markdown("Evaluate model generalizability across multi-epoch sleep cohorts.")

    col_bench_1, col_bench_2 = st.columns([1, 2])

    with col_bench_1:
        n_epochs = st.slider("Samples Per Stage for Cohort Test:", min_value=10, max_value=80, value=30, step=10)
        run_benchmark_btn = st.button("🚀 Run Cohort Benchmark Test", type="primary", use_container_width=True)

    if run_benchmark_btn or 'cohort_results' in st.session_state:
        if run_benchmark_btn:
            with st.spinner(f"Generating synthetic cohort ({n_epochs * 5} total epochs) and evaluating..."):
                t_start = time.time()
                X_cohort, y_cohort, _ = generate_dataset(n_samples_per_stage=n_epochs, fs=100, duration=30.0, random_state=123)
                
                preds = []
                for i in range(len(y_cohort)):
                    p_idx, _, _ = run_model_inference(model, X_cohort[i])
                    preds.append(p_idx)
                
                y_preds = np.array(preds)
                y_trues = y_cohort
                elapsed = time.time() - t_start
                
                # Compute metrics
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
                acc = accuracy_score(y_trues, y_preds)
                kappa = cohen_kappa_score(y_trues, y_preds)
                cm = confusion_matrix(y_trues, y_preds, labels=[0, 1, 2, 3, 4])
                report_dict = classification_report(y_trues, y_preds, target_names=STAGE_NAMES, output_dict=True)

                st.session_state['cohort_results'] = {
                    'acc': acc,
                    'kappa': kappa,
                    'cm': cm,
                    'report_dict': report_dict,
                    'elapsed': elapsed,
                    'total': len(y_trues)
                }

        res = st.session_state['cohort_results']

        with col_bench_2:
            mb1, mb2, mb3 = st.columns(3)
            with mb1:
                st.metric("Overall Accuracy", f"{res['acc']*100:.2f}%")
            with mb2:
                st.metric("Cohen's Kappa (kappa)", f"{res['kappa']:.3f}")
            with mb3:
                st.metric("Evaluated Epochs", f"{res['total']} (in {res['elapsed']:.2f}s)")

        st.markdown("---")
        c_cm1, c_cm2 = st.columns([1.2, 1.4])

        with c_cm1:
            st.markdown("**Confusion Matrix (Normalized %):**")
            cm_norm = res['cm'].astype('float') / res['cm'].sum(axis=1)[:, np.newaxis] * 100.0
            
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            import seaborn as sns
            sns.heatmap(
                cm_norm, annot=True, fmt='.1f', cmap='Blues', cbar=False,
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES, ax=ax_cm,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'}
            )
            ax_cm.set_xlabel("Predicted Sleep Stage", fontweight='bold')
            ax_cm.set_ylabel("True Sleep Stage", fontweight='bold')
            ax_cm.set_title("Sleep Stage Confusion Matrix (%)", fontweight='bold', pad=10)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with c_cm2:
            st.markdown("**Per-Stage Precision, Recall, and F1-Scores:**")
            rep = res['report_dict']
            table_data = []
            for stg in STAGE_NAMES:
                table_data.append({
                    "Stage": stg,
                    "Precision (%)": f"{rep[stg]['precision']*100:.1f}%",
                    "Recall (%)": f"{rep[stg]['recall']*100:.1f}%",
                    "F1-Score (%)": f"{rep[stg]['f1-score']*100:.1f}%",
                    "Epochs Tested": int(rep[stg]['support'])
                })
            st.table(pd.DataFrame(table_data))


# =========================================================
# TAB 3: Model Architecture & Mathematical Formulations
# =========================================================
with tab3:
    st.subheader("🧠 Deep Learning Architecture & Mathematical Formulations")
    st.markdown("""
    The system utilizes a hybrid **1D-CNN + BiLSTM + Scaled Additive Self-Attention** network designed specifically
    for time-series electrophysiological signals.
    """)

    col_arch1, col_arch2 = st.columns([1.3, 1])

    with col_arch1:
        st.markdown("### Architectural Pipeline")
        st.markdown("""
        1. **Dual-Branch 1D-CNN Feature Extractor**:
           - **Branch A (Fine-Grained)**: Kernel size $k_1 = 7$, stride $s_1 = 2$. Specialized in capturing high-frequency transient waveforms (Sleep Spindles: 12-14 Hz, Beta rhythms, Vertex spikes).
           - **Branch B (Coarse-Grained)**: Kernel size $k_2 = 31$, stride $s_2 = 2$. Receptive to broad low-frequency morphologies (Delta waves: 0.5-2 Hz, K-Complexes).
           - **Feature Fusion**: Concatenation of fine and coarse maps followed by convolution and adaptive average pooling to downsample 3,000 raw time points into sequence tokens $T=46$.
        
        2. **Bidirectional Long Short-Term Memory (BiLSTM)**:
           - Processes feature tokens in both forward $(\overrightarrow{h}_t)$ and reverse $(\overleftarrow{h}_t)$ directions to contextualize wave transitions:
        """)
        st.latex(r"h_t = \left[ \overrightarrow{h}_t \,\|\, \overleftarrow{h}_t \right] \in \mathbb{R}^{2 \cdot d_{hidden}}")

        st.markdown("""
        3. **Scaled Additive Self-Attention Layer**:
           - Learns a scalar salience score for each time window $t \in [1, T]$:
        """)
        st.latex(r"u_t = \tanh(W_a h_t + b_a)")
        st.latex(r"\alpha_t = \frac{\exp(u_t^\top v_a)}{\sum_{k=1}^T \exp(u_k^\top v_a)}")
        st.latex(r"c = \sum_{t=1}^T \alpha_t h_t")

        st.markdown("""
        4. **Clinical Frequency Band Attribution**:
           - Attention weights $\alpha(t)$ are mapped onto continuous Short-Time Fourier Transform (STFT) power distributions:
        """)
        st.latex(r"\text{Importance}(b) = \frac{\sum_{t} \alpha(t) \cdot P_b(t)}{\sum_{b'}\sum_{t} \alpha(t) \cdot P_{b'}(t)}")

    with col_arch2:
        st.markdown("### Hyperparameter Configuration")
        hyper_df = pd.DataFrame([
            {"Parameter": "Sampling Rate (fs)", "Configured Value": "100 Hz"},
            {"Parameter": "Epoch Length", "Configured Value": "30.0 seconds (3,000 samples)"},
            {"Parameter": "Channels", "Configured Value": "3 (Fpz-Cz, Pz-Oz, EOG)"},
            {"Parameter": "CNN Base Filters", "Configured Value": "32 per branch (64 fused)"},
            {"Parameter": "Condensed Steps (T)", "Configured Value": "46 temporal tokens"},
            {"Parameter": "BiLSTM Hidden Units", "Configured Value": "64 per direction (128 total)"},
            {"Parameter": "Attention Hidden Dim", "Configured Value": "64 units"},
            {"Parameter": "Dropout Rate", "Configured Value": "0.3"},
            {"Parameter": "Optimizer", "Configured Value": "AdamW (weight decay: 1e-4)"},
            {"Parameter": "Learning Rate", "Configured Value": "1e-3 (ReduceLROnPlateau)"}
        ])
        st.table(hyper_df)


# =========================================================
# TAB 4: Project Report & AASM Standards
# =========================================================
with tab4:
    st.subheader("📑 Project Report & AASM Standards Reference")
    
    st.markdown("""
    ### Project Report Overview
    - **Title**: *Explainable Sleep Stage Classification from EEG Signals Using Attention-Based Deep Learning*
    - **Report Document**: 65-page comprehensive academic dissertation compliant with university guidelines.
    - **PDF Status**: Compiled with official JAIN University / FET emblem and formatted per IEEE guidelines.
    """)
    
    if os.path.exists("PROJECT_REPORT_YOGESH_25BTRCM058.pdf"):
        st.success("✅ **PROJECT_REPORT_YOGESH_25BTRCM058.pdf** is available in your project directory (Size: ~1.06 MB, 65 Pages).")
        with open("PROJECT_REPORT_YOGESH_25BTRCM058.pdf", "rb") as f_pdf:
            st.download_button(
                label="📥 Download Full 65-Page Report (PDF)",
                data=f_pdf.read(),
                file_name="PROJECT_REPORT_YOGESH_25BTRCM058.pdf",
                mime="application/pdf",
                key="btn_download_tab4"
            )

    st.markdown("---")
    st.markdown("### American Academy of Sleep Medicine (AASM) Scoring Manual Reference")
    
    aasm_df = pd.DataFrame([
        {
            "Stage": "Wake (W)",
            "Spectral Characteristics": "Alpha rhythm (8-12 Hz) in posterior leads with eye closure; Beta (>13 Hz) frontal",
            "Key Diagnostic Transients": "Eye blinks (0.5-1 Hz), rapid eye movements with high muscle tone",
            "Amplitude Criteria": "10 - 50 µV"
        },
        {
            "Stage": "Stage N1",
            "Spectral Characteristics": "Alpha attenuation (<50% of epoch); dominant low-amplitude Theta (4-7 Hz)",
            "Key Diagnostic Transients": "Vertex Sharp Waves (<200 ms), Slow Rolling Eye Movements (SEMs)",
            "Amplitude Criteria": "< 50 µV"
        },
        {
            "Stage": "Stage N2",
            "Spectral Characteristics": "Theta background rhythm (4-7 Hz)",
            "Key Diagnostic Transients": "Sleep Spindles (12-14 Hz, >=0.5s duration) and biphasic K-Complexes (>=0.5s)",
            "Amplitude Criteria": "K-complexes >= 75 µV"
        },
        {
            "Stage": "Stage N3 (SWS)",
            "Spectral Characteristics": "High-amplitude slow Delta waves (0.5-2 Hz)",
            "Key Diagnostic Transients": "Slow Wave Activity (SWA) occupying >= 20% of epoch duration",
            "Amplitude Criteria": "Peak-to-peak amplitude > 75 µV"
        },
        {
            "Stage": "Stage REM",
            "Spectral Characteristics": "Desynchronized Low-Amplitude Mixed Frequency (LAMF) background",
            "Key Diagnostic Transients": "Sawtooth waves (2-6 Hz), Rapid Eye Movements (bursts in EOG), muscle atonia",
            "Amplitude Criteria": "Low voltage (< 40 µV)"
        }
    ])
    st.table(aasm_df)

# Footer
st.markdown("---")
st.markdown(
    "<center><small style='color: #64748b;'>"
    "NeuroSleep AI • Explainable Sleep Staging from EEG Signals • "
    "Developed by <b>YOGESH.L</b> (USN: <b>25BTRCM058</b>) • "
    "JAIN (Deemed-to-be University)"
    "</small></center>",
    unsafe_allow_html=True
)
