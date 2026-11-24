"""
main.py - End-to-End CLI for Explainable Sleep Stage Classification
===================================================================
A menu-driven CLI interface and automated pipeline to:
  1. Generate synthetic AASM-compliant multi-channel EEG datasets.
  2. Train 1D-CNN + BiLSTM + Self-Attention Deep Learning Model.
  3. Evaluate performance (Accuracy, Precision, Recall, Macro-F1, Confusion Matrix).
  4. Generate and save publication-grade Explainability plots & Clinical Reports.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split

# Local project imports
from data_generator import generate_dataset, STAGE_NAMES, CHANNEL_NAMES
from model import (
    TORCH_AVAILABLE,
    ExplainableSleepNet,
    EEGDataset,
    train_model,
    evaluate_model,
    FallbackSpectralSleepClassifier
)
from explainability import (
    plot_epoch_explanation,
    correlate_attention_with_bands,
    generate_clinical_explanation,
    FREQUENCY_BANDS,
    BAND_COLORS
)

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader


OUTPUT_DIR = "outputs"
DATASET_PATH = "eeg_sleep_dataset.npz"


def ensure_output_dir():
    """Creates the output directory for artifacts if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_and_save_dataset(n_samples_per_stage=100, fs=100, duration=30.0):
    """Generates synthetic EEG dataset and persists to disk."""
    ensure_output_dir()
    print(f"\n[*] Generating Synthetic EEG Dataset ({n_samples_per_stage} epochs per stage)...")
    print(f"    - Stages: {', '.join(STAGE_NAMES)}")
    print(f"    - Sampling Rate: {fs} Hz | Epoch Length: {duration}s")
    
    X, y, meta = generate_dataset(
        n_samples_per_stage=n_samples_per_stage,
        fs=fs,
        duration=duration,
        random_state=42
    )
    
    np.savez_compressed(
        DATASET_PATH,
        X=X,
        y=y,
        fs=fs,
        duration=duration,
        channels=np.array(CHANNEL_NAMES)
    )
    print(f"[+] Dataset saved to '{DATASET_PATH}' | Total Epochs: {len(y)} | Shape: {X.shape}")
    return X, y, meta


def load_dataset():
    """Loads dataset from file, or prompts generation if missing."""
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Dataset '{DATASET_PATH}' not found. Generating default dataset...")
        return generate_and_save_dataset(n_samples_per_stage=80)
    
    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data['X']
    y = data['y']
    meta = {
        'fs': int(data['fs']),
        'duration_seconds': float(data['duration']),
        'channels': list(data['channels']),
        'stage_names': STAGE_NAMES,
        'total_epochs': len(y)
    }
    print(f"[+] Loaded dataset from '{DATASET_PATH}' ({len(y)} epochs, shape: {X.shape})")
    return X, y, meta


def plot_training_history(history, save_path):
    """Plots and saves loss and accuracy learning curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', lw=1.8, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r--s', label='Validation Loss', lw=1.8, markersize=4)
    ax1.set_title('Cross-Entropy Loss Progression', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Accuracy', lw=1.8, markersize=4)
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'g--^', label='Val Accuracy', lw=1.8, markersize=4)
    ax2.set_title('Classification Accuracy Progression', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved training history curves: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generates and saves normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=STAGE_NAMES,
        yticklabels=STAGE_NAMES,
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title('Sleep Stage Confusion Matrix (%)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Stage', fontsize=11, fontweight='semibold')
    plt.ylabel('Ground Truth Stage', fontsize=11, fontweight='semibold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved confusion matrix heatmap: {save_path}")


def plot_average_band_importance_per_stage(X, attentions, y_true, fs, save_path):
    """Computes and plots average frequency band importance profile for each sleep stage."""
    n_classes = len(STAGE_NAMES)
    bands = list(FREQUENCY_BANDS.keys())
    matrix = np.zeros((n_classes, len(bands)))
    counts = np.zeros(n_classes)
    
    for i in range(len(y_true)):
        stage_idx = y_true[i]
        band_imp = correlate_attention_with_bands(X[i, 0], attentions[i], fs=fs)
        for b_idx, b_name in enumerate(bands):
            matrix[stage_idx, b_idx] += band_imp[b_name]
        counts[stage_idx] += 1
        
    for c in range(n_classes):
        if counts[c] > 0:
            matrix[c] /= counts[c]
            
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_indices = np.arange(n_classes)
    width = 0.18
    
    for b_idx, b_name in enumerate(bands):
        scores = matrix[:, b_idx] * 100
        ax.bar(
            x_indices + (b_idx - 1.5) * width,
            scores,
            width,
            label=f"{b_name} ({FREQUENCY_BANDS[b_name][0]}-{FREQUENCY_BANDS[b_name][1]} Hz)",
            color=BAND_COLORS[b_name],
            edgecolor='black',
            linewidth=0.5
        )
        
    ax.set_xticks(x_indices)
    ax.set_xticklabels(STAGE_NAMES, fontsize=11, fontweight='bold')
    ax.set_ylabel("Attention-Weighted Relative Power (%)", fontsize=11)
    ax.set_xlabel("Sleep Stage", fontsize=11, fontweight='bold')
    ax.set_title("Cross-Stage Neurophysiological Band Importance Alignment", fontsize=13, fontweight='bold', pad=12)
    ax.legend(title="Canonical Frequency Band", fontsize=9)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved average band importance plot: {save_path}")


def run_pipeline(epochs=12, batch_size=16, n_samples_per_stage=80):
    """Executes the full automated training, evaluation, and explainability pipeline."""
    ensure_output_dir()
    print("\n" + "="*70)
    print("   EXPLAINABLE SLEEP STAGE CLASSIFIER - AUTOMATED PIPELINE")
    print("="*70)

    # 1. Dataset Generation or Loading
    X, y, meta = load_dataset()
    fs = meta['fs']
    
    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Further split train into train/val
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )

    print(f"\n[+] Dataset Partitioned:")
    print(f"    - Training Set   : {X_train_sub.shape[0]} epochs")
    print(f"    - Validation Set : {X_val.shape[0]} epochs")
    print(f"    - Test Set       : {X_test.shape[0]} epochs")

    # 2. Model Training
    if TORCH_AVAILABLE:
        print("\n[*] Initializing PyTorch ExplainableSleepNet (1D-CNN + BiLSTM + Attention)...")
        train_ds = EEGDataset(X_train_sub, y_train_sub)
        val_ds   = EEGDataset(X_val, y_val)
        test_ds  = EEGDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = ExplainableSleepNet(in_channels=X.shape[1], num_classes=len(STAGE_NAMES))
        history = train_model(model, train_loader, val_loader, epochs=epochs, lr=1e-3)
        plot_training_history(history, os.path.join(OUTPUT_DIR, "training_curves.png"))

        # 3. Model Evaluation
        print("\n[*] Evaluating on Unseen Test Dataset...")
        y_true, y_pred, y_probas, attentions = evaluate_model(model, test_loader)
    else:
        print("\n[!] PyTorch not found. Running with Heuristic Rule Baseline Classifier...")
        clf = FallbackSpectralSleepClassifier(n_classes=len(STAGE_NAMES))
        y_pred, y_probas, attentions = clf.predict_with_attention(X_test)
        y_true = y_test

    # Metrics
    acc = np.mean(y_true == y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred)
    print("\n" + "="*70)
    print(f"TEST ACCURACY: {acc:.2f}% | COHEN'S KAPPA: {kappa:.4f}")
    print("="*70)
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=STAGE_NAMES)
    print(report)

    # Save metrics JSON
    summary_metrics = {
        'test_accuracy': float(acc),
        'cohen_kappa': float(kappa),
        'total_test_epochs': len(y_true),
        'stages': STAGE_NAMES
    }
    with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json"), 'w') as f:
        json.dump(summary_metrics, f, indent=4)

    # Save Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # Save Cross-Stage Band Importance Analysis
    plot_average_band_importance_per_stage(
        X_test, attentions, y_true, fs,
        os.path.join(OUTPUT_DIR, "spectral_band_importance.png")
    )

    # 4. Explainability Analysis for Representative Epochs (One per stage)
    print("\n[*] Generating Multi-Modal Explainability Diagnostic Plots for Each Sleep Stage...")
    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        matching_indices = np.where((y_true == stage_idx) & (y_pred == stage_idx))[0]
        if len(matching_indices) == 0:
            matching_indices = np.where(y_true == stage_idx)[0]
            
        sample_idx = matching_indices[0]
        save_file = os.path.join(OUTPUT_DIR, f"attention_analysis_stage_{stage_name}.png")
        
        plot_epoch_explanation(
            signals=X_test[sample_idx],
            attention_weights=attentions[sample_idx],
            stage_true=y_true[sample_idx],
            stage_pred=y_pred[sample_idx],
            probas=y_probas[sample_idx],
            fs=fs,
            channel_names=CHANNEL_NAMES,
            save_path=save_file
        )
        
        # Clinical Narrative for sample
        band_imp = correlate_attention_with_bands(X_test[sample_idx, 0], attentions[sample_idx], fs=fs)
        narrative = generate_clinical_explanation(
            pred_idx=y_pred[sample_idx],
            confidence=y_probas[sample_idx, y_pred[sample_idx]],
            band_importance=band_imp,
            attention_weights=attentions[sample_idx],
            true_idx=y_true[sample_idx]
        )
        print(f"\n{narrative}")

    print("\n" + "="*70)
    print(f"[+] All artifacts generated and saved into '{OUTPUT_DIR}/':")
    print(f"    1. {OUTPUT_DIR}/training_curves.png")
    print(f"    2. {OUTPUT_DIR}/confusion_matrix.png")
    print(f"    3. {OUTPUT_DIR}/spectral_band_importance.png")
    print(f"    4. {OUTPUT_DIR}/attention_analysis_stage_Wake.png")
    print(f"    5. {OUTPUT_DIR}/attention_analysis_stage_N1.png")
    print(f"    6. {OUTPUT_DIR}/attention_analysis_stage_N2.png")
    print(f"    7. {OUTPUT_DIR}/attention_analysis_stage_N3.png")
    print(f"    8. {OUTPUT_DIR}/attention_analysis_stage_REM.png")
    print(f"    9. {OUTPUT_DIR}/evaluation_summary.json")
    print("="*70)


def interactive_menu():
    """Runs interactive command line menu."""
    while True:
        print("\n" + "="*50)
        print("   EXPLAINABLE SLEEP STAGE CLASSIFIER")
        print("="*50)
        print(" [1] Generate Synthetic EEG Dataset (W, N1, N2, N3, REM)")
        print(" [2] Train Attention SleepNet Model (1D-CNN + BiLSTM + Attention)")
        print(" [3] Run Full Automated Pipeline (Train, Evaluate & Explain)")
        print(" [4] Run Explainability Analysis on Saved Test Epochs")
        print(" [5] Exit")
        print("="*50)
        
        choice = input("Enter option [1-5]: ").strip()
        
        if choice == '1':
            n = input("Enter epochs per stage (default 80): ").strip()
            n_epochs = int(n) if n.isdigit() else 80
            generate_and_save_dataset(n_samples_per_stage=n_epochs)
        elif choice == '2':
            ep = input("Enter training epochs (default 12): ").strip()
            num_ep = int(ep) if ep.isdigit() else 12
            run_pipeline(epochs=num_ep)
        elif choice == '3':
            run_pipeline(epochs=12)
        elif choice == '4':
            run_pipeline(epochs=5)
        elif choice == '5':
            print("Exiting. Goodbye!")
            sys.exit(0)
        else:
            print("[!] Invalid option. Please choose 1 to 5.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Explainable Sleep Stage Classification CLI")
    parser.add_argument('--all', action='store_true', help="Run full pipeline automatically")
    parser.add_argument('--generate', action='store_true', help="Generate synthetic dataset only")
    parser.add_argument('--epochs', type=int, default=12, help="Number of training epochs")
    parser.add_argument('--samples_per_stage', type=int, default=80, help="Epochs per sleep stage")
    args = parser.parse_args()

    if args.generate:
        generate_and_save_dataset(n_samples_per_stage=args.samples_per_stage)
    elif args.all:
        run_pipeline(epochs=args.epochs, n_samples_per_stage=args.samples_per_stage)
    else:
        # Default to interactive menu if no direct flags provided
        interactive_menu()
