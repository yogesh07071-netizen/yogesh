"""
model.py - Attention-Based Deep Learning Model for Sleep Stage Classification
=============================================================================
Architecture:
  1. Multi-Scale 1D-CNN:
     - Branch A (High Frequency): Small receptive field for Sleep Spindles, Beta, Vertex waves.
     - Branch B (Low Frequency): Large receptive field for Slow Delta waves and K-Complexes.
     - Max Pooling + Conv blocks condensing 3000 time-steps into temporal sequence tokens.
  2. Bidirectional LSTM (BiLSTM):
     - Captures temporal dependencies and transitions across the 30-second epoch.
  3. Scaled Additive Self-Attention Layer:
     - Computes time-step importance weights alpha_t.
     - Aggregates hidden states into a salient context vector.
  4. Classification Head:
     - Dense MLP with Dropout to output 5-stage sleep logits (Wake, N1, N2, N3, REM).
  5. Explainability Output:
     - Returns both class logits and raw temporal attention distribution for interpretability.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union, List

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TemporalAttention(nn.Module):
        """
        Scaled Additive Self-Attention Layer.
        Learns an attention weight for each temporal step in the feature sequence:
            u_t = tanh(W * h_t + b)
            alpha_t = exp(u_t^T * v) / sum_k(exp(u_k^T * v))
            context = sum_t(alpha_t * h_t)
        """
        def __init__(self, hidden_dim: int):
            super(TemporalAttention, self).__init__()
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1, bias=False)
            )

        def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
            Returns:
                context_vector: Tensor of shape (batch_size, hidden_dim)
                attention_weights: Tensor of shape (batch_size, seq_len)
            """
            # energy shape: (batch_size, seq_len, 1)
            energy = self.projection(hidden_states)
            # attention_weights shape: (batch_size, seq_len)
            attention_weights = F.softmax(energy.squeeze(-1), dim=-1)
            
            # Weighted sum over temporal dimension
            # (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_dim) -> (batch_size, 1, hidden_dim)
            context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
            
            return context_vector, attention_weights


    class MultiScaleCNNBlock(nn.Module):
        """
        Dual-branch 1D CNN with fine (high-frequency) and coarse (low-frequency) kernels.
        """
        def __init__(self, in_channels: int = 3, base_filters: int = 32):
            super(MultiScaleCNNBlock, self).__init__()
            
            # Branch 1: Fine-grained for spindles / beta / vertex spikes (small kernel)
            self.branch_fine = nn.Sequential(
                nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
            )
            
            # Branch 2: Coarse-grained for slow delta waves and K-complexes (large kernel)
            self.branch_coarse = nn.Sequential(
                nn.Conv1d(in_channels, base_filters, kernel_size=31, stride=2, padding=15),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
            )
            
            # Downsampling & fusion conv block
            fused_channels = base_filters * 2
            self.fusion_block = nn.Sequential(
                nn.Conv1d(fused_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
                nn.Dropout(0.2),
                
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(46)  # Standardized sequence length T=46
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Input shape:  (B, C, L) e.g., (B, 3, 3000)
            Output shape: (B, T, D) e.g., (B, 46, 128)
            """
            out_fine = self.branch_fine(x)
            out_coarse = self.branch_coarse(x)
            fused = torch.cat([out_fine, out_coarse], dim=1)
            out = self.fusion_block(fused)
            # Transpose to (batch_size, seq_len, features) for sequential modeling
            return out.transpose(1, 2)


    class ExplainableSleepNet(nn.Module):
        """
        Complete End-to-End Deep Learning Architecture:
        1D CNN -> BiLSTM -> Temporal Self-Attention -> Classification Head.
        """
        def __init__(self, in_channels: int = 3, num_classes: int = 5, lstm_hidden: int = 64):
            super(ExplainableSleepNet, self).__init__()
            
            self.feature_extractor = MultiScaleCNNBlock(in_channels=in_channels, base_filters=32)
            
            # BiLSTM: 128 input dim from CNN, lstm_hidden in each direction -> 2*lstm_hidden
            self.bilstm = nn.LSTM(
                input_size=128,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
            bilstm_output_dim = lstm_hidden * 2
            self.attention = TemporalAttention(hidden_dim=bilstm_output_dim)
            
            # Classification MLP
            self.classifier = nn.Sequential(
                nn.Linear(bilstm_output_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: Multi-channel EEG tensor of shape (batch_size, n_channels, n_samples)
            Returns:
                logits: Class scores of shape (batch_size, num_classes)
                attention_weights: Temporal importance weights of shape (batch_size, seq_len)
            """
            # Extract localized frequency & temporal tokens
            conv_features = self.feature_extractor(x)  # (B, T=46, 128)
            
            # Bidirectional sequential modeling
            lstm_out, _ = self.bilstm(conv_features)    # (B, T=46, 128)
            
            # Attention pooling
            context_vector, attention_weights = self.attention(lstm_out)  # (B, 128), (B, T)
            
            # Final logits
            logits = self.classifier(context_vector)     # (B, num_classes)
            
            return logits, attention_weights


    class EEGDataset(Dataset):
        """PyTorch Dataset wrapper for multi-channel EEG signals."""
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).long()

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.X[idx], self.y[idx]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Executes training loop with AdamW optimizer and cross-entropy loss.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training the deep learning model.")
        
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print(f"\n[+] Starting Model Training on device: {device.upper()}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item() * signals.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                logits, _ = model(signals)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * signals.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:5.2f}%")

    print("-" * 65)
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates model on test dataset and extracts predictions, probabilities, and attention weights.
    
    Returns:
        y_true, y_pred, y_probas, all_attentions
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probas = []
    all_attentions = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            logits, attentions = model(signals)
            probas = F.softmax(logits, dim=-1)
            preds = torch.argmax(probas, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probas.append(probas.cpu().numpy())
            all_attentions.append(attentions.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probas = np.concatenate(all_probas, axis=0)
    attentions = np.concatenate(all_attentions, axis=0)

    return y_true, y_pred, y_probas, attentions


# --- Robust Fallback Baseline (when PyTorch is absent or for fast CPU benchmark) ---
class FallbackSpectralSleepClassifier:
    """
    Physiological rule-based spectral power and feature extractor baseline
    providing identical API interface if PyTorch is not available.
    """
    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass  # Heuristic clinical threshold model
        
    def predict_with_attention(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy import signal
        N, C, L = X.shape
        probas = np.zeros((N, self.n_classes))
        attentions = np.zeros((N, 46))
        
        for i in range(N):
            sig = X[i, 0]  # Fpz-Cz channel
            # Segment into 46 sub-windows
            window_size = L // 46
            sub_powers = []
            for w in range(46):
                seg = sig[w * window_size : (w + 1) * window_size]
                sub_powers.append(np.var(seg))
            sub_powers = np.array(sub_powers)
            att_weights = np.exp(sub_powers - np.max(sub_powers))
            att_weights = att_weights / (np.sum(att_weights) + 1e-8)
            attentions[i] = att_weights

            # Spectral power ratios
            freqs, psd = signal.welch(sig, fs=100, nperseg=256)
            delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
            theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
            alpha = np.mean(psd[(freqs >= 8) & (freqs < 12)])
            beta  = np.mean(psd[(freqs >= 12) & (freqs < 30)])

            # Rule heuristic
            scores = np.array([
                alpha * 1.5 + beta,    # Wake
                theta * 1.2 + 0.1,     # N1
                theta + 1.2 * beta,    # N2 (spindles in beta)
                delta * 2.5,           # N3 (dominant slow wave)
                theta * 0.9 + 0.8      # REM
            ])
            p = np.exp(scores - np.max(scores))
            probas[i] = p / np.sum(p)
            
        preds = np.argmax(probas, axis=-1)
        return preds, probas, attentions


if __name__ == '__main__':
    print("[*] Testing ExplainableSleepNet Architecture...")
    if TORCH_AVAILABLE:
        dummy_input = torch.randn(4, 3, 3000)
        net = ExplainableSleepNet(in_channels=3, num_classes=5)
        out_logits, out_attn = net(dummy_input)
        print(f"[+] PyTorch Model Instantiated Successfully:")
        print(f"    - Input Shape        : {dummy_input.shape}")
        print(f"    - Logits Output      : {out_logits.shape} (Batch, Classes)")
        print(f"    - Attention Weights  : {out_attn.shape} (Batch, SequenceSteps)")
        print(f"    - Total Parameters   : {sum(p.numel() for p in net.parameters()):,}")
    else:
        print("[!] PyTorch not found. Fallback spectral baseline available.")
