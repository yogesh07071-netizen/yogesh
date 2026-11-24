"""
data_generator.py - Synthetic Multi-Channel EEG Signal Generator for Sleep Staging
==================================================================================
Simulates multi-channel EEG signals for standard AASM sleep stages:
  - Wake (W)   : High Alpha (8-12 Hz) posterior, Beta (>13 Hz) frontal, eye blinks, EMG artifacts.
  - N1 Stage   : Alpha attenuation (<50%), dominant Theta (4-8 Hz), Vertex sharp waves.
  - N2 Stage   : Dominant Theta/Delta background, Sleep Spindles (12-14 Hz), K-Complexes.
  - N3 Stage   : Slow Wave Sleep (SWS), high-amplitude Delta waves (0.5-4 Hz, >75 uV).
  - REM Stage  : Low-amplitude mixed frequency (LAMF), Sawtooth waves (2-6 Hz), Rapid eye movements.

Sampling rate: 100 Hz (Standard downsampled Sleep-EDF rate).
Standard epoch length: 30 seconds (3,000 samples per channel).
Montage: ['EEG_Fpz_Cz', 'EEG_Pz_Oz', 'EOG_horizontal']
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, List, Optional


STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_TO_LABEL = {name: idx for idx, name in enumerate(STAGE_NAMES)}
LABEL_TO_STAGE = {idx: name for idx, name in enumerate(STAGE_NAMES)}

FREQUENCY_BANDS = {
    'Delta': (0.5, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 12.0),
    'Beta': (12.0, 30.0),
    'Gamma': (30.0, 45.0)
}

CHANNEL_NAMES = ['EEG_Fpz_Cz', 'EEG_Pz_Oz', 'EOG_horizontal']


def generate_colored_noise(n_samples: int, exponent: float = 1.0, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Generates 1/f^exponent colored noise (pink noise background typical for brain signals).
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # Generate white noise in frequency domain
    white = rng.randn(n_samples)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0  # avoid division by zero at DC
    
    # Scale amplitudes by 1 / f^(exponent/2)
    s_filter = 1.0 / (freqs ** (exponent / 2.0))
    s_filter[0] = 0.0
    
    fft_colored = fft_white * s_filter
    colored = np.fft.irfft(fft_colored, n=n_samples)
    
    # Standardize variance
    colored = (colored - np.mean(colored)) / (np.std(colored) + 1e-8)
    return colored


def generate_sinusoid(freq: float, duration: float, fs: int, phase: float = 0.0) -> np.ndarray:
    """Generates a sinusoidal wave of given frequency and duration."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t + phase)


def inject_sleep_spindle(duration: float = 1.0, fs: int = 100, freq: float = 13.0, amplitude: float = 40.0) -> np.ndarray:
    """
    Generates a classical waxing-and-waning sleep spindle (11-16 Hz, typical ~13 Hz).
    Enveloped with a Tukey or Gaussian window.
    """
    n_samples = int(duration * fs)
    t = np.linspace(-1, 1, n_samples)
    envelope = np.exp(-4 * (t ** 2))  # Gaussian envelope
    carrier = np.sin(2 * np.pi * freq * np.linspace(0, duration, n_samples))
    spindle = amplitude * envelope * carrier
    return spindle


def inject_k_complex(duration: float = 1.2, fs: int = 100, amplitude: float = 120.0) -> np.ndarray:
    """
    Generates a biphasic K-complex: sharp negative deflection followed by slower positive wave.
    Duration > 0.5 sec, amplitude typically > 75-100 uV.
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, np.pi, n_samples)
    # Biphasic asymmetric shape: sharp dip (-), then smooth broader rise (+)
    wave = -np.sin(2 * t) * np.exp(-1.5 * t)
    wave = wave / (np.max(np.abs(wave)) + 1e-8) * amplitude
    return wave


def inject_vertex_wave(duration: float = 0.4, fs: int = 100, amplitude: float = 50.0) -> np.ndarray:
    """Generates a sharp Vertex Wave prominent in late N1 / transition to N2."""
    n_samples = int(duration * fs)
    t = np.linspace(-1, 1, n_samples)
    wave = -np.exp(-12 * (t ** 2)) * amplitude
    return wave


def inject_sawtooth_wave(duration: float = 1.5, fs: int = 100, freq: float = 3.5, amplitude: float = 30.0) -> np.ndarray:
    """Generates triangular/sawtooth waves (2-6 Hz) characteristic of REM sleep."""
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    wave = amplitude * signal.sawtooth(2 * np.pi * freq * t, width=0.8)
    return wave


def inject_eye_blink(duration: float = 0.5, fs: int = 100, amplitude: float = 80.0) -> np.ndarray:
    """Simulates high-amplitude blink artifacts captured in frontal EEG and EOG."""
    n_samples = int(duration * fs)
    t = np.linspace(-np.pi, np.pi, n_samples)
    blink = (np.cos(t) + 1.0) / 2.0 * amplitude
    return blink


def generate_single_epoch(stage: str, fs: int = 100, duration: float = 30.0, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Generates a 3-channel 30-second epoch for the requested sleep stage.
    
    Returns:
        np.ndarray of shape (n_channels, n_samples)
    """
    if rng is None:
        rng = np.random.RandomState()
    
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # Baseline channels: [0: Fpz-Cz, 1: Pz-Oz, 2: EOG_horizontal]
    # Background pink noise for all channels
    ch_fpz_cz = generate_colored_noise(n_samples, exponent=1.2, rng=rng) * 10.0
    ch_pz_oz  = generate_colored_noise(n_samples, exponent=1.2, rng=rng) * 10.0
    ch_eog    = generate_colored_noise(n_samples, exponent=1.0, rng=rng) * 12.0

    if stage == 'Wake':
        # Wake: Posterior Alpha rhythm (8-12 Hz) dominant in Pz-Oz with high amplitude
        alpha_freq = rng.uniform(9.0, 11.0)
        alpha_wave = generate_sinusoid(alpha_freq, duration, fs, phase=rng.uniform(0, 2*np.pi))
        alpha_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t)  # waxing and waning
        ch_pz_oz += alpha_wave * alpha_mod * rng.uniform(30.0, 50.0)
        ch_fpz_cz += alpha_wave * 0.3 * alpha_mod * rng.uniform(10.0, 20.0)
        
        # Frontal Beta rhythm (14-25 Hz)
        beta_freq = rng.uniform(16.0, 22.0)
        ch_fpz_cz += generate_sinusoid(beta_freq, duration, fs) * rng.uniform(10.0, 20.0)
        
        # Eye blinks and saccades in EOG and leaking into Fpz-Cz
        num_blinks = rng.randint(2, 6)
        for _ in range(num_blinks):
            start_idx = rng.randint(0, n_samples - int(0.6 * fs))
            blink = inject_eye_blink(duration=0.5, fs=fs, amplitude=rng.uniform(70.0, 130.0))
            bl_len = len(blink)
            ch_eog[start_idx:start_idx+bl_len] += blink
            ch_fpz_cz[start_idx:start_idx+bl_len] += blink * 0.45  # frontal volume conduction

    elif stage == 'N1':
        # N1: Alpha drops below 50%; dominant low-amplitude Theta (4-7 Hz)
        theta_freq = rng.uniform(4.5, 6.5)
        ch_fpz_cz += generate_sinusoid(theta_freq, duration, fs) * rng.uniform(15.0, 28.0)
        ch_pz_oz  += generate_sinusoid(theta_freq, duration, fs) * rng.uniform(12.0, 22.0)
        
        # Residual intermittent low alpha
        ch_pz_oz  += generate_sinusoid(10.0, duration, fs) * rng.uniform(5.0, 12.0)
        
        # Vertex sharp transients (1 to 3 per epoch)
        num_v_waves = rng.randint(1, 3)
        for _ in range(num_v_waves):
            pos = rng.randint(int(3 * fs), n_samples - int(3 * fs))
            v_wave = inject_vertex_wave(duration=0.35, fs=fs, amplitude=rng.uniform(35.0, 65.0))
            vw_len = len(v_wave)
            ch_fpz_cz[pos:pos+vw_len] += v_wave
        
        # Slow rolling eye movements in EOG (0.2 - 0.8 Hz)
        sem = generate_sinusoid(0.35, duration, fs, phase=rng.uniform(0, np.pi)) * rng.uniform(25.0, 45.0)
        ch_eog += sem

    elif stage == 'N2':
        # N2: Theta background (4-7 Hz) + Sleep Spindles + K-Complexes
        theta_freq = rng.uniform(4.5, 6.0)
        ch_fpz_cz += generate_sinusoid(theta_freq, duration, fs) * rng.uniform(18.0, 30.0)
        ch_pz_oz  += generate_sinusoid(theta_freq, duration, fs) * rng.uniform(15.0, 25.0)
        
        # Inject 2-4 Sleep Spindles (12-14 Hz)
        num_spindles = rng.randint(2, 5)
        for _ in range(num_spindles):
            pos = rng.randint(int(1 * fs), n_samples - int(2.5 * fs))
            spindle_freq = rng.uniform(12.5, 14.5)
            spindle_dur = rng.uniform(0.7, 1.4)
            sp = inject_sleep_spindle(duration=spindle_dur, fs=fs, freq=spindle_freq, amplitude=rng.uniform(35.0, 60.0))
            slen = len(sp)
            ch_fpz_cz[pos:pos+slen] += sp
            ch_pz_oz[pos:pos+slen] += sp * 0.7
        
        # Inject 1-3 K-Complexes
        num_kc = rng.randint(1, 3)
        for _ in range(num_kc):
            pos = rng.randint(int(2 * fs), n_samples - int(3 * fs))
            kc_dur = rng.uniform(0.9, 1.4)
            kc = inject_k_complex(duration=kc_dur, fs=fs, amplitude=rng.uniform(90.0, 150.0))
            kclen = len(kc)
            ch_fpz_cz[pos:pos+kclen] += kc
            ch_pz_oz[pos:pos+kclen] += kc * 0.5

    elif stage == 'N3':
        # N3 (Slow Wave Sleep): High-amplitude Delta (0.5 - 2.5 Hz, amplitude > 75 uV)
        # Dominates > 20% of epoch duration (typically 40-70% in synthetic deep sleep)
        delta_freq = rng.uniform(0.7, 1.8)
        delta_wave = generate_sinusoid(delta_freq, duration, fs, phase=rng.uniform(0, 2*np.pi))
        delta_sub  = generate_sinusoid(delta_freq * 1.5, duration, fs, phase=rng.uniform(0, 2*np.pi))
        
        # High amplitude envelope
        deep_delta = (delta_wave * 0.7 + delta_sub * 0.3) * rng.uniform(85.0, 140.0)
        ch_fpz_cz += deep_delta
        ch_pz_oz  += deep_delta * 0.8
        ch_eog    += deep_delta * 0.35  # volume conduction to ocular leads

    elif stage == 'REM':
        # REM: Low-amplitude mixed frequency (LAMF) resembling Wake/N1 without alpha dominance
        # Theta + Beta + transient sawtooth waves + Rapid eye movements
        theta_freq = rng.uniform(4.5, 7.0)
        ch_fpz_cz += generate_sinusoid(theta_freq, duration, fs) * rng.uniform(10.0, 18.0)
        ch_pz_oz  += generate_sinusoid(theta_freq, duration, fs) * rng.uniform(8.0, 15.0)
        
        # Sawtooth waves (2-6 Hz)
        num_saw = rng.randint(1, 3)
        for _ in range(num_saw):
            pos = rng.randint(int(2 * fs), n_samples - int(3 * fs))
            saw = inject_sawtooth_wave(duration=1.2, fs=fs, freq=rng.uniform(2.8, 4.5), amplitude=rng.uniform(20.0, 35.0))
            slen = len(saw)
            ch_fpz_cz[pos:pos+slen] += saw
            
        # Rapid eye movement bursts in EOG
        num_rems = rng.randint(3, 7)
        for _ in range(num_rems):
            pos = rng.randint(int(1 * fs), n_samples - int(2 * fs))
            rem_dur = rng.uniform(0.3, 0.7)
            rem_wave = inject_sinusoid_burst(rem_dur, fs, freq=rng.uniform(1.0, 2.5), amplitude=rng.uniform(60.0, 110.0), rng=rng)
            rlen = len(rem_wave)
            ch_eog[pos:pos+rlen] += rem_wave
            ch_fpz_cz[pos:pos+rlen] += rem_wave * 0.25

    # Bandpass filter signal between 0.5 Hz and 40 Hz to match clinical PSG montage
    signals = np.stack([ch_fpz_cz, ch_pz_oz, ch_eog], axis=0)
    filtered = filter_eeg_signals(signals, fs=fs, lowcut=0.5, highcut=40.0)
    
    # Normalize per channel (Z-score standard scaling for ML models)
    normed = (filtered - np.mean(filtered, axis=-1, keepdims=True)) / (np.std(filtered, axis=-1, keepdims=True) + 1e-6)
    return normed.astype(np.float32)


def inject_sinusoid_burst(duration: float, fs: int, freq: float, amplitude: float, rng: np.random.RandomState) -> np.ndarray:
    """Produces rapid isolated deflections for EOG bursts."""
    n_samples = int(duration * fs)
    t = np.linspace(-1, 1, n_samples)
    env = np.exp(-3 * (t ** 2))
    carrier = np.sin(2 * np.pi * freq * np.linspace(0, duration, n_samples) + rng.uniform(0, np.pi))
    return amplitude * env * carrier


def filter_eeg_signals(signals: np.ndarray, fs: int = 100, lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
    """Applies a 4th-order Butterworth bandpass filter along the last axis."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, signals, axis=-1)


def generate_dataset(
    n_samples_per_stage: int = 100,
    fs: int = 100,
    duration: float = 30.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generates a balanced multi-channel EEG sleep staging dataset.
    
    Args:
        n_samples_per_stage: Number of 30-second epochs per sleep stage.
        fs: Sampling rate in Hz.
        duration: Epoch length in seconds (standard 30s).
        random_state: Random seed for reproducibility.
        
    Returns:
        X: np.ndarray of shape (total_epochs, n_channels, n_samples)
        y: np.ndarray of shape (total_epochs,) with integer labels [0..4]
        metadata: Dictionary with dataset specifications and class mapping
    """
    rng = np.random.RandomState(random_state)
    all_epochs = []
    all_labels = []
    
    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        for _ in range(n_samples_per_stage):
            epoch_data = generate_single_epoch(stage_name, fs=fs, duration=duration, rng=rng)
            all_epochs.append(epoch_data)
            all_labels.append(stage_idx)
            
    X = np.stack(all_epochs, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    
    # Shuffle dataset
    indices = rng.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    metadata = {
        'fs': fs,
        'duration_seconds': duration,
        'n_samples_per_epoch': int(fs * duration),
        'channels': CHANNEL_NAMES,
        'stage_names': STAGE_NAMES,
        'stage_to_label': STAGE_TO_LABEL,
        'total_epochs': len(y),
        'n_classes': len(STAGE_NAMES)
    }
    
    return X, y, metadata


if __name__ == '__main__':
    print("[*] Testing EEG Signal Generator...")
    X, y, meta = generate_dataset(n_samples_per_stage=20, fs=100, duration=30.0, random_state=42)
    print(f"[+] Dataset generated successfully!")
    print(f"    - Input Shape X  : {X.shape} (Epochs, Channels, TimeSteps)")
    print(f"    - Labels Shape y : {y.shape}")
    print(f"    - Class Balance  : { {meta['stage_names'][i]: int(np.sum(y == i)) for i in range(5)} }")
    print(f"    - Sampling Rate  : {meta['fs']} Hz ({meta['duration_seconds']}s per epoch)")
