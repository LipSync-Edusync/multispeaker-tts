import os
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
from librosa.filters import mel as librosa_mel_fn
from pathlib import Path
from typing import Optional, Union

class AudioProcessor:
    # Audio processing module for TTS system handling all audio-related operations
    
    def __init__(self,
                 sample_rate: int = 22050,
                 num_mels: int = 80,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 mel_fmin: float = 0.0,
                 mel_fmax: float = 8000.0,
                 max_wav_value: float = 32768.0,
                 clip_norm: bool = True,
                 preemphasize: bool = True,
                 preemphasis: float = 0.97,
                 min_level_db: float = -100,
                 ref_level_db: float = 20,
                 signal_norm: bool = True,
                 symmetric_norm: bool = True,
                 power: float = 1.5,
                 griffin_lim_iters: int = 60,
                 **kwargs):
        """
        Initialize audio processor with given parameters.
        
        Args:
            sample_rate: Target audio sample rate
            num_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            mel_fmin: Minimum mel frequency
            mel_fmax: Maximum mel frequency
            max_wav_value: Maximum waveform value for normalization
            clip_norm: Clip normalized values to [-1, 1]
            preemphasize: Apply preemphasis
            preemphasis: Preemphasis coefficient
            min_level_db: Minimum dB value for normalization
            ref_level_db: Reference dB value for normalization
            signal_norm: Normalize audio signal
            symmetric_norm: Use symmetric normalization
            power: Exponent for amplifying the predicted magnitude
            griffin_lim_iters: Number of Griffin-Lim iterations
        """
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.max_wav_value = max_wav_value
        self.clip_norm = clip_norm
        self.preemphasize = preemphasize
        self.preemphasis = preemphasis
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.power = power
        self.griffin_lim_iters = griffin_lim_iters
        
        # Initialize mel basis
        self.mel_basis = librosa_mel_fn(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=mel_fmin,
            fmax=mel_fmax
        )
        self.inv_mel_basis = np.linalg.pinv(self.mel_basis)
        
        # Window for STFT
        self.window = torch.hann_window(win_length).float()
        
    def load_wav(self, path: Union[str, Path], sr: Optional[int] = None) -> np.ndarray:
        """
        Load audio file and resample if needed.
        
        Args:
            path: Path to audio file
            sr: Target sample rate (None to use default)
            
        Returns:
            waveform: Loaded audio as numpy array
        """
        target_sr = sr if sr is not None else self.sample_rate
        wav, orig_sr = sf.read(path, dtype='float32')
        
        if orig_sr != target_sr:
            wav = self.resample(wav, orig_sr, target_sr)
            
        if self.signal_norm:
            wav = self.normalize_wav(wav)
            
        return wav
    
    def resample(self, wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio waveform.
        
        Args:
            wav: Input waveform
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            resampled_wav: Resampled waveform
        """
        return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
    
    def normalize_wav(self, wav: np.ndarray) -> np.ndarray:
        """
        Normalize waveform to [-1, 1] range.
        
        Args:
            wav: Input waveform
            
        Returns:
            normalized_wav: Normalized waveform
        """
        if self.clip_norm:
            return np.clip(wav, -1.0, 1.0)
        return wav / np.max(np.abs(wav))
    
    def save_wav(self, wav: np.ndarray, path: str, sr: Optional[int] = None):
        """
        Save waveform to file.
        
        Args:
            wav: Waveform to save
            path: Output path
            sr: Sample rate (None to use default)
        """
        target_sr = sr if sr is not None else self.sample_rate
        wav = np.clip(wav, -1.0, 1.0)
        sf.write(path, wav, target_sr)
    
    def preemphasis(self, wav: np.ndarray) -> np.ndarray:
        """
        Apply preemphasis to waveform.
        
        Args:
            wav: Input waveform
            
        Returns:
            preemphasized_wav: Processed waveform
        """
        return signal.lfilter([1, -self.preemphasis], [1], wav)
    
    def inv_preemphasis(self, wav: np.ndarray) -> np.ndarray:
        """
        Invert preemphasis on waveform.
        
        Args:
            wav: Preemphasized waveform
            
        Returns:
            original_wav: Waveform with preemphasis removed
        """
        return signal.lfilter([1], [1, -self.preemphasis], wav)
    
    def stft(self, wav: np.ndarray) -> np.ndarray:
        """
        Compute STFT of waveform.
        
        Args:
            wav: Input waveform
            
        Returns:
            stft: Complex STFT matrix
        """
        if torch.is_tensor(wav):
            wav = wav.numpy()
            
        return librosa.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann'
        )
    
    def stft_torch(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT using PyTorch.
        
        Args:
            wav: Input waveform tensor
            
        Returns:
            stft: Complex STFT tensor
        """
        if len(wav.shape) == 1:  # Single waveform
            wav = wav.unsqueeze(0)
            
        if wav.device != self.window.device:
            self.window = self.window.to(wav.device)
            
        return torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
    
    def melspectrogram(self, wav: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from waveform.
        
        Args:
            wav: Input waveform
            
        Returns:
            mel: Mel spectrogram
        """
        if self.preemphasize:
            wav = self.preemphasis(wav)
            
        # Compute STFT
        stft = self.stft(wav)
        magnitude = np.abs(stft)
        
        # Compute mel spectrogram
        mel = np.dot(self.mel_basis, magnitude)
        mel = self.amp_to_db(mel)
        mel = self.normalize(mel)
        
        return mel
    
    def melspectrogram_torch(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram using PyTorch.
        
        Args:
            wav: Input waveform tensor
            
        Returns:
            mel: Mel spectrogram tensor
        """
        if self.preemphasize:
            wav = torch.cat([
                wav[:, 0:1],
                wav[:, 1:] - self.preemphasis * wav[:, :-1]
            ], dim=1)
            
        # Compute STFT
        stft = self.stft_torch(wav)
        magnitude = torch.abs(stft)
        
        # Convert to mel scale
        mel_basis = torch.from_numpy(self.mel_basis).float().to(magnitude.device)
        mel = torch.matmul(mel_basis, magnitude)
        
        # Convert to dB
        mel = self.amp_to_db_torch(mel)
        mel = self.normalize_torch(mel)
        
        return mel
    
    def amp_to_db(self, x: np.ndarray) -> np.ndarray:
        """
        Convert amplitude to decibels.
        
        Args:
            x: Input amplitude
            
        Returns:
            x_db: Converted dB value
        """
        return 20 * np.log10(np.maximum(1e-5, x))
    
    def amp_to_db_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert amplitude to decibels (PyTorch version).
        
        Args:
            x: Input amplitude tensor
            
        Returns:
            x_db: Converted dB tensor
        """
        return 20 * torch.log10(torch.clamp(x, min=1e-5))
    
    def db_to_amp(self, x: np.ndarray) -> np.ndarray:
        """
        Convert decibels to amplitude.
        
        Args:
            x: Input dB value
            
        Returns:
            x_amp: Converted amplitude
        """
        return np.power(10.0, x * 0.05)
    
    def normalize(self, S: np.ndarray) -> np.ndarray:
        """
        Normalize spectrogram.
        
        Args:
            S: Input spectrogram
            
        Returns:
            S_norm: Normalized spectrogram
        """
        if self.signal_norm:
            if self.symmetric_norm:
                S = np.clip(
                    (S - self.ref_level_db) / (-self.min_level_db),
                    -1.0, 0.0
                )
                S = (S + 1.0) * 0.5
            else:
                S = np.clip(
                    (S - self.ref_level_db) / (-self.min_level_db),
                    0.0, 1.0
                )
        return S
    
    def normalize_torch(self, S: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrogram (PyTorch version).
        
        Args:
            S: Input spectrogram tensor
            
        Returns:
            S_norm: Normalized spectrogram tensor
        """
        if self.signal_norm:
            if self.symmetric_norm:
                S = torch.clamp(
                    (S - self.ref_level_db) / (-self.min_level_db),
                    -1.0, 0.0
                )
                S = (S + 1.0) * 0.5
            else:
                S = torch.clamp(
                    (S - self.ref_level_db) / (-self.min_level_db),
                    0.0, 1.0
                )
        return S
    
    def denormalize(self, S: np.ndarray) -> np.ndarray:
        """
        Denormalize spectrogram.
        
        Args:
            S: Normalized spectrogram
            
        Returns:
            S_denorm: Denormalized spectrogram
        """
        if self.signal_norm:
            if self.symmetric_norm:
                S = S * 2.0 - 1.0
                S = (S * -self.min_level_db) + self.ref_level_db
            else:
                S = (S * -self.min_level_db) + self.ref_level_db
        return S
    
    def denormalize_torch(self, S: torch.Tensor) -> torch.Tensor:
        """
        Denormalize spectrogram (PyTorch version).
        
        Args:
            S: Normalized spectrogram tensor
            
        Returns:
            S_denorm: Denormalized spectrogram tensor
        """
        if self.signal_norm:
            if self.symmetric_norm:
                S = S * 2.0 - 1.0
                S = (S * -self.min_level_db) + self.ref_level_db
            else:
                S = (S * -self.min_level_db) + self.ref_level_db
        return S
    
    def mel_to_linear(self, mel: np.ndarray) -> np.ndarray:
        """
        Convert mel spectrogram back to linear scale.
        
        Args:
            mel: Mel spectrogram
            
        Returns:
            mag: Linear magnitude spectrogram
        """
        mel = self.db_to_amp(self.denormalize(mel))
        return np.dot(self.inv_mel_basis, mel)
    
    def griffin_lim(self, mel: np.ndarray) -> np.ndarray:
        """
        Reconstruct waveform from mel spectrogram using Griffin-Lim algorithm.
        
        Args:
            mel: Mel spectrogram
            
        Returns:
            wav: Reconstructed waveform
        """
        mag = self.mel_to_linear(mel)
        
        # Initialize with random phase
        angles = np.exp(2j * np.pi * np.random.rand(*mag.shape))
        complex_spec = mag * angles
        
        for _ in range(self.griffin_lim_iters):
            # Reconstruct waveform
            wav = librosa.istft(
                complex_spec,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window='hann'
            )
            
            # Recompute STFT
            new_spec = self.stft(wav)
            angles = np.exp(1j * np.angle(new_spec))
            complex_spec = mag * angles
            
        wav = librosa.istft(
            complex_spec,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann'
        )
        
        if self.preemphasize:
            wav = self.inv_preemphasis(wav)
            
        return wav
    
    def text_to_sequence(self, text: str) -> np.ndarray:
        """
        Convert text to sequence of character IDs.
        
        Args:
            text: Input text string
            
        Returns:
            sequence: Array of character IDs
        """
        # Basic implementation - should be replaced with proper text processing
        char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?-'"
        char_to_id = {c: i for i, c in enumerate(char_set)}
        
        sequence = []
        for c in text.lower():
            if c in char_to_id:
                sequence.append(char_to_id[c])
            else:
                sequence.append(0)  # Unknown character
        
        return np.array(sequence)
    
    def trim_silence(self, wav: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Trim leading and trailing silence from waveform.
        
        Args:
            wav: Input waveform
            threshold: Silence threshold
            
        Returns:
            trimmed_wav: Waveform with silence trimmed
        """
        # Find non-silent intervals
        intervals = librosa.effects.split(wav, top_db=40, frame_length=1024, hop_length=256)
        
        if len(intervals) == 0:
            return wav
            
        # Get first and last non-silent intervals
        start = intervals[0][0]
        end = intervals[-1][1]
        
        return wav[start:end]
    
    def get_mel(self, wav_path: str) -> np.ndarray:
        """
        Load audio file and compute mel spectrogram.
        
        Args:
            wav_path: Path to audio file
            
        Returns:
            mel: Mel spectrogram
        """
        wav = self.load_wav(wav_path)
        return self.melspectrogram(wav)