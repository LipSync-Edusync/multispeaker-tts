import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List
import soundfile as sf
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from __init__ import logger



class SpeakerVerificationDataset(Dataset):
    def __init__(self, data_root: str, audio_processor, num_speakers: int = 64, 
                 num_utterances: int = 5, min_duration: float = 1.0, 
                 max_duration: float = 3.0):
        """
        Args:
            data_root: Root directory containing speaker directories
            audio_processor: Audio processor instance
            num_speakers: Number of speakers per batch (N)
            num_utterances: Number of utterances per speaker (M)
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
        """
        self.ap = audio_processor
        self.num_speakers = num_speakers
        self.num_utterances = num_utterances
        
        
        # Collect all valid speakers and their utterances
        self.speakers = []
        self.speaker_to_utts = {}
        logger.debug(f" === reach test ===")
        
        for speaker in os.listdir(data_root):
            speaker_path = os.path.join(data_root, speaker)
            if not os.path.isdir(speaker_path):
                logger.debug(f"Skipping {speaker_path}, not in directory")
                continue
                
            utt_files = []
            for f in os.listdir(speaker_path):
                if f.endswith('.wav'):
                    path = os.path.join(speaker_path, f)
                    duration = self._get_duration(path)
                    if min_duration <= duration <= max_duration:
                        utt_files.append(path)
            
            if len(utt_files) >= num_utterances:
                self.speakers.append(speaker)
                self.speaker_to_utts[speaker] = utt_files
        
        if len(self.speakers) < num_speakers:
            raise ValueError(f"Only {len(self.speakers)} speakers meet requirements, but need {num_speakers}")

    def _get_duration(self, wav_path: str) -> float:
        with sf.SoundFile(wav_path) as f:
            return len(f) / f.samplerate

    def __len__(self) -> int:
        return len(self.speakers) // self.num_speakers

    def __getitem__(self, idx: int) -> Dict[str, List[np.ndarray]]:
        """Returns utterances from one speaker"""
        speaker = self.speakers[idx]
        utt_paths = random.sample(self.speaker_to_utts[speaker], self.num_utterances)
        
        mels = []
        for path in utt_paths:
            wav = self.ap.load_wav(path)
            mel = self.ap.melspectrogram(wav)
            mels.append(mel)
        
        return {
            'mels': mels,  # List of melspectrograms
            'speaker': speaker
        }

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function that creates proper N x M batch for GE2E loss
        Returns:
            Dictionary with:
            - mels: Tensor of shape (N*M, T, n_mels)
            - labels: Speaker IDs (N,)
        """
        # First pad all mels to max length in batch
        all_mels = [mel for item in batch for mel in item['mels']]
        max_len = max(m.shape[1] for m in all_mels)
        
        padded_mels = []
        for item in batch:
            for mel in item['mels']:
                pad_amount = max_len - mel.shape[1]
                padded = np.pad(mel, ((0, 0), (0, pad_amount)), 
                             mode='constant')
                padded_mels.append(padded)
        
        # Stack all utterances (N*M, n_mels, T)
        mels_tensor = torch.FloatTensor(np.stack(padded_mels))
        
        # Create speaker labels
        labels = [item['speaker'] for item in batch]
        
        return {
            'mels': mels_tensor,  # (N*M, n_mels, T)
            'labels': labels     # (N,)
        }