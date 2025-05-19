import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple
import soundfile as sf

class SpeakerVerificationDataset(Dataset):
    # Dataset for speaker verification/encoder training
    
    def __init__(self, data_root: str, audio_processor, num_utterances: int = 5, 
                 min_duration: float = 1.6, max_duration: float = 3.0):
        """
        Args:
            data_root: Root directory containing speaker directories
            audio_processor: Audio processor instance
            num_utterances: Number of utterances per speaker to include in each batch
            min_duration: Minimum duration of utterances in seconds
            max_duration: Maximum duration of utterances in seconds
        """
        self.ap = audio_processor
        self.num_utterances = num_utterances
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Collect speaker data
        self.speakers = []
        self.speaker_to_utts = {}
        
        speaker_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        for speaker in speaker_dirs:
            speaker_path = os.path.join(data_root, speaker)
            utt_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
            
            if len(utt_files) >= num_utterances:
                self.speakers.append(speaker)
                self.speaker_to_utts[speaker] = [
                    os.path.join(speaker_path, f) for f in utt_files
                ]
        
        # Calculate lengths for bucketing
        self.utt_lengths = {}
        for speaker, utts in self.speaker_to_utts.items():
            for utt in utts:
                duration = self._get_duration(utt)
                if self.min_duration <= duration <= self.max_duration:
                    bucket = int(duration * 100)  # Bucket by 10ms
                    if bucket not in self.utt_lengths:
                        self.utt_lengths[bucket] = []
                    self.utt_lengths[bucket].append((speaker, utt))
    
    def _get_duration(self, wav_path: str) -> float:
        # Get duration of audio file in seconds
        with sf.SoundFile(wav_path) as f:
            return len(f) / f.samplerate
    
    def __len__(self) -> int:
        return len(self.speakers)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get a batch of utterances from the same speaker
        speaker = self.speakers[idx]
        utt_paths = random.sample(self.speaker_to_utts[speaker], self.num_utterances)
        
        mels = []
        for path in utt_paths:
            wav = self.ap.load_wav(path)
            mel = self.ap.melspectrogram(wav)
            mels.append(mel)
        
        # Pad to max length in batch
        max_len = max(m.shape[1] for m in mels)
        padded_mels = np.stack([
            np.pad(m, ((0, 0), (0, max_len - m.shape[1])), 
            mode='constant'
        ) for m in mels])
        
        return {
            'mel': torch.FloatTensor(padded_mels),
            'speaker': speaker
        }
    
    def get_triplet(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get anchor, positive, negative samples for triplet loss
        # Anchor and positive from same speaker
        
        anchor_speaker = random.choice(self.speakers)
        anchor_path, pos_path = random.sample(self.speaker_to_utts[anchor_speaker], 2)
        
        # Negative from different speaker
        neg_speaker = random.choice([s for s in self.speakers if s != anchor_speaker])
        neg_path = random.choice(self.speaker_to_utts[neg_speaker])
        
        # Process audio
        anchor_mel = self.ap.melspectrogram(self.ap.load_wav(anchor_path))
        pos_mel = self.ap.melspectrogram(self.ap.load_wav(pos_path))
        neg_mel = self.ap.melspectrogram(self.ap.load_wav(neg_path))
        
        return (
            torch.FloatTensor(anchor_mel),
            torch.FloatTensor(pos_mel),
            torch.FloatTensor(neg_mel)
        )