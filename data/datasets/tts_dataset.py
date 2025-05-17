import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple

class TTSDataset(Dataset):
    # Dataset for TTS training with text-audio pairs
    
    def __init__(self, data_root: str, audio_processor, metadata_file: str = 'metadata.csv',
                 speaker_ids: Dict[str, int] = None, text_cleaner=None):
        """
        Args:
            data_root: Root directory containing audio files
            audio_processor: Audio processor instance
            metadata_file: File containing text-audio mappings
            speaker_ids: Mapping from speaker names to IDs
            text_cleaner: Text cleaning function
        """
        self.ap = audio_processor
        self.speaker_ids = speaker_ids if speaker_ids is not None else {}
        self.text_cleaner = text_cleaner
        
        # Load metadata
        self.metadata = []
        with open(os.path.join(data_root, metadata_file), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    wav_path = os.path.join(data_root, 'wavs', parts[0] + '.wav')
                    text = parts[1]
                    speaker = parts[2] if len(parts) > 2 else 'default'
                    
                    if os.path.exists(wav_path):
                        self.metadata.append({
                            'wav_path': wav_path,
                            'text': text,
                            'speaker': speaker
                        })
        
        # Build speaker ID mapping if not provided
        if not self.speaker_ids:
            speakers = set(item['speaker'] for item in self.metadata)
            self.speaker_ids = {speaker: i for i, speaker in enumerate(speakers)}
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.metadata[idx]
        
        # Load and process audio
        wav = self.ap.load_wav(item['wav_path'])
        mel = self.ap.melspectrogram(wav)
        
        # Process text
        text = item['text']
        if self.text_cleaner:
            text = self.text_cleaner(text)
        text_seq = self.ap.text_to_sequence(text)
        
        # Speaker embedding
        speaker_id = self.speaker_ids[item['speaker']]
        
        return {
            'text': torch.LongTensor(text_seq),
            'mel': torch.FloatTensor(mel),
            'speaker_id': torch.LongTensor([speaker_id]),
            'wav_path': item['wav_path']
        }
    
    def get_collate_fn(self):
        # Create collate function for DataLoader
        def collate_fn(batch):
            # Sort by text length (descending)
            batch = sorted(batch, key=lambda x: x['text'].shape[0], reverse=True)
            
            # Pad text sequences
            text_lengths = [x['text'].shape[0] for x in batch]
            max_text_len = max(text_lengths)
            text_padded = torch.LongTensor(len(batch), max_text_len)
            text_padded.zero_()
            for i, x in enumerate(batch):
                text_padded[i, :x['text'].shape[0]] = x['text']
            
            # Pad mel spectrograms
            num_mels = batch[0]['mel'].shape[0]
            mel_lengths = [x['mel'].shape[1] for x in batch]
            max_mel_len = max(mel_lengths)
            mel_padded = torch.FloatTensor(len(batch), num_mels, max_mel_len)
            mel_padded.zero_()
            for i, x in enumerate(batch):
                mel_padded[i, :, :x['mel'].shape[1]] = x['mel']
            
            # Speaker IDs
            speaker_ids = torch.LongTensor([x['speaker_id'] for x in batch])
            
            return {
                'text': text_padded,
                'text_lengths': torch.LongTensor(text_lengths),
                'mel': mel_padded,
                'mel_lengths': torch.LongTensor(mel_lengths),
                'speaker_ids': speaker_ids,
                'wav_paths': [x['wav_path'] for x in batch]
            }
        
        return collate_fn