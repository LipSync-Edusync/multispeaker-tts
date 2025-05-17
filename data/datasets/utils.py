# data/datasets/utils.py
import os
import random
import numpy as np
from typing import List, Dict, Tuple

def split_dataset(dataset, val_ratio=0.1, test_ratio=0.1):
    # Split dataset into train/val/test sets
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size
    
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    return train_indices, val_indices, test_indices

def create_speaker_mapping(data_root: str) -> Dict[str, int]:
    # Create mapping from speaker names to IDs 
    speakers = set()
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.wav'):
                speaker = os.path.basename(root)
                speakers.add(speaker)
    
    return {speaker: i for i, speaker in enumerate(sorted(speakers))}

def load_metadata(metadata_path: str) -> List[Dict]:
    # Load metadata file with text-audio mappings 
    metadata = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                metadata.append({
                    'wav_path': parts[0].strip(),
                    'text': parts[1].strip(),
                    'speaker': parts[2].strip() if len(parts) > 2 else None
                })
    return metadata

def text_cleaner(text: str) -> str:
    # Basic text cleaning function
    # Convert to lowercase
    
    text = text.lower()
    
    text = ''.join(c for c in text if c.isalnum() or c in " ,.!?-'")
    return text.strip()