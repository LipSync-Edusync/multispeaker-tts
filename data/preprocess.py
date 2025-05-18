# data/preprocess.py
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from data.datasets.utils import create_speaker_mapping, load_metadata
from data.datasets.tts_dataset import TTSDataset
from utils import AudioProcessor
import numpy as np

def preprocess_dataset(data_root: str, output_dir: str, audio_processor):
    # Preprocess TTS dataset and save processed files
    os.makedirs(output_dir, exist_ok=True)
    
    # Create speaker mapping
    speaker_mapping = create_speaker_mapping(data_root)
    with open(os.path.join(output_dir, 'speakers.json'), 'w') as f:
        json.dump(speaker_mapping, f)
    
    # Process metadata
    metadata_path = os.path.join(data_root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    metadata = load_metadata(metadata_path)
    
    # Create output directories
    mel_dir = os.path.join(output_dir, 'mels')
    wav_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    
    # Process each audio file
    processed_metadata = []
    for item in tqdm(metadata, desc="Processing dataset"):
        wav_path = os.path.join(data_root, item['wav_path'])
        if not os.path.exists(wav_path):
            continue
        
        # Load and process audio
        wav = audio_processor.load_wav(wav_path)
        mel = audio_processor.melspectrogram(wav)
        
        # Save processed files
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        mel_path = os.path.join(mel_dir, base_name + '.npy')
        wav_path_out = os.path.join(wav_dir, base_name + '.wav')
        
        np.save(mel_path, mel)
        audio_processor.save_wav(wav, wav_path_out)
        
        # Update metadata
        processed_metadata.append({
            'mel_path': mel_path,
            'wav_path': wav_path_out,
            'text': item['text'],
            'speaker': item['speaker']
        })
    
    # Save processed metadata
    with open(os.path.join(output_dir, 'metadata_processed.csv'), 'w') as f:
        for item in processed_metadata:
            f.write(f"{item['mel_path']}|{item['text']}|{item['speaker']}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, 
                       help='Root directory of raw dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed data')
    parser.add_argument('--sample_rate', type=int, default=22050,
                       help='Target sample rate')
    parser.add_argument('--n_fft', type=int, default=1024,
                       help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=256,
                       help='Hop length for STFT')
    args = parser.parse_args()
    
    
    ap = AudioProcessor(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )
    
    preprocess_dataset(args.data_root, args.output_dir, ap)