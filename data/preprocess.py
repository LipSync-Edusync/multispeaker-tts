import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from data.datasets.utils import create_speaker_mapping, load_metadata
from utils import AudioProcessor
import numpy as np
import logging

def preprocess_dataset(data_root: str, output_dir: str, audio_processor):
    logger = logging.getLogger("preprocess")
    logger.info(f"Starting preprocessing: data_root={data_root}, output_dir={output_dir}")

    # val
    if not os.path.isdir(data_root):
        raise NotADirectoryError(f"Data root directory does not exist: {data_root}")

    # create o/p dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory ready at: {output_dir}")

    # speaker mapping
    try:
        speaker_mapping = create_speaker_mapping(data_root)
        if not speaker_mapping:
            logger.warning("Speaker mapping is empty. Check your data_root contents.")
        with open(os.path.join(output_dir, 'speakers.json'), 'w') as f:
            json.dump(speaker_mapping, f, indent=2)
        logger.info(f"Saved speaker mapping to {os.path.join(output_dir, 'speakers.json')}")
    except Exception as e:
        logger.error(f"Failed to create speaker mapping: {e}")
        raise

    # metadata file
    metadata_path = os.path.join(data_root, 'metadata.csv')
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # load metadata
    try:
        metadata = load_metadata(metadata_path)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        raise

    if not metadata:
        raise ValueError("Metadata file is empty or invalid.")

    # o/p dirs for mel and wav files
    mel_dir = os.path.join(output_dir, 'mels')
    wav_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    logger.info(f"Created mel spectrogram dir: {mel_dir}")
    logger.info(f"Created wav dir: {wav_dir}")

    processed_metadata = []
    skipped_count = 0

    # process each audio entry
    for idx, item in enumerate(tqdm(metadata, desc="Processing dataset")):
        try:
            wav_path = os.path.join(data_root, item['wav_path'])
            if not os.path.isfile(wav_path):
                logger.warning(f"Missing wav file at {wav_path}, skipping entry {idx}")
                skipped_count += 1
                continue

            # Load audio
            wav = audio_processor.load_wav(wav_path)
            if wav is None or len(wav) == 0:
                logger.warning(f"Empty or corrupted audio file at {wav_path}, skipping entry {idx}")
                skipped_count += 1
                continue

            # Generate mel spectrogram
            mel = audio_processor.melspectrogram(wav)
            if mel is None or mel.size == 0:
                logger.warning(f"Failed to generate mel for {wav_path}, skipping entry {idx}")
                skipped_count += 1
                continue

            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            mel_path = os.path.join(mel_dir, base_name + '.npy')
            wav_path_out = os.path.join(wav_dir, base_name + '.wav')

            # Save mel and wav
            np.save(mel_path, mel)
            audio_processor.save_wav(wav, wav_path_out)

            processed_metadata.append({
                'mel_path': mel_path,
                'wav_path': wav_path_out,
                'text': item['text'],
                'speaker': item['speaker']
            })
        except Exception as e:
            logger.error(f"Error processing entry {idx} ({item.get('wav_path', 'unknown')}): {e}")
            skipped_count += 1
            continue

    if not processed_metadata:
        raise RuntimeError("No valid audio files were processed. Check your dataset and preprocessing parameters.")

    # Save processed metadata CSV
    processed_metadata_path = os.path.join(output_dir, 'metadata_processed.csv')
    try:
        with open(processed_metadata_path, 'w') as f:
            for item in processed_metadata:
                f.write(f"{item['mel_path']}|{item['text']}|{item['speaker']}\n")
        logger.info(f"Saved processed metadata to {processed_metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save processed metadata CSV: {e}")
        raise

    logger.info(f"Preprocessing completed. {len(processed_metadata)} files processed, {skipped_count} files skipped.")

if __name__ == '__main__':
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("preprocess_main")

    parser = argparse.ArgumentParser(description="Preprocess TTS dataset")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of raw dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Target sample rate')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=256, help='Hop length for STFT')

    args = parser.parse_args()

    try:
        ap = AudioProcessor(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length
        )
        preprocess_dataset(args.data_root, args.output_dir, ap)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)
