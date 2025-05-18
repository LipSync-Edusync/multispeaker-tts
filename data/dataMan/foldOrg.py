import os
import shutil
from pathlib import Path
from tqdm import tqdm

VOX_SRC = Path(r"D:\Work\lipsync\multispeaker-tts\datasets\vox1_dev_wav")
DEST = Path(r"D:\Work\lipsync\multispeaker-tts\datasets\speaker_verification_data")

DEST.mkdir(parents=True, exist_ok=True)

for speaker_dir in tqdm(sorted(VOX_SRC.iterdir()), desc="Processing speakers"):
    if not speaker_dir.is_dir():
        continue
    speaker_id = speaker_dir.name
    target_dir = DEST / speaker_id
    target_dir.mkdir(exist_ok=True)

    for wav_path in speaker_dir.rglob("*.wav"):
        subfolder = wav_path.parent.name
        filename = wav_path.name
        unique_name = f"{subfolder}_{filename}"
        dest_file = target_dir / unique_name
        shutil.copy2(wav_path, dest_file)
