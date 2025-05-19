import os
import shutil
from pathlib import Path
from tqdm import tqdm

TEST_DIR = Path(r"/home/oem/Lipsync-Edusync/datasets/speaker_verification_unrefined/vox1_test_wav/wav")
TEST_DEST = Path(r"/home/oem/Lipsync-Edusync/datasets/speaker_verification_test")

VOX_SRC = TEST_DIR
DEST = TEST_DEST

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
