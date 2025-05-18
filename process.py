import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

data_root = r"D:/work/lipsync/multispeaker-tts/datasets/speaker_verification_data"
output_dir = r"processed/processed_speaker_verification"

command = [
    sys.executable, "-m", "data.preprocess",
    "--data_root", data_root,
    "--output_dir", output_dir,
    "--sample_rate", "16000",
    "--n_fft", "1024",
    "--hop_length", "256"
]

subprocess.run(command)
