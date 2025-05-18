import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

data_root = r"processed/processed_speaker_verification"
checkpoint_dir = r"speaker_checkpoints"
batch_size = "64"
num_utterances = "5"
epochs = "100"
learning_rate = "1e-4"

command = [
    sys.executable, "-m", "training.train_speaker_encoder",
    "--data_root", data_root,
    "--checkpoint_dir", checkpoint_dir,
    "--batch_size", batch_size,
    "--num_utterances", num_utterances,
    "--epochs", epochs,
    "--lr", learning_rate
]

subprocess.run(command)

