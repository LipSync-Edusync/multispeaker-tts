import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

data_root = r"/home/oem/Lipsync-Edusync/datasets/speaker_verification_test/processed"
checkpoint_dir = r"checkpoints/speaker_encoder"
num_speakers = "4" # N
num_utterances = "5" # M
epochs = "1"
learning_rate = "1e-4"

command = [
    sys.executable, "-m", "training.train_speaker_encoder",
    "--data_root", data_root,
    "--checkpoint_dir", checkpoint_dir,
    "--num_speakers", num_speakers,
    "--num_utterances", num_utterances,
    "--epochs", epochs,
    "--lr", learning_rate
]

subprocess.run(command)

