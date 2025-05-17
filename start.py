import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "data"))
sys.path.insert(0, str(Path(__file__).parent / "demo"))
sys.path.insert(0, str(Path(__file__).parent / "inference"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
sys.path.insert(0, str(Path(__file__).parent / "papers"))
sys.path.insert(0, str(Path(__file__).parent / "training"))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from demo.app import main

if __name__ == "__main__":
    main()