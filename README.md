# Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech (TTS)

Implementation of the NeurIPS 2018 paper:  
**Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis**  
by Ye Jia, Yu Zhang, Ron J. Weiss, et al. ([arXiv:1806.04558](https://arxiv.org/abs/1806.04558))

This system synthesizes speech for seen and unseen speakers by leveraging a pre-trained speaker encoder and a Tacotron 2-based TTS synthesizer.

---

## Quickstart

### Installation

```bash
git clone https://github.com/your-username/multispeaker-tts.git
cd multispeaker-tts
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
