import streamlit as st
import numpy as np
import soundfile as sf
from inference import MultispeakerTTS
from utils import AudioProcessor
from st_audiorec import st_audiorec

# Load config and models
@st.cache(allow_output_mutation=True)
def load_tts():
    config = {
        'audio': {
            'sample_rate': 22050,
            'num_mels': 80,
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'mel_fmin': 0.0,
            'mel_fmax': 8000.0
        },
        'speaker_encoder': {
            'input_dim': 40,
            'hidden_dim': 768,
            'proj_dim': 256,
            'num_layers': 3
        },
        'synthesizer': {
            'num_chars': 256,
            'n_mels': 80,
            'speaker_embed_dim': 256
        },
        'vocoder': {
            'n_mels': 80,
            'residual_channels': 512,
            'gate_channels': 512,
            'skip_channels': 256,
            'kernel_size': 3,
            'n_layers': 30,
            'dropout': 0.05
        }
    }
    return MultispeakerTTS(config)

tts = load_tts()

# UI
st.title("Multispeaker Text-to-Speech Synthesis")
st.write("""
This demo shows a transfer learning approach to multispeaker TTS, where a speaker verification 
model is used to condition a Tacotron 2 synthesizer and WaveNet vocoder.
""")

# Input components
text_input = st.text_area("Enter text to synthesize:", "Hello, this is a demonstration of multispeaker text-to-speech synthesis.")
# reference_audio = st.file_uploader("Upload reference audio for speaker characteristics:", type=["wav", "mp3"])

# reference audio (upload or record)
st.subheader("Reference Audio for Speaker Embedding")

col1, col2 = st.columns(2)

with col1:
    reference_audio = st.file_uploader("Upload reference audio:", type=["wav", "mp3"])

with col2:
    st.write("Or record your voice:")
    recorded_audio = st_audiorec()

selected_audio = None
if reference_audio is not None:
    selected_audio = reference_audio
elif recorded_audio is not None:
    selected_audio = recorded_audio

if st.button("Synthesize") and reference_audio is not None:
    # Process reference audio
    audio, sr = sf.read(reference_audio)
    if sr != tts.config['audio']['sample_rate']:
        st.warning(f"Resampling from {sr}Hz to {tts.config['audio']['sample_rate']}Hz")
        audio = tts.ap.resample(audio, sr, tts.config['audio']['sample_rate'])
    
    # Synthesize
    with st.spinner("Synthesizing..."):
        waveform = tts.synthesize(text_input, audio)
    
    # Display audio
    st.audio(waveform, sample_rate=tts.config['audio']['sample_rate'])
    st.success("Synthesis complete!")