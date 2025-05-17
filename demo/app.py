import streamlit as st
import numpy as np
import soundfile as sf
from st_audiorec import st_audiorec

#load config and models
@st.cache(allow_output_mutation=True)
def load_tts():
    config = {
        'audio': {
            'sample_rate': 22050,
        },
        'speaker_encoder': {
            
        },
        'synthesizer': {
            
        },
        'vocoder': {
            
        }
    }
    return config

tts = load_tts()

# ui
st.title("Multispeaker Text-to-Speech Synthesis")
st.write("This demo shows a transfer learning approach to multispeaker TTS, where a speaker verification model is used to condition a Tacotron 2 synthesizer and WaveNet vocoder.")

# input components
text_input = st.text_area("Enter text to synthesize:", "Hello, this is a demonstration of multispeaker text-to-speech synthesis.")

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

if st.button("Synthesize") and selected_audio is not None:
    # process ref audio
    audio, sr = sf.read(selected_audio)
    if sr != tts.config['audio']['sample_rate']:
        st.warning(f"Resampling from {sr}Hz to {tts.config['audio']['sample_rate']}Hz")
        # audio = tts.ap.resample(audio, sr, tts.config['audio']['sample_rate']) #implement
        
    # synthesize
    # with st.spinner("Synthesizing..."):
        # waveform = tts.synthesize(text_input, audio) implement
    
    # display
    st.audio(selected_audio, sample_rate=tts.config['audio']['sample_rate'])
    st.success("Synthesys Complete")