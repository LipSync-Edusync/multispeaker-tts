import torch
import numpy as np
from models import SpeakerEncoder
from models import Synthesizer
from models import WaveNet
from utils.audio import AudioProcessor

class MultispeakerTTS:
    """End-to-end multispeaker TTS system"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.ap = AudioProcessor(**config['audio'])
        self.speaker_encoder = self._load_speaker_encoder()
        self.synthesizer = self._load_synthesizer()
        self.vocoder = self._load_vocoder()
        
    def _load_speaker_encoder(self):
        model = SpeakerEncoder(**self.config['speaker_encoder'])
        checkpoint = torch.load(self.config['speaker_encoder_path'], map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model.to(self.device)
    
    def _load_synthesizer(self):
        model = Synthesizer(**self.config['synthesizer'])
        checkpoint = torch.load(self.config['synthesizer_path'], map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model.to(self.device)
    
    def _load_vocoder(self):
        model = WaveNet(**self.config['vocoder'])
        checkpoint = torch.load(self.config['vocoder_path'], map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model.to(self.device)
    
    def synthesize(self, text, reference_audio):
        """
        Synthesize speech from text using reference audio for speaker characteristics
        
        Args:
            text: Input text to synthesize
            reference_audio: Reference audio waveform (numpy array) for speaker characteristics
        """
        # Get speaker embedding from reference audio
        with torch.no_grad():
            # Preprocess reference audio
            mel = self.ap.melspectrogram(reference_audio)
            mel = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
            
            # Get speaker embedding
            speaker_embed = self.speaker_encoder(mel)
            
            # Preprocess text
            text_seq = self.ap.text_to_sequence(text)
            text_seq = torch.LongTensor(text_seq).unsqueeze(0).to(self.device)
            
            # Synthesize mel spectrogram
            mel_outputs, _, _ = self.synthesizer(text_seq, speaker_embed)
            mel = mel_outputs.squeeze(0).transpose(0, 1)
            
            # Synthesize waveform
            mel = mel.unsqueeze(0)
            waveform = self.vocoder.inference(mel)
            
        return waveform.squeeze().cpu().numpy()