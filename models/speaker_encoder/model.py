import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path   
sys.path.append(str(Path(__file__).parent.parent))
from __init__ import logger

class SpeakerEncoder(nn.Module):
    # speaker encoder networks based on LSTM architecture
    
    def __init__(self, audio_config, hidden_dim=768, proj_dim=256, num_layers=3):
        super().__init__()
        self.audio_config = audio_config
        self.lstm = nn.LSTM(
            input_size=audio_config['num_mels'],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first= True
        )
        self.projection = nn.Linear(hidden_dim, proj_dim)
        self.num_layers = num_layers
        self.hidden_dims = hidden_dim
        self.embedding_dim = proj_dim
      
    def forward(self, x):
        # x: (batch, seq_len, input_dim) | (N*M, T, num_mels)
        lstm_out, _ = self.lstm(x) # (batch, seq_len, hidden_dim) (N*M, T, hidden_dim)
        # o/p from final frame
        last_out = lstm_out[:, -1, :] # (batch, hidden_dim) | (N*M, hidden_dim)
        embedding = self.projection(last_out) # (batch, proj_dim) | (N*M, proj_dim)
        # L2 norm
        embedding = F.normalize(embedding, p=2, dim=1)
        logger.debug(f"Speaker Encoder output shape: {embedding.shape}")
        return embedding

    @torch.no_grad()
    def embed_utterance(self, utterance):
        # embed : single utter -> ovrlap_add processing
        # 800ms win -> 50% ovrlap
        
        window_size = int(0.8 * self.sample_rate)
        hop_size = window_size // 2
        
        #split into windows
        windows = []
        for start in range(0, len(utterance) - window_size + 1, hop_size):
            window = utterance[start:start+window_size]
            windows.append(window)
        
        if not windows:
            return torch.zeros(self.proj_dim)
        
        # process each win
        embeddings = []
        for window in windows:
            mel = self.audio_to_mel(window)
            emb = self.forward(mel.unsqueeze(0))
            embeddings.append(emb)
        
        embedding = torch.mean(torch.stack(embeddings), dim=0)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0)
    
    
            
        