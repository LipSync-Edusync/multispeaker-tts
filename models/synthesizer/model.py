import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import LocationSensitiveAttention

class Encoder(nn.Module):
    # text encoder with CBHG architecture
    def __init__(self, num_chars, embedding_dim=256, encoder_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.conv_bank = nn.ModuleList([
            nn.Conv1d(embedding_dim, encoder_dim, kernel_size=k, padding=k//2)
            for k in range(1, 9)
        ])
        self.conv_proj = nn.Sequential(
            nn.Conv1d(encoder_dim * 8, encoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(encoder_dim, encoder_dim//2, bidirectional=True, batch_first=True)
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        # conv bank
        conv_outputs = []
        for conv in self.conv_bank:
            conv_output = conv(x)
            conv_outputs.append(F.relu(conv_output))
        
        x = torch.cat(conv_outputs, dim=1)  # (batch, 8*encoder_dim, seq_len)
        x = self.conv_proj(x)  # (batch, encoder_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, encoder_dim)
        
        # bidirectional lstm
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # (batch, seq_len, encoder_dim)
        return x

class Decoder(nn.Module):
    # autoregressive decoder with attention
    def __init__(self, encoder_dim=256, decoder_dim=1024, 
                 n_mels=80, speaker_embed_dim=256):
        super().__init__()
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, decoder_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.attention = LocationSensitiveAttention(decoder_dim, encoder_dim)
        self.lstm1 = nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim)
        self.lstm2 = nn.LSTMCell(decoder_dim, decoder_dim)
        
        # speaker conditioning
        self.speaker_proj = nn.Linear(speaker_embed_dim, decoder_dim)
        
        # mel prediction
        self.mel_proj = nn.Linear(decoder_dim, n_mels)
        
        # stop token prediction
        self.stop_proj = nn.Linear(decoder_dim, 1)
        
    def forward(self, encoder_outputs, speaker_embed, mels=None):
        batch_size = encoder_outputs.size(0)
        
        # initialize states
        h1 = torch.zeros(batch_size, self.lstm1.hidden_size).to(encoder_outputs.device)
        c1 = torch.zeros(batch_size, self.lstm1.hidden_size).to(encoder_outputs.device)
        h2 = torch.zeros(batch_size, self.lstm2.hidden_size).to(encoder_outputs.device)
        c2 = torch.zeros(batch_size, self.lstm2.hidden_size).to(encoder_outputs.device)
        
        # speaker conditioning
        speaker_embed = self.speaker_proj(speaker_embed)
        
        # initialize attention
        self.attention.init_states(encoder_outputs)
        
        # outputs
        mel_outputs, stop_outputs = [], []
        
        # first input is zeros
        prev_mel = torch.zeros(batch_size, self.mel_proj.out_features).to(encoder_outputs.device)
        
        # autoregressive decoding
        for i in range(mels.size(1) if mels is not None else 1000):
            # prenet
            prenet_out = self.prenet(prev_mel)
            
            # attention
            context, attention_weights = self.attention(prenet_out + speaker_embed)
            
            # lstm cells
            h1, c1 = self.lstm1(torch.cat([prenet_out, context], dim=1), (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            
            # projections
            mel_out = self.mel_proj(h2)
            stop_out = torch.sigmoid(self.stop_proj(h2))
            
            mel_outputs.append(mel_out)
            stop_outputs.append(stop_out)
            
            # teacher forcing or next step
            if mels is not None and i < mels.size(1) - 1:
                prev_mel = mels[:, i+1, :]
            else:
                prev_mel = mel_out
                
            # stop if all sequences have finished
            if mels is None and torch.all(stop_out > 0.5):
                break
                
        return torch.stack(mel_outputs, dim=1), torch.stack(stop_outputs, dim=1)

class Synthesizer(nn.Module):
    # complete Tacotron 2 based synthesizer with speaker conditioning
    
    def __init__(self, num_chars, n_mels=80, speaker_embed_dim=256):
        super().__init__()
        self.encoder = Encoder(num_chars)
        self.decoder = Decoder(
            encoder_dim=self.encoder.lstm.hidden_size,
            n_mels=n_mels,
            speaker_embed_dim=speaker_embed_dim
        )
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Dropout(0.5)
        )
        
    def forward(self, text, speaker_embed, mels=None):
        # encode text
        encoder_outputs = self.encoder(text)
        
        # decode mel spectrogram
        mel_outputs, stop_outputs = self.decoder(encoder_outputs, speaker_embed, mels)
        
        # post-processing
        residual = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + residual.transpose(1, 2)
        
        return mel_outputs, mel_outputs_postnet, stop_outputs