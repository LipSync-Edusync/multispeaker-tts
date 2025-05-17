import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConvBlock(nn.Module):
    # dilated convolution block with gated activation
    def __init__(self, residual_channels, gate_channels, skip_channels, 
                 kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(
            residual_channels, 
            2 * gate_channels,  # for gated activation
            kernel_size, 
            padding=(kernel_size * dilation - dilation) // 2,
            dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Conv1d(gate_channels, residual_channels, 1)
        self.skip_proj = nn.Conv1d(gate_channels, skip_channels, 1)
        
    def forward(self, x):
        residual = x
        x = self.conv(x)
        
        # Split for gated activation
        gate, filter = torch.split(x, x.size(1) // 2, dim=1)
        x = torch.tanh(filter) * torch.sigmoid(gate)
        
        x = self.dropout(x)
        
        # Projections
        residual_out = self.residual_proj(x) + residual
        skip_out = self.skip_proj(x)
        
        return residual_out, skip_out

class WaveNet(nn.Module):
    # WaveNet vocoder conditioned on mel spectrograms
    
    def __init__(self, n_mels=80, residual_channels=512, 
                 gate_channels=512, skip_channels=256, 
                 kernel_size=3, n_layers=30, dropout=0.05):
        super().__init__()
        
        # Initial convolutions
        self.input_conv = nn.Conv1d(1, residual_channels, 1)
        
        # Mel conditioning network
        self.cond_net = nn.Sequential(
            nn.Conv1d(n_mels, gate_channels, 1),
            nn.ReLU(),
            nn.Conv1d(gate_channels, gate_channels, 1)
        )
        
        # Dilated convolution blocks
        self.dilated_convs = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 10)  # Cycle through 1, 2, 4, ..., 512
            self.dilated_convs.append(
                DilatedConvBlock(
                    residual_channels, gate_channels, skip_channels,
                    kernel_size, dilation, dropout
                )
            )
        
        # Output network
        self.out_net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, 256, 1)  # 256 for 8-bit mu-law
        )
        
    def forward(self, x, mel):
        """
        Args:
            x: (batch, 1, seq_len) - input waveform
            mel: (batch, n_mels, seq_len) - conditioning mel spectrogram
        """
        # Upsample mel to match audio length
        mel = F.interpolate(mel, size=x.size(2), mode='linear')
        
        # Initial processing
        x = self.input_conv(x)
        cond = self.cond_net(mel)
        
        # Residual and skip connections
        skip_connections = []
        for conv in self.dilated_convs:
            x, skip = conv(x)
            x = x + cond  # Add conditioning
            skip_connections.append(skip)
        
        # Sum skips and process
        x = torch.sum(torch.stack(skip_connections), dim=0)
        x = self.out_net(x)
        
        return x