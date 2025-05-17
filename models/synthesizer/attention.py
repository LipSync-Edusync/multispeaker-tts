import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LocationSensitiveAttention(nn.Module):
    # location-sensitive attention mechanism from Tacotron 2: cumulative attention weights.
    
    def __init__(self, decoder_dim, encoder_dim, attention_dim=128, attention_filters=32, attention_kernel_size=31):
        """
        Args:
            decoder_dim: Dimension of decoder hidden states
            encoder_dim: Dimension of encoder hidden states
            attention_dim: Dimension of attention hidden space
            attention_filters: Number of filters for location features
            attention_kernel_size: Kernel size for location convolution
        """
        super().__init__()
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        
        # location layer
        self.location_conv = nn.Conv1d(
            in_channels=2,  # for prev attention weights ( cat of wgts and cum_weights)
            out_channels=attention_filters,
            kernel_size=attention_kernel_size,
            padding=(attention_kernel_size - 1) // 2,
            bias=False
        )
        self.location_proj = nn.Linear(attention_filters, attention_dim, bias=False)
        
        # content-based attention
        self.query_proj = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.value_proj = nn.Linear(encoder_dim, 1, bias=False)
        
        # learned parameters
        self.score_mask_value = -float("inf")
        self.register_buffer('prev_weights', None)
        self.register_buffer('prev_weights_cum', None)
        
    def init_states(self, encoder_outputs):
        # init attention states for a new seq
        batch_size = encoder_outputs.size(0)
        max_time = encoder_outputs.size(1)
        
        # irevious attention weights( initially uniform)
        self.prev_weights = torch.zeros(batch_size, max_time).to(encoder_outputs.device)
        self.prev_weights_cum = torch.zeros(batch_size, max_time).to(encoder_outputs.device)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: Current decoder hidden state (batch, decoder_dim)
            encoder_outputs: Encoder outputs (batch, time, encoder_dim)
            mask: Binary mask for padded positions (batch, time)
        Returns:
            context: Attention context vector (batch, encoder_dim)
            attention_weights: Softmax attention weights (batch, time)
        """
        batch_size = encoder_outputs.size(0)
        max_time = encoder_outputs.size(1)
        
        # project decoder hidden state (query)
        query = self.query_proj(decoder_hidden)  # (batch, attention_dim)
        
        # project encoder outputs (keys)
        keys = self.key_proj(encoder_outputs)  # (batch, time, attention_dim)
        
        # process location features
        # prev_weights: (batch, time)
        # prev_weights_cum: (batch, time)
        location_features = torch.stack([self.prev_weights, self.prev_weights_cum], dim=1)  # (batch, 2, time)
        processed_location = self.location_conv(location_features)  # (batch, attention_filters, time)
        processed_location = processed_location.transpose(1, 2)  # (batch, time, attention_filters)
        processed_location = self.location_proj(processed_location)  # (batch, time, attention_dim)
        
        # compute attention scores
        # (batch, 1, attention_dim) + (batch, time, attention_dim) + (batch, time, attention_dim)
        energies = self.value_proj(
            torch.tanh(query.unsqueeze(1) + keys + processed_location)
        ).squeeze(-1)  # (batch, time)
        
        # apply mask to padded positions
        if mask is not None:
            energies = energies.masked_fill(mask, self.score_mask_value)
        
        # normalize attention weights
        attention_weights = F.softmax(energies, dim=1)  # (batch, time)
        
        # update cumulative attention weights
        self.prev_weights_cum += attention_weights
        
        # compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, encoder_dim)
        context = context.squeeze(1)  # (batch, encoder_dim)
        
        # store current attention weights for next step
        self.prev_weights = attention_weights
        
        return context, attention_weights

    def get_alignment_energies(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Get alignment energies (for visualization)
        Args:
            decoder_hidden: Current decoder hidden state (batch, decoder_dim)
            encoder_outputs: Encoder outputs (batch, time, encoder_dim)
            mask: Binary mask for padded positions (batch, time)
        Returns:
            energies: Unnormalized attention energies (batch, time)
        """
        # project decoder hidden state (query)
        query = self.query_proj(decoder_hidden)  # (batch, attention_dim)
        
        # project encoder outputs (keys)
        keys = self.key_proj(encoder_outputs)  # (batch, time, attention_dim)
        
        # process location features
        location_features = torch.stack([self.prev_weights, self.prev_weights_cum], dim=1)  # (batch, 2, time)
        processed_location = self.location_conv(location_features)  # (batch, attention_filters, time)
        processed_location = processed_location.transpose(1, 2)  # (batch, time, attention_filters)
        processed_location = self.location_proj(processed_location)  # (batch, time, attention_dim)
        
        # compute attention scores
        energies = self.value_proj(
            torch.tanh(query.unsqueeze(1) + keys + processed_location)
        ).squeeze(-1)  # (batch, time)
        
        # apply mask to padded positions
        if mask is not None:
            energies = energies.masked_fill(mask, self.score_mask_value)
            
        return energies