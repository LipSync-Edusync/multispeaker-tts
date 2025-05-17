import torch
import torch.nn as nn
import torch.nn.functional as F

class SynthesizerLoss(nn.Module):
    """
    Loss function for Tacotron 2 synthesizer with:
    - Mel spectrogram reconstruction loss (L1 + L2)
    - Stop token prediction loss
    - Optional guided attention loss
    """
    
    def __init__(self, l1_weight=1.0, l2_weight=1.0, stop_token_weight=1.0, 
                 guided_attn_weight=0.0, guided_attn_sigma=0.2):
        """
        Args:
            l1_weight: Weight for L1 loss
            l2_weight: Weight for L2 loss
            stop_token_weight: Weight for stop token loss
            guided_attn_weight: Weight for guided attention loss
            guided_attn_sigma: Sigma for guided attention loss
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.stop_token_weight = stop_token_weight
        self.guided_attn_weight = guided_attn_weight
        self.guided_attn_sigma = guided_attn_sigma
        
        # Loss functions
        self.l1_loss = nn.L1Loss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, outputs, targets, input_lengths, output_lengths, attn_weights=None):
        """
        Compute all loss components.
        
        Args:
            outputs: Dictionary containing:
                - mel_outputs: Predicted mel spectrograms [B, n_mels, T_out]
                - mel_outputs_postnet: Post-net predicted mels [B, n_mels, T_out]
                - stop_logits: Stop token predictions [B, T_out]
            targets: Dictionary containing:
                - mels: Ground truth mel spectrograms [B, n_mels, T_out]
                - stop_targets: Ground truth stop tokens [B, T_out]
            input_lengths: Lengths of input sequences [B]
            output_lengths: Lengths of output sequences [B]
            attn_weights: Attention weights [B, T_out, T_in] if guided attention is used
            
        Returns:
            loss_dict: Dictionary containing all loss components and total loss
        """
        mel_targets = targets['mels']
        stop_targets = targets['stop_targets']
        
        mel_outputs = outputs['mel_outputs']
        mel_outputs_postnet = outputs['mel_outputs_postnet']
        stop_logits = outputs['stop_logits']
        
        # Mask for variable length sequences
        mask = self._create_mask(output_lengths, mel_targets.size(2))
        
        # Mel reconstruction losses (pre and post-net)
        l1_loss = self._compute_mel_loss(mel_outputs, mel_targets, mask, self.l1_loss)
        l1_loss_postnet = self._compute_mel_loss(mel_outputs_postnet, mel_targets, mask, self.l1_loss)
        
        l2_loss = self._compute_mel_loss(mel_outputs, mel_targets, mask, self.mse_loss)
        l2_loss_postnet = self._compute_mel_loss(mel_outputs_postnet, mel_targets, mask, self.mse_loss)
        
        # Stop token loss
        stop_loss = self._compute_stop_token_loss(stop_logits, stop_targets, mask)
        
        # Total mel loss (average of pre and post-net)
        total_mel_loss = 0.5 * (l1_loss + l1_loss_postnet) * self.l1_weight + \
                         0.5 * (l2_loss + l2_loss_postnet) * self.l2_weight
        
        # Guided attention loss (if enabled)
        guided_attn_loss = 0.0
        if self.guided_attn_weight > 0 and attn_weights is not None:
            guided_attn_loss = self._compute_guided_attention_loss(
                attn_weights, input_lengths, output_lengths
            )
        
        # Total loss
        total_loss = total_mel_loss + stop_loss * self.stop_token_weight + \
                     guided_attn_loss * self.guided_attn_weight
        
        return {
            'loss': total_loss,
            'mel_loss': total_mel_loss,
            'l1_loss': l1_loss,
            'l1_loss_postnet': l1_loss_postnet,
            'l2_loss': l2_loss,
            'l2_loss_postnet': l2_loss_postnet,
            'stop_loss': stop_loss,
            'guided_attn_loss': guided_attn_loss,
        }
    
    def _create_mask(self, lengths, max_len):
        # Create mask for variable length sequences
        
        device = lengths.device
        mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1)  # [B, 1, T]
    
    def _compute_mel_loss(self, predictions, targets, mask, loss_fn):
        # Compute mel spectrogram loss with masking
        
        loss = loss_fn(predictions, targets)
        loss = loss.masked_select(mask).mean()
        return loss
    
    def _compute_stop_token_loss(self, predictions, targets, mask):
        # Compute stop token prediction loss with masking
        # Convert mask to match stop token shape [B, T]
        
        mask = mask.squeeze(1)
        loss = self.bce_loss(predictions, targets)
        loss = loss.masked_select(mask).mean()
        return loss
    
    def _compute_guided_attention_loss(self, attn_weights, input_lengths, output_lengths):
        """
        Compute guided attention loss to encourage diagonal alignments.
        
        Args:
            attn_weights: Attention weights [B, T_out, T_in]
            input_lengths: Lengths of input sequences [B]
            output_lengths: Lengths of output sequences [B]
        """
        device = attn_weights.device
        batch_size = attn_weights.size(0)
        max_input_len = attn_weights.size(2)
        max_output_len = attn_weights.size(1)
        
        # Create grid of positions
        input_pos = torch.arange(max_input_len, device=device).view(1, -1)
        output_pos = torch.arange(max_output_len, device=device).view(-1, 1)
        
        # Create attention mask
        input_mask = (input_pos < input_lengths.view(-1, 1)).unsqueeze(1)  # [B, 1, T_in]
        output_mask = (output_pos < output_lengths.view(-1, 1)).unsqueeze(2)  # [B, T_out, 1]
        mask = input_mask & output_mask  # [B, T_out, T_in]
        
        # Create guided attention weights (Gaussian around diagonal)
        sigma = self.guided_attn_sigma
        attn_guide = 1.0 - torch.exp(-((input_pos / input_lengths.view(-1, 1, 1) - output_pos / output_lengths.view(-1, 1, 1)) ** 2) / (2 * sigma ** 2))
        
        # Apply mask and compute loss
        loss = attn_weights * attn_guide
        loss = loss.masked_select(mask).mean()
        
        return loss