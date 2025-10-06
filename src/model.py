"""Policy network implementation for RLHF."""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Transformer-based policy network with value head."""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.policy_head = nn.Linear(hidden_size, vocab_size)
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
        
        Returns:
            logits: [batch_size, seq_length, vocab_size]
            values: [batch_size, seq_length, 1]
        """
        batch_size, seq_length = input_ids.shape
        
        # Embedding with positional encoding
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_length, :]
        
        # Transformer encoding
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Policy and value heads
        logits = self.policy_head(x)
        values = self.value_head(x)
        
        return logits, values