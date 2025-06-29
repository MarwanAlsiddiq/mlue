import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple
import math

class TradingConfig(PretrainedConfig):
    def __init__(
        self,
        input_dim: int = 11,  # Number of features (Open, High, Low, Close, Volume, Marketcap/Quote volume, RSI, MACD, MACD_Signal, MACD_Hist, EMA)
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

class TradingTransformer(nn.Module):
    def __init__(self, config: TradingConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.embedding = nn.Linear(11, config.hidden_dim)  # Explicitly set input_dim to 11
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(config.hidden_dim, config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Embed input features
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Take the mean across sequence dimension
        x = x.mean(dim=1)
        
        # Final classification
        x = self.classifier(x)
        
        # Ensure output shape is (batch_size, 1)
        return x.view(-1, 1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, hidden_dim)
        """
        # Get sequence length
        seq_len = x.size(1)
        
        # Get positional encoding for sequence length
        pe = self.pe[:, :seq_len, :]
        
        # Add positional encoding
        x = x + pe
        
        return self.dropout(x)
