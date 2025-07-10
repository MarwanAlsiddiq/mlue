import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple
import math

class TradingConfig(PretrainedConfig):
    def __init__(self):
        self.input_dim = 10  # open, high, low, close, volume, marketcap, rsi, macd, macd_signal, macd_hist
        self.hidden_dim = 64
        self.num_layers = 4
        self.num_heads = 4
        self.dropout = 0.1
        self.learning_rate = 0.0001
        self.weight_decay = 0.001
        self.class_weights = torch.tensor([1.0, 2.0])  # Give more weight to positive class
        self.batch_size = 32
        self.window_size = 16
        self.epochs = 15
        self.patience = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TradingTransformer(nn.Module):
    def __init__(self, config: TradingConfig):
        super().__init__()
        self.config = config
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(config.hidden_dim, config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Classification head with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.BatchNorm1d(config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 4, 1)
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
        Forward pass of the Transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        """
        # Embed input features
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        x = self.transformer(x)
        
        # Take mean across sequence length dimension
        x = x.mean(dim=1)
        
        # Pass through classifier
        x = self.classifier(x)
        
        # Return logits with shape (batch_size, 1)
        return x.squeeze(-1)

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
