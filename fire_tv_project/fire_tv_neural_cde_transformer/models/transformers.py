# models/transformers.py
import torch
import torch.nn as nn
import math

class BehavioralSequenceTransformer(nn.Module):
    """Enhanced Transformer for Fire TV behavioral sequence analysis"""
    
    def __init__(self, feature_dim, d_model=512, nhead=8, num_layers=6, patch_size=4):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.patch_size = patch_size
        
        # Input projection to d_model
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True  # CRITICAL: This makes it expect (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        ROBUST: Handles various input shapes from the hybrid model
        Input: Can be (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
        Output: (hidden_state, attention_weights)
        """
        # Debug: Print actual shape for troubleshooting
        print(f"🔍 Transformer received input shape: {x.shape}")
        
        # Handle different input shapes
        if len(x.shape) == 2:
            # Case: (batch_size, feature_dim) - your expected case
            batch_size, feature_dim = x.shape
            # Add sequence dimension: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, feature_dim)
            
        elif len(x.shape) == 3:
            # Case: (batch_size, seq_len, feature_dim) - from Neural CDE
            batch_size, seq_len, feature_dim = x.shape
            # Already has sequence dimension, use as-is
            
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected 2D or 3D tensor.")
        
        # Ensure feature_dim matches what we expect
        if x.shape[-1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {x.shape[-1]}")
        
        # Project to d_model
        x = self.input_projection(x)  # Shape: (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer (with batch_first=True, it expects (batch, seq, feature))
        transformer_output = self.transformer(x)  # Shape: (batch_size, seq_len, d_model)
        
        # Global average pooling over sequence dimension to get fixed-size output
        hidden_state = transformer_output.mean(dim=1)  # Shape: (batch_size, d_model)
        
        # Apply output projection
        hidden_state = self.output_projection(hidden_state)  # Shape: (batch_size, d_model)
        
        print(f"✅ Transformer output shape: {hidden_state.shape}")
        
        # Return hidden state and None for attention weights
        return hidden_state, None


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer with batch dimension first (to match training)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, max_len, d_model]

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]  # Broadcast correctly

class FireTVRecommendationTransformer(nn.Module):
    """Placeholder for recommendation transformer"""
    
    def __init__(self, content_vocab_size, psychological_traits, d_model, max_seq_len):
        super().__init__()
        self.content_vocab_size = content_vocab_size
        self.psychological_traits = psychological_traits
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Simple placeholder implementation
        self.recommendation_head = nn.Linear(psychological_traits, content_vocab_size)
        
    def forward(self, content_history, traits, target_recommendations=None):
        """Simple recommendation generation"""
        recommendations = self.recommendation_head(traits)
        return recommendations, None  # Return (recommendations, attention_weights)
