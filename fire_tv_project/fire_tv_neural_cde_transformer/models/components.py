# models/components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class PathwayAttention(nn.Module):
    """Pathway attention mechanism for behavior-aware processing"""
    
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.pathway_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, features):
        # Self-attention
        attended_features, attention_weights = self.attention(
            features, features, features
        )
        
        # Residual connection
        attended_features = self.layer_norm(attended_features + features)
        
        # Pathway gating
        pathway_scores = self.pathway_gate(attended_features)
        
        # Apply pathway filtering
        filtered_features = attended_features * pathway_scores
        
        return filtered_features, attention_weights

class MultiModalFusion(nn.Module):
    """Multi-modal fusion layer for different data types"""
    
    def __init__(self, modal_dims, output_dim):
        super().__init__()
        
        self.modal_projections = nn.ModuleDict({
            modal: nn.Linear(dim, output_dim)
            for modal, dim in modal_dims.items()
        })
        
        self.attention_weights = nn.Parameter(
            torch.ones(len(modal_dims)) / len(modal_dims)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, modal_features):
        # Project each modality to common dimension
        projected_features = []
        for modal, features in modal_features.items():
            projected = self.modal_projections[modal](features)
            projected_features.append(projected)
        
        # Weighted fusion
        stacked_features = torch.stack(projected_features, dim=1)
        weights = F.softmax(self.attention_weights, dim=0)
        fused_features = torch.sum(
            stacked_features * weights.view(1, -1, 1), dim=1
        )
        
        # Final fusion layer
        output = self.fusion_layer(fused_features)
        
        return output

class PsychologicalTraitDecoder(nn.Module):
    """FIXED: Simplified decoder that returns a tensor instead of complex dictionary"""
    
    def __init__(self, input_dim, num_traits=15):  # Match your actual label count
        super().__init__()
        
        # Simplified architecture that directly outputs trait predictions
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_traits),  # Direct output to trait count
            nn.Sigmoid()  # Ensure outputs are in [0,1] range
        )
    
    def forward(self, features):
        """Returns a simple tensor of shape (batch_size, num_traits)"""
        return self.decoder(features)
