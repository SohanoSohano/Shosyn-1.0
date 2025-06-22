# models/hybrid_model.py (Enhanced with Batch Norm and Residual Connections)
import torch
import torch.nn as nn
from .neural_cde import LayerNormNeuralCDE
from .transformers import BehavioralSequenceTransformer

# --- NEW: Helper for Residual Block ---
class ResidualBlock(nn.Module):
    """
    A basic Residual Block for a feed-forward network.
    It includes a Linear layer, BatchNorm, GELU activation, and Dropout,
    with a skip connection.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.4):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # BatchNorm1d is used because we expect (batch_size, num_features)
        self.bn = nn.BatchNorm1d(output_dim) 
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Projection layer for the shortcut connection if input and output dimensions differ
        self.shortcut = nn.Identity()
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Store the input for the skip connection
        residual = self.shortcut(x)
        
        # Apply the main path transformations
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Add the residual connection
        return x + residual

class HybridFireTVSystem(nn.Module):
    """
    A focused version of the Hybrid Model for training on the synthetic dataset,
    now enhanced with Batch Normalization and Residual Connections in the fusion layer.
    """
    
    def __init__(self, config):
        super().__init__()
        
        print("🔥 INITIALIZING FOCUSED HYBRID MODEL for Synthetic Training 🔥")
        
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim

        self.neural_cde = LayerNormNeuralCDE(
            input_dim=self.input_dim, 
            hidden_dim=config.neural_cde.hidden_dim,
            dropout_rate=config.neural_cde.dropout_rate,
            vector_field_layers=config.neural_cde.vector_field_layers
        )
        
        self.behavioral_transformer = BehavioralSequenceTransformer(
            feature_dim=self.input_dim,
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            num_layers=config.transformer.num_layers,
            patch_size=config.transformer.patch_size
        )
        
        # --- Fusion Network with Residual Blocks and BatchNorm ---
        cde_output_dim = config.neural_cde.hidden_dim
        transformer_output_dim = config.transformer.d_model
        fused_input_dim = cde_output_dim + transformer_output_dim
        
        print(f"🔧 Initializing focused fusion layer with input dimension: {fused_input_dim}")
        print("   --- ENHANCED with Residual Blocks and BatchNorm ---")

        self.final_fusion = nn.Sequential(
            # First Residual Block: Input fused_input_dim, Output 512
            ResidualBlock(fused_input_dim, 512, dropout_rate=0.5),
            # Second Residual Block: Input 512, Output 256
            ResidualBlock(512, 256, dropout_rate=0.4),
            # Final Linear layer to project to the output dimension
            nn.Linear(256, self.output_dim)
        )

        # --- Define Dropout layers for initial components (Unchanged) ---
        print("   Adding Dropout layers (p=0.4) for regularization.")
        self.cde_dropout = nn.Dropout(p=0.4)
        self.transformer_dropout = nn.Dropout(p=0.4)

    def forward(self, data: dict):
        behavioral_features = data['features']
        
        if behavioral_features.dim() == 2:
            behavioral_features = behavioral_features.unsqueeze(1)

        timestamps = torch.zeros(behavioral_features.shape[0], behavioral_features.shape[1], device=behavioral_features.device)
        
        # --- Core model processing ---
        cde_output = self.neural_cde(behavioral_features, timestamps)
        cde_features = cde_output[0] if isinstance(cde_output, tuple) else cde_output
        
        transformer_output, _ = self.behavioral_transformer(behavioral_features)
        
        # --- Apply Dropout Regularization ---
        cde_features = self.cde_dropout(cde_features)
        transformer_features = self.transformer_dropout(transformer_features)
        
        # Squeeze out the sequence dimension if it's 1
        if cde_features.dim() == 3:
            cde_features = cde_features.squeeze(1)
        if transformer_features.dim() == 3:
            transformer_features = transformer_features.squeeze(1)

        # --- Combine the core features ---
        combined_features = torch.cat([cde_features, transformer_features], dim=-1)
        
        # --- Generate final prediction using the enhanced final_fusion ---
        predicted_traits = self.final_fusion(combined_features)
        
        return {
            'psychological_traits': predicted_traits
        }
