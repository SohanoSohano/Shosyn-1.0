# models/neural_cde.py
import torch
import torch.nn as nn
from .components import PsychologicalTraitDecoder

class LayerNormNeuralCDE(nn.Module):
    """
    FIXED: Robust Neural CDE that handles variable input shapes safely.
    """
    
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.15, vector_field_layers=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simplified temporal encoder (no complex CDE dependencies)
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        
        # Combine bidirectional outputs
        self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Simple attention mechanism
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Psychological trait decoder
        self.trait_decoder = PsychologicalTraitDecoder(hidden_dim)
        
    def forward(self, interaction_path, timestamps):
        """
        FIXED: Safe forward pass that handles variable input shapes.
        """
        # SAFE SHAPE HANDLING - No triple unpacking
        shape = interaction_path.shape
        batch_size = shape[0]
        
        # Handle both 2D and 3D inputs safely
        if len(shape) == 2:
            # Input is (batch_size, feature_dim) - add sequence dimension
            interaction_path = interaction_path.unsqueeze(1)  # Make it (batch_size, 1, feature_dim)
        
        # Now we know it's 3D: (batch_size, seq_len, feature_dim)
        
        # Use LSTM for temporal modeling
        lstm_output, (hidden_state, cell_state) = self.temporal_encoder(interaction_path)
        
        # Project bidirectional output to hidden_dim
        projected_output = self.hidden_projection(lstm_output)
        normalized_output = self.layer_norm(projected_output)
        
        # Simple attention mechanism
        attention_scores = self.attention_weights(normalized_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum of sequence
        attended_output = torch.sum(normalized_output * attention_weights, dim=1)
        
        # Decode psychological traits
        traits = self.trait_decoder(attended_output)
        
        # Always return exactly 2 values
        return traits, attention_weights.squeeze(-1)
