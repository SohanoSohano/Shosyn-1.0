# models/neural_cde.py
import torch
import torch.nn as nn
import torchcde
from .components import PsychologicalTraitDecoder

class LayerNormNeuralCDE(nn.Module):
    """Enhanced Neural CDE with LayerNorm for Fire TV behavioral analysis"""
    
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.15, 
                 vector_field_layers=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Vector field network
        self.vector_field = self._build_vector_field(vector_field_layers, dropout_rate)
        
        # Initial state encoder
        self.initial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Multi-head attention for feature refinement
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout_rate, batch_first=True
        )
        
        # Psychological trait decoder
        self.trait_decoder = PsychologicalTraitDecoder(hidden_dim)
        
    def _build_vector_field(self, num_layers, dropout_rate):
        """Build vector field network with LayerNorm"""
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                ])
            elif i == num_layers - 1:
                layers.append(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim * self.input_dim)
                )
            else:
                layers.extend([
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                ])
        
        return nn.Sequential(*layers)
    
    def forward(self, interaction_path, timestamps):
        """Forward pass through Neural CDE"""
        batch_size, seq_len, feature_dim = interaction_path.shape
        
        # Create control path for CDE
        try:
            coeffs = torchcde.natural_cubic_spline_coeffs(
                torch.cat([timestamps.unsqueeze(-1), interaction_path], dim=-1)
            )
            control_path = torchcde.CubicSpline(coeffs)
        except:
            coeffs = torchcde.linear_interpolation_coeffs(
                torch.cat([timestamps.unsqueeze(-1), interaction_path], dim=-1)
            )
            control_path = torchcde.LinearInterpolation(coeffs)
        
        # Initial hidden state
        z0 = self.initial_encoder(interaction_path[:, 0])
        
        # CDE function
        class CDEFunc(nn.Module):
            def __init__(self, vector_field, input_dim):
                super().__init__()
                self.vector_field = vector_field
                self.input_dim = input_dim
                
            def forward(self, t, z):
                try:
                    dXdt = control_path.derivative(t)[:, 1:]
                    f_z = self.vector_field(z).view(z.shape[0], -1, self.input_dim)
                    return torch.bmm(f_z, dXdt.unsqueeze(-1)).squeeze(-1)
                except:
                    return torch.zeros_like(z)
        
        cde_func = CDEFunc(self.vector_field, self.input_dim)
        
        # Solve CDE
        try:
            t_eval = timestamps[0]
            z_trajectory = torchcde.cdeint(
                X=control_path,
                func=cde_func,
                z0=z0,
                t=t_eval,
                method="dopri5",
                rtol=1e-4,
                atol=1e-6,
                adjoint=True
            )
            final_hidden = z_trajectory[-1]
        except:
            final_hidden = z0
        
        # Apply attention mechanism
        final_hidden_attended, attention_weights = self.attention(
            final_hidden.unsqueeze(1), final_hidden.unsqueeze(1), final_hidden.unsqueeze(1)
        )
        final_hidden = final_hidden_attended.squeeze(1)
        
        # Decode psychological traits
        traits = self.trait_decoder(final_hidden)
        
        return traits, attention_weights
