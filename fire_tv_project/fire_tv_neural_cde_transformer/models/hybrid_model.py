# models/hybrid_model.py (Focused Version for Synthetic Data Training)
import torch
import torch.nn as nn
from .neural_cde import LayerNormNeuralCDE
from .transformers import BehavioralSequenceTransformer

class HybridFireTVSystem(nn.Module):
    """
    A focused version of the Hybrid Model for training on the synthetic dataset.
    This version concentrates on the core task: predicting psychological traits
    from user behavior, and incorporates strong regularization.
    """
    
    def __init__(self, config):
        super().__init__()
        
        print("🔥 INITIALIZING FOCUSED HYBRID MODEL for Synthetic Training 🔥")
        
        self.config = config
        # --- MODIFICATION: Dimensions are now simpler ---
        self.input_dim = config.input_dim  # Expected to be 9 (behavioral features)
        self.output_dim = config.output_dim # Expected to be 3 (psychological traits)

        # --- Core components remain the same ---
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
        
        # --- REMOVED: Unnecessary Components for this training phase ---
        # The tmdb_processor and content_embedding_processor are not needed
        # because the synthetic data loader does not provide this data.
        # self.tmdb_processor = ...
        # self.content_embedding_processor = ...
        
        # --- MODIFICATION: A simplified fusion network ---
        # It now fuses only the outputs of the two core components.
        # The input dimension is calculated based on the CDE and Transformer hidden dimensions.
        cde_output_dim = config.neural_cde.hidden_dim
        transformer_output_dim = config.transformer.d_model
        fused_input_dim = cde_output_dim + transformer_output_dim
        
        print(f"🔧 Initializing focused fusion layer with input dimension: {fused_input_dim}")

        self.final_fusion = nn.Sequential(
            nn.Linear(fused_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.5), # Regularization for the fusion layer
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.4), # More regularization
            nn.Linear(256, self.output_dim)
            # REMOVED: Sigmoid is removed. We'll use BCEWithLogitsLoss in the trainer,
            # which is more numerically stable and expects raw logits.
        )

        # --- MODIFICATION: Define Dropout layers for regularization ---
        # As per previous discussions and best practices [3].
        print("   Adding Dropout layers (p=0.4) for regularization.")
        self.cde_dropout = nn.Dropout(p=0.4)
        self.transformer_dropout = nn.Dropout(p=0.4)

    def forward(self, data: dict):
        """
        --- MODIFICATION: Simplified forward pass ---
        Accepts a simple dictionary from our new SyntheticFireTVDataset.
        """
        # The input 'features' tensor is expected to have a shape like (batch_size, num_features)
        behavioral_features = data['features']
        
        # To use with CDE/Transformer, we need a sequence dimension. We unsqueeze to add it.
        # Shape becomes (batch_size, 1, num_features) - a sequence of length 1.
        if behavioral_features.dim() == 2:
            behavioral_features = behavioral_features.unsqueeze(1)

        # We need a dummy timestamps tensor for the CDE.
        timestamps = torch.zeros(behavioral_features.shape[0], behavioral_features.shape[1], device=behavioral_features.device)
        
        # --- Core model processing ---
        cde_output = self.neural_cde(behavioral_features, timestamps)
        cde_features = cde_output[0] if isinstance(cde_output, tuple) else cde_output
        
        transformer_output, _ = self.behavioral_transformer(behavioral_features)
        
        # --- Apply Dropout Regularization ---
        cde_features = self.cde_dropout(cde_features)
        transformer_features = self.transformer_dropout(transformer_output)
        
        # Squeeze out the sequence dimension if it's 1
        if cde_features.dim() == 3 and cde_features.shape[1] == 1:
            cde_features = cde_features.squeeze(1)
        if transformer_features.dim() == 3 and transformer_features.shape[1] == 1:
            transformer_features = transformer_features.squeeze(1)

        # --- Combine the core features ---
        combined_features = torch.cat([cde_features, transformer_features], dim=-1)
        
        # --- Generate final prediction ---
        predicted_traits = self.final_fusion(combined_features)
        
        # Return a dictionary that matches what the trainer expects for loss calculation
        return {
            'psychological_traits': predicted_traits
        }
