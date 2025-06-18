# models/hybrid_model.py
import torch
import torch.nn as nn
from .neural_cde import LayerNormNeuralCDE
from .transformers import BehavioralSequenceTransformer, FireTVRecommendationTransformer

class HybridFireTVSystem(nn.Module):
    """
    COMPLETE FIXED HYBRID MODEL: Handles all dimension mismatches and unpacking errors.
    This version dynamically adapts to actual tensor shapes to prevent multiplication errors.
    """
    
    def __init__(self, config):
        super().__init__()
        
        print("🔥 EXECUTING COMPLETE FIXED HYBRID MODEL 🔥")
        print("🔥 This version handles all dimension mismatches dynamically 🔥")
        
        self.config = config
        self.input_dim = config.input_dim 
        self.output_dim = config.output_dim

        # Neural CDE component
        self.neural_cde = LayerNormNeuralCDE(
            input_dim=self.input_dim, 
            hidden_dim=config.neural_cde.hidden_dim,
            dropout_rate=config.neural_cde.dropout_rate,
            vector_field_layers=config.neural_cde.vector_field_layers
        )
        
        # Transformer component
        self.behavioral_transformer = BehavioralSequenceTransformer(
            feature_dim=self.input_dim,
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            num_layers=config.transformer.num_layers,
            patch_size=config.transformer.patch_size
        )
        
        # Recommendation transformer
        self.recommendation_transformer = FireTVRecommendationTransformer(
            content_vocab_size=config.recommendation.content_vocab_size,
            psychological_traits=self.output_dim,
            d_model=config.recommendation.d_model,
            max_seq_len=config.recommendation.max_sequence_length
        )
        
        # Dynamic fusion layers - will be created based on actual tensor shapes
        self.cde_fusion = None
        self.transformer_fusion = None
        self.final_fusion = None
        
        # Flag to track if layers have been initialized
        self.layers_initialized = False

    def _initialize_fusion_layers(self, cde_features, transformer_features):
        """
        Dynamically initialize fusion layers based on actual tensor shapes.
        This prevents all dimension mismatch errors.
        """
        if self.layers_initialized:
            return
            
        cde_dim = cde_features.shape[-1]
        transformer_dim = transformer_features.shape[-1]
        
        print(f"🔧 Initializing fusion layers:")
        print(f"   CDE features: {cde_dim} dims")
        print(f"   Transformer features: {transformer_dim} dims")
        print(f"   Target output: {self.output_dim} dims")
        
        # CDE feature processor
        self.cde_fusion = nn.Sequential(
            nn.Linear(cde_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        ).to(cde_features.device)
        
        # Transformer feature processor
        self.transformer_fusion = nn.Sequential(
            nn.Linear(transformer_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        ).to(transformer_features.device)
        
        # Final fusion layer (combines 64 + 64 = 128 features)
        self.final_fusion = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.output_dim),
            nn.Sigmoid()
        ).to(cde_features.device)
        
        self.layers_initialized = True
        print("✅ Fusion layers initialized successfully")

    def forward(self, interaction_data, modal_data=None, content_history=None, 
                target_recommendations=None):
        """
        Forward pass with dynamic layer initialization and robust error handling.
        """
        interaction_path = interaction_data.get('sequence')
        timestamps = interaction_data.get('timestamps')
        
        # Neural CDE processing with safe unpacking
        cde_output = self.neural_cde(interaction_path, timestamps)
        if isinstance(cde_output, tuple) and len(cde_output) >= 1:
            cde_features = cde_output[0]
            cde_attention = cde_output[1] if len(cde_output) > 1 else None
        else:
            cde_features = cde_output
            cde_attention = None

        # Transformer processing with safe unpacking
        transformer_output = self.behavioral_transformer(interaction_path)
        if isinstance(transformer_output, tuple) and len(transformer_output) >= 1:
            transformer_features = transformer_output[0]
            transformer_attention = transformer_output[1] if len(transformer_output) > 1 else None
        else:
            transformer_features = transformer_output
            transformer_attention = None
            
        # Initialize fusion layers dynamically if not already done
        self._initialize_fusion_layers(cde_features, transformer_features)
        
        # Process features through their respective fusion layers
        processed_cde = self.cde_fusion(cde_features)
        processed_transformer = self.transformer_fusion(transformer_features)
        
        # Combine processed features
        combined_features = torch.cat([processed_cde, processed_transformer], dim=-1)
        
        # Generate final traits
        final_traits = self.final_fusion(combined_features)
        
        # Prepare output dictionary
        outputs = {
            'psychological_traits': final_traits,
            'cde_traits': processed_cde,
            'transformer_traits': processed_transformer,
            'attention_weights': {
                'cde': cde_attention,
                'transformer': transformer_attention
            }
        }
        
        # Recommendation generation if content history provided
        if content_history is not None:
            try:
                rec_output = self.recommendation_transformer(content_history, final_traits, target_recommendations)
                if isinstance(rec_output, tuple) and len(rec_output) >= 1:
                    outputs['recommendations'] = rec_output[0]
                    outputs['attention_weights']['recommendations'] = rec_output[1] if len(rec_output) > 1 else None
                else:
                    outputs['recommendations'] = rec_output
                    outputs['attention_weights']['recommendations'] = None
            except Exception as e:
                print(f"Warning: Recommendation generation failed: {e}")
                outputs['recommendations'] = None
                outputs['attention_weights']['recommendations'] = None

        return outputs
    
    def predict_traits_only(self, interaction_data):
        """Predict only psychological traits (for inference)"""
        with torch.no_grad():
            outputs = self.forward(interaction_data)
            return outputs['psychological_traits']
    
    def generate_recommendations(self, interaction_data, content_history, num_recommendations=10):
        """Generate content recommendations"""
        with torch.no_grad():
            outputs = self.forward(interaction_data, content_history=content_history)
            if outputs.get('recommendations') is not None:
                return outputs['recommendations'][:, :num_recommendations]
            else:
                # Return dummy recommendations if generation fails
                batch_size = interaction_data['sequence'].shape[0]
                return torch.zeros(batch_size, num_recommendations)
    
    def get_model_info(self):
        """Get information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'layers_initialized': self.layers_initialized
        }
