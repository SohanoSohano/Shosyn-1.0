# models/hybrid_model.py
import torch
import torch.nn as nn
from .neural_cde import LayerNormNeuralCDE
from .transformers import (
    BehavioralSequenceTransformer, 
    MultiModalFireTVTransformer,
    FireTVRecommendationTransformer
)
from .components import MultiModalFusion

class HybridFireTVSystem(nn.Module):
    """FIXED: Hybrid system handling current model output format"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Neural CDE component
        self.neural_cde = LayerNormNeuralCDE(
            input_dim=config.neural_cde.input_dim,
            hidden_dim=config.neural_cde.hidden_dim,
            dropout_rate=config.neural_cde.dropout_rate,
            vector_field_layers=config.neural_cde.vector_field_layers
        )
        
        # Transformer components
        self.behavioral_transformer = BehavioralSequenceTransformer(
            feature_dim=config.neural_cde.input_dim,
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            num_layers=config.transformer.num_layers,
            patch_size=config.transformer.patch_size
        )
        
        self.recommendation_transformer = FireTVRecommendationTransformer(
            content_vocab_size=config.recommendation.content_vocab_size,
            psychological_traits=config.recommendation.psychological_traits,
            d_model=config.recommendation.d_model,
            max_seq_len=config.recommendation.max_sequence_length
        )
        
        # FIXED: Simple fusion layer that works with trait predictions
        self.trait_fusion = nn.Sequential(
            nn.Linear(config.recommendation.psychological_traits * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.recommendation.psychological_traits),
            nn.Sigmoid()
        )
        
    def forward(self, interaction_data, modal_data=None, content_history=None, 
                target_recommendations=None):
        """FIXED: Forward pass handling current model outputs (2 values each)"""
        
        interaction_path = interaction_data['sequence']
        timestamps = interaction_data['timestamps']
        
        # FIXED: Handle 2-value returns from current models
        try:
            # Neural CDE processing (returns traits, attention)
            cde_traits, cde_attention = self.neural_cde(interaction_path, timestamps)
            
            # Transformer processing (returns traits, attention)  
            transformer_traits, transformer_attention = self.behavioral_transformer(interaction_path)
            
            # Handle different output formats
            if isinstance(cde_traits, dict):
                # If traits are returned as dictionary, extract values
                cde_features = torch.cat([
                    trait['mean'] if isinstance(trait, dict) else trait 
                    for trait in cde_traits.values()
                ], dim=-1)
            else:
                # If traits are returned as tensor
                cde_features = cde_traits
            
            if isinstance(transformer_traits, dict):
                # If traits are returned as dictionary, extract values
                transformer_features = torch.cat([
                    trait['mean'] if isinstance(trait, dict) else trait 
                    for trait in transformer_traits.values()
                ], dim=-1)
            else:
                # If traits are returned as tensor
                transformer_features = transformer_traits
            
            # FIXED: Fuse trait predictions directly
            combined_features = torch.cat([cde_features, transformer_features], dim=-1)
            final_traits = self.trait_fusion(combined_features)
            
            outputs = {
                'psychological_traits': final_traits,
                'cde_traits': cde_traits,
                'transformer_traits': transformer_traits,
                'attention_weights': {
                    'cde': cde_attention,
                    'transformer': transformer_attention
                }
            }
            
            # Recommendation generation if content history provided
            if content_history is not None:
                recommendations, rec_attention = self.recommendation_transformer(
                    content_history, final_traits, target_recommendations
                )
                outputs['recommendations'] = recommendations
                outputs['attention_weights']['recommendations'] = rec_attention
            
            return outputs
            
        except ValueError as e:
            if "not enough values to unpack" in str(e):
                print(f"Model output format error: {e}")
                print("Falling back to simple trait prediction...")
                
                # Fallback: use only one model
                cde_output = self.neural_cde(interaction_path, timestamps)
                if isinstance(cde_output, tuple):
                    cde_traits = cde_output[0]
                    cde_attention = cde_output[1] if len(cde_output) > 1 else None
                else:
                    cde_traits = cde_output
                    cde_attention = None
                
                return {
                    'psychological_traits': cde_traits,
                    'cde_traits': cde_traits,
                    'attention_weights': {'cde': cde_attention}
                }
            else:
                raise e
    
    def predict_traits_only(self, interaction_data):
        """Predict only psychological traits (for inference)"""
        with torch.no_grad():
            outputs = self.forward(interaction_data)
            return outputs['psychological_traits']
    
    def generate_recommendations(self, interaction_data, content_history, num_recommendations=10):
        """Generate content recommendations"""
        with torch.no_grad():
            outputs = self.forward(interaction_data, content_history=content_history)
            return outputs['recommendations'][:, :num_recommendations]
