# config/model_config.py
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class NeuralCDEConfig:
    """Configuration for Neural CDE component"""
    input_dim: int = 49
    hidden_dim: int = 128
    dropout_rate: float = 0.15
    vector_field_layers: int = 3
    
@dataclass
class TransformerConfig:
    """Configuration for Transformer components"""
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    patch_size: int = 4
    
@dataclass
class MultiModalConfig:
    """Configuration for multimodal transformer"""
    input_dims: Dict[str, int] = None
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    
    def __post_init__(self):
        if self.input_dims is None:
            self.input_dims = {
                'navigation': 15,
                'content': 12, 
                'device': 10,
                'temporal': 12
            }

@dataclass
class RecommendationConfig:
    """Configuration for recommendation transformer"""
    content_vocab_size: int = 10000
    psychological_traits: int = 20
    d_model: int = 512
    max_sequence_length: int = 100

@dataclass
class HybridModelConfig:
    """Complete hybrid model configuration"""
    neural_cde: NeuralCDEConfig = None
    transformer: TransformerConfig = None
    multimodal: MultiModalConfig = None
    recommendation: RecommendationConfig = None
    
    def __post_init__(self):
        if self.neural_cde is None:
            self.neural_cde = NeuralCDEConfig()
        if self.transformer is None:
            self.transformer = TransformerConfig()
        if self.multimodal is None:
            self.multimodal = MultiModalConfig()
        if self.recommendation is None:
            self.recommendation = RecommendationConfig()
