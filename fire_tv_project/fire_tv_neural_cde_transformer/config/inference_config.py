# config/inference_config.py
from dataclasses import dataclass, field

@dataclass
class NeuralCDEConfig:
    # Based on where it's used in hybrid_model.py:
    # LayerNormNeuralCDE(input_dim=self.input_dim, hidden_dim=config.neural_cde.hidden_dim, ...)
    hidden_dim: int = 128 # Based on your previous error, this was likely 512, not 64 or 256.
    dropout_rate: float = 0.15 # Default value from previous configs, verify if changed
    vector_field_layers: int = 3 # Default value from previous configs, verify if changed

@dataclass
class TransformerConfig:
    # Based on where it's used in hybrid_model.py:
    # BehavioralSequenceTransformer(feature_dim=self.input_dim, d_model=config.transformer.d_model, ...)
    d_model: int = 512 # Based on your previous error, this was likely 512, not 128.
    nhead: int = 8 # Default value from previous configs, verify if changed
    num_layers: int = 6 # Based on your previous error (layers 4 & 5), this was likely 6.
    patch_size: int = 4 # Default value from previous configs, verify if changed

@dataclass
class InferenceConfig:
    """
    Configuration for loading the model during inference.
    These values MUST match the configuration used during training.
    """
    # Based on your initial model setup:
    input_dim: int = 18 # Your feature_columns size (e.g., 18 features)
    output_dim: int = 15 # Your label_columns size (e.g., 15 psychological traits)
    
    neural_cde: NeuralCDEConfig = field(default_factory=NeuralCDEConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    cde_fusion_input_dim: int = 15 # Because traits is 15-dim
