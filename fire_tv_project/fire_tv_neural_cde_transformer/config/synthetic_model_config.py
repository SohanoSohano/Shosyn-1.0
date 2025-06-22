# config/synthetic_model_config.py

class HybridModelConfig:
    """
    Configuration for the model trained on the FOCUSED SYNTHETIC DATASET.
    This config matches the architecture of 'best_synthetic_trained_model.pth'.
    """
    # --- MODIFIED to match the reality of your trained model ---
    input_dim = 9   # We trained on 9 behavioral features.
    output_dim = 3  # We trained to predict 3 psychological traits.
    
    class neural_cde:
        hidden_dim = 15 # The actual output dim of the CDE
        dropout_rate = 0.2
        vector_field_layers = 3
        
    class transformer:
        d_model = 512 # The actual output dim of the Transformer
        nhead = 8
        num_layers = 6
        patch_size = 4
