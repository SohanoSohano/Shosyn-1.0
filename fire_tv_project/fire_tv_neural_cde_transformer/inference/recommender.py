# inference/recommender.py
import torch
import numpy as np
import pandas as pd
from typing import List, Dict


from models.hybrid_model import HybridFireTVSystem
from config.inference_config import InferenceConfig

class RecommendationService:
    """
    A robust service to load the trained model and generate real-time recommendations.
    """
    def __init__(self, model_path: str, item_catalog_path: str, device: str = 'cpu'):
        print("🔄 Initializing Recommendation Service...")
        self.device = torch.device(device)
        
        model_config = InferenceConfig()
        
        # 1. Instantiate the model by passing the single config object
        print(f"   - Loading model from {model_path}...")
        self.model = HybridFireTVSystem(config=model_config).to(self.device)
        
        print("   - Manually initializing dynamic fusion layers...")
        # Create dummy tensors with the correct shapes to trigger initialization
        dummy_cde_features = torch.randn(1, model_config.output_dim).to(self.device)
        dummy_transformer_features = torch.randn(1, model_config.transformer.d_model).to(self.device)
        # Call the private method to build the layers
        self.model._initialize_fusion_layers(dummy_cde_features, dummy_transformer_features)

        # Load the saved state dictionary [8]
        checkpoint = torch.load(model_path, map_location=self.device)

        #print(f"Loaded checkpoint keys: {checkpoint.keys()}")

        self.model.load_state_dict(checkpoint)
        
        # 2. Set the model to evaluation mode [1]
        self.model.to(self.device)
        self.model.eval()
        print("   - Model loaded and set to evaluation mode.")
        
        # 3. Load the item catalog
        # In a real system, this would be a connection to a database like Elasticsearch [11]
        print(f"   - Loading item catalog from {item_catalog_path}...")
        self.item_catalog = pd.read_csv(item_catalog_path)
        # For simplicity, we assume the catalog has pre-computed features
        # In a real system, these would be embeddings or TMDb features
        self.item_features = self._get_item_features()
        print(f"✅ Service initialized with {len(self.item_catalog)} items.")

    def _preprocess_user_history(self, user_interactions: List[Dict]) -> torch.Tensor:
        """
        Converts a user's recent interaction history into a tensor for the model.
        This must match the preprocessing done during training [6].
        """
        # Example: user_interactions = [{'feature_1': 0.5, ...}, {'feature_1': 0.8, ...}]
        # This function should create a tensor of shape [1, sequence_length, num_features]
        
        # For simplicity, let's create a dummy tensor. Replace with your real logic.
        sequence_length = 50 # Must match model's expected input
        num_features = 18
        
        # In a real app, you would build this tensor from the actual interactions
        history_tensor = torch.randn(1, sequence_length, num_features)
        
        return history_tensor.to(self.device)

    def _get_item_features(self) -> torch.Tensor:
        """
        Retrieves or generates feature vectors for all items in the catalog.
        """
        # In a real system, these would be embeddings from a separate model or TMDb features.
        # For now, we'll create dummy features.
        num_items = len(self.item_catalog)
        num_item_features = 15 # This should match the psychological traits dimension
        
        # Create dummy "personality" features for each item
        item_features_tensor = torch.randn(num_items, num_item_features)
        return item_features_tensor.to(self.device)

    def get_recommendations(self, user_id: str, user_history: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        The main inference function to generate recommendations for a user.
        """
        # 1. Preprocess the user's history
        processed_history = self._preprocess_user_history(user_history)
        
        # Create dummy inputs for other model requirements
        timestamps = torch.arange(processed_history.shape[1]).float().unsqueeze(0).to(self.device)
        interaction_data = {'sequence': processed_history, 'timestamps': timestamps}
        dummy_tmdb_features = torch.randn(1, 70).to(self.device)
        dummy_embeddings = torch.randn(1, 384).to(self.device)

        # 2. Run inference to get psychological traits
        # Use torch.inference_mode() for performance [1]
        with torch.inference_mode():
            model_output = self.model(
                interaction_data,
                tmdb_features=dummy_tmdb_features,
                content_embeddings=dummy_embeddings
            )
            user_trait_vector = model_output['psychological_traits'] # Shape: [1, 15]

        # 3. Score all items against the user's traits
        # We use a simple dot product for similarity scoring [3]
        # This calculates how well each item's personality matches the user's
        scores = torch.matmul(self.item_features, user_trait_vector.T).squeeze()

        # 4. Rank items and get the top_k
        ranked_indices = torch.argsort(scores, descending=True)
        top_indices = ranked_indices[:top_k].cpu().numpy()

        # 5. Format the output
        recommendations = self.item_catalog.iloc[top_indices].copy()
        
        # Add the score to the output
        recommendations['score'] = scores[top_indices].cpu().numpy()
        
        return recommendations.to_dict(orient='records')
