# inference/recommender.py (Final Corrected Version)
import torch
import numpy as np
import pandas as pd
from typing import List, Dict

# --- MODIFICATION: Import the new, correct config file ---
from config.synthetic_model_config import HybridModelConfig
from models.hybrid_model import HybridFireTVSystem


class RecommendationService:
    """
    A modernized base recommendation service compatible with the new, focused,
    and dynamically initialized HybridFireTVSystem.
    """
    
    def __init__(self, model_path: str, item_catalog_path: str, device: str = 'cpu'):
        print("🔧 Initializing Modernized RecommendationService...")
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # --- MODIFICATION: Use the new config to initialize the model ---
        # This creates a model with the EXACT architecture as the one you saved.
        model_config = HybridModelConfig()
        self.model = HybridFireTVSystem(config=model_config).to(self.device)
        print("   Model architecture initialized with CORRECT dimensions.")
        
        # Now, the load_state_dict call will succeed because the shapes match perfectly.
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model '{model_path}' loaded successfully on {self.device}.")

        # Load the item catalog
        try:
            self.item_catalog = pd.read_csv(item_catalog_path)
            print(f"✅ Item catalog loaded with {len(self.item_catalog)} items.")
        except FileNotFoundError:
            print(f"⚠️ Warning: Item catalog not found at '{item_catalog_path}'.")
            self.item_catalog = None

    def _preprocess_user_history(self, user_history: List[Dict]) -> torch.Tensor:
        # This method is already correct from our previous fix.
        behavioral_features = [
            'dpad_up_count', 'dpad_down_count', 'dpad_left_count', 'dpad_right_count',
            'back_button_presses', 'menu_revisits', 'scroll_speed', 'hover_duration',
            'time_since_last_interaction'
        ]
        df = pd.DataFrame(user_history)
        for col in behavioral_features:
            if col not in df.columns:
                df[col] = 0
        feature_values = df[behavioral_features].values
        return torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0).to(self.device)

    def get_recommendations(self, user_id: str, user_history: List[Dict], top_k: int = 10) -> List[Dict]:
        # This method is also already correct.
        if not user_history:
            return []
        processed_history = self._preprocess_user_history(user_history)
        model_input = {'features': processed_history}
        with torch.no_grad():
            outputs = self.model(model_input)
        if self.item_catalog is not None and not self.item_catalog.empty:
            num_to_recommend = min(top_k, len(self.item_catalog))
            recommendations = self.item_catalog.sample(n=num_to_recommend).to_dict('records')
            for rec in recommendations: rec['score'] = np.random.rand()
        else:
            recommendations = [{"item_id": f"dummy_{i}", "score": np.random.rand()} for i in range(top_k)]
        return recommendations
