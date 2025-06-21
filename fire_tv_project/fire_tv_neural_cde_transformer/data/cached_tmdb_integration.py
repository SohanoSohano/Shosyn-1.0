import torch
import pickle
import os
from typing import Dict, List
from data.tmdb_integration import TMDbIntegration

class CachedTMDbIntegration:
    """
    Cached version of TMDb integration for faster training
    """
    
    def __init__(self, features_cache_path: str, embeddings_cache_path: str):
        self.features_cache_path = features_cache_path
        self.embeddings_cache_path = embeddings_cache_path
        
        print("🔄 Loading TMDb feature cache...")
        with open(features_cache_path, 'rb') as f:
            self.features_cache = pickle.load(f)
        
        with open(embeddings_cache_path, 'rb') as f:
            self.embeddings_cache = pickle.load(f)
        
        print(f"✅ Loaded {len(self.features_cache)} cached TMDb features")
        print(f"✅ Loaded {len(self.embeddings_cache)} cached content embeddings")
    
    def create_tmdb_features(self, content_ids: List[str]) -> torch.Tensor:
        """
        Get pre-computed TMDb features from cache
        """
        features = []
        for content_id in content_ids:
            if content_id in self.features_cache:
                features.append(self.features_cache[content_id])
            else:
                # Default feature vector if not found
                features.append(torch.zeros(70))
        
        return torch.stack(features)
    
    def create_content_embeddings(self, content_ids: List[str]) -> torch.Tensor:
        """
        Get pre-computed content embeddings from cache
        """
        embeddings = []
        for content_id in content_ids:
            if content_id in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[content_id])
            else:
                # Default embedding vector if not found
                embeddings.append(torch.zeros(384))
        
        return torch.stack(embeddings)
    
    def fetch_tmdb_data(self, content_ids: List[str], content_mapping: Dict) -> Dict:
        """
        Dummy method for compatibility - not used in cached version
        """
        return {content_id: {'genres': ['Drama'], 'rating': 5.0} for content_id in content_ids}
