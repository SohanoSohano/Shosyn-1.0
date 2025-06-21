import torch
import numpy as np
from typing import List, Dict

class DummyTMDbIntegration:
    """
    Dummy TMDb integration that generates consistent synthetic features
    without any API calls for maximum training speed
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Predefined genre list (same as your real integration)
        self.genre_list = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
            'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller',
            'War', 'Western', 'Biography'
        ]
        
        # Common genres for realistic distribution
        self.common_genres = ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller']
        
        print("🎭 Dummy TMDb Integration initialized")
        print(f"   Device: {device}")
        print(f"   Genres: {len(self.genre_list)} categories")
    
    def create_tmdb_features(self, content_ids: List[str]) -> torch.Tensor:
        """
        Create consistent 70-dimensional TMDb features using content_id hashing
        """
        features = []
        
        for content_id in content_ids:
            # Use content_id hash for deterministic but varied features
            seed = hash(content_id) % 100000
            np.random.seed(seed)
            
            # Create realistic feature distribution
            feature_vector = np.random.normal(0.0, 0.3, 70)  # 70-dim features
            
            # Add some structure to make features more realistic
            # First 20 dims: genre-related features
            feature_vector[:20] = np.random.uniform(-1, 1, 20)
            
            # Next 20 dims: rating/popularity features  
            feature_vector[20:40] = np.random.normal(0.5, 0.2, 20)
            
            # Last 30 dims: metadata features
            feature_vector[40:] = np.random.normal(0.0, 0.1, 30)
            
            # Ensure reasonable bounds
            feature_vector = np.clip(feature_vector, -2.0, 2.0)
            
            features.append(torch.tensor(feature_vector, dtype=torch.float32))
        
        return torch.stack(features)
    
    def create_content_embeddings(self, content_ids: List[str]) -> torch.Tensor:
        """
        Create consistent 384-dimensional content embeddings using content_id hashing
        """
        embeddings = []
        
        for content_id in content_ids:
            # Use content_id hash for deterministic embeddings
            seed = hash(content_id) % 100000
            np.random.seed(seed)
            
            # Create 384-dim embedding (typical for sentence transformers)
            embedding = np.random.normal(0.0, 0.1, 384)
            
            # Normalize to unit vector (common for embeddings)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            embeddings.append(torch.tensor(embedding, dtype=torch.float32))
        
        return torch.stack(embeddings)
    
    def fetch_tmdb_data(self, content_ids: List[str], content_mapping: Dict) -> Dict:
        """
        Generate dummy TMDb data for compatibility with existing code
        """
        tmdb_data = {}
        
        for content_id in content_ids:
            # Use content_id hash for consistent genre assignment
            seed = hash(content_id) % 100000
            np.random.seed(seed)
            
            # Assign 1-3 genres randomly but consistently
            num_genres = np.random.randint(1, 4)
            assigned_genres = np.random.choice(self.common_genres, size=num_genres, replace=False).tolist()
            
            # Generate consistent rating
            rating = np.random.uniform(3.0, 9.0)  # Realistic movie rating range
            
            tmdb_data[content_id] = {
                'genres': assigned_genres,
                'rating': rating,
                'vote_average': rating,
                'popularity': np.random.uniform(1.0, 100.0),
                'title': f"Movie_{content_id}",
                'overview': f"Generated movie for {content_id}"
            }
        
        return tmdb_data

# Convenience function for quick integration
def create_dummy_tmdb_features(content_ids: List[str], device='cuda') -> tuple:
    """
    Quick function to generate dummy features without class instantiation
    """
    dummy_integration = DummyTMDbIntegration(device)
    tmdb_features = dummy_integration.create_tmdb_features(content_ids)
    content_embeddings = dummy_integration.create_content_embeddings(content_ids)
    tmdb_data = dummy_integration.fetch_tmdb_data(content_ids, {})
    
    return tmdb_features, content_embeddings, tmdb_data
