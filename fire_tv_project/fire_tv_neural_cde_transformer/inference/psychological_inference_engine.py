# inference/psychological_inference_engine.py (TRIAL 2)
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from models.hybrid_model import HybridFireTVSystem
from config.synthetic_model_config import HybridModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PsychologicalInferenceEngine:
    """
    Core inference engine that converts user behavior into psychological traits
    and finds content with matching psychological profiles.
    """
    
    def __init__(self, model_path: str, tmdb_cache_path: str, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.movie_vectors = None
        self.movie_metadata = {}
        self.movie_ids = []
        
        # Load and process movie catalog
        self._initialize_movie_profiles(tmdb_cache_path)
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device with fallback logic."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("Using CPU for inference")
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained PyTorch model with error handling."""
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model with correct configuration
            model_config = HybridModelConfig()
            model = HybridFireTVSystem(config=model_config).to(self.device)
            
            # Load state dict with proper error handling
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            logger.info(f"✅ Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _initialize_movie_profiles(self, tmdb_cache_path: str):
        """Initialize movie psychological profiles (offline 'baking' phase)."""
        try:
            with open(tmdb_cache_path, 'r') as f:
                tmdb_catalog = json.load(f)
            
            # Flatten catalog and remove duplicates
            all_movies = []
            for genre_movies in tmdb_catalog.values():
                all_movies.extend(genre_movies)
            
            unique_movies = {movie['content_id']: movie for movie in all_movies}.values()
            logger.info(f"Processing {len(unique_movies)} unique movies for psychological profiling...")
            
            movie_vectors_list = []
            
            with torch.inference_mode():  # More efficient than torch.no_grad()
                for movie in unique_movies:
                    # Simulate ideal audience behavior for this movie
                    simulated_behavior = self._simulate_ideal_audience(movie)
                    
                    # Convert to model input format
                    feature_tensor = self._behavior_to_tensor(simulated_behavior)
                    
                    # Get psychological profile
                    model_input = {'features': feature_tensor}
                    psychological_vector = self.model(model_input)['psychological_traits']
                    
                    movie_vectors_list.append(psychological_vector)
                    self.movie_ids.append(movie['content_id'])
                    
                    # Store enriched metadata
                    self.movie_metadata[movie['content_id']] = {
                        'title': movie.get('title', 'Unknown'),
                        'content_genre': movie.get('content_genre', 'Unknown'),
                        'release_year': movie.get('release_year', None),
                        'tmdb_popularity': movie.get('tmdb_popularity', 0),
                        'tmdb_vote_average': movie.get('tmdb_vote_average', 0)
                    }
            
            # Store as single tensor for efficient similarity computation
            self.movie_vectors = torch.cat(movie_vectors_list, dim=0)
            logger.info("✅ Movie psychological profiles created successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize movie profiles: {e}")
            raise RuntimeError(f"Movie profile initialization failed: {e}")
    
    def _simulate_ideal_audience(self, movie: Dict) -> Dict:
        """Simulate ideal audience behavior based on movie genre."""
        genre = movie.get('content_genre', 'Drama')
        
        # Define genre-specific behavior patterns
        genre_patterns = {
            'Action': {'engagement': 0.9, 'frustration': 0.2, 'exploration': 0.7},
            'Science Fiction': {'engagement': 0.9, 'frustration': 0.2, 'exploration': 0.8},
            'Thriller': {'engagement': 0.85, 'frustration': 0.3, 'exploration': 0.6},
            'Comedy': {'engagement': 0.6, 'frustration': 0.1, 'exploration': 0.4},
            'Romance': {'engagement': 0.5, 'frustration': 0.1, 'exploration': 0.3},
            'Horror': {'engagement': 0.8, 'frustration': 0.4, 'exploration': 0.5},
            'Documentary': {'engagement': 0.75, 'frustration': 0.1, 'exploration': 0.6},
            'Drama': {'engagement': 0.7, 'frustration': 0.2, 'exploration': 0.5}
        }
        
        pattern = genre_patterns.get(genre, genre_patterns['Drama'])
        
        # Generate behavioral features with some randomness
        return {
            'dpad_up_count': max(0, np.random.normal(pattern['engagement'] * 10, 2)),
            'dpad_down_count': max(0, np.random.normal(pattern['engagement'] * 10, 2)),
            'dpad_left_count': max(0, np.random.normal(pattern['engagement'] * 5, 1)),
            'dpad_right_count': max(0, np.random.normal(pattern['engagement'] * 5, 1)),
            'back_button_presses': max(0, np.random.normal(pattern['frustration'] * 5, 1)),
            'menu_revisits': max(0, np.random.normal(pattern['frustration'] * 3, 1)),
            'scroll_speed': max(50, np.random.normal(150 - pattern['engagement'] * 50, 10)),
            'hover_duration': max(0.5, np.random.normal(3 - pattern['frustration'] * 2, 0.5)),
            'time_since_last_interaction': max(1, np.random.exponential(15 - pattern['engagement'] * 10))
        }
    
    def _behavior_to_tensor(self, behavior: Dict) -> torch.Tensor:
        """Convert behavior dictionary to model input tensor."""
        # Ensure correct feature order (must match training data)
        feature_order = [
            'dpad_up_count', 'dpad_down_count', 'dpad_left_count', 'dpad_right_count',
            'back_button_presses', 'menu_revisits', 'scroll_speed', 'hover_duration',
            'time_since_last_interaction'
        ]
        
        feature_values = [float(behavior.get(feature, 0)) for feature in feature_order]
        return torch.tensor([feature_values], dtype=torch.float32).to(self.device)
    
    # Add this debug code to your inference engine
    def debug_model_inputs(self, user_behavior):
        feature_tensor = self._behavior_to_tensor(user_behavior)
        print(f"Debug - Input tensor: {feature_tensor}")
        
        with torch.no_grad():
            # Get intermediate outputs
            model_input = {'features': feature_tensor}
            outputs = self.model(model_input)
            print(f"Debug - Raw model output: {outputs['psychological_traits']}")



    def predict_user_psychology(self, user_behavior: Dict) -> torch.Tensor:
        """Convert user behavior into psychological trait vector."""
        try:
            self.debug_model_inputs(user_behavior)  
            feature_tensor = self._behavior_to_tensor(user_behavior)
            model_input = {'features': feature_tensor}
            
            with torch.inference_mode():
                psychological_traits = self.model(model_input)['psychological_traits']
            
            return psychological_traits
            
        except Exception as e:
            logger.error(f"Failed to predict user psychology: {e}")
            raise RuntimeError(f"Psychology prediction failed: {e}")
    
    def get_recommendations(self, user_psychological_vector: torch.Tensor, top_k: int = 10) -> List[Dict]:
        """Find movies with similar psychological profiles."""
        try:
            if self.movie_vectors is None:
                raise RuntimeError("Movie profiles not initialized")
            
            # Normalize vectors for cosine similarity
            user_norm = F.normalize(user_psychological_vector, p=2, dim=1)
            movie_norms = F.normalize(self.movie_vectors, p=2, dim=1)
            
            # Compute similarity scores
            similarity_scores = torch.matmul(user_norm, movie_norms.T).squeeze(0)
            
            # Get top K recommendations
            top_k_scores, top_k_indices = torch.topk(similarity_scores, k=min(top_k, len(self.movie_ids)))
            
            recommendations = []
            for i in range(len(top_k_indices)):
                idx = top_k_indices[i].item()
                movie_id = self.movie_ids[idx]
                metadata = self.movie_metadata[movie_id]
                
                recommendations.append({
                    'content_id': movie_id,
                    'similarity_score': float(top_k_scores[i].item()),
                    'title': metadata['title'],
                    'content_genre': metadata['content_genre'],
                    'release_year': metadata['release_year'],
                    'tmdb_popularity': metadata['tmdb_popularity'],
                    'tmdb_vote_average': metadata['tmdb_vote_average'],
                    'confidence_level': self._calculate_confidence(top_k_scores[i].item())
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            raise RuntimeError(f"Recommendation generation failed: {e}")
    
    def _calculate_confidence(self, similarity_score: float) -> str:
        """Convert similarity score to human-readable confidence level."""
        if similarity_score >= 0.8:
            return "Very High"
        elif similarity_score >= 0.6:
            return "High"
        elif similarity_score >= 0.4:
            return "Medium"
        elif similarity_score >= 0.2:
            return "Low"
        else:
            return "Very Low"
