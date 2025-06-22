# inference/recommendation_pipeline.py (TRIAL 2)
import logging
import pandas as pd
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from .psychological_inference_engine import PsychologicalInferenceEngine
from .data_processor import FirebaseDataProcessor

logger = logging.getLogger(__name__)

class RecommendationPipeline:
    """
    Main pipeline that orchestrates the entire recommendation process:
    Firebase Data -> Behavioral Features -> Psychological Traits -> Content Recommendations
    """
    
    def __init__(self, 
                 model_path: str = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\best_performance_model.pth",
                 tmdb_cache_path: str = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\fire_tv_neural_cde_transformer_instance_version\Shosyn-1.0\fire_tv_project\fire_tv_neural_cde_transformer\tmdb_local_catalog.json",
                 device: str = 'auto'):
        """
        Initialize the recommendation pipeline.
        
        Args:
            model_path: Path to the trained PyTorch model
            tmdb_cache_path: Path to the TMDb movie catalog cache
            device: Computing device ('auto', 'cpu', 'cuda')
        """
        try:
            # Initialize components
            self.data_processor = FirebaseDataProcessor()
            self.inference_engine = PsychologicalInferenceEngine(
                model_path=model_path,
                tmdb_cache_path=tmdb_cache_path,
                device=device
            )
            
            logger.info("✅ Recommendation pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize recommendation pipeline: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    
    def recommend_for_user(self, 
                          user_df: pd.DataFrame, 
                          user_id: str,
                          top_k: int = 10,
                          use_sessions: bool = True) -> Dict[str, Any]:
        """
        Generate recommendations for a single user based on their interaction history.
        
        Args:
            user_df: DataFrame containing user's interaction logs
            user_id: Unique identifier for the user
            top_k: Number of recommendations to return
            use_sessions: Whether to process data as separate sessions
            
        Returns:
            Dictionary containing recommendations and metadata
        """
        try:
            logger.info(f"Generating recommendations for user {user_id}")
            
            if user_df.empty:
                logger.warning(f"No data available for user {user_id}")
                return self._get_fallback_recommendations(user_id, top_k)
            
            # Process user data into behavioral features
            if use_sessions:
                session_features_list = self.data_processor.process_session_data(user_df)
                # Use the most recent session for recommendations
                behavioral_features = session_features_list[-1] if session_features_list else {}
            else:
                behavioral_features = self.data_processor.process_user_data(user_df)
            
            # Convert behavioral features to psychological traits
            psychological_vector = self.inference_engine.predict_user_psychology(behavioral_features)
            
            # Get content recommendations
            recommendations = self.inference_engine.get_recommendations(psychological_vector, top_k)
            
            # Prepare response
            response = {
                "user_id": user_id,
                "timestamp": pd.Timestamp.now().isoformat(),
                "behavioral_summary": behavioral_features,
                "psychological_profile": self._vector_to_profile(psychological_vector),
                "recommendations": recommendations,
                "metadata": {
                    "total_interactions": len(user_df),
                    "recommendation_count": len(recommendations),
                    "processing_method": "session-based" if use_sessions else "aggregate",
                    "model_confidence": self._calculate_overall_confidence(recommendations)
                }
            }
            
            logger.info(f"Successfully generated {len(recommendations)} recommendations for user {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return self._get_error_response(user_id, str(e))
    
    def batch_recommend(self, 
                       user_data_dict: Dict[str, pd.DataFrame], 
                       top_k: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Generate recommendations for multiple users in batch.
        
        Args:
            user_data_dict: Dictionary mapping user_id to their DataFrame
            top_k: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to their recommendation response
        """
        results = {}
        
        for user_id, user_df in user_data_dict.items():
            try:
                results[user_id] = self.recommend_for_user(user_df, user_id, top_k)
            except Exception as e:
                logger.error(f"Failed to process user {user_id}: {e}")
                results[user_id] = self._get_error_response(user_id, str(e))
        
        logger.info(f"Batch processing completed for {len(results)} users")
        return results
    
    def _vector_to_profile(self, psychological_vector) -> Dict[str, float]:
        """Convert psychological vector to interpretable profile."""
        # Assuming the model outputs 3 traits: engagement, frustration, exploration
        vector_cpu = psychological_vector.cpu().numpy().flatten()
        
        # Apply sigmoid to get 0-1 range
        import numpy as np
        normalized = 1 / (1 + np.exp(-vector_cpu))
        
        return {
            "engagement_level": float(normalized[0]) if len(normalized) > 0 else 0.5,
            "frustration_level": float(normalized[1]) if len(normalized) > 1 else 0.5,
            "exploration_tendency": float(normalized[2]) if len(normalized) > 2 else 0.5
        }
    
    def _calculate_overall_confidence(self, recommendations: List[Dict]) -> str:
        """Calculate overall confidence based on recommendation scores."""
        if not recommendations:
            return "No Data"
        
        avg_score = sum(rec['similarity_score'] for rec in recommendations) / len(recommendations)
        
        if avg_score >= 0.7:
            return "High"
        elif avg_score >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _get_fallback_recommendations(self, user_id: str, top_k: int) -> Dict[str, Any]:
        """Return popular content when no user data is available."""
        # Get top popular movies from the inference engine
        try:
            # Use a neutral psychological profile
            import torch
            neutral_vector = torch.zeros(1, 3).to(self.inference_engine.device)
            recommendations = self.inference_engine.get_recommendations(neutral_vector, top_k)
            
            return {
                "user_id": user_id,
                "timestamp": pd.Timestamp.now().isoformat(),
                "behavioral_summary": {},
                "psychological_profile": {"engagement_level": 0.5, "frustration_level": 0.5, "exploration_tendency": 0.5},
                "recommendations": recommendations,
                "metadata": {
                    "total_interactions": 0,
                    "recommendation_count": len(recommendations),
                    "processing_method": "fallback_popular",
                    "model_confidence": "Low"
                }
            }
        except Exception as e:
            logger.error(f"Fallback recommendations failed: {e}")
            return self._get_error_response(user_id, "No data available and fallback failed")
    
    def _get_error_response(self, user_id: str, error_message: str) -> Dict[str, Any]:
        """Return error response structure."""
        return {
            "user_id": user_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "error": error_message,
            "recommendations": [],
            "metadata": {
                "total_interactions": 0,
                "recommendation_count": 0,
                "processing_method": "error",
                "model_confidence": "None"
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        try:
            # Test with dummy data
            dummy_behavior = {
                'dpad_up_count': 5.0, 'dpad_down_count': 5.0,
                'dpad_left_count': 2.0, 'dpad_right_count': 2.0,
                'back_button_presses': 1.0, 'menu_revisits': 0.0,
                'scroll_speed': 100.0, 'hover_duration': 2.0,
                'time_since_last_interaction': 5.0
            }
            
            psychological_vector = self.inference_engine.predict_user_psychology(dummy_behavior)
            recommendations = self.inference_engine.get_recommendations(psychological_vector, 3)
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "movie_profiles_count": len(self.inference_engine.movie_ids),
                "test_recommendations": len(recommendations),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat()
            }
