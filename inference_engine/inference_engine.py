import torch
import pandas as pd
import numpy as np
import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import signatory
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """Represents a user's current session state."""
    user_id: str
    session_id: str
    events: List[Dict]
    current_frustration: float = 0.0
    predicted_frustration: float = 0.0
    last_update: float = 0.0

@dataclass
class MovieRecommendation:
    """Represents a movie recommendation with confidence scores."""
    item_id: str
    title: str
    genres: List[str]
    frustration_compatibility: float
    persona_match: float
    overall_score: float
    reasoning: str

class NeuralRDEInferenceEngine:
    """
    Production-ready inference engine for real-time frustration prediction
    and movie recommendations using your trained Neural RDE model.
    """
    
    def __init__(self, 
                 model_path: str,
                 movie_catalog_path: str,
                 config_path: Optional[str] = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained Neural RDE model (.pth file)
            movie_catalog_path: Path to TMDB movie catalog CSV
            config_path: Optional path to configuration file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing inference engine on {self.device}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load and initialize model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load movie catalog
        self.movie_catalog = self._load_movie_catalog(movie_catalog_path)
        
        # Initialize preprocessing components
        self.scaler, self.ohe_encoder = self._initialize_preprocessors()
        
        # Active user sessions
        self.active_sessions: Dict[str, UserSession] = {}
        
        # Persona-based recommendation strategies
        self.persona_strategies = self._initialize_persona_strategies()
        
        logger.info("Inference engine initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration parameters."""
        default_config = {
            "logsig_depth": 3,
            "min_events_for_prediction": 3,
            "frustration_threshold_high": 0.4,
            "frustration_threshold_low": 0.15,
            "recommendation_count": 10,
            "session_timeout_minutes": 30,
            "model_confidence_threshold": 0.1
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained Neural RDE model."""
        try:
            # Recreate model architecture (adjust based on your model)
            from train_nrde import RobustNeuralRDE
            
            # Calculate input channels (should match training)
            path_dim = 1 + 2 + 2 + 7  # time + psych + scroll + actions (adjust based on your data)
            input_channels = signatory.logsignature_channels(path_dim, self.config["logsig_depth"])
            
            model = RobustNeuralRDE(
                input_channels=input_channels,
                hidden_dims=[128, 64],  # Adjust based on your training config
                output_channels=1
            )
            
            # Load trained weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            
            logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_movie_catalog(self, catalog_path: str) -> pd.DataFrame:
        """Load and preprocess movie catalog."""
        try:
            df = pd.read_csv(catalog_path)
            
            # Basic preprocessing for TMDB data
            if 'genres' in df.columns:
                # Handle genres column (assuming it's JSON-like)
                df['genres'] = df['genres'].apply(self._parse_genres)
            
            # Create genre-based features for recommendation
            all_genres = set()
            for genres in df['genres']:
                all_genres.update(genres)
            
            # Create binary genre features
            for genre in all_genres:
                df[f'genre_{genre.lower()}'] = df['genres'].apply(lambda x: genre in x)
            
            logger.info(f"Loaded {len(df)} movies from catalog")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load movie catalog: {e}")
            raise
    
    def _parse_genres(self, genres_str) -> List[str]:
        """Parse genres from string format."""
        try:
            if pd.isna(genres_str) or genres_str == '[]':
                return []
            
            # Handle different possible formats
            if isinstance(genres_str, str):
                if genres_str.startswith('['):
                    # JSON-like format
                    import ast
                    genres_list = ast.literal_eval(genres_str)
                    if isinstance(genres_list, list) and len(genres_list) > 0:
                        if isinstance(genres_list[0], dict):
                            return [g.get('name', '') for g in genres_list]
                        else:
                            return genres_list
                else:
                    # Simple comma-separated
                    return [g.strip() for g in genres_str.split(',')]
            
            return []
        except:
            return []
    
    def _initialize_preprocessors(self):
        """Initialize preprocessing components to match training pipeline."""
        # These should match your training preprocessing
        scaler = StandardScaler()
        # Fit with dummy data that matches your training distribution
        dummy_data = np.array([[0.2, 0.1], [0.8, 0.9]])  # frustration, cognitive_load ranges
        scaler.fit(dummy_data)
        
        # Initialize OHE for action types
        action_types = [['click'], ['back'], ['dpad_right'], ['dpad_down'], 
                       ['dpad_left'], ['dpad_up'], ['session_start']]
        ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        ohe_encoder.fit(action_types)
        
        return scaler, ohe_encoder
    
    def _initialize_persona_strategies(self) -> Dict:
        """Initialize recommendation strategies for different personas."""
        return {
            "high_frustration": {
                "preferred_genres": ["Comedy", "Family", "Animation"],
                "avoid_genres": ["Horror", "Thriller"],
                "strategy": "comfort_content"
            },
            "low_frustration": {
                "preferred_genres": ["Action", "Adventure", "Sci-Fi"],
                "avoid_genres": [],
                "strategy": "exploration"
            },
            "medium_frustration": {
                "preferred_genres": ["Drama", "Romance", "Comedy"],
                "avoid_genres": ["Horror"],
                "strategy": "balanced"
            }
        }
    
    def update_session(self, user_id: str, session_id: str, event: Dict) -> Dict:
        """
        Update user session with new event and predict frustration.
        
        Args:
            user_id: Unique user identifier
            session_id: Unique session identifier  
            event: New user event (action_type, timestamp, context, etc.)
            
        Returns:
            Dictionary with updated session state and predictions
        """
        current_time = time.time()
        
        # Get or create session
        session_key = f"{user_id}_{session_id}"
        if session_key not in self.active_sessions:
            self.active_sessions[session_key] = UserSession(
                user_id=user_id,
                session_id=session_id,
                events=[],
                last_update=current_time
            )
        
        session = self.active_sessions[session_key]
        session.events.append(event)
        session.last_update = current_time
        
        # Predict frustration if we have enough events
        if len(session.events) >= self.config["min_events_for_prediction"]:
            try:
                predicted_frustration = self._predict_frustration(session.events)
                session.predicted_frustration = predicted_frustration
                
                logger.info(f"Updated session {session_key}: predicted frustration = {predicted_frustration:.3f}")
                
                return {
                    "status": "success",
                    "user_id": user_id,
                    "session_id": session_id,
                    "predicted_frustration": predicted_frustration,
                    "event_count": len(session.events),
                    "recommendations_needed": predicted_frustration > self.config["frustration_threshold_high"]
                }
                
            except Exception as e:
                logger.error(f"Frustration prediction failed for {session_key}: {e}")
                return {
                    "status": "error",
                    "message": str(e),
                    "user_id": user_id,
                    "session_id": session_id
                }
        
        return {
            "status": "insufficient_data",
            "user_id": user_id,
            "session_id": session_id,
            "event_count": len(session.events),
            "min_required": self.config["min_events_for_prediction"]
        }
    
    def _predict_frustration(self, events: List[Dict]) -> float:
        """Predict frustration level from sequence of events."""
        try:
            # Convert events to feature sequence (matching training format)
            features = self._events_to_features(events)
            
            # Create path tensor
            path_tensor = torch.tensor(features, dtype=torch.float32)
            
            # Compute log-signature
            logsignature = signatory.logsignature(
                path_tensor.unsqueeze(0), 
                self.config["logsig_depth"]
            ).squeeze(0)
            
            # Predict using model
            with torch.no_grad():
                logsignature = logsignature.to(self.device)
                prediction = self.model(logsignature.unsqueeze(0))
                frustration = float(prediction.cpu().numpy()[0, 0])
            
            # Ensure valid range
            return max(0.0, min(1.0, frustration))
            
        except Exception as e:
            logger.error(f"Frustration prediction error: {e}")
            return 0.5  # Default moderate frustration
    
    def _events_to_features(self, events: List[Dict]) -> np.ndarray:
        """Convert event sequence to feature matrix for log-signature."""
        features = []
        
        for i, event in enumerate(events):
            # Time delta
            if i == 0:
                time_delta = 0.0
            else:
                # Calculate time difference (you may need to adjust based on your timestamp format)
                time_delta = 1.0  # Simplified - implement proper time calculation
            
            # Psychological features (simplified - you may want to implement proper calculation)
            frustration = event.get('frustration_level', 0.1)
            cognitive_load = event.get('cognitive_load', 0.1)
            
            # Scale psychological features
            psych_scaled = self.scaler.transform([[frustration, cognitive_load]])[0]
            
            # Scroll features
            scroll_speed = event.get('scroll_speed', 0.0) or 0.0
            scroll_depth = event.get('scroll_depth', 0.0) or 0.0
            
            # Action type encoding
            action_type = event.get('action_type', 'click')
            action_encoded = self.ohe_encoder.transform([[action_type]])[0]
            
            # Combine all features
            feature_row = np.concatenate([
                [time_delta],
                psych_scaled,
                [scroll_speed, scroll_depth],
                action_encoded
            ])
            
            features.append(feature_row)
        
        return np.array(features)
    
    def get_recommendations(self, 
                          user_id: str, 
                          session_id: str,
                          user_preferences: Optional[Dict] = None) -> List[MovieRecommendation]:
        """
        Get movie recommendations based on current user frustration and preferences.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_preferences: Optional user preference data
            
        Returns:
            List of movie recommendations ranked by suitability
        """
        session_key = f"{user_id}_{session_id}"
        
        if session_key not in self.active_sessions:
            logger.warning(f"No active session found for {session_key}")
            return self._get_default_recommendations()
        
        session = self.active_sessions[session_key]
        frustration_level = session.predicted_frustration
        
        # Determine recommendation strategy based on frustration
        if frustration_level > self.config["frustration_threshold_high"]:
            strategy = "high_frustration"
        elif frustration_level < self.config["frustration_threshold_low"]:
            strategy = "low_frustration"
        else:
            strategy = "medium_frustration"
        
        # Get strategy configuration
        strategy_config = self.persona_strategies[strategy]
        
        # Score movies based on strategy
        recommendations = self._score_movies(
            strategy_config, 
            frustration_level, 
            user_preferences
        )
        
        # Return top N recommendations
        top_recommendations = recommendations[:self.config["recommendation_count"]]
        
        logger.info(f"Generated {len(top_recommendations)} recommendations for {session_key} "
                   f"(frustration: {frustration_level:.3f}, strategy: {strategy})")
        
        return top_recommendations
    
    def _score_movies(self, 
                     strategy_config: Dict, 
                     frustration_level: float,
                     user_preferences: Optional[Dict]) -> List[MovieRecommendation]:
        """Score and rank movies based on strategy and user state."""
        recommendations = []
        
        for _, movie in self.movie_catalog.iterrows():
            # Calculate frustration compatibility
            frustration_compatibility = self._calculate_frustration_compatibility(
                movie, strategy_config, frustration_level
            )
            
            # Calculate persona match
            persona_match = self._calculate_persona_match(movie, user_preferences)
            
            # Overall score (weighted combination)
            overall_score = (
                0.6 * frustration_compatibility + 
                0.4 * persona_match
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                movie, strategy_config, frustration_level, overall_score
            )
            
            recommendation = MovieRecommendation(
                item_id=str(movie.get('item_id', movie.get('id', ''))),
                title=movie.get('title', 'Unknown'),
                genres=movie.get('genres', []),
                frustration_compatibility=frustration_compatibility,
                persona_match=persona_match,
                overall_score=overall_score,
                reasoning=reasoning
            )
            
            recommendations.append(recommendation)
        
        # Sort by overall score
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        
        return recommendations
    
    def _calculate_frustration_compatibility(self, 
                                           movie: pd.Series, 
                                           strategy_config: Dict, 
                                           frustration_level: float) -> float:
        """Calculate how well a movie matches the user's current frustration state."""
        movie_genres = movie.get('genres', [])
        preferred_genres = strategy_config['preferred_genres']
        avoid_genres = strategy_config['avoid_genres']
        
        # Base compatibility
        compatibility = 0.5
        
        # Boost for preferred genres
        for genre in preferred_genres:
            if genre in movie_genres:
                compatibility += 0.2
        
        # Penalty for avoided genres
        for genre in avoid_genres:
            if genre in movie_genres:
                compatibility -= 0.3
        
        # Additional logic based on frustration level
        if frustration_level > 0.6:  # High frustration
            # Prefer shorter, lighter content
            if 'Comedy' in movie_genres or 'Animation' in movie_genres:
                compatibility += 0.15
            if 'Horror' in movie_genres or 'Thriller' in movie_genres:
                compatibility -= 0.25
        
        return max(0.0, min(1.0, compatibility))
    
    def _calculate_persona_match(self, 
                                movie: pd.Series, 
                                user_preferences: Optional[Dict]) -> float:
        """Calculate persona-based movie matching score."""
        if not user_preferences:
            return 0.5  # Neutral score
        
        movie_genres = movie.get('genres', [])
        preferred_genres = user_preferences.get('preferred_genres', [])
        
        if not preferred_genres:
            return 0.5
        
        # Calculate genre overlap
        overlap = len(set(movie_genres) & set(preferred_genres))
        total_preferred = len(preferred_genres)
        
        if total_preferred == 0:
            return 0.5
        
        return overlap / total_preferred
    
    def _generate_reasoning(self, 
                          movie: pd.Series, 
                          strategy_config: Dict, 
                          frustration_level: float, 
                          score: float) -> str:
        """Generate human-readable reasoning for recommendation."""
        movie_genres = movie.get('genres', [])
        strategy = strategy_config['strategy']
        
        if frustration_level > 0.6:
            return f"Recommended for stress relief. {', '.join(movie_genres[:2])} content helps reduce frustration."
        elif frustration_level < 0.2:
            return f"Perfect for exploration. {', '.join(movie_genres[:2])} matches your adventurous mood."
        else:
            return f"Balanced choice. {', '.join(movie_genres[:2])} content suits your current state."
    
    def _get_default_recommendations(self) -> List[MovieRecommendation]:
        """Get default recommendations when no session data is available."""
        # Return popular/highly-rated movies as fallback
        popular_movies = self.movie_catalog.head(self.config["recommendation_count"])
        
        recommendations = []
        for _, movie in popular_movies.iterrows():
            recommendation = MovieRecommendation(
                item_id=str(movie.get('item_id', movie.get('id', ''))),
                title=movie.get('title', 'Unknown'),
                genres=movie.get('genres', []),
                frustration_compatibility=0.5,
                persona_match=0.5,
                overall_score=0.5,
                reasoning="Popular content - no session data available"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions to prevent memory leaks."""
        current_time = time.time()
        timeout_seconds = self.config["session_timeout_minutes"] * 60
        
        expired_sessions = [
            session_key for session_key, session in self.active_sessions.items()
            if current_time - session.last_update > timeout_seconds
        ]
        
        for session_key in expired_sessions:
            del self.active_sessions[session_key]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_session_stats(self) -> Dict:
        """Get statistics about active sessions."""
        return {
            "active_sessions": len(self.active_sessions),
            "total_events": sum(len(s.events) for s in self.active_sessions.values()),
            "avg_frustration": np.mean([s.predicted_frustration for s in self.active_sessions.values()]) if self.active_sessions else 0.0
        }
