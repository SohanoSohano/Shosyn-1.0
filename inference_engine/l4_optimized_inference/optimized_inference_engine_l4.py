import torch
import pandas as pd
import numpy as np
import json
import logging
import time
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
import signatory
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
from enhanced_psychological_tracker import EnhancedPsychologicalTracker, PsychologicalState
import threading
from concurrent.futures import Future 
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Represents a user's current session state with both psychological dimensions."""
    user_id: str
    session_id: str
    events: List[Dict]
    genre_history: List[str] = field(default_factory=list)
    genre_affinity: Dict[str, float] = field(default_factory=dict) # <-- ADDED for new strategy
    current_frustration: float = 0.0
    current_cognitive_load: float = 0.1
    predicted_frustration: float = 0.0
    predicted_cognitive_load: float = 0.1
    last_update: float = 0.0

@dataclass
class MovieRecommendation:
    """Enhanced movie recommendation with cognitive load considerations."""
    item_id: str
    title: str
    genres: List[str]
    frustration_compatibility: float
    cognitive_compatibility: float
    persona_match: float
    overall_score: float
    reasoning: str

class MultiTargetNeuralRDE(torch.nn.Module):
    """Multi-target Neural RDE model for inference."""
    
    def __init__(self, input_channels, hidden_dims, output_channels=2):
        super(MultiTargetNeuralRDE, self).__init__()
        
        layers = []
        in_dim = input_channels
        
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, h_dim))
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Dropout(0.1))
            in_dim = h_dim
        
        self.shared_layers = torch.nn.Sequential(*layers)
        
        self.frustration_head = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(in_dim // 2, 1)
        )
        
        self.cognitive_load_head = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // 2),
            torch.nn.Tanh(), 
            torch.nn.Linear(in_dim // 2, 1)
        )

    def forward(self, logsig):
        shared_features = self.shared_layers(logsig)
        frustration = self.frustration_head(shared_features)
        cognitive_load = self.cognitive_load_head(shared_features)
        output = torch.cat([frustration, cognitive_load], dim=1)
        output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
        return output

    def __init__(self, 
                model_path: str,
                movie_catalog_path: str,
                config_path: Optional[str] = None,
                batch_size: int = 32,        # NEW: L4 batch processing parameter
                max_wait_ms: int = 10):      # NEW: L4 timing parameter
        """
        Initialize the multi-target inference engine with L4 optimization support.
        
        Args:
            model_path: Path to trained Multi-Target Neural RDE model (.pth file)
            movie_catalog_path: Path to TMDB movie catalog CSV
            config_path: Optional path to configuration file
            batch_size: Batch size for L4 GPU optimization (default: 32)
            max_wait_ms: Maximum wait time for batch processing in milliseconds (default: 10)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing multi-target inference engine on {self.device}")
        
        # NEW: Store L4 optimization parameters
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        
        self._initialize_l4_optimizations()

        self.calibration_enabled = True
        self.prediction_errors = {}  # Track prediction errors for calibration

        # Load configuration
        self.config = self._load_config(config_path)
        
        # Top K Shuffling Config.
        self.config.update({
            "enable_top_k_shuffling": True,
            "shuffle_method": "weighted",  # Options: "simple", "weighted"
            "shuffle_consistency_factor": 0.3,  # 30% chance to keep original top movie
            "shuffle_k_size": 5,  # Number of top recommendations to shuffle
            "shuffle_weight_exponent": 3.0,  # Controls how much to favor higher scores
            "min_recommendations_for_shuffle": 3  # Only shuffle if we have at least this many
        })

        # Load and initialize model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load movie catalog
        self.movie_catalog = self._load_movie_catalog(movie_catalog_path)
        
        # Initialize preprocessing components
        self.scaler, self.ohe_encoder = self._initialize_preprocessors()
        
        # Active user sessions
        self.active_sessions: Dict[str, UserSession] = {}
        
        # Prediction history for smoothing
        self.prediction_history = {}
        
        # Initialize Kalman smoother
        self.kalman_smoother = KalmanSmoother(
            process_noise=0.01,  # Low process noise for stable predictions
            measurement_noise=0.1  # Moderate measurement noise
        )

        # Enhanced recommendation strategies for multi-target
        self.recommendation_strategies = self._initialize_multitarget_strategies()
        
        # Setup prediction logging
        self.setup_prediction_logging()

        # Initialize enhanced psychological tracker
        self.psychological_tracker = EnhancedPsychologicalTracker()
        
        # Enhanced recommendation strategies with temporal awareness
        self.enhanced_strategies = self._initialize_enhanced_strategies()
        
        logger.info("Multi-target inference engine initialized successfully with L4 optimization support")

    
    def _update_genre_affinity(self, session: UserSession, genres: List[str]):
        """Update the user's in-session genre affinity."""
        for genre in genres:
            if genre in session.genre_affinity:
                session.genre_affinity[genre] += 1
            else:
                session.genre_affinity[genre] = 1

    def _adjust_inputs_with_genre_heuristics(self, event: Dict, session: UserSession) -> Tuple[float, float]:
        """
        Improves the raw psychological inputs based on genre interaction heuristics.
        """
        base_frustration = event.get('frustration_level', 0.1)
        base_cognitive_load = event.get('cognitive_load', 0.1)

        focused_item = event.get("focused_item", {})

        current_genres = focused_item.get("genres", [])
        
        self._update_genre_affinity(session, current_genres)

        if not current_genres:
            return base_frustration, base_cognitive_load

        # Define genre categories
        high_cognitive_genres = {"Documentary", "Drama", "Sci-Fi", "Thriller", "Mystery"}
        low_cognitive_genres = {"Comedy", "Family", "Animation", "Romance"}
        high_frustration_genres = {"Horror", "Thriller"}

        # 1. Adjust Cognitive Load
        if any(g in high_cognitive_genres for g in current_genres):
            base_cognitive_load *= 1.25
        elif any(g in low_cognitive_genres for g in current_genres):
            base_cognitive_load *= 0.80

        # 2. Adjust Frustration
        if len(session.genre_history) > 2 and event.get('action_type') in ['dpad_right', 'dpad_down']:
            last_three_genres = set(session.genre_history[-3:])
            if len(last_three_genres.intersection(high_frustration_genres)) >= 2:
                base_frustration *= 1.30

        if base_frustration > 0.4 and any(g in low_cognitive_genres for g in current_genres):
            base_frustration *= 0.70

        session.genre_history.extend(current_genres)
        
        adjusted_frustration = max(0.0, min(1.0, base_frustration))
        adjusted_cognitive_load = max(0.0, min(1.0, base_cognitive_load))

        return adjusted_frustration, adjusted_cognitive_load
    
    def _calibrate_predictions(self, session_key: str, frustration: float, cognitive_load: float) -> Tuple[float, float]:
        """Calibrate predictions based on historical accuracy using post-hoc methods."""
        
        if not self.calibration_enabled or session_key not in self.prediction_errors:
            return frustration, cognitive_load
        
        errors = self.prediction_errors[session_key]
        
        if len(errors['frustration']) >= 3:  # Need minimum samples for calibration
            # Simple linear calibration (Platt Scaling approach)
            avg_frustration_error = np.mean(errors['frustration'])
            avg_cognitive_error = np.mean(errors['cognitive'])
            
            # Apply calibration correction
            calibrated_frustration = min(frustration + avg_frustration_error * 0.3, 1.0)
            calibrated_cognitive = min(cognitive_load + avg_cognitive_error * 0.3, 1.0)
            
            return max(calibrated_frustration, 0.05), max(calibrated_cognitive, 0.1)
        
        return frustration, cognitive_load

    def _update_prediction_errors(self, session_key: str, predicted_f: float, predicted_c: float, 
                                 actual_f: float, actual_c: float):
        """Track prediction errors for calibration."""
        if session_key not in self.prediction_errors:
            self.prediction_errors[session_key] = {'frustration': [], 'cognitive': []}
        
        # Store recent errors (keep last 10)
        self.prediction_errors[session_key]['frustration'].append(actual_f - predicted_f)
        self.prediction_errors[session_key]['cognitive'].append(actual_c - predicted_c)
        
        # Keep only recent errors
        if len(self.prediction_errors[session_key]['frustration']) > 10:
            self.prediction_errors[session_key]['frustration'] = self.prediction_errors[session_key]['frustration'][-10:]
            self.prediction_errors[session_key]['cognitive'] = self.prediction_errors[session_key]['cognitive'][-10:]

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration with corrected thresholds."""
        default_config = {
            "logsig_depth": 2,
            "min_events_for_prediction": 3,
            "frustration_threshold_high": 0.06,    # Lowered for earlier detection
            "frustration_threshold_low": 0.03,     # Lowered for sensitivity
            "cognitive_load_threshold_high": 0.12, # Lowered for earlier detection
            "cognitive_load_threshold_low": 0.08,  # Lowered for sensitivity
            "recommendation_count": 10,
            "session_timeout_minutes": 30,
            "frustration_scale_factor": 6.0,       # INCREASED from 4.0
            "cognitive_scale_factor": 20.0,        # INCREASED from 15.0
            "calibration_enabled": True,
            "calibration_weight": 0.4,             # INCREASED from 0.3
            "pattern_aware_scaling": True,         # NEW: Enable pattern-based scaling
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def setup_prediction_logging(self):
        """Setup logging for predictions and errors."""
        self.prediction_logger = logging.getLogger('predictions')
        handler = logging.FileHandler('prediction_logs.json')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.prediction_logger.addHandler(handler)
        self.prediction_logger.setLevel(logging.INFO)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained Multi-Target Neural RDE model with robust size mismatch handling."""
        try:
            # First, load the checkpoint to inspect its structure
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine the correct architecture from the checkpoint
            first_layer_key = 'shared_layers.0.weight'
            if first_layer_key in checkpoint:
                expected_input_channels = checkpoint[first_layer_key].shape[1]
                
                # Determine hidden dimensions from checkpoint structure
                if 'shared_layers.6.weight' in checkpoint:
                    # Model has 3 hidden layers: [128, 64, 32]
                    hidden_dims = [128, 64, 32]
                else:
                    # Model has 2 hidden layers: [128, 64]
                    hidden_dims = [128, 64]
                
                logger.info(f"Detected model architecture: input_channels={expected_input_channels}, hidden_dims={hidden_dims}")
            else:
                raise ValueError("Cannot determine model architecture from checkpoint")
            
            # Create model with correct architecture
            model = MultiTargetNeuralRDE(
                input_channels=expected_input_channels,
                hidden_dims=hidden_dims,
                output_channels=2
            )
            
            # Robust loading with size mismatch handling
            self._load_state_dict_with_mismatch_handling(model, checkpoint)
            
            model.to(self.device)
            logger.info(f"Multi-target model loaded successfully from {model_path}")
            logger.info(f"Model architecture: input_channels={expected_input_channels}, hidden_dims={hidden_dims}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_state_dict_with_mismatch_handling(self, model, checkpoint):
        """Load state dict with robust size mismatch handling."""
        loaded_state_dict = checkpoint
        model_state_dict = model.state_dict()
        
        # Create filtered state dict
        filtered_state_dict = {}
        skipped_keys = []
        
        for k, v in loaded_state_dict.items():
            if k in model_state_dict:
                if v.size() == model_state_dict[k].size():
                    # Sizes match - load the parameter
                    filtered_state_dict[k] = v
                else:
                    # Size mismatch - keep model's original parameter
                    logger.warning(f"Size mismatch for {k}: checkpoint {v.size()} vs model {model_state_dict[k].size()}")
                    filtered_state_dict[k] = model_state_dict[k]
                    skipped_keys.append(k)
            else:
                # Key not in model - skip it
                logger.warning(f"Skipping unexpected key: {k}")
                skipped_keys.append(k)
        
        # Add any missing keys from model
        for k, v in model_state_dict.items():
            if k not in filtered_state_dict:
                logger.warning(f"Using random initialization for missing key: {k}")
                filtered_state_dict[k] = v
        
        # Load with strict=False
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if skipped_keys:
            logger.info(f"Successfully loaded model with {len(skipped_keys)} mismatched parameters")
            logger.info(f"Skipped keys: {skipped_keys}")
        else:
            logger.info("Model loaded perfectly - all parameters matched")
    
    def _load_movie_catalog(self, catalog_path: str) -> pd.DataFrame:
        """Load and preprocess movie catalog."""
        try:
            df = pd.read_csv(catalog_path)
            
            # Basic preprocessing for TMDB data
            if 'genres' in df.columns:
                df['genres'] = df['genres'].apply(self._parse_genres)
            
            # Create genre-based features for recommendation
            all_genres = set()
            for genres in df['genres']:
                all_genres.update(genres)
            
            # Create binary genre features
            for genre in all_genres:
                df[f'genre_{genre.lower()}'] = df['genres'].apply(lambda x: genre in x)
            
            # Add complexity scores for cognitive load matching
            df['complexity_score'] = self._calculate_content_complexity(df)
            
            logger.info(f"Loaded {len(df)} movies from catalog")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load movie catalog: {e}")
            raise
    
    def _calculate_content_complexity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate content complexity scores for cognitive load matching."""
        complexity_scores = []
        
        for _, movie in df.iterrows():
            score = 0.5  # Base complexity
            genres = movie.get('genres', [])
            
            # Genre-based complexity
            if 'Documentary' in genres:
                score += 0.3
            if 'Drama' in genres:
                score += 0.2
            if 'Sci-Fi' in genres:
                score += 0.2
            if 'Comedy' in genres:
                score -= 0.2
            if 'Animation' in genres:
                score -= 0.3
            if 'Family' in genres:
                score -= 0.2
            
            # Runtime-based complexity (if available)
            runtime = movie.get('runtime', 120)
            if runtime > 150:
                score += 0.1
            elif runtime < 90:
                score -= 0.1
            
            complexity_scores.append(max(0.1, min(1.0, score)))
        
        return pd.Series(complexity_scores)
    
    def _parse_genres(self, genres_str) -> List[str]:
        """Parse genres from string format."""
        try:
            if pd.isna(genres_str) or genres_str == '[]':
                return []
            
            if isinstance(genres_str, str):
                if genres_str.startswith('['):
                    import ast
                    genres_list = ast.literal_eval(genres_str)
                    if isinstance(genres_list, list) and len(genres_list) > 0:
                        if isinstance(genres_list[0], dict):
                            return [g.get('name', '') for g in genres_list]
                        else:
                            return genres_list
                else:
                    return [g.strip() for g in genres_str.split(',')]
            
            return []
        except:
            return []
    
    def _initialize_preprocessors(self):
        """Initialize preprocessing components to match training pipeline."""
        # These should match your training preprocessing
        scaler = StandardScaler()
        dummy_data = np.array([[0.2, 0.1], [0.8, 0.9]])  # frustration, cognitive_load ranges
        scaler.fit(dummy_data)
        
        # Initialize OHE for action types
        action_types = [['click'], ['back'], ['dpad_right'], ['dpad_down'], 
                       ['dpad_left'], ['dpad_up'], ['session_start']]
        ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        ohe_encoder.fit(action_types)
        
        return scaler, ohe_encoder
    
    def _initialize_multitarget_strategies(self) -> Dict:
        """Initialize recommendation strategies for different psychological states."""
        return {
            "high_frustration_high_cognitive": {
                "preferred_genres": ["Comedy", "Animation", "Family"],
                "avoid_genres": ["Horror", "Thriller", "Documentary", "Drama"],
                "complexity_preference": "low",
                "strategy": "simple_comfort_content",
                "description": "Simple, comforting content to reduce both stress and mental load"
            },
            "high_frustration_low_cognitive": {
                "preferred_genres": ["Comedy", "Family", "Animation"],
                "avoid_genres": ["Horror", "Thriller", "Action"],
                "complexity_preference": "low",
                "strategy": "comfort_content",
                "description": "Comforting, easy-to-follow content to reduce frustration"
            },
            "low_frustration_high_cognitive": {
                "preferred_genres": ["Documentary", "Drama", "Sci-Fi"],
                "avoid_genres": ["Animation", "Family"],
                "complexity_preference": "high",
                "strategy": "complex_exploration",
                "description": "Intellectually engaging content for focused viewing"
            },
            "low_frustration_low_cognitive": {
                "preferred_genres": ["Action", "Adventure", "Comedy"],
                "avoid_genres": [],
                "complexity_preference": "medium",
                "strategy": "balanced_exploration",
                "description": "Engaging content for relaxed exploration"
            },
            "recovery_comfort_content": {
                "preferred_genres": ["Comedy", "Animation", "Family", "Feel-good"],
                "avoid_genres": ["Horror", "Thriller", "Drama", "Documentary"],
                "complexity_preference": "very_low",
                "strategy": "gentle_recovery",
                "description": "Gentle, healing content for stress recovery"
            },
            "intervention_content": {
                "preferred_genres": ["Comedy", "Animation", "Family"],
                "avoid_genres": ["Horror", "Thriller", "Action", "Drama"],
                "complexity_preference": "very_low",
                "strategy": "immediate_relief",
                "description": "Immediate stress relief content"
            },
            "energy_boost_content": {
                "preferred_genres": ["Comedy", "Adventure", "Animation"],
                "avoid_genres": ["Horror", "Thriller"],
                "complexity_preference": "low",
                "strategy": "gentle_engagement",
                "description": "Engaging but not overwhelming content"
            }
        }


    def _initialize_enhanced_strategies(self) -> Dict:
        """Initialize enhanced strategies with temporal awareness."""
        return {
            "recovery_comfort_content": {
                "preferred_genres": ["Comedy", "Animation", "Family", "Feel-good"],
                "avoid_genres": ["Horror", "Thriller", "Drama", "Documentary"],
                "complexity_preference": "very_low",
                "strategy": "gentle_recovery",
                "description": "Gentle, healing content for stress recovery"
            },
            "intervention_content": {
                "preferred_genres": ["Comedy", "Animation", "Family"],
                "avoid_genres": ["Horror", "Thriller", "Action", "Drama"],
                "complexity_preference": "very_low",
                "strategy": "immediate_relief",
                "description": "Immediate stress relief content"
            },
            "energy_boost_content": {
                "preferred_genres": ["Comedy", "Adventure", "Animation"],
                "avoid_genres": ["Horror", "Thriller"],
                "complexity_preference": "low",
                "strategy": "gentle_engagement",
                "description": "Engaging but not overwhelming content"
            },
            "high_frustration_high_cognitive": {
                "preferred_genres": ["Comedy", "Animation", "Family"],
                "avoid_genres": ["Horror", "Thriller", "Documentary", "Drama"],
                "complexity_preference": "low",
                "strategy": "simple_comfort_content",
                "description": "Simple, comforting content to reduce both stress and mental load"
            },
            "high_frustration_low_cognitive": {
                "preferred_genres": ["Comedy", "Family", "Animation"],
                "avoid_genres": ["Horror", "Thriller", "Action"],
                "complexity_preference": "low",
                "strategy": "comfort_content",
                "description": "Comforting, easy-to-follow content to reduce frustration"
            },
            "low_frustration_high_cognitive": {
                "preferred_genres": ["Documentary", "Drama", "Sci-Fi"],
                "avoid_genres": ["Animation", "Family"],
                "complexity_preference": "high",
                "strategy": "complex_exploration",
                "description": "Intellectually engaging content for focused viewing"
            },
            "low_frustration_low_cognitive": {
                "preferred_genres": ["Action", "Adventure", "Comedy"],
                "avoid_genres": [],
                "complexity_preference": "medium",
                "strategy": "balanced_exploration",
                "description": "Engaging content for relaxed exploration"
            }
        }
    
    def _apply_enhanced_smoothing(self, session_key: str, enhanced_state: 'PsychologicalState') -> tuple:
        """Apply Kalman smoothing with temporal awareness."""
        
        # Apply Kalman smoothing
        smoothed_frustration, smoothed_cognitive = self.kalman_smoother.smooth_prediction(
            session_key, 
            enhanced_state.current_frustration, 
            enhanced_state.current_cognitive
        )
        
        # Apply trend-based adjustments (keep existing logic)
        frustration_adjustment = 0.0
        cognitive_adjustment = 0.0
        
        if enhanced_state.frustration_trend == 'increasing':
            frustration_adjustment = 0.01  # Reduced from 0.02 due to Kalman smoothing
        elif enhanced_state.frustration_trend == 'decreasing':
            frustration_adjustment = -0.01
        
        if enhanced_state.cognitive_trend == 'increasing':
            cognitive_adjustment = 0.01
        elif enhanced_state.cognitive_trend == 'decreasing':
            cognitive_adjustment = -0.01
        
        # Apply final adjustments
        final_frustration = max(0.0, min(1.0, smoothed_frustration + frustration_adjustment))
        final_cognitive = max(0.0, min(1.0, smoothed_cognitive + cognitive_adjustment))
        
        return final_frustration, final_cognitive

    
    def _enhanced_recommendation_trigger(self, enhanced_state: PsychologicalState) -> bool:
        """Enhanced logic for when to trigger recommendations."""
        # Immediate intervention needed
        if enhanced_state.stress_recovery_phase:
            return True
        
        # Increasing stress trends
        if (enhanced_state.frustration_trend == 'increasing' and 
            enhanced_state.current_frustration > 0.08):
            return True
        
        # High cognitive load with increasing trend
        if (enhanced_state.cognitive_trend == 'increasing' and 
            enhanced_state.current_cognitive > 0.15):
            return True
        
        # Fall back to existing logic
        return (enhanced_state.current_frustration > self.config["frustration_threshold_high"] or 
                enhanced_state.current_cognitive > self.config["cognitive_load_threshold_high"])    

    def _apply_prediction_smoothing(self, session_key: str, frustration: float, cognitive_load: float) -> Tuple[float, float]:
        """Apply smoothing over recent predictions."""
        if session_key not in self.prediction_history:
            self.prediction_history[session_key] = {'frustration': [], 'cognitive': []}
        
        history = self.prediction_history[session_key]
        
        # Add current predictions
        history['frustration'].append(frustration)
        history['cognitive'].append(cognitive_load)
        
        # Keep only last 3 predictions
        SMOOTHING_WINDOW = 3
        history['frustration'] = history['frustration'][-SMOOTHING_WINDOW:]
        history['cognitive'] = history['cognitive'][-SMOOTHING_WINDOW:]
        
        # Calculate smoothed values
        smoothed_frustration = sum(history['frustration']) / len(history['frustration'])
        smoothed_cognitive = sum(history['cognitive']) / len(history['cognitive'])
        
        return smoothed_frustration, smoothed_cognitive
    
    def update_session(self, user_id: str, session_id: str, event: Dict) -> Dict:
        """
        Enhanced session update with temporal psychological tracking and calibration.
        L4 GPU optimized for real-time performance.
        
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
        self._current_session_key = session_key  # Store for calibration
        if session_key not in self.active_sessions:
            self.active_sessions[session_key] = UserSession(
                user_id=user_id,
                session_id=session_id,
                events=[],
                last_update=current_time
            )
        
        session = self.active_sessions[session_key]
        
        # --- NEW STRATEGY IMPLEMENTATION ---
        # Apply genre-based heuristics to the incoming event's psychological values
        adj_frustration, adj_cognitive = self._adjust_inputs_with_genre_heuristics(event, session)
        event['frustration_level'] = adj_frustration
        event['cognitive_load'] = adj_cognitive
        # --- END OF NEW STRATEGY ---
        
        session.events.append(event)
        session.last_update = current_time
        
        # Predict both targets if we have enough events
        if len(session.events) >= self.config["min_events_for_prediction"]:
            try:
                # L4 GPU OPTIMIZATION: Use batch processing for better performance
                if not hasattr(self, '_l4_batch_queue'):
                    self._l4_batch_queue = []
                    self._l4_batch_sessions = []
                    self._l4_batch_futures = []
                
                # Add to L4 batch queue
                future = Future()
                self._l4_batch_queue.append(session.events)
                self._l4_batch_sessions.append(session_key)
                self._l4_batch_futures.append(future)
                
                # Process batch when optimal size reached or timeout
                if len(self._l4_batch_queue) >= 16:  # L4 sweet spot for Tensor Cores
                    self._process_l4_batch()
                else:
                    # Set timer for partial batch processing (5ms for real-time)
                    threading.Timer(0.005, self._process_l4_batch).start()
                
                # Wait for L4 batch result
                try:
                    predicted_frustration, predicted_cognitive_load = future.result(timeout=0.05)
                except:
                    # Fallback to original method if batch processing fails
                    predicted_frustration, predicted_cognitive_load = self._predict_psychological_state(session.events)
                
                # Get temporal psychological state
                enhanced_state = self.psychological_tracker.update_psychological_state(
                    session_key, predicted_frustration, predicted_cognitive_load, current_time
                )
                
                # Apply smoothing with temporal awareness
                smoothed_frustration, smoothed_cognitive = self._apply_enhanced_smoothing(
                    session_key, enhanced_state
                )
                
                # ENHANCED: Apply calibration after smoothing
                calibrated_frustration, calibrated_cognitive = self._enhanced_calibrate_predictions(
                    session_key, smoothed_frustration, smoothed_cognitive, session.events
                )
                
                # ENHANCED: Track prediction errors for calibration if actual values available
                if 'frustration_level' in event and 'cognitive_load' in event:
                    self._update_prediction_errors(
                        session_key,
                        calibrated_frustration, calibrated_cognitive,
                        event['frustration_level'], event['cognitive_load']
                    )
                
                session.predicted_frustration = calibrated_frustration
                session.predicted_cognitive_load = calibrated_cognitive
                session.enhanced_psychological_state = enhanced_state
                
                logger.info(f"Enhanced session update {session_key}: "
                        f"frustration={calibrated_frustration:.3f} (trend: {enhanced_state.frustration_trend}), "
                        f"cognitive={calibrated_cognitive:.3f} (trend: {enhanced_state.cognitive_trend}), "
                        f"recovery_phase={enhanced_state.stress_recovery_phase}")
                
                return {
                    "status": "success",
                    "user_id": user_id,
                    "session_id": session_id,
                    "predicted_frustration": calibrated_frustration,
                    "predicted_cognitive_load": calibrated_cognitive,
                    "psychological_trends": {
                        "frustration_trend": enhanced_state.frustration_trend,
                        "cognitive_trend": enhanced_state.cognitive_trend,
                        "recovery_phase": enhanced_state.stress_recovery_phase,
                        "session_duration": enhanced_state.session_duration,
                        # ADD: Confidence metrics
                        "frustration_trend_confidence": enhanced_state.frustration_trend_confidence,
                        "cognitive_trend_confidence": enhanced_state.cognitive_trend_confidence
                    },
                    "event_count": len(session.events),
                    "recommendations_needed": self._enhanced_recommendation_trigger(enhanced_state),
                    "calibration_applied": hasattr(self, 'prediction_errors') and session_key in getattr(self, 'prediction_errors', {}),
                    # ADD: Smoothing information
                    "smoothing_applied": True,
                    # L4 GPU: Performance metrics
                    "l4_optimized": True
                }

                
            except Exception as e:
                logger.error(f"Enhanced prediction failed for {session_key}: {e}")
                return {
                    "status": "error",
                    "message": str(e),
                    "user_id": user_id,
                    "session_id": session_id,
                    "event_count": len(session.events)
                }
        
        return {
            "status": "insufficient_data",
            "user_id": user_id,
            "session_id": session_id,
            "event_count": len(session.events)
        }

    # L4 GPU OPTIMIZATION: Add batch processing method
    def _process_l4_batch(self):
        """Process batch of events using L4 GPU optimization"""
        if not hasattr(self, '_l4_batch_queue') or not self._l4_batch_queue:
            return
        
        try:
            import torch
            from torch.cuda.amp import autocast
            
            # Prepare batch tensors for L4
            batch_events = self._l4_batch_queue.copy()
            batch_tensors = []
            
            for events in batch_events:
                # Convert events to tensor (adjust based on your model input format)
                event_features = self._events_to_tensor_features(events)
                batch_tensors.append(event_features)
            
            if batch_tensors:
                # Stack into batch tensor for L4 processing
                batch_tensor = torch.stack(batch_tensors)
                
                # L4 optimization: Use half precision for Tensor Cores
                if torch.cuda.is_available():
                    batch_tensor = batch_tensor.half().cuda()
                    
                    # L4 optimized inference with mixed precision
                    with torch.no_grad(), autocast():
                        if hasattr(self, 'model') and self.model is not None:
                            batch_predictions = self.model(batch_tensor)
                            batch_predictions = batch_predictions.float().cpu().numpy()
                        else:
                            # Fallback if model not loaded
                            batch_predictions = [[0.5, 0.5] for _ in batch_tensors]
                else:
                    # CPU fallback
                    batch_predictions = [[0.5, 0.5] for _ in batch_tensors]
                
                # Return results to futures
                for i, future in enumerate(self._l4_batch_futures):
                    if i < len(batch_predictions):
                        future.set_result((batch_predictions[i][0], batch_predictions[i][1]))
                    else:
                        future.set_result((0.5, 0.5))
            
            # Clear batch queues
            self._l4_batch_queue.clear()
            self._l4_batch_sessions.clear()
            self._l4_batch_futures.clear()
            
        except Exception as e:
            logger.error(f"L4 batch processing failed: {e}")
            # Set fallback results for all futures
            for future in self._l4_batch_futures:
                if not future.done():
                    future.set_result((0.5, 0.5))
            
            # Clear batch queues
            self._l4_batch_queue.clear()
            self._l4_batch_sessions.clear()
            self._l4_batch_futures.clear()

    def _events_to_tensor_features(self, events):
        """Convert events to tensor features for L4 processing"""
        # Adjust this based on your actual model input requirements
        features = []
        for event in events[-10:]:  # Use last 10 events
            event_features = [
                event.get('frustration_level', 0.1),
                event.get('cognitive_load', 0.1),
                event.get('timestamp', time.time()) % 86400,  # Time of day
                len(str(event.get('action_type', ''))),  # Action complexity
            ]
            features.extend(event_features)
        
        # Pad to expected size (adjust based on your model)
        while len(features) < 40:  # 10 events * 4 features
            features.append(0.0)
        
        return torch.tensor(features[:40], dtype=torch.float32)

    # L4 GPU OPTIMIZATION: Add initialization method (call this in your __init__)
    def _initialize_l4_optimizations(self):
        """Initialize L4 GPU optimizations"""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Enable L4-specific optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                # Load and optimize model for L4 if available
                if hasattr(self, 'model') and self.model is not None:
                    self.model = self.model.half().cuda()  # Use FP16 for L4 Tensor Cores
                    
                    # Warm up L4 Tensor Cores
                    dummy_input = torch.randn(16, 40).half().cuda()  # Adjust size based on your model
                    with torch.no_grad():
                        for _ in range(3):
                            _ = self.model(dummy_input)
                    
                    logger.info("L4 GPU optimizations applied successfully")
                else:
                    logger.warning("Model not found - L4 optimizations partially applied")
            else:
                logger.warning("CUDA not available - L4 optimizations skipped")
                
        except ImportError:
            logger.warning("PyTorch not available - L4 optimizations skipped")
        except Exception as e:
            logger.error(f"L4 optimization initialization failed: {e}")


        
    def _predict_psychological_state(self, events: List[Dict]) -> Tuple[float, float]:
        """Predict both frustration level and cognitive load with dynamic scaling based on recent behavior."""
        try:
            # Convert events to feature sequence (matching training format)
            features = self._events_to_features(events)
            
            # Create path tensor
            path_tensor = torch.tensor(features, dtype=torch.float32)
            
            # Compute log-signature with correct depth
            logsignature = signatory.logsignature(
                path_tensor.unsqueeze(0), 
                self.config["logsig_depth"]  # Use depth 2
            ).squeeze(0)
            
            # Predict using model
            with torch.no_grad():
                logsignature = logsignature.to(self.device)
                predictions = self.model(logsignature.unsqueeze(0))
                frustration = float(predictions.cpu().numpy()[0, 0])
                cognitive_load = float(predictions.cpu().numpy()[0, 1])
            
            # Get base scaling factors
            base_frustration_scale = self.config["frustration_scale_factor"]
            base_cognitive_scale = self.config["cognitive_scale_factor"]
            
            # Previously ENHANCED: Dynamic scaling based on recent behavior
            # ENHANCED: Context-aware scaling
            recent_actions = [e.get('action_type', '') for e in events[-5:]]  # Increased window
            back_action_count = recent_actions.count('back')
            click_action_count = recent_actions.count('click')
            
            # Behavioral pattern analysis
            stress_indicators = back_action_count
            engagement_indicators = click_action_count
            
            # Dynamic scaling based on behavioral context
            if stress_indicators >= 3:  # High stress pattern
                frustration_boost = 2.5  # Increased from 1.5
                cognitive_boost = 2.0    # Increased from 1.3
            elif stress_indicators >= 2:
                frustration_boost = 2.0  # Increased from 1.2
                cognitive_boost = 1.7    # Increased from 1.1
            elif stress_indicators >= 1:
                frustration_boost = 1.5
                cognitive_boost = 1.3
            else:
                # Low stress - reduce over-prediction for relaxed users
                frustration_boost = 0.8  # NEW: Scale down for relaxed users
                cognitive_boost = 0.9
            
            # Apply dynamic scaling
            dynamic_frustration_scale = base_frustration_scale * frustration_boost
            dynamic_cognitive_scale = base_cognitive_scale * cognitive_boost
            
            # Scale predictions with dynamic factors
            scaled_frustration = min(frustration * dynamic_frustration_scale, 1.0)
            scaled_cognitive_load = min(cognitive_load * dynamic_cognitive_scale, 1.0)
            
            # APPLY CALIBRATION HERE
            session_key = getattr(self, '_current_session_key', None)
            if session_key:
                scaled_frustration, scaled_cognitive_load = self._calibrate_predictions(
                    session_key, scaled_frustration, scaled_cognitive_load
                )
            
            # Apply minimum thresholds to prevent unrealistically low values
            scaled_frustration = max(scaled_frustration, 0.05)
            scaled_cognitive_load = max(scaled_cognitive_load, 0.1)
            
            # ENHANCED: Log prediction with dynamic scaling details
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_count': len(events),
                'raw_frustration': frustration,
                'raw_cognitive_load': cognitive_load,
                'scaled_frustration': scaled_frustration,
                'scaled_cognitive_load': scaled_cognitive_load,
                'frustration_boost': frustration_boost,
                'cognitive_boost': cognitive_boost,
                'back_actions': back_action_count,
                'base_frustration_scale': base_frustration_scale,
                'base_cognitive_scale': base_cognitive_scale,
                'dynamic_frustration_scale': dynamic_frustration_scale,
                'dynamic_cognitive_scale': dynamic_cognitive_scale,
                'last_action': events[-1].get('action_type') if events else None
            }
            
            self.prediction_logger.info(json.dumps(log_entry))
            
            scaled_frustration, scaled_cognitive_load = self._apply_adaptive_bounds(
                scaled_frustration, scaled_cognitive_load, events
            )

            return scaled_frustration, scaled_cognitive_load
            
        except Exception as e:
            logger.error(f"Psychological state prediction error: {e}")
            return 0.3, 0.2  # Higher default values for better intervention

    
    def _events_to_features(self, events: List[Dict]) -> np.ndarray:
        """Revert to original feature extraction to match model dimensions."""
        features = []
        
        for i, event in enumerate(events):
            # Time delta
            if i == 0:
                time_delta = 0.0
            else:
                time_delta = 1.0  # Simplified - implement proper time calculation
            
            # Psychological features (ORIGINAL - no weighting)
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
            
            # Combine all features (ORIGINAL FORMAT)
            feature_row = np.concatenate([
                [time_delta],
                psych_scaled,
                [scroll_speed, scroll_depth],
                action_encoded
            ])
            
            features.append(feature_row)
        
        return np.array(features)

    
    def get_recommendations(self, user_id: str, session_id: str, 
                        user_preferences: Optional[Dict] = None) -> List[MovieRecommendation]:
        """Enhanced recommendations with temporal psychological awareness."""
        
        session_key = f"{user_id}_{session_id}"
        
        if session_key not in self.active_sessions:
            return self._get_default_recommendations()
        
        session = self.active_sessions[session_key]
        
        # FIXED: Initialize variables at the start
        enhanced_state = None
        frustration_level = session.predicted_frustration
        cognitive_load_level = session.predicted_cognitive_load
        
        # Use enhanced psychological state if available
        if hasattr(session, 'enhanced_psychological_state') and session.enhanced_psychological_state:
            enhanced_state = session.enhanced_psychological_state
            strategy = self._determine_enhanced_strategy(session) # <-- Pass the whole session
            strategy_config = self.enhanced_strategies.get(
                strategy, 
                self.recommendation_strategies.get('low_frustration_low_cognitive', {})
            )
            # Update with enhanced state values
            frustration_level = enhanced_state.current_frustration
            cognitive_load_level = enhanced_state.current_cognitive
            
            logger.info(f"Using enhanced strategy '{strategy}' for session {session_key}")
        else:
            # Fall back to basic strategy
            strategy = self._determine_strategy(frustration_level, cognitive_load_level)
            strategy_config = self.recommendation_strategies.get(strategy, {})
            logger.info(f"Using basic strategy '{strategy}' for session {session_key}")
        
        # Generate recommendations
        try:
            recommendations = self._score_movies_multitarget(
                strategy_config, 
                frustration_level,
                cognitive_load_level,
                user_preferences,
                session # <-- Pass the session
            )
            
            return recommendations[:self.config["recommendation_count"]]
            
        except Exception as e:
            logger.error(f"Recommendation generation failed for {session_key}: {e}")
            return self._get_default_recommendations()

    
    def _determine_strategy(self, frustration: float, cognitive_load: float) -> str:
        """Improved strategy determination with better thresholds and logic."""
        
        # DEBUG: Log the decision process
        logger.info(f"Strategy detection: frustration={frustration:.3f}, cognitive_load={cognitive_load:.3f}")
        
        # FIXED THRESHOLDS - More sensitive to user state
        frustration_high_threshold = 0.08  # Lower from 0.3
        frustration_medium_threshold = 0.05  # New medium threshold
        
        cognitive_high_threshold = 0.15  # Lower from 0.4
        cognitive_medium_threshold = 0.11  # New medium threshold
        
        # Categorize frustration level
        if frustration >= frustration_high_threshold:
            frustration_category = "high"
        elif frustration >= frustration_medium_threshold:
            frustration_category = "medium"
        else:
            frustration_category = "low"
        
        # Categorize cognitive load level
        if cognitive_load >= cognitive_high_threshold:
            cognitive_category = "high"
        elif cognitive_load >= cognitive_medium_threshold:
            cognitive_category = "medium"
        else:
            cognitive_category = "low"
        
        # IMPROVED STRATEGY MAPPING
        strategy_map = {
            ("high", "high"): "high_frustration_high_cognitive",
            ("high", "medium"): "high_frustration_high_cognitive",  # Treat as high cognitive
            ("high", "low"): "high_frustration_low_cognitive",
            ("medium", "high"): "high_frustration_high_cognitive",  # Treat as high frustration
            ("medium", "medium"): "high_frustration_low_cognitive",  # Moderate stress = comfort content
            ("medium", "low"): "low_frustration_low_cognitive",
            ("low", "high"): "low_frustration_high_cognitive",
            ("low", "medium"): "low_frustration_low_cognitive",
            ("low", "low"): "low_frustration_low_cognitive"
        }
        
        strategy = strategy_map.get((frustration_category, cognitive_category), "low_frustration_low_cognitive")
        
        logger.info(f"Selected strategy: {strategy} (frustration: {frustration_category}, cognitive: {cognitive_category})")
        
        return strategy
    
    def _score_movies_multitarget(self, 
                                strategy_config: Dict, 
                                frustration_level: float,
                                cognitive_load_level: float,
                                user_preferences: Optional[Dict],
                                session: UserSession) -> List[MovieRecommendation]:
        """Enhanced movie scoring with temporal awareness and top-k shuffling for discovery."""
        import random
        import time
        
        recommendations = []
        
        # Get the top genre from the session's affinity scores
        top_genre = None
        if session.genre_affinity:
            top_genre = max(session.genre_affinity, key=session.genre_affinity.get)
            
        for _, movie in self.movie_catalog.iterrows():
            # Calculate base compatibility scores
            frustration_compatibility = self._calculate_frustration_compatibility(
                movie, strategy_config, frustration_level
            )
            
            cognitive_compatibility = self._calculate_cognitive_compatibility(
                movie, strategy_config, cognitive_load_level
            )
            
            persona_match = self._calculate_persona_match(movie, user_preferences)

            # --- NEW STRATEGY IMPLEMENTATION ---
            # Boost score based on genre affinity
            genre_affinity_boost = 0.0
            if top_genre and top_genre in movie.get('genres', []):
                genre_affinity_boost = 0.15 # Add a significant boost for matching the top genre
            # --- END OF NEW STRATEGY ---
            
            # ADD DIVERSITY FACTORS TO CREATE SCORE VARIATION
            movie_genres = movie.get('genres', [])
            movie_id = movie.get('item_id', movie.get('id', 0))
            
            # Add genre-specific scoring variations
            genre_bonus = 0.0
            if 'Animation' in movie_genres:
                genre_bonus += 0.03
            if 'Comedy' in movie_genres:
                genre_bonus += 0.02
            if 'Action' in movie_genres:
                genre_bonus += 0.01
            if 'Family' in movie_genres:
                genre_bonus += 0.025
            
            popularity_factor = (hash(str(movie_id)) % 100) / 1000.0
            
            runtime = movie.get('runtime', 120)
            runtime_factor = 0.0
            if 90 <= runtime <= 120:
                runtime_factor = 0.015
            elif runtime > 150:
                runtime_factor = -0.005 if frustration_level > 0.2 else 0.02
            
            base_score = (
                0.4 * frustration_compatibility + 
                0.3 * cognitive_compatibility +
                0.3 * persona_match
            )
            
            scaled_base_score = base_score * 0.85
            diversity_bonus = genre_bonus + popularity_factor + runtime_factor
            
            overall_score = scaled_base_score + diversity_bonus + genre_affinity_boost
            
            overall_score = max(0.0, min(overall_score, 0.98))
            
            enhanced_state = getattr(session, 'enhanced_psychological_state', None)
            
            if enhanced_state:
                reasoning = self._generate_enhanced_reasoning(
                    movie, enhanced_state, strategy_config, overall_score
                )
            else:
                reasoning = self._generate_diverse_reasoning(
                    movie, strategy_config, 0.1, 0.1, overall_score
                )
            
            recommendation = MovieRecommendation(
                item_id=str(movie.get('item_id', movie.get('id', ''))),
                title=movie.get('title', 'Unknown'),
                genres=movie.get('genres', []),
                frustration_compatibility=frustration_compatibility,
                cognitive_compatibility=cognitive_compatibility,
                persona_match=persona_match,
                overall_score=round(overall_score, 3),
                reasoning=reasoning
            )
            
            recommendations.append(recommendation)
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Apply diversity filter first
        filtered_recommendations = self._apply_advanced_diversity_filter(recommendations)
        
        # NEW: Apply top-k shuffling for discovery
        if self.config.get("enable_top_k_shuffling", True):
            final_recommendations = self._apply_discovery_shuffling(
                filtered_recommendations, 
                session_key=f"{session.user_id}_{session.session_id}"
            )
            return final_recommendations
        else:
            return filtered_recommendations

    def _apply_discovery_shuffling(self, recommendations: List[MovieRecommendation], 
                                session_key: str) -> List[MovieRecommendation]:
        """Apply discovery shuffling with proper error handling"""
        if len(recommendations) < self.config.get("min_recommendations_for_shuffle", 3):
            logger.info(f"Skipping shuffle - only {len(recommendations)} recommendations")
            return recommendations
        
        shuffle_method = self.config.get("shuffle_method", "weighted")
        k = min(self.config.get("shuffle_k_size", 8), len(recommendations))
        
        # Store original order for analytics
        original_top_k = recommendations[:k]
        
        # FIXED: Ensure shuffle methods always return a list
        if shuffle_method == "weighted":
            shuffled_recommendations = self._weighted_shuffle_top_k(recommendations, k)
        else:
            shuffled_recommendations = self._simple_shuffle_top_k(recommendations, k)
        
        # FIXED: Add safety check
        if shuffled_recommendations is None:
            logger.warning(f"Shuffle method returned None, using original recommendations")
            shuffled_recommendations = recommendations
        
        # Track analytics with safety check
        if len(shuffled_recommendations) >= k:
            self._track_shuffle_analytics(original_top_k, shuffled_recommendations[:k], session_key)
        else:
            logger.warning(f"Shuffled recommendations too short: {len(shuffled_recommendations)}")
        
        logger.info(f"Applied {shuffle_method} shuffle to top-{k} for {session_key}")
        return shuffled_recommendations

    def _track_shuffle_analytics(self, original_top_k, shuffled_top_k, session_key):
        """Track shuffle analytics with safety checks"""
        # FIXED: Add safety checks
        if not original_top_k or not shuffled_top_k:
            logger.warning("Cannot track analytics - empty recommendation lists")
            return
        
        position_changes = 0
        min_length = min(len(original_top_k), len(shuffled_top_k))
        
        for i in range(min_length):
            if original_top_k[i].item_id != shuffled_top_k[i].item_id:
                position_changes += 1
        
        analytics = {
            'session_key': session_key,
            'timestamp': time.time(),
            'k_size': min_length,
            'position_changes': position_changes,
            'discovery_rate': position_changes / min_length if min_length > 0 else 0,
            'original_top_score': original_top_k[0].overall_score if original_top_k else 0,
            'shuffled_top_score': shuffled_top_k[0].overall_score if shuffled_top_k else 0
        }
        
        self.shuffle_analytics.append(analytics)


    def _simple_shuffle_top_k(self, recommendations: List[MovieRecommendation], 
                            k: int) -> List[MovieRecommendation]:
        """Simple random shuffle - FIXED VERSION"""
        if k <= 1 or len(recommendations) <= 1:
            return recommendations
        
        # FIXED: Ensure we always return a list
        try:
            top_k = recommendations[:k]
            remaining = recommendations[k:]
            random.shuffle(top_k)
            return top_k + remaining
        except Exception as e:
            logger.warning(f"Simple shuffle failed: {e}, returning original list")
            return recommendations


    def _weighted_shuffle_top_k(self, recommendations: List[MovieRecommendation], 
                            k: int) -> List[MovieRecommendation]:
        """
        Complete weighted shuffle with consistency factor and robust error handling.
        Implements Option 3 hybrid approach with all fixes.
        """
        import random
        
        # Basic validation
        if k <= 1 or len(recommendations) <= 1:
            return recommendations
        
        if k > len(recommendations):
            k = len(recommendations)
        
        try:
            # NEW: Consistency factor - sometimes keep the original top movie
            consistency_factor = self.config.get("shuffle_consistency_factor", 0.3)
            
            if random.random() < consistency_factor:
                # CONSISTENCY MODE: Keep original top movie, shuffle positions 2-k only
                top_1 = recommendations[:1]
                remaining_k = recommendations[1:k] if k > 1 else []
                after_k = recommendations[k:]
                
                if len(remaining_k) <= 1:
                    return recommendations  # Not enough to shuffle
                
                # Shuffle only positions 2-k with weighted selection
                weight_exponent = self.config.get("shuffle_weight_exponent", 2.5)
                weights = []
                
                for rec in remaining_k:
                    if hasattr(rec, 'overall_score') and rec.overall_score is not None:
                        weight = max(0.1, (float(rec.overall_score) ** weight_exponent) * 100)
                    else:
                        weight = 1.0  # Default weight if score missing
                    weights.append(weight)
                
                # Weighted shuffle of positions 2-k
                shuffled_remaining = []
                available_recs = remaining_k.copy()
                available_weights = weights.copy()
                
                while available_recs and len(shuffled_remaining) < len(remaining_k):
                    try:
                        # Ensure weights are valid
                        if not available_weights or all(w <= 0 for w in available_weights):
                            # Fallback to simple shuffle if weights are invalid
                            shuffled_remaining.extend(available_recs)
                            break
                        
                        # Weighted random selection
                        selected_rec = random.choices(available_recs, weights=available_weights, k=1)[0]
                        shuffled_remaining.append(selected_rec)
                        
                        # Remove selected item
                        idx = available_recs.index(selected_rec)
                        available_recs.pop(idx)
                        available_weights.pop(idx)
                        
                    except (ValueError, IndexError, TypeError) as e:
                        # Fallback: add remaining items in order
                        shuffled_remaining.extend(available_recs)
                        break
                
                return top_1 + shuffled_remaining + after_k
            
            # FULL SHUFFLE MODE: Original weighted shuffle logic
            top_k = recommendations[:k]
            remaining = recommendations[k:]
            
            # Create weights based on scores with robust error handling
            weights = []
            weight_exponent = self.config.get("shuffle_weight_exponent", 2.5)
            
            for rec in top_k:
                try:
                    if hasattr(rec, 'overall_score') and rec.overall_score is not None:
                        score = float(rec.overall_score)
                        weight = max(0.1, (score ** weight_exponent) * 100)
                    else:
                        weight = 1.0  # Default weight
                except (ValueError, TypeError):
                    weight = 1.0  # Fallback weight
                weights.append(weight)
            
            # Validate weights
            if not weights or all(w <= 0 for w in weights):
                # Fallback to simple shuffle if all weights are invalid
                logger.warning("Invalid weights detected, falling back to simple shuffle")
                random.shuffle(top_k)
                return top_k + remaining
            
            # Weighted random selection to create new order
            shuffled_top_k = []
            available_recs = top_k.copy()
            available_weights = weights.copy()
            
            while available_recs and len(shuffled_top_k) < k:
                try:
                    # Double-check weights before selection
                    if not available_weights or all(w <= 0 for w in available_weights):
                        shuffled_top_k.extend(available_recs)
                        break
                    
                    # Weighted random selection
                    selected_rec = random.choices(available_recs, weights=available_weights, k=1)[0]
                    shuffled_top_k.append(selected_rec)
                    
                    # Remove selected item
                    idx = available_recs.index(selected_rec)
                    available_recs.pop(idx)
                    available_weights.pop(idx)
                    
                except (ValueError, IndexError, TypeError) as e:
                    # Fallback: add remaining items in order
                    logger.warning(f"Weighted selection failed: {e}, adding remaining items")
                    shuffled_top_k.extend(available_recs)
                    break
            
            return shuffled_top_k + remaining
            
        except Exception as e:
            # Ultimate fallback: simple shuffle
            logger.error(f"Weighted shuffle completely failed: {e}, falling back to simple shuffle")
            try:
                top_k = recommendations[:k]
                remaining = recommendations[k:]
                random.shuffle(top_k)
                return top_k + remaining
            except Exception as final_e:
                logger.error(f"Even simple shuffle failed: {final_e}, returning original list")
                return recommendations


    def _log_shuffle_analytics(self, original_recommendations: List[MovieRecommendation],
                            shuffled_recommendations: List[MovieRecommendation],
                            k: int, session_key: str):
        """Log analytics about the shuffling for monitoring."""
        if not original_recommendations or not shuffled_recommendations:
            return
        
        # Calculate position changes
        position_changes = 0
        for i in range(min(k, len(original_recommendations), len(shuffled_recommendations))):
            if original_recommendations[i].item_id != shuffled_recommendations[i].item_id:
                position_changes += 1
        
        # Log the analytics
        logger.info(f"Shuffle Analytics for {session_key}: "
                f"Changed {position_changes}/{k} positions, "
                f"Discovery rate: {(position_changes/k)*100:.1f}%")
        
        # Optional: Store analytics for later analysis
        if not hasattr(self, 'shuffle_analytics'):
            self.shuffle_analytics = []
        
        self.shuffle_analytics.append({
            'session_key': session_key,
            'timestamp': time.time(),
            'k_size': k,
            'position_changes': position_changes,
            'discovery_rate': position_changes / k,
            'top_original_score': original_recommendations[0].overall_score if original_recommendations else 0,
            'top_shuffled_score': shuffled_recommendations[0].overall_score if shuffled_recommendations else 0
        })

    def get_shuffle_analytics(self) -> Dict:
        """Get analytics about shuffling performance."""
        if not hasattr(self, 'shuffle_analytics') or not self.shuffle_analytics:
            return {
                'total_shuffles': 0,
                'avg_discovery_rate': 0.0,
                'avg_position_changes': 0.0
            }
        
        analytics = self.shuffle_analytics
        
        return {
            'total_shuffles': len(analytics),
            'avg_discovery_rate': sum(a['discovery_rate'] for a in analytics) / len(analytics),
            'avg_position_changes': sum(a['position_changes'] for a in analytics) / len(analytics),
            'last_24h_shuffles': len([a for a in analytics if time.time() - a['timestamp'] < 86400]),
            'config': {
                'shuffle_method': self.config.get('shuffle_method', 'weighted'),
                'shuffle_k_size': self.config.get('shuffle_k_size', 8),
                'weight_exponent': self.config.get('shuffle_weight_exponent', 2.0)
            }
        }


    def _apply_advanced_diversity_filter(self, recommendations: List[MovieRecommendation]) -> List[MovieRecommendation]:
        """Apply advanced diversity filtering to ensure varied recommendations."""
        diverse_recommendations = []
        genre_counts = {}
        director_counts = {}  # If you have director data
        MAX_PER_GENRE = 2
        MAX_PER_DIRECTOR = 1
        
        for rec in recommendations:
            # Check genre diversity
            primary_genre = rec.genres[0] if rec.genres else 'Unknown'
            genre_count = genre_counts.get(primary_genre, 0)
            
            # Check if we can add this recommendation
            can_add = True
            
            if genre_count >= MAX_PER_GENRE:
                # Only add if score is significantly higher than existing ones
                existing_scores = [r.overall_score for r in diverse_recommendations 
                                if r.genres and r.genres[0] == primary_genre]
                if existing_scores and rec.overall_score <= max(existing_scores) + 0.05:
                    can_add = False
            
            if can_add:
                diverse_recommendations.append(rec)
                genre_counts[primary_genre] = genre_count + 1
            
            # Stop when we have enough diverse recommendations
            if len(diverse_recommendations) >= self.config["recommendation_count"]:
                break
        
        return diverse_recommendations

    def _generate_diverse_reasoning(self, 
                                movie: pd.Series, 
                                strategy_config: Dict, 
                                frustration_level: float,
                                cognitive_load_level: float,
                                score: float) -> str:
        """Generate varied reasoning messages with more granular explanations."""
        movie_genres = movie.get('genres', [])
        primary_genre = movie_genres[0] if movie_genres else 'Unknown'
        
        # Create varied reasoning based on score ranges and genres
        if score > 0.9:
            if 'Animation' in movie_genres:
                return f"Outstanding match! {primary_genre} content offers exceptional stress relief and joy."
            elif 'Comedy' in movie_genres:
                return f"Perfect choice! {primary_genre} delivers exactly the mood boost you need right now."
            else:
                return f"Excellent recommendation! {primary_genre} content perfectly matches your current state."
        
        elif score > 0.85:
            if frustration_level > 0.15:
                return f"Great for unwinding. {primary_genre} content provides effective stress relief."
            else:
                return f"Highly recommended! {primary_genre} content aligns beautifully with your preferences."
        
        elif score > 0.8:
            if 'Family' in movie_genres:
                return f"Wonderful option. {primary_genre} content offers comfort and familiarity."
            else:
                return f"Strong match! {primary_genre} content suits your psychological profile well."
        
        elif score > 0.75:
            return f"Good choice. {primary_genre} content provides balanced entertainment for your current mood."
        
        elif score > 0.7:
            return f"Solid option. {primary_genre} offers {', '.join(movie_genres[:2])} entertainment."
        
        else:
            return f"Decent pick. {primary_genre} content provides light {', '.join(movie_genres[:2])} viewing."

    
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
            if 'Comedy' in movie_genres or 'Animation' in movie_genres:
                compatibility += 0.15
            if 'Horror' in movie_genres or 'Thriller' in movie_genres:
                compatibility -= 0.25
        
        return max(0.0, min(1.0, compatibility))
    
    def _calculate_cognitive_compatibility(self,
                                         movie: pd.Series,
                                         strategy_config: Dict,
                                         cognitive_load_level: float) -> float:
        """Calculate how well a movie matches the user's current cognitive load."""
        complexity_preference = strategy_config['complexity_preference']
        movie_complexity = movie.get('complexity_score', 0.5)
        
        # Base compatibility
        compatibility = 0.5
        
        # Match complexity to cognitive state
        if complexity_preference == "low":
            # Prefer simple content when cognitive load is high
            compatibility += (1.0 - movie_complexity) * 0.4
        elif complexity_preference == "high":
            # Prefer complex content when cognitive load is low
            compatibility += movie_complexity * 0.4
        else:  # medium
            # Prefer moderate complexity
            complexity_distance = abs(movie_complexity - 0.5)
            compatibility += (1.0 - complexity_distance * 2) * 0.3
        
        # Additional adjustments based on cognitive load level
        if cognitive_load_level > 0.7:  # Very high cognitive load
            if movie_complexity < 0.3:  # Very simple content
                compatibility += 0.2
        elif cognitive_load_level < 0.2:  # Very low cognitive load
            if movie_complexity > 0.7:  # Complex content
                compatibility += 0.2
        
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
    
    def _generate_multitarget_reasoning(self, 
                                    movie: pd.Series, 
                                    strategy_config: Dict, 
                                    frustration_level: float,
                                    cognitive_load_level: float,
                                    score: float) -> str:
        """Generate reasoning based on actual strategy selected."""
        movie_genres = movie.get('genres', [])
        strategy_description = strategy_config.get('description', '')
        
        # Use the strategy description to create appropriate reasoning
        if 'comfort' in strategy_description.lower():
            return f"Perfect for stress relief. {', '.join(movie_genres[:2])} content helps reduce frustration and provides comfort."
        elif 'simple' in strategy_description.lower():
            return f"Ideal for mental break. {', '.join(movie_genres[:2])} content is easy to follow and relaxing."
        elif 'complex' in strategy_description.lower():
            return f"Great for engaged viewing. {', '.join(movie_genres[:2])} content offers intellectual stimulation."
        elif 'balanced' in strategy_description.lower():
            return f"Perfect balance. {', '.join(movie_genres[:2])} content suits your relaxed, alert mood."
        else:
            return f"Good match. {', '.join(movie_genres[:2])} content aligns with your current psychological state."
    
    def _get_default_recommendations(self) -> List[MovieRecommendation]:
        """Get default recommendations when no session data is available."""
        popular_movies = self.movie_catalog.head(self.config["recommendation_count"])
        
        recommendations = []
        for _, movie in popular_movies.iterrows():
            recommendation = MovieRecommendation(
                item_id=str(movie.get('item_id', movie.get('id', ''))),
                title=movie.get('title', 'Unknown'),
                genres=movie.get('genres', []),
                frustration_compatibility=0.5,
                cognitive_compatibility=0.5,
                persona_match=0.5,
                overall_score=0.5,
                reasoning="Popular content - no session data available"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def cleanup_expired_sessions(self):
        """Enhanced cleanup including Kalman smoother states."""
        current_time = time.time()
        expired_sessions = []
        
        for session_key, session in self.active_sessions.items():
            if current_time - session.last_update > self.config["session_timeout_minutes"] * 60:
                expired_sessions.append(session_key)
        
        for session_key in expired_sessions:
            # Clean up main session
            del self.active_sessions[session_key]
            
            # Clean up Kalman smoother state
            self.kalman_smoother.cleanup_session(session_key)
            
            # Clean up psychological tracker state
            if hasattr(self.psychological_tracker, 'frustration_history'):
                self.psychological_tracker.frustration_history.pop(session_key, None)
                self.psychological_tracker.cognitive_history.pop(session_key, None)
                self.psychological_tracker.session_start_times.pop(session_key, None)
                self.psychological_tracker.peak_states.pop(session_key, None)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    
    def get_session_stats(self) -> Dict:
        """Get statistics about active sessions."""
        if not self.active_sessions:
            return {
                "active_sessions": 0,
                "total_events": 0,
                "avg_frustration": 0.0,
                "avg_cognitive_load": 0.0
            }
        
        return {
            "active_sessions": len(self.active_sessions),
            "total_events": sum(len(s.events) for s in self.active_sessions.values()),
            "avg_frustration": np.mean([s.predicted_frustration for s in self.active_sessions.values()]),
            "avg_cognitive_load": np.mean([s.predicted_cognitive_load for s in self.active_sessions.values()])
        }

    def _determine_enhanced_strategy(self, session: UserSession) -> str:
        """Enhanced strategy selection with temporal awareness and genre affinity."""
        
        enhanced_state = getattr(session, 'enhanced_psychological_state', None)
        if not enhanced_state:
            return "low_frustration_low_cognitive"

        # Check for strong genre affinity
        if session.genre_affinity:
            top_genre = max(session.genre_affinity, key=session.genre_affinity.get)
            if session.genre_affinity[top_genre] > 3: # If user has interacted with a genre 4+ times
                return f"focused_search_{top_genre.lower()}"

        # Priority 1: Recovery phase - gentle healing content
        if enhanced_state.stress_recovery_phase:
            logger.info("Selected recovery strategy: user is in stress recovery phase")
            return "recovery_comfort_content"
        
        # Priority 2: Increasing frustration - immediate intervention
        if (enhanced_state.frustration_trend == 'increasing' and 
            enhanced_state.current_frustration > 0.08):
            logger.info("Selected intervention strategy: frustration is increasing")
            return "intervention_content"
        
        # Priority 3: Decreasing cognitive capacity - energy boost
        if (enhanced_state.cognitive_trend == 'increasing' and 
            enhanced_state.current_cognitive > 0.15):
            logger.info("Selected energy boost strategy: cognitive load increasing")
            return "energy_boost_content"
        
        # Priority 4: Long session with stable high stress
        if (enhanced_state.session_duration > 300 and  # 5 minutes
            enhanced_state.current_frustration > 0.12):
            logger.info("Selected sustained comfort strategy: long session with stress")
            return "recovery_comfort_content"
        
        # Fall back to existing strategy logic
        return self._determine_strategy(
            enhanced_state.current_frustration,
            enhanced_state.current_cognitive
        )

    def _generate_enhanced_reasoning(self, movie: pd.Series, enhanced_state: PsychologicalState, 
                                strategy_config: Dict, overall_score: float) -> str:
        """Generate contextual, meaningful reasoning with temporal awareness."""
        
        movie_genres = movie.get('genres', [])
        primary_genre = movie_genres[0] if movie_genres else 'Unknown'
        strategy = strategy_config.get('strategy', 'unknown')
        
        # Recovery phase reasoning
        if enhanced_state.stress_recovery_phase:
            if overall_score > 0.9:
                return f"Perfect for your recovery journey. {movie.get('title', 'This content')} provides gentle " \
                    f"{primary_genre} to help you continue healing from stress."
            else:
                return f"Good for recovery. {primary_genre} content offers comfort as you unwind."
        
        # Increasing frustration trend reasoning
        if enhanced_state.frustration_trend == 'increasing':
            if 'Comedy' in movie_genres or 'Animation' in movie_genres:
                return f"Immediate stress relief! {movie.get('title', 'This')} offers comforting " \
                    f"{primary_genre} to help break the frustration cycle before it builds further."
            else:
                return f"Stress intervention. {primary_genre} content designed to halt increasing frustration."
        
        # Cognitive overload reasoning
        if enhanced_state.cognitive_trend == 'increasing':
            complexity_score = movie.get('complexity_score', 0.5)
            if complexity_score < 0.3:
                return f"Mental break time! {movie.get('title', 'This content')} provides easy " \
                    f"{primary_genre} viewing to give your mind a rest."
            else:
                return f"Light engagement. {primary_genre} content that won't add to your mental load."
        
        # Long session reasoning
        if enhanced_state.session_duration > 600:  # 10 minutes
            return f"Perfect for extended viewing. {movie.get('title', 'This')} offers " \
                f"sustained {primary_genre} entertainment for your longer session."
        
        # Trend-aware reasoning based on score
        if overall_score > 0.95:
            if enhanced_state.frustration_trend == 'decreasing':
                return f"Excellent choice as you're feeling better! {movie.get('title', 'This')} " \
                    f"provides uplifting {primary_genre} to maintain your improving mood."
            else:
                return f"Outstanding match! {movie.get('title', 'This')} offers exceptional " \
                    f"{primary_genre} perfectly suited to your current state."
        
        elif overall_score > 0.9:
            return f"Great match for your current mood. {movie.get('title', 'This')} provides " \
                f"excellent {primary_genre} that aligns with your psychological state."
        
        elif overall_score > 0.85:
            return f"Good choice for right now. {primary_genre} content that fits your " \
                f"current emotional and mental state well."
        
        else:
            return f"Solid option. {primary_genre} content that provides appropriate " \
                f"entertainment for your current psychological profile."

    def _enhanced_recommendation_trigger(self, enhanced_state: PsychologicalState) -> bool:
        """Enhanced logic for when to trigger recommendations."""
        # Immediate intervention needed
        if enhanced_state.stress_recovery_phase:
            return True
        
        # Increasing stress trends
        if (enhanced_state.frustration_trend == 'increasing' and 
            enhanced_state.current_frustration > 0.08):
            return True
        
        # High cognitive load with increasing trend
        if (enhanced_state.cognitive_trend == 'increasing' and 
            enhanced_state.current_cognitive > 0.15):
            return True
        
        # Fall back to existing logic
        return (enhanced_state.current_frustration > self.config.get("frustration_threshold_high", 0.08) or 
                enhanced_state.current_cognitive > self.config.get("cognitive_load_threshold_high", 0.15))

    def _enhanced_calibrate_predictions(self, session_key: str, frustration: float, 
                                    cognitive_load: float, events: List[Dict]) -> Tuple[float, float]:
        """Enhanced calibration with pattern-aware adjustments."""
        
        # Base calibration
        calibrated_f, calibrated_c = self._calibrate_predictions(session_key, frustration, cognitive_load)
        
        # Pattern-based adjustments
        if len(events) >= 3:
            recent_actual_f = [e.get('frustration_level', 0) for e in events[-3:]]
            recent_actual_c = [e.get('cognitive_load', 0) for e in events[-3:]]
            
            # Trend-aware calibration
            if len(recent_actual_f) >= 2:
                actual_f_trend = recent_actual_f[-1] - recent_actual_f[0]
                actual_c_trend = recent_actual_c[-1] - recent_actual_c[0]
                
                # If actual values are trending up, boost predictions
                if actual_f_trend > 0.1:
                    calibrated_f = min(calibrated_f * 1.3, 1.0)
                if actual_c_trend > 0.1:
                    calibrated_c = min(calibrated_c * 1.3, 1.0)
                
                # If actual values are trending down, reduce predictions
                if actual_f_trend < -0.1:
                    calibrated_f = max(calibrated_f * 0.8, 0.05)
                if actual_c_trend < -0.1:
                    calibrated_c = max(calibrated_c * 0.8, 0.1)
        
        return calibrated_f, calibrated_c

    def _apply_adaptive_bounds(self, frustration: float, cognitive_load: float, 
                            events: List[Dict]) -> Tuple[float, float]:
        """Apply adaptive bounds based on user behavior context."""
        
        # Analyze recent behavior for context
        if len(events) >= 3:
            recent_actions = [e.get('action_type', '') for e in events[-3:]]
            back_actions = recent_actions.count('back')
            
            # High stress context - allow higher predictions
            if back_actions >= 2:
                max_frustration = 0.9
                max_cognitive = 0.85
                min_frustration = 0.2
                min_cognitive = 0.15
            # Normal context
            else:
                max_frustration = 0.7
                max_cognitive = 0.6
                min_frustration = 0.05
                min_cognitive = 0.1
        else:
            # Default bounds
            max_frustration = 0.8
            max_cognitive = 0.7
            min_frustration = 0.05
            min_cognitive = 0.1
        
        # Apply adaptive bounds
        bounded_frustration = max(min_frustration, min(frustration, max_frustration))
        bounded_cognitive = max(min_cognitive, min(cognitive_load, max_cognitive))
        
        return bounded_frustration, bounded_cognitive

    def _optimize_memory_for_l4(self):
        """Optimize memory usage for L4's 24GB VRAM"""
        # Keep frequently accessed data on GPU
        if hasattr(self, 'movie_embeddings'):
            self.movie_embeddings_gpu = torch.tensor(
                self.movie_embeddings, dtype=torch.half
            ).cuda()
        
        # Pre-allocate common tensor sizes
        self._preallocated_tensors = {
            'single': torch.zeros(1, self.input_size, dtype=torch.half).cuda(),
            'batch_16': torch.zeros(16, self.input_size, dtype=torch.half).cuda(),
            'batch_32': torch.zeros(32, self.input_size, dtype=torch.half).cuda(),
        }



class KalmanSmoother:
    """Kalman filter for smoothing psychological state predictions."""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.process_noise = process_noise  # How much we expect values to change
        self.measurement_noise = measurement_noise  # How much we trust new measurements
        
        # State tracking for each session
        self.session_states = {}
    
    def smooth_prediction(self, session_key: str, new_frustration: float, new_cognitive: float) -> tuple:
        """Apply Kalman smoothing to new predictions."""
        
        if session_key not in self.session_states:
            # Initialize state for new session
            self.session_states[session_key] = {
                'frustration': {
                    'estimate': new_frustration,
                    'error_covariance': 1.0
                },
                'cognitive': {
                    'estimate': new_cognitive,
                    'error_covariance': 1.0
                }
            }
            return new_frustration, new_cognitive
        
        # Apply Kalman filtering for both dimensions
        smoothed_frustration = self._kalman_update(
            self.session_states[session_key]['frustration'],
            new_frustration
        )
        
        smoothed_cognitive = self._kalman_update(
            self.session_states[session_key]['cognitive'],
            new_cognitive
        )
        
        return smoothed_frustration, smoothed_cognitive
    
    def _kalman_update(self, state: dict, measurement: float) -> float:
        """Single dimension Kalman update."""
        
        # Prediction step
        predicted_estimate = state['estimate']
        predicted_error_cov = state['error_covariance'] + self.process_noise
        
        # Update step
        kalman_gain = predicted_error_cov / (predicted_error_cov + self.measurement_noise)
        
        # Updated estimate
        updated_estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        updated_error_cov = (1 - kalman_gain) * predicted_error_cov
        
        # Store updated state
        state['estimate'] = updated_estimate
        state['error_covariance'] = updated_error_cov
        
        return updated_estimate
    
    def cleanup_session(self, session_key: str):
        """Remove session state when session expires."""
        if session_key in self.session_states:
            del self.session_states[session_key]
