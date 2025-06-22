# inference/data_processor.py (TRIAL 2)
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FirebaseDataProcessor:
    """
    Processes Firebase log data and converts it into behavioral features
    suitable for psychological inference.
    """
    
    def __init__(self):
        self.required_features = [
            'dpad_up_count', 'dpad_down_count', 'dpad_left_count', 'dpad_right_count',
            'back_button_presses', 'menu_revisits', 'scroll_speed', 'hover_duration',
            'time_since_last_interaction'
        ]
    
    def process_user_data(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """
        Process a single user's interaction data into behavioral features.
        
        Args:
            user_df: DataFrame containing all interactions for one user
            
        Returns:
            Dictionary of behavioral features ready for model inference
        """
        try:
            if user_df.empty:
                logger.warning("Empty user data provided")
                return self._get_default_features()
            
            # Calculate each behavioral feature
            features = {}
            
            # 1. Button press counts
            button_counts = self._calculate_button_counts(user_df)
            features.update(button_counts)
            
            # 2. Scroll speed (average of burst speeds)
            features['scroll_speed'] = self._calculate_scroll_speed(user_df)
            
            # 3. Hover duration (average)
            features['hover_duration'] = self._calculate_hover_duration(user_df)
            
            # 4. Time since last interaction (maximum gap)
            features['time_since_last_interaction'] = self._calculate_time_gaps(user_df)
            
            # Validate and clean features
            features = self._validate_features(features)
            
            logger.info(f"Processed user data: {len(user_df)} interactions -> {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error processing user data: {e}")
            return self._get_default_features()
    
    def _calculate_button_counts(self, user_df: pd.DataFrame) -> Dict[str, int]:
        """Calculate individual button press counts."""
        movement_df = user_df[user_df['event_type'] == 'movements']
        
        if movement_df.empty:
            return {
                'dpad_up_count': 0, 'dpad_down_count': 0,
                'dpad_left_count': 0, 'dpad_right_count': 0,
                'back_button_presses': 0, 'menu_revisits': 0
            }
        
        button_counts = movement_df['button'].value_counts()
        
        return {
            'dpad_up_count': int(button_counts.get('Up', 0)),
            'dpad_down_count': int(button_counts.get('Down', 0)),
            'dpad_left_count': int(button_counts.get('Left', 0)),
            'dpad_right_count': int(button_counts.get('Right', 0)),
            'back_button_presses': int(button_counts.get('Back', 0)),
            'menu_revisits': int(button_counts.get('Menu', 0))
        }
    
    def _calculate_scroll_speed(self, user_df: pd.DataFrame) -> float:
        """Calculate average scroll speed from d-pad bursts."""
        movement_df = user_df[user_df['event_type'] == 'movements']
        dpad_df = movement_df[movement_df['button'].isin(['Up', 'Down', 'Left', 'Right'])]
        
        if len(dpad_df) < 2:
            return 100.0  # Default scroll speed
        
        # Calculate time differences between consecutive d-pad presses
        dpad_df = dpad_df.sort_values('timestamp').copy()
        dpad_df['time_diff'] = dpad_df['timestamp'].diff().dt.total_seconds().fillna(0)
        
        # Identify bursts (gaps > 2 seconds indicate new burst)
        dpad_df['burst_id'] = (dpad_df['time_diff'] > 2.0).cumsum()
        
        burst_speeds = []
        for burst_id, burst_group in dpad_df.groupby('burst_id'):
            if len(burst_group) > 1:
                duration = (burst_group['timestamp'].max() - burst_group['timestamp'].min()).total_seconds()
                if duration > 0:
                    speed = len(burst_group) / duration
                    burst_speeds.append(speed)
        
        return float(np.mean(burst_speeds)) if burst_speeds else 100.0
    
    def _calculate_hover_duration(self, user_df: pd.DataFrame) -> float:
        """Calculate average hover duration."""
        hover_df = user_df[user_df['event_type'] == 'hover_durations']
        
        if hover_df.empty:
            return 2.0  # Default hover duration
        
        # Convert duration from milliseconds to seconds
        durations = hover_df['duration'] / 1000.0
        return float(durations.mean())
    
    def _calculate_time_gaps(self, user_df: pd.DataFrame) -> float:
        """Calculate maximum time gap between interactions."""
        if len(user_df) < 2:
            return 5.0  # Default time gap
        
        user_df_sorted = user_df.sort_values('timestamp')
        time_diffs = user_df_sorted['timestamp'].diff().dt.total_seconds().dropna()
        
        return float(time_diffs.max()) if not time_diffs.empty else 5.0
    
    def _validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and clean feature values."""
        validated = {}
        
        for feature_name in self.required_features:
            value = features.get(feature_name, 0.0)
            
            # Handle NaN and infinite values
            if pd.isna(value) or np.isinf(value):
                value = 0.0
            
            # Apply reasonable bounds
            if feature_name in ['dpad_up_count', 'dpad_down_count', 'dpad_left_count', 'dpad_right_count']:
                value = max(0, min(value, 1000))  # Max 1000 presses
            elif feature_name in ['back_button_presses', 'menu_revisits']:
                value = max(0, min(value, 100))   # Max 100 presses
            elif feature_name == 'scroll_speed':
                value = max(10, min(value, 500))  # 10-500 presses/sec
            elif feature_name == 'hover_duration':
                value = max(0.1, min(value, 30))  # 0.1-30 seconds
            elif feature_name == 'time_since_last_interaction':
                value = max(0.1, min(value, 3600)) # 0.1-3600 seconds
            
            validated[feature_name] = float(value)
        
        return validated
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values for error cases."""
        return {
            'dpad_up_count': 0.0, 'dpad_down_count': 0.0,
            'dpad_left_count': 0.0, 'dpad_right_count': 0.0,
            'back_button_presses': 0.0, 'menu_revisits': 0.0,
            'scroll_speed': 100.0, 'hover_duration': 2.0,
            'time_since_last_interaction': 5.0
        }
    
    def process_session_data(self, user_df: pd.DataFrame, session_timeout_minutes: int = 30) -> List[Dict[str, float]]:
        """
        Split user data into sessions and process each session separately.
        
        Args:
            user_df: DataFrame containing all interactions for one user
            session_timeout_minutes: Minutes of inactivity to define new session
            
        Returns:
            List of feature dictionaries, one per session
        """
        try:
            if user_df.empty:
                return [self._get_default_features()]
            
            # Sort by timestamp
            user_df_sorted = user_df.sort_values('timestamp').copy()
            
            # Calculate time gaps
            user_df_sorted['time_gap'] = user_df_sorted['timestamp'].diff().dt.total_seconds().fillna(0)
            
            # Mark session boundaries (gaps > timeout)
            session_timeout_seconds = session_timeout_minutes * 60
            user_df_sorted['new_session'] = user_df_sorted['time_gap'] > session_timeout_seconds
            user_df_sorted['session_id'] = user_df_sorted['new_session'].cumsum()
            
            # Process each session
            session_features = []
            for session_id, session_df in user_df_sorted.groupby('session_id'):
                features = self.process_user_data(session_df)
                features['session_id'] = int(session_id)
                session_features.append(features)
            
            logger.info(f"Split user data into {len(session_features)} sessions")
            return session_features
            
        except Exception as e:
            logger.error(f"Error processing session data: {e}")
            return [self._get_default_features()]
