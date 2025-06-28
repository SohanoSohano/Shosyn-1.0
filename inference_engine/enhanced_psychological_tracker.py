import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class PsychologicalState:
    """Enhanced psychological state with temporal awareness."""
    current_frustration: float
    current_cognitive: float
    frustration_trend: str  # 'increasing', 'decreasing', 'stable'
    cognitive_trend: str
    stress_recovery_phase: bool
    session_duration: float
    peak_frustration: float
    peak_cognitive: float

class EnhancedPsychologicalTracker:
    """Enhanced psychological state tracking with temporal patterns."""
    
    def __init__(self):
        self.frustration_history: Dict[str, List[Dict]] = {}
        self.cognitive_history: Dict[str, List[Dict]] = {}
        self.session_start_times: Dict[str, float] = {}
        self.peak_states: Dict[str, Dict] = {}
        
        # Trend detection parameters
        self.trend_window = 3  # Number of recent measurements to consider
        self.trend_threshold = 0.015  # Minimum change to detect trend
        self.recovery_threshold = 0.03  # Minimum drop from peak to detect recovery
        self.confidence_threshold = 0.6 # Minimum confidence for trend detection
    
    def update_psychological_state(self, session_key: str, frustration: float, 
                                 cognitive_load: float, timestamp: float) -> PsychologicalState:
        """Track psychological state evolution over time."""
        
        # Initialize if new session
        if session_key not in self.frustration_history:
            self._initialize_session(session_key, timestamp)
        
        # Add current state to history
        self._add_to_history(session_key, frustration, cognitive_load, timestamp)
        
        # Calculate trends and patterns
        frustration_trend = self._calculate_trend(self.frustration_history[session_key])
        cognitive_trend = self._calculate_trend(self.cognitive_history[session_key])
        
        # FIXED: Calculate recovery phase BEFORE using it in debug print
        stress_recovery = (self._detect_recovery_phase(session_key, frustration, cognitive_load) or 
                  self._detect_sustained_recovery(session_key))
        
        # DEBUG: Print trend calculation details
        if len(self.frustration_history[session_key]) >= 2:
            recent_f = [h['value'] for h in self.frustration_history[session_key][-2:]]
            recent_c = [h['value'] for h in self.cognitive_history[session_key][-2:]]
            
            f_change = recent_f[-1] - recent_f[0] if len(recent_f) >= 2 else 0
            c_change = recent_c[-1] - recent_c[0] if len(recent_c) >= 2 else 0
            
            print(f"DEBUG {session_key}: F_change={f_change:.3f}, C_change={c_change:.3f}, "
                  f"F_trend={frustration_trend}, C_trend={cognitive_trend}, "
                  f"Recovery={stress_recovery}")
        
        # Calculate session duration
        session_duration = timestamp - self.session_start_times[session_key]
        
        # Update peak states
        self._update_peak_states(session_key, frustration, cognitive_load)
        
        return PsychologicalState(
            current_frustration=frustration,
            current_cognitive=cognitive_load,
            frustration_trend=frustration_trend,
            cognitive_trend=cognitive_trend,
            stress_recovery_phase=stress_recovery,
            session_duration=session_duration,
            peak_frustration=self.peak_states[session_key]['frustration'],
            peak_cognitive=self.peak_states[session_key]['cognitive']
        )
    
    def _initialize_session(self, session_key: str, timestamp: float):
        """Initialize tracking for a new session."""
        self.frustration_history[session_key] = []
        self.cognitive_history[session_key] = []
        self.session_start_times[session_key] = timestamp
        self.peak_states[session_key] = {'frustration': 0.0, 'cognitive': 0.0}
    
    def _add_to_history(self, session_key: str, frustration: float, 
                       cognitive_load: float, timestamp: float):
        """Add current measurements to history."""
        self.frustration_history[session_key].append({
            'value': frustration,
            'timestamp': timestamp
        })
        self.cognitive_history[session_key].append({
            'value': cognitive_load,
            'timestamp': timestamp
        })
        
        # Keep only recent history (last 10 measurements)
        if len(self.frustration_history[session_key]) > 10:
            self.frustration_history[session_key] = self.frustration_history[session_key][-10:]
            self.cognitive_history[session_key] = self.cognitive_history[session_key][-10:]
    
    def _calculate_trend(self, history: List[Dict]) -> str:
        """Enhanced trend calculation with better sensitivity and stability."""
        if len(history) < 2:
            return 'stable'
        
        # Use last 3 points if available for more stable trend detection
        window_size = min(3, len(history))
        recent_values = [h['value'] for h in history[-window_size:]]
        
        if len(recent_values) >= 3:
            # Use linear regression for more accurate trend
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]
            
            # Adjusted thresholds for better sensitivity
            if slope > 0.015:  # Reduced from 0.02
                return 'increasing'
            elif slope < -0.015:  # Reduced from -0.02
                return 'decreasing'
            else:
                return 'stable'
        else:
            # Fallback to simple comparison
            change = recent_values[-1] - recent_values[0]
            if change > 0.02:
                return 'increasing'
            elif change < -0.02:
                return 'decreasing'
            else:
                return 'stable'

    
    def _detect_recovery_phase(self, session_key: str, current_frustration: float, 
                              current_cognitive: float) -> bool:
        """Improved recovery phase detection."""
        if len(self.frustration_history[session_key]) < 2:
            return False
        
        # Check recent history for recovery pattern
        frustration_values = [h['value'] for h in self.frustration_history[session_key]]
        
        if len(frustration_values) >= 2:
            # Check if we had high stress and it's now decreasing
            max_recent_frustration = max(frustration_values[-3:])
            
            # Recovery conditions (made more sensitive)
            had_high_stress = max_recent_frustration > 0.12  # Lowered from 0.2
            significant_drop = current_frustration < max_recent_frustration - self.recovery_threshold
            
            # Additional condition: check for sustained decrease
            if len(frustration_values) >= 3:
                recent_trend = frustration_values[-1] < frustration_values[-2] < frustration_values[-3]
                return had_high_stress and (significant_drop or recent_trend)
            else:
                return had_high_stress and significant_drop
        
        return False
    
    def _detect_sustained_recovery(self, session_key: str) -> bool:
        """Detect sustained recovery pattern across multiple events."""
        if len(self.frustration_history[session_key]) < 3:
            return False
        
        # Get last 3 frustration values
        recent_values = [h['value'] for h in self.frustration_history[session_key][-3:]]
        
        # Check for sustained decrease pattern
        if len(recent_values) == 3:
            # Pattern: high → medium → low
            return (recent_values[0] > 0.2 and 
                    recent_values[1] < recent_values[0] - 0.05 and
                    recent_values[2] < recent_values[1] - 0.05)
        
        return False


    def _update_peak_states(self, session_key: str, frustration: float, cognitive_load: float):
        """Update peak psychological states for the session."""
        self.peak_states[session_key]['frustration'] = max(
            self.peak_states[session_key]['frustration'], frustration
        )
        self.peak_states[session_key]['cognitive'] = max(
            self.peak_states[session_key]['cognitive'], cognitive_load
        )
    
    def get_session_summary(self, session_key: str) -> Optional[Dict]:
        """Get comprehensive session summary."""
        if session_key not in self.frustration_history:
            return None
        
        frustration_values = [h['value'] for h in self.frustration_history[session_key]]
        cognitive_values = [h['value'] for h in self.cognitive_history[session_key]]
        
        return {
            'session_duration': time.time() - self.session_start_times[session_key],
            'avg_frustration': np.mean(frustration_values),
            'avg_cognitive': np.mean(cognitive_values),
            'peak_frustration': self.peak_states[session_key]['frustration'],
            'peak_cognitive': self.peak_states[session_key]['cognitive'],
            'total_measurements': len(frustration_values),
            'current_trend': {
                'frustration': self._calculate_trend(self.frustration_history[session_key]),
                'cognitive': self._calculate_trend(self.cognitive_history[session_key])
            }
        }

    def _calculate_trend_confidence(self, history: List[Dict]) -> float:
        """Calculate confidence in trend detection."""
        if len(history) < 3:
            return 0.5
        
        recent_values = [h['value'] for h in history[-3:]]
        
        # Calculate variance to determine confidence
        variance = np.var(recent_values)
        
        # Lower variance = higher confidence in trend
        confidence = max(0.3, min(1.0, 1.0 - variance * 10))
        return confidence
