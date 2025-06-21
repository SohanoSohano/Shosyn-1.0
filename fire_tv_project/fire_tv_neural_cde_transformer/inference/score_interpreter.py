# inference/score_interpreter.py

import numpy as np
from typing import Dict, List, Tuple

class ScoreInterpreter:
    """Enhanced score interpretation with confidence levels and descriptions"""
    
    def __init__(self):
        self.score_bands = {
            "excellent": {"min": 6.5, "color": "#4CAF50", "description": "Highly recommended based on your preferences"},
            "good": {"min": 5.5, "color": "#8BC34A", "description": "Good match for your viewing style"},
            "fair": {"min": 4.5, "color": "#FFC107", "description": "Might interest you"},
            "poor": {"min": 0.0, "color": "#F44336", "description": "Not recommended for you"}
        }
    
    def interpret_score(self, score: float) -> Dict[str, str]:
        """Convert numerical score to interpretable band with metadata"""
        for band_name, band_info in self.score_bands.items():
            if score >= band_info["min"]:
                return {
                    "band": band_name.title(),
                    "score": round(score, 2),
                    "color": band_info["color"],
                    "description": band_info["description"],
                    "confidence": self._calculate_confidence(score)
                }
        return self.score_bands["poor"]
    
    def _calculate_confidence(self, score: float) -> str:
        """Calculate confidence level based on score magnitude"""
        if score >= 7.0:
            return "Very High"
        elif score >= 6.0:
            return "High"
        elif score >= 5.0:
            return "Medium"
        elif score >= 4.0:
            return "Low"
        else:
            return "Very Low"
    
    def batch_interpret(self, scores: List[float]) -> List[Dict]:
        """Interpret multiple scores at once"""
        return [self.interpret_score(score) for score in scores]
