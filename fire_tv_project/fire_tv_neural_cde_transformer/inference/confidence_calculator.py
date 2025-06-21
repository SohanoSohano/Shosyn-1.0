# inference/confidence_calculator.py

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats

class ConfidenceCalculator:
    """Calculate confidence intervals and uncertainty measures for recommendations"""
    
    def __init__(self, bootstrap_samples: int = 100):
        self.bootstrap_samples = bootstrap_samples
    
    def calculate_recommendation_confidence(self, 
                                         user_traits: np.ndarray, 
                                         item_features: np.ndarray,
                                         base_score: float) -> Dict[str, float]:
        """Calculate confidence metrics for a recommendation"""
        
        # Simulate uncertainty through trait vector perturbation
        trait_std = np.std(user_traits) * 0.1  # 10% of trait standard deviation
        confidence_scores = []
        
        for _ in range(self.bootstrap_samples):
            # Add small random perturbations to simulate uncertainty
            perturbed_traits = user_traits + np.random.normal(0, trait_std, user_traits.shape)
            perturbed_score = np.dot(item_features, perturbed_traits)
            confidence_scores.append(perturbed_score)
        
        confidence_scores = np.array(confidence_scores)
        
        # Calculate confidence metrics
        mean_score = np.mean(confidence_scores)
        std_score = np.std(confidence_scores)
        ci_lower, ci_upper = np.percentile(confidence_scores, [2.5, 97.5])
        
        # Calculate confidence level based on variance
        confidence_level = self._calculate_confidence_level(std_score, base_score)
        
        return {
            "base_score": float(base_score),
            "mean_score": float(mean_score),
            "std_score": float(std_score),
            "confidence_interval_lower": float(ci_lower),
            "confidence_interval_upper": float(ci_upper),
            "confidence_level": confidence_level,
            "uncertainty": float(std_score / abs(mean_score)) if abs(mean_score) > 0 else 1.0
        }
    
    def _calculate_confidence_level(self, std_score: float, base_score: float) -> str:
        """Determine confidence level based on score variance"""
        relative_std = std_score / abs(base_score) if abs(base_score) > 0 else 1.0
        
        if relative_std < 0.05:
            return "Very High"
        elif relative_std < 0.10:
            return "High"
        elif relative_std < 0.20:
            return "Medium"
        elif relative_std < 0.35:
            return "Low"
        else:
            return "Very Low"
    
    def batch_calculate_confidence(self, 
                                 user_traits: np.ndarray, 
                                 all_item_features: np.ndarray,
                                 base_scores: List[float]) -> List[Dict]:
        """Calculate confidence for multiple recommendations"""
        results = []
        
        for i, (item_features, base_score) in enumerate(zip(all_item_features, base_scores)):
            confidence_metrics = self.calculate_recommendation_confidence(
                user_traits, item_features, base_score
            )
            results.append(confidence_metrics)
        
        return results
