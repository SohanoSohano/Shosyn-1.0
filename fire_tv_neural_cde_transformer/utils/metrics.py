# utils/metrics.py
import torch
import numpy as np

def calculate_trait_metrics(predictions, targets):
    """Calculate metrics for trait predictions"""
    
    # Simple correlation metric for now
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    
    correlations = []
    for i in range(predictions_np.shape[1]):
        corr = np.corrcoef(predictions_np[:, i], targets_np[:, i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    return {
        'mean_correlation': np.mean(correlations) if correlations else 0.0,
        'mse': torch.nn.functional.mse_loss(predictions, targets).item()
    }

def calculate_recommendation_metrics(predictions, targets):
    """Calculate recommendation metrics"""
    # Placeholder for recommendation metrics
    return {'recommendation_accuracy': 0.0}
