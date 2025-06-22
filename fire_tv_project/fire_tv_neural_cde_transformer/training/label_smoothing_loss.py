# training/label_smoothing_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingBCELoss(nn.Module):
    """
    Label smoothing for multi-label classification using BCE.
    Based on the search results for multi-label problems.
    """
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        """
        Apply label smoothing to targets and compute BCE loss.
        
        Args:
            pred: Model predictions (logits)
            target: Ground truth labels (0 or 1)
        """
        # Apply sigmoid to get probabilities
        pred_probs = torch.sigmoid(pred)
        
        # Apply label smoothing: 
        # 1 becomes (1 - smoothing + smoothing/2) = (1 - smoothing/2)
        # 0 becomes (0 + smoothing/2) = smoothing/2
        target_smooth = target * (1 - self.smoothing) + self.smoothing / 2
        
        # Compute BCE loss with smoothed targets
        eps = 1e-8  # For numerical stability
        loss = -(target_smooth * torch.log(pred_probs + eps) + 
                (1 - target_smooth) * torch.log(1 - pred_probs + eps))
        
        return loss.mean()

# Update your training script to use this loss:
# criterion = LabelSmoothingBCELoss(smoothing=0.1)
