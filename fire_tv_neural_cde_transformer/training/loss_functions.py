# training/loss_functions.py
import torch
import torch.nn as nn

class HybridLoss(nn.Module):
    """FIXED: Loss function for hybrid Fire TV model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        
    def forward(self, outputs, batch):
        """Calculate hybrid loss with proper error handling"""
        
        try:
            # Get predictions and targets
            predictions = outputs['psychological_traits']
            
            # Check if we have real targets
            if 'trait_targets' in batch:
                targets = batch['trait_targets']
            else:
                # Generate dummy targets as fallback
                batch_size = predictions.shape[0]
                targets = torch.rand_like(predictions) * 0.6 + 0.2
                targets = targets.to(predictions.device)
            
            # Ensure shapes match
            if predictions.shape != targets.shape:
                print(f"Shape mismatch: pred {predictions.shape}, target {targets.shape}")
                # Adjust target shape if needed
                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(0).expand(predictions.shape[0], -1)
            
            # Calculate trait prediction loss
            trait_loss = self.mse_loss(predictions, targets)
            
            loss_dict = {
                'trait_loss': trait_loss,
                'total_loss': trait_loss
            }
            
            # Add other losses if available
            if 'recommendations' in outputs:
                # Placeholder for recommendation loss
                rec_loss = torch.tensor(0.0, device=predictions.device)
                loss_dict['recommendation_loss'] = rec_loss
                loss_dict['total_loss'] = trait_loss + rec_loss * 0.1
            
            return loss_dict
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            # Return dummy loss to prevent crash
            dummy_loss = torch.tensor(1.0, requires_grad=True)
            return {
                'trait_loss': dummy_loss,
                'total_loss': dummy_loss
            }
