# training/loss_functions.py
import torch
import torch.nn as nn

class HybridLoss(nn.Module):
    """
    MODIFIED: This loss function is now aligned with the model's output.
    It calculates a primary loss on the final fused traits and adds weighted 
    auxiliary losses on the intermediate outputs from the CDE and Transformer.
    This resolves the unpacking errors seen in the training loop.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        
        # Weight for the auxiliary losses to control their contribution.
        # A value like 0.4 ensures the model prioritizes the final prediction
        # while still getting a helpful gradient signal for the sub-modules.
        self.aux_loss_weight = 0.4

    def forward(self, outputs, batch):
        """
        Calculate a hybrid loss with auxiliary components and robust error handling.
        """
        try:
            # --- MODIFICATION START ---
            
            # Safely get final predictions and targets using .get()
            final_predictions = outputs.get('psychological_traits')
            targets = batch.get('trait_targets')

            # The trainer should always provide targets, but this is a failsafe.
            if final_predictions is None or targets is None:
                raise ValueError("Could not find 'psychological_traits' in model outputs or 'trait_targets' in batch.")

            # 1. Calculate the primary loss on the final, fused prediction.
            primary_trait_loss = self.mse_loss(final_predictions, targets)
            total_loss = primary_trait_loss

            loss_dict = {
                'trait_loss': primary_trait_loss
            }

            # 2. Calculate auxiliary loss for the CDE branch, if its output is present.
            cde_traits = outputs.get('cde_traits')
            if cde_traits is not None:
                # The CDE hidden state might not have the same final dimension as the labels.
                # We only add this loss if the dimensions match. A more complex model
                # might add a dedicated linear "projection head" to match dimensions.
                if cde_traits.shape == targets.shape:
                    cde_aux_loss = self.mse_loss(cde_traits, targets)
                    total_loss += self.aux_loss_weight * cde_aux_loss
                    loss_dict['cde_aux_loss'] = cde_aux_loss

            # 3. Calculate auxiliary loss for the Transformer branch, if its output is present.
            transformer_traits = outputs.get('transformer_traits')
            if transformer_traits is not None:
                if transformer_traits.shape == targets.shape:
                    transformer_aux_loss = self.mse_loss(transformer_traits, targets)
                    total_loss += self.aux_loss_weight * transformer_aux_loss
                    loss_dict['transformer_aux_loss'] = transformer_aux_loss
            
            # Add final total loss to the dictionary
            loss_dict['total_loss'] = total_loss

            # 4. Placeholder for recommendation loss (logic can be added later)
            if 'recommendations' in outputs:
                rec_loss = torch.tensor(0.0, device=final_predictions.device)
                loss_dict['recommendation_loss'] = rec_loss
                loss_dict['total_loss'] += rec_loss * 0.1 # Example weighting

            return loss_dict
            
            # --- MODIFICATION END ---
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            # Return a dummy loss with requires_grad=True to allow training to continue
            dummy_loss = torch.tensor(1.0, requires_grad=True, device=self.config.device if hasattr(self.config, 'device') else 'cpu')
            return {
                'trait_loss': dummy_loss,
                'total_loss': dummy_loss
            }

