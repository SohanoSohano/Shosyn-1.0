# training/ensemble_trainer.py
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from tqdm import tqdm

class EnsembleTrainer:
    """
    Train multiple models with different configurations for ensemble learning.
    """
    
    def __init__(self, model_class, config, num_models=3, device='cuda'):
        self.device = device
        self.models = []
        self.optimizers = []
        self.schedulers = []
        self.num_models = num_models
        
        print(f"🔥 Initializing Ensemble of {num_models} models")
        
        for i in range(num_models):
            # Create model with different random seed
            torch.manual_seed(42 + i * 100)
            model = model_class(config).to(device)
            
            # Use slightly different hyperparameters for diversity
            base_lr = config.learning_rate
            lr_multiplier = 0.8 + 0.4 * i / (num_models - 1)  # 0.8 to 1.2
            lr = base_lr * lr_multiplier
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=config.weight_decay * (0.5 + i * 0.5 / num_models)
            )
            
            # Different scheduler configurations
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=3 + i  # Different patience for each model
            )
            
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
            
            print(f"   Model {i+1}: lr={lr:.2e}, weight_decay={config.weight_decay * (0.5 + i * 0.5 / num_models):.2e}")
    
    def train_ensemble(self, train_loader, val_loader, num_epochs, criterion):
        """Train all models in the ensemble."""
        best_ensemble_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Ensemble Epoch {epoch}/{num_epochs} ---")
            
            # Train each model
            ensemble_train_losses = []
            for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                train_loss = self._train_single_model(model, optimizer, train_loader, criterion)
                ensemble_train_losses.append(train_loss)
                print(f"   Model {i+1} Train Loss: {train_loss:.6f}")
            
            # Validate ensemble
            ensemble_val_loss = self._validate_ensemble(val_loader, criterion)
            print(f"   Ensemble Validation Loss: {ensemble_val_loss:.6f}")
            
            # Update schedulers
            for scheduler in self.schedulers:
                scheduler.step(ensemble_val_loss)
            
            # Save best ensemble
            if ensemble_val_loss < best_ensemble_loss:
                best_ensemble_loss = ensemble_val_loss
                self._save_ensemble(epoch)
                print(f"   💡 Ensemble improved! Saved to ensemble_epoch_{epoch}/")
        
        return best_ensemble_loss
    
    def _train_single_model(self, model, optimizer, train_loader, criterion):
        """Train a single model for one epoch."""
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            features, labels = batch['features'].to(self.device), batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            outputs = model({'features': features})
            loss = criterion(outputs['psychological_traits'], labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_ensemble(self, val_loader, criterion):
        """Validate the ensemble by averaging predictions."""
        for model in self.models:
            model.eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch['features'].to(self.device), batch['labels'].to(self.device)
                
                # Get predictions from all models
                predictions = []
                for model in self.models:
                    outputs = model({'features': features})
                    predictions.append(outputs['psychological_traits'])
                
                # Average the predictions (ensemble prediction)
                ensemble_pred = torch.stack(predictions).mean(dim=0)
                loss = criterion(ensemble_pred, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _save_ensemble(self, epoch):
        """Save all models in the ensemble."""
        import os
        save_dir = f"models/ensemble_epoch_{epoch}"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{save_dir}/model_{i}.pth")
    
    def predict_ensemble(self, features):
        """Make ensemble predictions."""
        for model in self.models:
            model.eval()
        
        with torch.no_grad():
            predictions = []
            for model in self.models:
                outputs = model({'features': features})
                predictions.append(outputs['psychological_traits'])
            
            # Return averaged prediction
            return torch.stack(predictions).mean(dim=0)
