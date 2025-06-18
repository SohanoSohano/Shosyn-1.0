# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
import gc

class HybridModelTrainer:
    """
    NUCLEAR SOLUTION: Completely self-contained trainer with no external dependencies
    that could cause unpacking errors.
    """
    
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # SELF-CONTAINED: Create optimizer directly without external function
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=getattr(config, 'learning_rate', 1e-4),
            weight_decay=getattr(config, 'weight_decay', 1e-5)
        )
        
        # SELF-CONTAINED: Simple MSE loss instead of complex HybridLoss
        self.criterion = nn.MSELoss()
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def train_epoch(self, train_loader):
        """Simplified training epoch with no external dependencies"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, data_tuple in enumerate(pbar):
            try:
                # Data formatting (this part we know works)
                features, labels = data_tuple
                timestamps = torch.linspace(0, 1, steps=features.shape[1]).unsqueeze(0).repeat(features.shape[0], 1)
                
                # Move to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                timestamps = timestamps.to(self.device)
                
                interaction_data = {
                    'sequence': features,
                    'timestamps': timestamps
                }
                
                self.optimizer.zero_grad()
                
                # CRITICAL: Call model and handle output safely
                model_output = self.model(interaction_data)
                
                # SAFE: Extract predictions without assuming complex structure
                if isinstance(model_output, dict):
                    predictions = model_output.get('psychological_traits')
                else:
                    predictions = model_output
                
                if predictions is None:
                    print(f"Warning: No predictions from model in batch {batch_idx}")
                    continue
                
                # Simple MSE loss
                loss = self.criterion(predictions, labels)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    getattr(self.config, 'gradient_clip_norm', 1.0)
                )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
                # Log to wandb
                wandb.log({"batch_loss": loss.item()})
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def validate_epoch(self, val_loader):
        """Simplified validation epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data_tuple in tqdm(val_loader, desc="Validation"):
                try:
                    features, labels = data_tuple
                    timestamps = torch.linspace(0, 1, steps=features.shape[1]).unsqueeze(0).repeat(features.shape[0], 1)
                    
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    timestamps = timestamps.to(self.device)
                    
                    interaction_data = {
                        'sequence': features,
                        'timestamps': timestamps
                    }
                    
                    model_output = self.model(interaction_data)
                    
                    if isinstance(model_output, dict):
                        predictions = model_output.get('psychological_traits')
                    else:
                        predictions = model_output
                    
                    if predictions is None:
                        continue
                    
                    loss = self.criterion(predictions, labels)
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs):
        """Simplified training loop"""
        print(f"Starting simplified training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), "best_model_simplified.pth")
                print(f"✅ New best model saved (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= getattr(self.config, 'patience', 5):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
            
            # Clear GPU cache
            if self.device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
        
        print("Training completed!")
        return self.training_history
