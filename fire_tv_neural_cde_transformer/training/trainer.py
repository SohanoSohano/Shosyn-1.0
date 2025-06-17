# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional

from .loss_functions import HybridLoss
from .optimizers import create_hybrid_optimizer
from utils.metrics import calculate_trait_metrics, calculate_recommendation_metrics

class HybridModelTrainer:
    """Trainer for the hybrid Fire TV Neural CDE + Transformer system"""
    
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = HybridLoss(config)
        
        # Optimizer and scheduler
        self.optimizer, self.schedulers = create_hybrid_optimizer(model, config)
        
        # Tracking
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'trait_metrics': [], 'rec_metrics': []
        }
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        trait_losses = []
        rec_losses = []
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = self.model(
                    interaction_data=batch['interaction_data'],
                    modal_data=batch.get('modal_data'),
                    content_history=batch.get('content_history'),
                    target_recommendations=batch.get('target_recommendations')
                )
                
                # Calculate loss
                loss_dict = self.criterion(outputs, batch)
                total_loss_item = loss_dict['total_loss']
                
                # Backward pass
                total_loss_item.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # Track losses
                total_loss += total_loss_item.item()
                trait_losses.append(loss_dict['trait_loss'].item())
                if 'recommendation_loss' in loss_dict:
                    rec_losses.append(loss_dict['recommendation_loss'].item())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{total_loss_item.item():.4f}",
                    'Trait': f"{loss_dict['trait_loss'].item():.4f}"
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        avg_trait_loss = np.mean(trait_losses) if trait_losses else 0
        avg_rec_loss = np.mean(rec_losses) if rec_losses else 0
        
        return {
            'total_loss': avg_loss,
            'trait_loss': avg_trait_loss,
            'recommendation_loss': avg_rec_loss
        }
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = self._move_batch_to_device(batch)
                
                try:
                    outputs = self.model(
                        interaction_data=batch['interaction_data'],
                        modal_data=batch.get('modal_data'),
                        content_history=batch.get('content_history'),
                        target_recommendations=batch.get('target_recommendations')
                    )
                    
                    loss_dict = self.criterion(outputs, batch)
                    total_loss += loss_dict['total_loss'].item()
                    
                    # Collect predictions for metrics
                    all_predictions.append(outputs['psychological_traits'].cpu())
                    if 'trait_targets' in batch:
                        all_targets.append(batch['trait_targets'].cpu())
                    
                except Exception as e:
                    continue
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = {}
        if all_targets:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            metrics = calculate_trait_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            for scheduler in self.schedulers.values():
                if hasattr(scheduler, 'step'):
                    scheduler.step(val_loss)
            
            # Track history
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['val_loss'].append(val_loss)
            self.training_history['trait_metrics'].append(val_metrics)
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self._save_best_model()
                print(f"✅ New best model saved (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Logging
            self._log_metrics(epoch, train_metrics, val_loss, val_metrics)
            
            # Progress report
            if (epoch + 1) % 10 == 0:
                self._print_progress_report(epoch, train_metrics, val_loss, val_metrics)
        
        print("Training completed!")
        return self.training_history
    
    def _move_batch_to_device(self, batch):
        """FIXED: Move batch data to device with proper error handling"""
        
        if not isinstance(batch, dict):
            print(f"Warning: Expected dict, got {type(batch)}")
            return batch
        
        device_batch = {}
        
        for key, value in batch.items():
            try:
                if isinstance(value, torch.Tensor):
                    device_batch[key] = value.to(self.device)
                elif isinstance(value, dict):
                    device_batch[key] = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in value.items()
                    }
                elif isinstance(value, list):
                    # Handle lists (like user_ids, session_ids)
                    device_batch[key] = value
                else:
                    device_batch[key] = value
            except Exception as e:
                print(f"Error moving {key} to device: {e}")
                device_batch[key] = value
        
        return device_batch
    
    def _save_best_model(self):
        """Save the best model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }, self.config.model_save_path)
    
    def _log_metrics(self, epoch, train_metrics, val_loss, val_metrics):
        """Log metrics to wandb"""
        log_dict = {
            'epoch': epoch,
            'train/total_loss': train_metrics['total_loss'],
            'train/trait_loss': train_metrics['trait_loss'],
            'val/total_loss': val_loss,
        }
        
        # Add validation metrics
        for metric_name, metric_value in val_metrics.items():
            log_dict[f'val/{metric_name}'] = metric_value
        
        # Add learning rates
        for name, scheduler in self.schedulers.items():
            if hasattr(scheduler, 'get_last_lr'):
                log_dict[f'lr/{name}'] = scheduler.get_last_lr()[0]
        
        wandb.log(log_dict)
    
    def _print_progress_report(self, epoch, train_metrics, val_loss, val_metrics):
        """Print progress report"""
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"├── Train Loss: {train_metrics['total_loss']:.6f}")
        print(f"├── Val Loss: {val_loss:.6f}")
        print(f"├── Best Loss: {self.best_loss:.6f}")
        print(f"└── Patience: {self.patience_counter}/{self.config.patience}")
        
        if val_metrics:
            print("Validation Metrics:")
            for metric, value in val_metrics.items():
                print(f"  {metric}: {value:.4f}")
