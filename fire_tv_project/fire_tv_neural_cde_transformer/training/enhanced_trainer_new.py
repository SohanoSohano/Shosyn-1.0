# training/enhanced_trainer_new.py (Upgraded for Synthetic Data Training)
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import time
import gc

class EnhancedHybridModelTrainer:
    """
    An enhanced, flexible trainer for the Hybrid Fire TV System.
    --- MODIFIED to be compatible with the focused synthetic data training task. ---
    """

    # --- MODIFICATION: Updated __init__ signature ---
    # It now accepts the optimizer directly and makes TMDb-related components optional.
    # This resolves the TypeError while maintaining flexibility for future use.
    def __init__(self, model, optimizer, config, device, tmdb_integration=None, content_mapping=None):
        print("🔧 Initializing Upgraded EnhancedHybridModelTrainer...")
        self.model = model
        self.optimizer = optimizer  # <-- Accepts the externally created optimizer with weight_decay
        self.config = config
        self.device = device
        
        # --- MODIFICATION: Conditional setup for TMDb components ---
        # These are not needed for the synthetic training run.
        self.is_synthetic_run = (tmdb_integration is None)
        if self.is_synthetic_run:
            print("   Running in FOCUSED SYNTHETIC mode. TMDb and multi-objective loss are disabled.")
            # Use BCEWithLogitsLoss, which is ideal for the focused model's raw output (logits).
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            print("   Running in FULL PRODUCTION mode.")
            self.tmdb_integration = tmdb_integration
            self.content_mapping = content_mapping
            self._setup_production_losses()

        # --- Features from your original code (retained for their power) ---
        self.scaler = GradScaler() if getattr(config, 'use_mixed_precision', True) else None
        self.accumulation_steps = getattr(config, 'gradient_accumulation_steps', 2)
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = getattr(config, 'patience', 3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        
        print("✅ Trainer initialized successfully.")
        
    def _setup_production_losses(self):
        """Sets up the complex, multi-objective loss for the full model."""
        self.loss_weights = self.config.loss_weights
        self.primary_criterion = nn.MSELoss()
        self.rating_criterion = nn.MSELoss()
        # This assumes genre balancing is configured as in your original file.
        # For simplicity, we'll use a standard BCE loss here if not configured.
        self.content_criterion = nn.BCEWithLogitsLoss() 
        print(f"   Production loss weights configured: {self.loss_weights}")

    def _calculate_loss(self, outputs, batch):
        """
        --- MODIFICATION: Calculates loss based on the run mode. ---
        """
        if self.is_synthetic_run:
            # --- FOCUSED SYNTHETIC LOSS ---
            predicted_traits = outputs['psychological_traits']
            labels = batch['labels'].to(self.device)
            loss = self.criterion(predicted_traits, labels)
            return loss, {'total_loss': loss.item()}
        else:
            # --- FULL PRODUCTION LOSS (Logic adapted from your original file) ---
            labels = batch['labels'].to(self.device)
            # You would need to pass genre/rating targets in the batch for this to work.
            # This part is kept conceptually for future use.
            primary_loss = self.primary_criterion(outputs['psychological_traits'], labels)
            # ... calculation for rating_loss, genre_loss, etc. ...
            total_loss = primary_loss * self.loss_weights['traits'] # + other weighted losses
            return total_loss, {'total_loss': total_loss.item(), 'traits_loss': primary_loss.item()}

    def _run_one_epoch(self, data_loader, is_training: bool):
        """
        --- MODIFICATION: A unified loop for both training and validation. ---
        """
        self.model.train(is_training)
        total_loss = 0.0
        
        epoch_type = "Training" if is_training else "Validation"
        progress_bar = tqdm(data_loader, desc=f"{epoch_type} Epoch", leave=False)

        for batch in progress_bar:
            # The synthetic data loader provides a simple dictionary.
            features = batch['features'].to(self.device)
            model_input = {'features': features}

            # Forward pass
            with torch.set_grad_enabled(is_training):
                with autocast(enabled=self.scaler is not None):
                    outputs = self.model(model_input)
                    loss, loss_dict = self.calculate_loss(outputs, batch)
                    
                    if self.is_synthetic_run and len(self.accumulation_steps) > 1:
                        loss = loss / self.accumulation_steps

            # Backward pass and optimization (only if training)
            if is_training:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    if (progress_bar.n + 1) % self.accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    loss.backward()
                    if (progress_bar.n + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps if self.is_synthetic_run else loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss

    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop."""
        print(f"🚀 Starting training for {num_epochs} epochs in {'SYNTHETIC' if self.is_synthetic_run else 'PRODUCTION'} mode...")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            
            train_loss = self._run_one_epoch(train_loader, is_training=True)
            val_loss = self._run_one_epoch(val_loader, is_training=False)
            
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}")
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                model_path = "models/best_synthetic_trained_model.pth"
                torch.save(self.model.state_dict(), model_path)
                print(f"💡 Validation loss improved. Model saved to {model_path}")
            else:
                self.patience_counter += 1
                print(f"⏳ No improvement for {self.patience_counter}/{self.patience} epochs")
            
            if self.patience_counter >= self.patience:
                print(f"🛑 Early stopping triggered at epoch {epoch}.")
                break
        
        print("\n🎉 Training finished!")
        wandb.log({"best_validation_loss": self.best_loss})
        return {"best_validation_loss": self.best_loss}

