# scripts/train_enhanced_model.py
import sys
import os
import torch
import wandb
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.synthetic_model_config import HybridModelConfig
from data.synthetic_data_loader import SyntheticFireTVDataset
from data.behavioral_augmentation import AugmentedBehavioralDataset
from models.hybrid_model import HybridFireTVSystem
from training.label_smoothing_loss import LabelSmoothingBCELoss
from training.ensemble_trainer import EnsembleTrainer

def main():
    print("🚀🚀🚀 ENHANCED TRAINING WITH ADVANCED TECHNIQUES 🚀🚀🚀")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project="enhanced-psychological-model", name="advanced-techniques-v1")
    
    # Load configuration
    config = HybridModelConfig()
    
    # Enhanced training parameters
    config.learning_rate = 1e-4
    config.weight_decay = 1e-3
    config.num_epochs = 50
    config.batch_size = 32
    
    # Load and augment dataset
    print("📊 Loading and augmenting dataset...")
    base_dataset = SyntheticFireTVDataset(r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\fire_tv_neural_cde_transformer_instance_version\Shosyn-1.0\fire_tv_project\fire_tv_neural_cde_transformer\fire_tv_synthetic_dataset_v3_tmdb.csv")
    augmented_dataset = AugmentedBehavioralDataset(base_dataset, augmentation_factor=2)
    
    # Split dataset
    train_size = int(0.85 * len(augmented_dataset))
    val_size = len(augmented_dataset) - train_size
    train_dataset, val_dataset = random_split(augmented_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    print(f"   Data augmentation: 3x increase (original + 2 augmented versions)")
    
    # Choose training method
    use_ensemble = True  # Set to False for single model training
    
    if use_ensemble:
        print("\n🔥 Training Ensemble of Models...")
        ensemble_trainer = EnsembleTrainer(
            model_class=HybridFireTVSystem,
            config=config,
            num_models=3,
            device=device
        )
        
        criterion = LabelSmoothingBCELoss(smoothing=0.1)
        best_loss = ensemble_trainer.train_ensemble(train_loader, val_loader, config.num_epochs, criterion)
        
    else:
        print("\n🔥 Training Single Enhanced Model...")
        model = HybridFireTVSystem(config=config).to(device)
        
        # Enhanced optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
            eps=1e-8
        )
        
        # OneCycleLR scheduler
        total_steps = len(train_loader) * config.num_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=5e-4,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        # Label smoothing loss
        criterion = LabelSmoothingBCELoss(smoothing=0.1)
        
        # Training loop with all enhancements
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(1, config.num_epochs + 1):
            print(f"\n--- Enhanced Epoch {epoch}/{config.num_epochs} ---")
            
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                features, labels = batch['features'].to(device), batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model({'features': features})
                loss = criterion(outputs['psychological_traits'], labels)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()  # Step after each batch for OneCycleLR
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    features, labels = batch['features'].to(device), batch['labels'].to(device)
                    outputs = model({'features': features})
                    loss = criterion(outputs['psychological_traits'], labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Logging
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "epoch": epoch,
                "lr": current_lr
            })
            
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, LR = {current_lr:.2e}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "models/enhanced_best_model.pth")
                print(f"💡 Validation loss improved! Model saved.")
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                break
    
    print(f"\n🎉 Enhanced training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()
