# scripts/train_hybrid_model_tmdb.py (Advanced Performance Version)
import sys
import os
import torch
import torch.nn as nn # Needed for gradient clipping
import wandb
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

# --- Project Path Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports---
from config.synthetic_model_config import HybridModelConfig
from data.synthetic_data_loader import SyntheticFireTVDataset
from models.hybrid_model import HybridFireTVSystem # Assumes you've added the ResidualBlock
from training.enhanced_trainer_new import EnhancedHybridModelTrainer # Assumes you've modified the scheduler call
from config.training_config import TrainingConfig

# --- Environment Setup ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    print("🚀🚀🚀 ADVANCED PERFORMANCE TRAINING RUN 🚀🚀🚀")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project="firetv-performance-run")
    
    # --- Load Configs & Data (Unchanged) ---
    model_config = HybridModelConfig()
    training_config = TrainingConfig()
    full_dataset = SyntheticFireTVDataset(r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\fire_tv_neural_cde_transformer_instance_version\Shosyn-1.0\fire_tv_project\fire_tv_neural_cde_transformer\fire_tv_synthetic_dataset_v3_tmdb.csv")
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, num_workers=4, pin_memory=True)
    
    model = HybridFireTVSystem(config=model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # This will be the max_lr for OneCycleLR
        betas=(0.9, 0.999),
        weight_decay=1e-3,
        eps=1e-8
    )

    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 50
    patience = 10 # Your increased patience
    patience_counter = 0
    best_val_loss = float('inf')

    # OneCycleLR - very powerful scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=5e-4,  # Peak learning rate (5x base lr)
        total_steps=total_steps,
        pct_start=0.3,  # Warm up for 30% of training
        anneal_strategy='cos',  # Cosine annealing
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,  # max_lr/div_factor = initial lr
        final_div_factor=1e4  # min_lr = initial_lr/final_div_factor
    )
    
    
    # --- Let's define the training loop directly here for clarity ---


    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        # Training Loop
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for batch in train_pbar:
            features, labels = batch['features'].to(device), batch['labels'].to(device)
            
            # --- STRATEGY 3: Input Noise Injection ---
            if model.training:
                features += torch.randn_like(features) * 0.01

            model_input = {'features': features}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(model_input)
                loss = criterion(outputs['psychological_traits'], labels)
            
            scaler.scale(loss).backward()
            
            # --- STRATEGY 4: Gradient Clipping ---
            scaler.unscale_(optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.step()
            scheduler.step() # Step the scheduler every batch
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            
        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                features, labels = batch['features'].to(device), batch['labels'].to(device)
                model_input = {'features': features}
                outputs = model(model_input)
                loss = criterion(outputs['psychological_traits'], labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch, "lr": scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Validation Loss = {avg_val_loss:.6f}")

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_performance_model.pth")
            print(f"💡 Validation loss improved. Model saved to models/best_performance_model.pth")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{patience} epochs.")
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch}.")
            break

    print("\n🎉 Advanced Training finished!")
    wandb.finish()

if __name__ == "__main__":
    main()
