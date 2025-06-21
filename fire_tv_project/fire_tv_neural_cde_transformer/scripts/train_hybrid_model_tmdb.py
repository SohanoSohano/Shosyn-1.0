# scripts/train_hybrid_model_tmdb.py (Modified for Synthetic Data Training)
import sys
import os
import torch
import wandb
import pandas as pd
from torch.utils.data import DataLoader, random_split

# --- Project Path Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- MODIFICATION: Import the new Synthetic Data Loader ---
from data.synthetic_data_loader import SyntheticFireTVDataset # <-- NEW
from config.model_config import HybridModelConfig
from config.training_config import TrainingConfig
from models.hybrid_model import HybridFireTVSystem # Assuming this is your model class
from training.enhanced_trainer_new import EnhancedHybridModelTrainer

# --- Environment Setup ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def main():
    print("🧪 Training Hybrid Model on HIGH-FIDELITY SYNTHETIC DATA 🧪")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    wandb.init(project="firetv-synthetic-data-training")
    
    # --- Load Configurations ---
    model_config = HybridModelConfig()
    training_config = TrainingConfig()

    # --- MODIFICATION: Set path to your new synthetic dataset ---
    DATASET_PATH = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\fire_tv_neural_cde_transformer_instance_version\Shosyn-1.0\fire_tv_project\fire_tv_neural_cde_transformer\fire_tv_synthetic_dataset_v3_tmdb.csv"
    print(f"Loading synthetic dataset from: {DATASET_PATH}")
    
    # --- MODIFICATION: Use the new SyntheticFireTVDataset ---
    # This loader is designed specifically for the synthetic data's structure.
    full_dataset = SyntheticFireTVDataset(DATASET_PATH)
    
    # --- Split dataset into training and validation sets ---
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=training_config.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )
    
    print(f"🔥 DataLoaders created successfully!")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")

    # --- MODIFICATION: Adjust Model Config for new data structure ---
    # The new loader provides features and labels directly.
    # Get a sample to determine dimensions.
    sample = full_dataset[0]
    feature_dim = sample['features'].shape[0]
    label_dim = sample['labels'].shape[0]
    
    model_config.input_dim = feature_dim
    model_config.output_dim = label_dim
    
    print(f"   Input feature dimension: {model_config.input_dim}")
    print(f"   Output label dimension:  {model_config.output_dim}")

    # --- Initialize Model ---
    # IMPORTANT: Remember to add nn.Dropout layers inside your HybridFireTVSystem class
    # in 'models/hybrid_model.py' to add regularization.
    model = HybridFireTVSystem(config=model_config).to(device)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- MODIFICATION: Configure Optimizer with Weight Decay ---
    # This is a critical step to prevent overfitting.
    print("🔧 Configuring optimizer with AdamW and Weight Decay...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=0.01  # <-- NEW: Adds L2 regularization
    )

    # --- MODIFICATION: Adjust Loss Weights ---
    # We are prioritizing the psychological trait prediction.
    print("⚖️ Adjusting loss weights to prioritize trait prediction...")
    training_config.loss_weights = {
        "traits": 2.5,   # <-- INCREASED
        "genre": 0.1,    # <-- DECREASED
        "affinity": 0.2, # Keep or adjust
        "rating": 0.1    # Keep or adjust
    }
    print(f"   New Loss Weights: {training_config.loss_weights}")

    # --- Initialize Trainer ---
    # Note: The trainer no longer needs tmdb_integration or content_mapping for this synthetic run,
    # as all data is self-contained. You may need to adjust your trainer's __init__ method
    # or pass None for those arguments if they are optional.
    # We assume here the trainer can be simplified for this run.
    
    # You might need a simplified trainer or adjust the existing one.
    # For now, we assume EnhancedHybridModelTrainer can handle this.
    trainer = EnhancedHybridModelTrainer(
        model=model,
        optimizer=optimizer,
        config=training_config, # Pass the full config object
        device=device
        # Remove tmdb_integration, content_mapping if they are not needed for loss calculation
    )

    print("\n🚀🚀🚀 Starting training on SYNTHETIC data... 🚀🚀🚀")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config.num_epochs
    )
    
    # --- Save the newly trained model ---
    model_path = "models/synthetic_trained_hybrid_model.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    print(f"\n💾 Model trained on synthetic data saved to {model_path}")
    print("🎉 Training completed successfully!")
    
    wandb.finish()
    return history

if __name__ == "__main__":
    main()
