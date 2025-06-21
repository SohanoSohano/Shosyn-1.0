# scripts/train_hybrid_model_tmdb.py
import sys
import os
import torch
import wandb
import pandas as pd

from torch.utils.data import DataLoader, random_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from data.fire_tv_dataset import FireTVDataset
from config.model_config import HybridModelConfig
from config.training_config import TrainingConfig
from data.streaming_data_loader import create_streaming_data_loaders
from data.tmdb_integration import TMDbIntegration
from models.hybrid_model import HybridFireTVSystem
from training.enhanced_trainer_new import EnhancedHybridModelTrainer

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def resume_from_checkpoint(model, optimizer, device):
    """Resume training from the best checkpoint"""
    checkpoint_path = "best_tmdb_enhanced_model.pth"
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint with optimizer state
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
        else:
            # Simple model state dict
            model.load_state_dict(checkpoint)
        
        print(f"✅ Resumed from epoch {start_epoch} with best loss {best_loss:.6f}")
        return start_epoch, best_loss
    else:
        print("No checkpoint found, starting from scratch")
        return start_epoch, best_loss

def create_sample_tmdb_mapping():
    """Create sample TMDb mapping - replace with your actual mapping"""
    sample_data = {
        'content_id': [f'content_{i}' for i in range(100)],
        'tmdb_id': [550 + i for i in range(100)]  # Sample TMDb IDs starting from Fight Club (550)
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('content_to_tmdb_mapping.csv', index=False)
    return dict(zip(df['content_id'], df['tmdb_id']))

def main():
    print("🧪 Fire TV + TMDb Hybrid Model Training 🧪")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project="firetv-tmdb-hybrid-superior")
    
    model_config = HybridModelConfig()
    training_config = TrainingConfig()

    # ENABLE DUMMY TMDB FOR MAXIMUM SPEED
    training_config.use_dummy_tmdb = True
    training_config.use_tmdb_cache = False
    
    print("🔧 Training Configuration:")
    print(f"   Dummy TMDb: {training_config.use_dummy_tmdb}")
    print(f"   TMDb Cache: {training_config.use_tmdb_cache}")
    print(f"   Batch Size: {training_config.batch_size}")

    # Override specific settings if needed for this run
    training_config.batch_size = 32  # Keep your optimal batch size
    training_config.use_tmdb_cache = False  # Enable caching
    training_config.num_epochs = 50
    
    # Print configuration for verification
    print("🔧 Training Configuration:")
    print(f"   Batch Size: {training_config.batch_size}")
    print(f"   Learning Rate: {training_config.learning_rate}")
    print(f"   TMDb Cache: {training_config.use_tmdb_cache}")
    print(f"   Cache Dir: {training_config.tmdb_cache_dir}")
    print(f"   Mixed Precision: {training_config.use_mixed_precision}")
    print(f"   Genre Balancing: {training_config.enable_genre_balancing}")
    print(f"   Loss Weights: {training_config.loss_weights}")

    
    # Data configuration (same as before)
    DATASET_PATH = "/home/ubuntu/fire_tv_data/fire_tv_sampled_2gb.csv"
    
    # Column definitions (same as before)
    all_cols = ['user_id', 'session_id', 'interaction_timestamp', 'interaction_type', 'dpad_up_count', 'dpad_down_count', 'dpad_left_count', 'dpad_right_count', 'back_button_presses', 'menu_revisits', 'scroll_speed', 'hover_duration', 'time_since_last_interaction', 'cpu_usage_percent', 'wifi_signal_strength', 'network_latency_ms', 'device_temperature', 'battery_level', 'time_of_day', 'day_of_week', 'content_id', 'content_type', 'content_genre', 'release_year', 'search_sophistication_pattern', 'navigation_efficiency_score', 'recommendation_engagement_pattern', 'cognitive_load_indicator', 'decision_confidence_score', 'frustration_level', 'attention_span_indicator', 'exploration_tendency_score', 'platform_loyalty_score', 'social_influence_factor', 'price_sensitivity_score', 'content_diversity_preference', 'session_engagement_level', 'ui_adaptation_speed', 'temporal_consistency_pattern', 'multi_platform_behavior_indicator', 'voice_command_usage_frequency', 'return_likelihood_score']
    
    label_columns = ['cognitive_load_indicator', 'decision_confidence_score', 'frustration_level', 'attention_span_indicator', 'exploration_tendency_score', 'platform_loyalty_score', 'social_influence_factor', 'price_sensitivity_score', 'content_diversity_preference', 'session_engagement_level', 'ui_adaptation_speed', 'temporal_consistency_pattern', 'multi_platform_behavior_indicator', 'voice_command_usage_frequency', 'return_likelihood_score']
    
    categorical_cols = ['search_sophistication_pattern', 'recommendation_engagement_pattern', 'interaction_type', 'content_type']
    
    cols_to_exclude_from_features = label_columns + categorical_cols + ['user_id', 'session_id', 'interaction_timestamp', 'content_id', 'content_genre']
    
    feature_columns = [col for col in all_cols if col not in cols_to_exclude_from_features]
    
    print(f"Identified {len(feature_columns)} feature columns and {len(label_columns)} label columns.")

    # --- Create the Dataset and DataLoader ---
    # Instantiate your new FireTVDataset class.
    # This loads the entire 2GB dataset (or a DataFrame reference) into memory ONCE.
    full_dataset = FireTVDataset(DATASET_PATH, feature_columns, label_columns)
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size # CORRECT validation size calculation
    
    # Use torch.utils.data.random_split for splitting
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create standard PyTorch DataLoaders
    # These will use the FireTVDataset and handle batching, shuffling, and multi-processing.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config.batch_size, # e.g., 32
        num_workers=4, # Keep this, good for CPU-GPU data transfer
        pin_memory=True, # Improves data transfer to GPU
        shuffle=True # Shuffle training data
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=training_config.batch_size,
        num_workers=0, # Use same num_workers for consistency
        pin_memory=False,
        shuffle=False # Do NOT shuffle validation data
    )
    
    print(f"🔥 DataLoader created successfully!")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    print(f"   Training batches per epoch: {len(train_loader):,}")
    print(f"   Validation batches per epoch: {len(val_loader):,}")

    # Initialize TMDb integration 
    TMDB_API_KEY = "c799fe85bcebb074eff49aa01dc6cdb0"  # Get from https://www.themoviedb.org/settings/api
    
    if TMDB_API_KEY != "c799fe85bcebb074eff49aa01dc6cdb0":
        print("⚠️  WARNING: Please set your actual TMDb API key!")
        print("   Get one free at: https://www.themoviedb.org/settings/api")
        return
    
    tmdb_integration = TMDbIntegration(TMDB_API_KEY)
    
    # Load or create content mapping
    try:
        content_mapping = pd.read_csv('/home/ubuntu/fire_tv_data/production_content_mapping_20250619_010621.csv')
        content_mapping = dict(zip(content_mapping['content_id'], content_mapping['tmdb_id']))
    except:
        print("Creating sample TMDb content mapping...")
        content_mapping = create_sample_tmdb_mapping()
    
    # Load cache
    tmdb_integration.load_cache('/home/ubuntu/fire_tv_data/tmdb_cache.json')

    # Create enhanced model
    model_config.input_dim = len(feature_columns)
    model_config.output_dim = len(label_columns)
    model = HybridFireTVSystem(config=model_config).to(device)

    # Add this debugging code in your training script after model creation:
    print("Checking model output dimensions...")
    dummy_input = {
        'sequence': torch.randn(16, 18).to(device),  # batch_size=16, features=18
        'timestamps': torch.linspace(0, 1, 18).unsqueeze(0).expand(16, -1).to(device)
    }
    dummy_tmdb = torch.randn(16, 70).to(device)
    dummy_embeddings = torch.randn(16, 384).to(device)

    with torch.no_grad():
        test_outputs = model(dummy_input, tmdb_features=dummy_tmdb, content_embeddings=dummy_embeddings)
        for key, value in test_outputs.items():
            print(f"{key}: {value.shape}")



    #if hasattr(torch, 'compile'):
        #model = torch.compile(model, mode='max-autotune')
        #print("✅ Model compiled for maximum performance")

    # Resume from checkpoint
    start_epoch, best_loss = resume_from_checkpoint(model, None, device)

    
    print(f"✅ TMDb-enhanced model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # Enhanced trainer with TMDb
    trainer = EnhancedHybridModelTrainer(model, training_config, device, tmdb_integration, content_mapping, use_cache=training_config.use_tmdb_cache)
    
    trainer.best_loss = best_loss  # Set the best loss from checkpoint

    print("🚀🚀🚀 Starting TMDb-enhanced training... 🚀🚀🚀")
    history = trainer.train(train_loader, val_loader, training_config.num_epochs)
    
    # Save model and cache
    model_path = f"models/tmdb_enhanced_hybrid_model_{device.type}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    tmdb_integration.save_cache('/home/ubuntu/fire_tv_data/tmdb_cache.json')
    
    print(f"💾 TMDb-enhanced model saved to {model_path}")
    print("🎉 Training completed with TMDb data integration!")
    
    wandb.finish()
    return history

if __name__ == "__main__":
    main()
