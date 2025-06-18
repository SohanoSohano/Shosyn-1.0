# scripts/train_hybrid_model.py
import sys
import os
import torch
import torch.cuda
import gc
import wandb
from torch.utils.data import DataLoader

# Add project root to Python path for clean imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.model_config import HybridModelConfig
from config.training_config import TrainingConfig
from data.streaming_data_loader import create_streaming_data_loaders
from models.hybrid_model import HybridFireTVSystem
from training.trainer import HybridModelTrainer

def clear_gpu_memory():
    """Helper function to release and test GPU memory."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache(); torch.cuda.synchronize(); gc.collect()
            print("✅ GPU memory cleared.")
            return True
        except RuntimeError as e: print(f"❌ GPU memory clear failed: {e}"); return False
    return False

def get_safe_device():
    """Selects GPU if available and safe, otherwise defaults to CPU."""
    if torch.cuda.is_available():
        print("🔍 CUDA is available. Checking GPU status...")
        if clear_gpu_memory():
            try:
                device = torch.device('cuda')
                print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
                return device
            except Exception as e: print(f"❌ GPU initialization failed: {e}")
    print("💻 GPU not available or failed. Using CPU for training."); return torch.device('cpu')

def debug_data_loader(data_loader):
    """Prints the shape and type of the first few batches from a data loader."""
    print("🔍 Debugging data loader output...")
    try:
        for i, (features, labels) in enumerate(data_loader):
            print(f"  Batch {i}:"); print(f"    Features - Type: {type(features)}, Shape: {features.shape}, Dtype: {features.dtype}"); print(f"    Labels   - Type: {type(labels)}, Shape: {labels.shape}, Dtype: {labels.dtype}")
            if i >= 1: break
        print("✅ Data loader debug check passed.")
    except Exception as e:
        print(f"❌ Error during data loader debug: {e}"); import traceback; traceback.print_exc()

def main():
    """Main training script updated to handle large datasets via streaming."""
    print("🧪 Fire TV Psychological Recommendation Engine Training 🧪")
    print("=" * 60)
    train_loader = val_loader = model = trainer = None
    try:
        device = get_safe_device()
        wandb.init(project="firetv-psychological-recommendation-engine")
        model_config = HybridModelConfig()
        training_config = TrainingConfig()
        
        DATASET_PATH = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\fire_tv_production_dataset_parallel.csv"
        
        all_cols = ['user_id', 'session_id', 'interaction_timestamp', 'interaction_type', 'dpad_up_count', 'dpad_down_count', 'dpad_left_count', 'dpad_right_count', 'back_button_presses', 'menu_revisits', 'scroll_speed', 'hover_duration', 'time_since_last_interaction', 'cpu_usage_percent', 'wifi_signal_strength', 'network_latency_ms', 'device_temperature', 'battery_level', 'time_of_day', 'day_of_week', 'content_id', 'content_type', 'content_genre', 'release_year', 'search_sophistication_pattern', 'navigation_efficiency_score', 'recommendation_engagement_pattern', 'cognitive_load_indicator', 'decision_confidence_score', 'frustration_level', 'attention_span_indicator', 'exploration_tendency_score', 'platform_loyalty_score', 'social_influence_factor', 'price_sensitivity_score', 'content_diversity_preference', 'session_engagement_level', 'ui_adaptation_speed', 'temporal_consistency_pattern', 'multi_platform_behavior_indicator', 'voice_command_usage_frequency', 'return_likelihood_score']
        label_columns = ['cognitive_load_indicator', 'decision_confidence_score', 'frustration_level', 'attention_span_indicator', 'exploration_tendency_score', 'platform_loyalty_score', 'social_influence_factor', 'price_sensitivity_score', 'content_diversity_preference', 'session_engagement_level', 'ui_adaptation_speed', 'temporal_consistency_pattern', 'multi_platform_behavior_indicator', 'voice_command_usage_frequency', 'return_likelihood_score']
        categorical_cols = ['search_sophistication_pattern', 'recommendation_engagement_pattern', 'interaction_type', 'content_type']
        cols_to_exclude_from_features = label_columns + categorical_cols + ['user_id', 'session_id', 'interaction_timestamp', 'content_id', 'content_genre']
        feature_columns = [col for col in all_cols if col not in cols_to_exclude_from_features]
        print(f"Identified {len(feature_columns)} feature columns and {len(label_columns)} label columns.")

        print(f"📊 Loading data from {DATASET_PATH} using streaming...")
        train_loader, val_loader = create_streaming_data_loaders(data_path=DATASET_PATH, feature_cols=feature_columns, label_cols=label_columns, batch_size=training_config.batch_size, chunksize=50000)
        print("✅ Streaming data loader created successfully.")
        debug_data_loader(train_loader)

        # --- CORRECTED MODEL INITIALIZATION ---
        print("🏗️ Creating hybrid model...")
        
        # Step 1: Update the config object with the dynamic dimensions from your data.
        model_config.input_dim = len(feature_columns)
        model_config.output_dim = len(label_columns)
        
        # Step 2: Pass ONLY the single, updated config object to the model.
        # The argument name 'config' must match the __init__ method in HybridFireTVSystem.
        model = HybridFireTVSystem(config=model_config).to(device)
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        # --- END OF CORRECTION ---

        print("🎯 Initializing trainer...")
        trainer = HybridModelTrainer(model, training_config, device)
        print("✅ Trainer initialized.")

        print("🚀🚀🚀 Starting training... 🚀🚀🚀")
        history = trainer.train(train_loader, val_loader, training_config.num_epochs)
        print("✅ Training completed successfully!")

        model_path = f"models/final_hybrid_model_{device.type}.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}")
        return history

    except Exception as e:
        print(f"❌ An unexpected error occurred in the main training loop: {e}"); import traceback; traceback.print_exc(); return None
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        wandb.finish()
        print("🧹 Cleaned up resources and finished wandb session.")

if __name__ == "__main__":
    main()
