# scripts/train_hybrid_model_tmdb.py
import sys
import os
import torch
import wandb
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.model_config import HybridModelConfig
from config.training_config import TrainingConfig
from data.streaming_data_loader import create_streaming_data_loaders
from data.tmdb_integration import TMDbIntegration
from models.hybrid_model import HybridFireTVSystem
from training.enhanced_trainer import EnhancedHybridModelTrainer

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
    
    # Data configuration (same as before)
    DATASET_PATH = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\fire_tv_production_dataset_parallel.csv"
    
    # Column definitions (same as before)
    all_cols = ['user_id', 'session_id', 'interaction_timestamp', 'interaction_type', 'dpad_up_count', 'dpad_down_count', 'dpad_left_count', 'dpad_right_count', 'back_button_presses', 'menu_revisits', 'scroll_speed', 'hover_duration', 'time_since_last_interaction', 'cpu_usage_percent', 'wifi_signal_strength', 'network_latency_ms', 'device_temperature', 'battery_level', 'time_of_day', 'day_of_week', 'content_id', 'content_type', 'content_genre', 'release_year', 'search_sophistication_pattern', 'navigation_efficiency_score', 'recommendation_engagement_pattern', 'cognitive_load_indicator', 'decision_confidence_score', 'frustration_level', 'attention_span_indicator', 'exploration_tendency_score', 'platform_loyalty_score', 'social_influence_factor', 'price_sensitivity_score', 'content_diversity_preference', 'session_engagement_level', 'ui_adaptation_speed', 'temporal_consistency_pattern', 'multi_platform_behavior_indicator', 'voice_command_usage_frequency', 'return_likelihood_score']
    
    label_columns = ['cognitive_load_indicator', 'decision_confidence_score', 'frustration_level', 'attention_span_indicator', 'exploration_tendency_score', 'platform_loyalty_score', 'social_influence_factor', 'price_sensitivity_score', 'content_diversity_preference', 'session_engagement_level', 'ui_adaptation_speed', 'temporal_consistency_pattern', 'multi_platform_behavior_indicator', 'voice_command_usage_frequency', 'return_likelihood_score']
    
    categorical_cols = ['search_sophistication_pattern', 'recommendation_engagement_pattern', 'interaction_type', 'content_type']
    
    cols_to_exclude_from_features = label_columns + categorical_cols + ['user_id', 'session_id', 'interaction_timestamp', 'content_id', 'content_genre']
    
    feature_columns = [col for col in all_cols if col not in cols_to_exclude_from_features]
    
    print(f"Identified {len(feature_columns)} feature columns and {len(label_columns)} label columns.")

    # Create data loaders
    train_loader, val_loader = create_streaming_data_loaders(
        data_path=DATASET_PATH, 
        feature_cols=feature_columns, 
        label_cols=label_columns, 
        batch_size=training_config.batch_size, 
        chunksize=50000
    )

    # Initialize TMDb integration 
    TMDB_API_KEY = "c799fe85bcebb074eff49aa01dc6cdb0"  # Get from https://www.themoviedb.org/settings/api
    
    if TMDB_API_KEY != "c799fe85bcebb074eff49aa01dc6cdb0":
        print("⚠️  WARNING: Please set your actual TMDb API key!")
        print("   Get one free at: https://www.themoviedb.org/settings/api")
        return
    
    tmdb_integration = TMDbIntegration(TMDB_API_KEY)
    
    # Load or create content mapping
    try:
        content_mapping = pd.read_csv(r'C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\production_content_mapping_20250619_010621.csv')
        content_mapping = dict(zip(content_mapping['content_id'], content_mapping['tmdb_id']))
    except:
        print("Creating sample TMDb content mapping...")
        content_mapping = create_sample_tmdb_mapping()
    
    # Load cache
    tmdb_integration.load_cache('tmdb_cache.json')

    # Create enhanced model
    model_config.input_dim = len(feature_columns)
    model_config.output_dim = len(label_columns)
    model = HybridFireTVSystem(config=model_config).to(device)
    
    print(f"✅ TMDb-enhanced model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # Enhanced trainer with TMDb
    trainer = EnhancedHybridModelTrainer(model, training_config, device, tmdb_integration, content_mapping)
    
    print("🚀🚀🚀 Starting TMDb-enhanced training... 🚀🚀🚀")
    history = trainer.train(train_loader, val_loader, training_config.num_epochs)
    
    # Save model and cache
    model_path = f"models/tmdb_enhanced_hybrid_model_{device.type}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    tmdb_integration.save_cache('tmdb_cache.json')
    
    print(f"💾 TMDb-enhanced model saved to {model_path}")
    print("🎉 Training completed with TMDb data integration!")
    
    wandb.finish()
    return history

if __name__ == "__main__":
    main()
