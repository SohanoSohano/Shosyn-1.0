# config/training_config.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration for hybrid model"""
    batch_size: int = 16
    learning_rate: float = 1e-4  # Changed from 1e-3 (or default) to a safer value
    num_epochs: int = 100
    patience: int = 20
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Loss weights
    cde_loss_weight: float = 0.5
    transformer_loss_weight: float = 0.3
    recommendation_loss_weight: float = 0.2
    
    # Scheduler settings
    scheduler_factor: float = 0.7
    scheduler_patience: int = 8
    min_lr: float = 1e-6
    
    # Validation
    validation_split: float = 0.2
    save_best_model: bool = True
    model_save_path: str = "models/best_hybrid_model.pth"
