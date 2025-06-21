# config/training_config.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration for hybrid model"""
    batch_size: int = 32
    learning_rate: float = 2e-4  # Changed from 1e-3 to safer value
    num_epochs: int = 50
    patience: int = 5
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
    
    # NEW: TMDb Caching Configuration
    tmdb_cache_dir: str = "/home/ubuntu/fire_tv_data/tmdb_cache"
    use_tmdb_cache: bool = True
    precompute_features: bool = False  # Set to True when running precomputation

    # NEW: Dummy TMDb Integration
    use_dummy_tmdb: bool = True  # Enable dummy integration
    use_tmdb_cache: bool = False  # Disable when using dummy
    
    # NEW: Enhanced Training Features
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 2
    
    # NEW: Genre Balancing
    enable_genre_balancing: bool = True
    genre_loss_weight: float = 0.3  # Reduced from default to prevent dominance
    
    # NEW: Loss component weights (refined)
    loss_weights: dict = None
    
    def __post_init__(self):
        """Initialize complex attributes after dataclass creation"""
        if self.loss_weights is None:
            self.loss_weights = {
                'traits': 1.0,      # Primary objective (psychological traits)
                'affinity': 0.2,    # Content affinity
                'rating': 0.1,      # Rating prediction (reduced impact)
                'genre': self.genre_loss_weight  # Genre prediction (balanced weight)
            }
