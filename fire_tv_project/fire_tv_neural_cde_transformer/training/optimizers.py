# training/optimizers.py
import torch.optim as optim

def create_hybrid_optimizer(model, config):
    """Create optimizer and schedulers for hybrid model"""
    
    # Simple optimizer for now
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Simple scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr
    )
    
    schedulers = {'main': scheduler}
    
    return optimizer, schedulers
