# data/behavioral_augmentation.py
import torch
import numpy as np
from torch.utils.data import Dataset

class AugmentedBehavioralDataset(Dataset):
    """
    Wrapper dataset that applies data augmentation to behavioral features.
    """
    
    def __init__(self, base_dataset, augmentation_factor=2, noise_std=0.05):
        self.base_dataset = base_dataset
        self.augmentation_factor = augmentation_factor
        self.noise_std = noise_std
        
        # Feature indices for different augmentation strategies
        self.dpad_indices = [0, 1, 2, 3]  # dpad_up, down, left, right
        self.button_indices = [4, 5]      # back_button, menu_revisits
        self.timing_indices = [6, 7, 8]   # scroll_speed, hover_duration, time_since_last
        
    def __len__(self):
        return len(self.base_dataset) * (1 + self.augmentation_factor)
    
    def __getitem__(self, idx):
        # Get base sample
        base_idx = idx // (1 + self.augmentation_factor)
        aug_type = idx % (1 + self.augmentation_factor)
        
        base_sample = self.base_dataset[base_idx]
        features = base_sample['features'].clone()
        labels = base_sample['labels'].clone()
        
        if aug_type == 0:
            # Return original sample
            return {'features': features, 'labels': labels}
        
        # Apply augmentation based on type
        if aug_type == 1:
            features = self._add_gaussian_noise(features)
        elif aug_type == 2:
            features = self._swap_similar_features(features)
        
        return {'features': features, 'labels': labels}
    
    def _add_gaussian_noise(self, features):
        """Add realistic noise to behavioral features."""
        noise = torch.randn_like(features) * self.noise_std
        
        # Apply different noise levels to different feature types
        noise[self.dpad_indices] *= 2.0    # More noise for d-pad counts
        noise[self.button_indices] *= 1.5  # Medium noise for buttons
        noise[self.timing_indices] *= 0.8  # Less noise for timing features
        
        # Add noise and ensure non-negative values
        augmented = features + noise
        augmented = torch.clamp(augmented, min=0)
        
        return augmented
    
    def _swap_similar_features(self, features):
        """Swap similar behavioral features occasionally."""
        augmented = features.clone()
        
        # Swap left/right d-pad counts (20% chance)
        if torch.rand(1) < 0.2:
            augmented[2], augmented[3] = augmented[3], augmented[2]
        
        # Swap up/down d-pad counts (20% chance)
        if torch.rand(1) < 0.2:
            augmented[0], augmented[1] = augmented[1], augmented[0]
        
        return augmented
