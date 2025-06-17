# data/simple_data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleFireTVDataset(Dataset):
    """Simple dataset for testing"""
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.sequence_length = 20
        self.feature_dim = 49
        self.num_traits = 20
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Simple data generation
        sequence = torch.randn(self.sequence_length, self.feature_dim)
        timestamps = torch.cumsum(torch.rand(self.sequence_length), dim=0)
        targets = torch.rand(self.num_traits) * 0.6 + 0.2
        
        return {
            'interaction_data': {
                'sequence': sequence,
                'timestamps': timestamps
            },
            'trait_targets': targets
        }

def simple_collate_fn(batch):
    """Simple collate function that returns proper dictionary structure"""
    
    sequences = torch.stack([item['interaction_data']['sequence'] for item in batch])
    timestamps = torch.stack([item['interaction_data']['timestamps'] for item in batch])
    targets = torch.stack([item['trait_targets'] for item in batch])
    
    return {
        'interaction_data': {
            'sequence': sequences,
            'timestamps': timestamps
        },
        'trait_targets': targets
    }

def create_simple_data_loaders(batch_size=8, num_samples=800):
    """Create simple data loaders for testing"""
    
    dataset = SimpleFireTVDataset(num_samples)
    
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=simple_collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=simple_collate_fn,
        drop_last=True
    )
    
    return train_loader, val_loader
