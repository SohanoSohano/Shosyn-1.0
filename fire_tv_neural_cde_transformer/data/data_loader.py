# data/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class FireTVDataset(Dataset):
    """Dataset class for Fire TV interaction data"""
    
    def __init__(self, data_path):
        # Create dummy data since you don't have real data yet
        self.data = self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy Fire TV interaction data for testing"""
        num_samples = 1000
        sequence_length = 20
        feature_dim = 49
        
        data = []
        for i in range(num_samples):
            # Create dummy interaction sequence
            sequence = torch.randn(sequence_length, feature_dim)
            timestamps = torch.cumsum(torch.rand(sequence_length), dim=0)
            
            data.append({
                'interaction_data': {
                    'sequence': sequence,
                    'timestamps': timestamps
                },
                'user_id': f"user_{i}",
                'session_id': f"session_{i}",
                # Add dummy targets for training
                'trait_targets': torch.rand(20) * 0.6 + 0.2  # 20 psychological traits
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """FIXED: Proper collate function that returns dictionary structure"""
    
    # Extract components from batch
    interaction_sequences = []
    interaction_timestamps = []
    trait_targets = []
    user_ids = []
    session_ids = []
    
    for item in batch:
        interaction_sequences.append(item['interaction_data']['sequence'])
        interaction_timestamps.append(item['interaction_data']['timestamps'])
        trait_targets.append(item['trait_targets'])
        user_ids.append(item['user_id'])
        session_ids.append(item['session_id'])
    
    # Stack tensors
    try:
        # Pad sequences to same length
        max_seq_len = max(seq.shape[0] for seq in interaction_sequences)
        
        padded_sequences = []
        padded_timestamps = []
        
        for seq, ts in zip(interaction_sequences, interaction_timestamps):
            if seq.shape[0] < max_seq_len:
                # Pad with last values
                pad_len = max_seq_len - seq.shape[0]
                padded_seq = torch.cat([seq, seq[-1:].repeat(pad_len, 1)], dim=0)
                padded_ts = torch.cat([ts, ts[-1:].repeat(pad_len)], dim=0)
            else:
                padded_seq = seq
                padded_ts = ts
            
            padded_sequences.append(padded_seq)
            padded_timestamps.append(padded_ts)
        
        # Stack into batches
        batch_sequences = torch.stack(padded_sequences)
        batch_timestamps = torch.stack(padded_timestamps)
        batch_targets = torch.stack(trait_targets)
        
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # Fallback to first item only
        batch_sequences = interaction_sequences[0].unsqueeze(0)
        batch_timestamps = interaction_timestamps[0].unsqueeze(0)
        batch_targets = trait_targets[0].unsqueeze(0)
    
    # Return properly structured batch dictionary
    return {
        'interaction_data': {
            'sequence': batch_sequences,
            'timestamps': batch_timestamps
        },
        'trait_targets': batch_targets,
        'user_ids': user_ids,
        'session_ids': session_ids
    }

def create_data_loaders(data_path, batch_size=16, validation_split=0.2):
    """Create train and validation data loaders"""
    
    # Create dataset
    dataset = FireTVDataset(data_path)
    
    # Split into train and validation
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders with fixed collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    return train_loader, val_loader
