
# data/streaming_data_loader.py - Optimized for GPU utilization
import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
import warnings

class StreamingCsvDataset(IterableDataset):
    def __init__(self, file_path, feature_cols, label_cols, chunksize=100000):  # Increased chunksize
        super(StreamingCsvDataset).__init__()
        self.file_path = file_path
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.chunksize = chunksize

    def __iter__(self):
        file_iterator = pd.read_csv(self.file_path, chunksize=self.chunksize, low_memory=False)
        nan_warning_count = 0
        processed_count = 0
        skipped_count = 0

        for chunk in file_iterator:
            # Fill NaN values in non-critical columns with defaults
            chunk = chunk.fillna({
                'content_id': 'unknown',
                'content_type': 'unknown', 
                'content_genre': '[]',
                'release_year': 2020
            })
            
            # Vectorized processing instead of row-by-row iteration
            try:
                # Extract features and labels for entire chunk at once
                feature_data = chunk[self.feature_cols].values.astype(np.float32)
                label_data = chunk[self.label_cols].values.astype(np.float32)
                
                # Vectorized validity check
                valid_mask = ~(np.isnan(feature_data).any(axis=1) | 
                              np.isinf(feature_data).any(axis=1) |
                              np.isnan(label_data).any(axis=1) | 
                              np.isinf(label_data).any(axis=1))
                
                # Filter valid rows
                valid_features = feature_data[valid_mask]
                valid_labels = label_data[valid_mask]
                
                skipped_count += (~valid_mask).sum()
                processed_count += valid_mask.sum()
                
                if processed_count % 100000 == 0:  # Reduced logging frequency
                    print(f"Processed {processed_count} valid rows, skipped {skipped_count}")
                
                # Convert to tensors in batches for better memory efficiency
                batch_size = 1000  # Process in mini-batches
                for i in range(0, len(valid_features), batch_size):
                    end_idx = min(i + batch_size, len(valid_features))
                    
                    # Create tensors for batch
                    feature_batch = torch.from_numpy(valid_features[i:end_idx]).float()
                    label_batch = torch.from_numpy(valid_labels[i:end_idx]).float()
                    
                    # Yield individual samples from batch
                    for j in range(feature_batch.shape[0]):
                        yield feature_batch[j], label_batch[j]
                        
            except Exception as e:
                warnings.warn(f"Error processing chunk: {e}")
                continue

def create_streaming_data_loaders(data_path, feature_cols, label_cols, batch_size, chunksize=100000):
    # Create dataset with larger chunksize
    dataset = StreamingCsvDataset(data_path, feature_cols, label_cols, chunksize)
    
    # Optimized DataLoader for maximum GPU utilization
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,              # Increased workers for better CPU-GPU pipeline
        pin_memory=True,            # Keep for faster GPU transfer
        persistent_workers=True,    # Keep workers alive between epochs
        prefetch_factor=4,          # Prefetch more batches to keep GPU fed
        drop_last=True,             # Ensure consistent batch sizes
        multiprocessing_context='spawn'  # Better for large datasets
    )
    
    return dataloader, dataloader

