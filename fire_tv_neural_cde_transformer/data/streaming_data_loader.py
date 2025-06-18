# data/streaming_data_loader.py
import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
import warnings

class StreamingCsvDataset(IterableDataset):
    def __init__(self, file_path, feature_cols, label_cols, chunksize=1000):
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
            
            for _, row in chunk.iterrows():
                try:
                    # Extract only the feature and label columns (exclude problematic ones)
                    feature_values = row[self.feature_cols].values.astype(float)
                    label_values = row[self.label_cols].values.astype(float)
                    
                    features = torch.tensor(feature_values, dtype=torch.float32)
                    labels = torch.tensor(label_values, dtype=torch.float32)
                    
                    # Check for NaN or Infinity in the actual training data
                    if torch.isnan(features).any() or torch.isinf(features).any() or \
                       torch.isnan(labels).any() or torch.isinf(labels).any():
                        skipped_count += 1
                        if nan_warning_count < 3:  # Reduce warning spam
                            warnings.warn(f"Skipping row {processed_count} with invalid values")
                            nan_warning_count += 1
                        continue

                    processed_count += 1
                    if processed_count % 10000 == 0:
                        print(f"Processed {processed_count} valid rows, skipped {skipped_count}")
                    
                    yield features, labels
                    
                except (ValueError, TypeError) as e:
                    skipped_count += 1
                    continue

def create_streaming_data_loaders(data_path, feature_cols, label_cols, batch_size, chunksize=10000):
    dataset = StreamingCsvDataset(data_path, feature_cols, label_cols, chunksize)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    return dataloader, dataloader
