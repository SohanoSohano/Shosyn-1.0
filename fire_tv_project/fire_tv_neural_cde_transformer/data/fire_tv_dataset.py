# data/fire_tv_dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset

class FireTVDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the Fire TV data efficiently.
    This class handles reading the CSV and providing samples to the DataLoader.
    """
    def __init__(self, csv_path, feature_columns, label_columns):
        """
        Initializes the dataset by loading the data from the specified CSV file.
        
        Args:
            csv_path (str): The path to the 2GB sampled CSV file.
            feature_columns (list): A list of column names to be used as features.
            label_columns (list): A list of column names to be used as labels.
        """
        print(f"🔄 Loading dataset from {csv_path}...")
        # Using low_memory=False can help prevent dtype errors with large files
        self.data = pd.read_csv(csv_path, low_memory=False)
        
        # Store column names
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        
        # Convert data to numpy arrays for faster access
        self.features = self.data[self.feature_columns].values
        self.labels = self.data[self.label_columns].values
        
        print(f"✅ Dataset loaded with {len(self.data):,} rows.")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        This is the critical method that defines the epoch size for the DataLoader.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (features and labels) from the dataset at the given index.
        
        Args:
            idx (int): The index of the sample to retrieve.
            
        Returns:
            tuple: A tuple containing the feature tensor and the label tensor.
        """
        # Get the specific row of data
        feature_row = self.features[idx]
        label_row = self.labels[idx]
        
        # Convert to PyTorch tensors
        features_tensor = torch.tensor(feature_row, dtype=torch.float32)
        labels_tensor = torch.tensor(label_row, dtype=torch.float32)
        
        return features_tensor, labels_tensor
