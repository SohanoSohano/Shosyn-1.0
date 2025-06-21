# data/synthetic_data_loader.py (Final Corrected Version)
import pandas as pd
import torch
from torch.utils.data import Dataset

class SyntheticFireTVDataset(Dataset):
    """
    A custom PyTorch Dataset to load and process the TMDb-backed synthetic data.
    --- CORRECTED to use the exact 21 columns available in the generated dataset ---
    """
    def __init__(self, csv_path: str):
        print(f"Loading synthetic dataset from: {csv_path}")
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CRITICAL ERROR: The dataset file was not found at '{csv_path}'. Please check the path.")

        # --- MODIFICATION: Definitive feature and label lists ---
        # These lists are based on the exact columns confirmed to be in your dataset.
        
        # These are the 9 user interaction metrics that will be the INPUT to your model.
        self.behavioral_features = [
            'dpad_up_count', 'dpad_down_count', 'dpad_left_count', 'dpad_right_count',
            'back_button_presses', 'menu_revisits', 'scroll_speed', 'hover_duration',
            'time_since_last_interaction'
        ]
        
        # These are the 3 core psychological traits that are the TARGET LABELS for your model to predict.
        self.psychological_labels = [
            'session_engagement_level', 'frustration_level', 'exploration_tendency_score'
        ]
        
        # --- Robustness Check ---
        # This check verifies that all the columns we need actually exist in the DataFrame.
        required_columns = self.behavioral_features + self.psychological_labels
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"CRITICAL ERROR: Column '{col}' is missing from the dataset '{csv_path}'. "
                                 f"Please regenerate the dataset or correct the column name list.")

        # Pre-convert to PyTorch tensors for training efficiency
        self.X = torch.tensor(self.df[self.behavioral_features].values, dtype=torch.float32)
        self.y = torch.tensor(self.df[self.psychological_labels].values, dtype=torch.float32)
        
        print("✅ Dataset processed and ready for training.")
        print(f"   Input features shape (X): {self.X.shape}  (batch_size, num_behavioral_features)")
        print(f"   Target labels shape (y):  {self.y.shape}  (batch_size, num_psychological_traits)")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Returns a single sample (features and labels) from the dataset."""
        return {
            'features': self.X[idx],
            'labels': self.y[idx]
        }
