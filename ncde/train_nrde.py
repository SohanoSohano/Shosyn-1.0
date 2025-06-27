import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import signatory
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set a non-interactive backend for Matplotlib
import matplotlib
matplotlib.use('Agg')
sns.set_theme(style="whitegrid")

# --- Configuration & Hyperparameters ---
CONFIG = {
    "data_path": "/home/ubuntu/Shosyn-1.0/dataset/enriched_simulation_logs_500_new.csv",
    "batch_size": 64,  # Reduced for stability
    "learning_rate": 1e-5,  # Much more conservative
    "weight_decay": 1e-6,
    "epochs": 100,
    "logsig_depth": 2,  # Reduced depth for stability
    "mlp_hidden_dims": [128, 64, 32],  # Smaller model for stability
    "num_workers": 0,  # Single-threaded for debugging
    "clip_value": 0.5,  # More aggressive clipping
    "scheduler_patience": 5,
    "scheduler_factor": 0.5
}

def robust_standardize(data, epsilon=1e-8):
    """
    Robust standardization that prevents division by zero and NaN values.
    """
    data = np.array(data, dtype=np.float32)
    
    # Replace any existing NaNs or infs with zeros
    data = np.where(np.isfinite(data), data, 0.0)
    
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Prevent division by zero by setting minimum std
    std = np.maximum(std, epsilon)
    
    standardized = (data - mean) / std
    
    # Final safety check
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    
    return standardized, mean, std

class RobustStandardScaler:
    """
    A robust version of StandardScaler that never produces NaN values.
    """
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.mean_ = None
        self.scale_ = None
        self.fitted = False
    
    def fit(self, X):
        X = np.array(X, dtype=np.float32)
        X = np.where(np.isfinite(X), X, 0.0)
        
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.maximum(self.scale_, self.epsilon)
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        X = np.array(X, dtype=np.float32)
        X = np.where(np.isfinite(X), X, 0.0)
        
        result = (X - self.mean_) / self.scale_
        result = np.where(np.isfinite(result), result, 0.0)
        
        return result

# --- Robust Data Handling ---
class RobustNeuralRDE_Dataset(Dataset):
    def __init__(self, df, numerical_scaler, ohe_encoder, logsig_depth):
        self.df = df
        self.numerical_scaler = numerical_scaler
        self.ohe_encoder = ohe_encoder
        self.logsig_depth = logsig_depth
        self.session_ids = df['session_id'].unique()
        
        print(f"Dataset initialized with {len(self.session_ids)} sessions")

    def __len__(self):
        return len(self.session_ids)

    def __getitem__(self, idx):
        try:
            session_id = self.session_ids[idx]
            session_df = self.df[self.df['session_id'] == session_id].sort_values('timestamp')

            # Robust time delta calculation
            time_deltas = session_df['timestamp'].diff().dt.total_seconds().fillna(0).values
            time_deltas = np.maximum(time_deltas, 1e-6)  # Ensure strictly positive
            time_deltas = np.where(np.isfinite(time_deltas), time_deltas, 1e-6)

            # Robust psychological features
            psych_features = session_df[['frustration_level', 'cognitive_load']].fillna(0).values
            psych_features = np.where(np.isfinite(psych_features), psych_features, 0.0)
            scaled_psych_features = self.numerical_scaler.transform(psych_features)
            
            # Robust scroll features
            scroll_features = session_df[['scroll_speed', 'scroll_depth']].fillna(0).values
            scroll_features = np.where(np.isfinite(scroll_features), scroll_features, 0.0)
            
            # Robust action encoding
            action_types = session_df['action_type'].fillna('unknown').values.reshape(-1, 1)
            action_ohe = self.ohe_encoder.transform(action_types)

            # Combine all features
            features = np.hstack([scaled_psych_features, scroll_features, action_ohe])
            features = np.where(np.isfinite(features), features, 0.0)
            
            # Create path with time as first dimension
            path = np.hstack([time_deltas.reshape(-1, 1), features])
            path = np.where(np.isfinite(path), path, 0.0)
            
            # Ensure minimum path length for log-signature
            if len(path) < 2:
                # Duplicate the last row to ensure minimum length
                path = np.vstack([path, path[-1:]])
            
            path_tensor = torch.tensor(path, dtype=torch.float32)
            
            # Robust log-signature computation
            try:
                logsignature = signatory.logsignature(path_tensor.unsqueeze(0), self.logsig_depth).squeeze(0)
                logsignature = torch.where(torch.isfinite(logsignature), logsignature, torch.zeros_like(logsignature))
            except Exception as e:
                print(f"Log-signature computation failed for session {session_id}: {e}")
                # Fallback: create a zero vector of appropriate size
                path_dim = path.shape[1]
                logsig_dim = signatory.logsignature_channels(path_dim, self.logsig_depth)
                logsignature = torch.zeros(logsig_dim, dtype=torch.float32)

            # Robust target
            final_frustration = session_df['frustration_level'].iloc[-1]
            if not np.isfinite(final_frustration):
                final_frustration = 0.0
            
            target = torch.tensor([final_frustration], dtype=torch.float32)

            return logsignature, target
            
        except Exception as e:
            print(f"Error processing session {idx}: {e}")
            # Return safe fallback values
            path_dim = 1 + 2 + 2 + len(self.ohe_encoder.categories_[0])
            logsig_dim = signatory.logsignature_channels(path_dim, self.logsig_depth)
            return torch.zeros(logsig_dim, dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32)

def robust_collate_fn(batch):
    """Robust collate function that handles any potential issues."""
    try:
        logsigs, targets = zip(*batch)
        logsigs_tensor = torch.stack(logsigs)
        targets_tensor = torch.stack(targets)
        
        # Final safety check
        logsigs_tensor = torch.where(torch.isfinite(logsigs_tensor), logsigs_tensor, torch.zeros_like(logsigs_tensor))
        targets_tensor = torch.where(torch.isfinite(targets_tensor), targets_tensor, torch.zeros_like(targets_tensor))
        
        return logsigs_tensor, targets_tensor
    except Exception as e:
        print(f"Collate function error: {e}")
        # Return safe fallback
        batch_size = len(batch)
        logsig_dim = batch[0][0].shape[0]
        return torch.zeros(batch_size, logsig_dim), torch.zeros(batch_size, 1)

# --- Robust Model Definition ---
class RobustNeuralRDE(nn.Module):
    def __init__(self, input_channels, hidden_dims, output_channels):
        super(RobustNeuralRDE, self).__init__()
        layers = []
        in_dim = input_channels
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())  # Bounded activation
            layers.append(nn.Dropout(0.1))  # Regularization
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_channels))
        self.net = nn.Sequential(*layers)
        
        # Initialize weights to prevent initial instability
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, logsig):
        output = self.net(logsig)
        # Ensure output is finite
        output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
        return output

# --- Robust Training Loop ---
def robust_train_loop(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    valid_batches = 0
    
    for logsigs, targets in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        logsigs, targets = logsigs.to(device), targets.to(device)
        
        # Ensure inputs are finite
        logsigs = torch.where(torch.isfinite(logsigs), logsigs, torch.zeros_like(logsigs))
        targets = torch.where(torch.isfinite(targets), targets, torch.zeros_like(targets))
        
        predictions = model(logsigs)
        loss = loss_fn(predictions, targets)
        
        # Check if loss is finite before proceeding
        if torch.isfinite(loss):
            loss.backward()
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_value"])
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1
        else:
            print("Warning: Non-finite loss detected, but continuing with next batch")
    
    return total_loss / max(valid_batches, 1)

def robust_eval_loop(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for logsigs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            logsigs, targets = logsigs.to(device), targets.to(device)
            
            # Ensure inputs are finite
            logsigs = torch.where(torch.isfinite(logsigs), logsigs, torch.zeros_like(logsigs))
            targets = torch.where(torch.isfinite(targets), targets, torch.zeros_like(targets))
            
            predictions = model(logsigs)
            loss = loss_fn(predictions, targets)
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                valid_batches += 1
    
    return total_loss / max(valid_batches, 1)

# --- Main Function ---
def main():
    print("Starting Robust Neural RDE Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and clean data
    df = pd.read_csv(CONFIG["data_path"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Clean the dataframe
    print("Cleaning data...")
    numeric_columns = ['frustration_level', 'cognitive_load', 'scroll_speed', 'scroll_depth']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = np.where(np.isfinite(df[col]), df[col], 0)
    
    print("Preparing robust scalers and encoders...")
    
    # Use robust scaler
    numerical_scaler = RobustStandardScaler()
    numerical_scaler.fit(df[['frustration_level', 'cognitive_load']].values)
    
    # Robust one-hot encoder
    df['action_type'] = df['action_type'].fillna('unknown')
    all_action_types = df['action_type'].unique().reshape(-1, 1)
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(all_action_types)
    
    # Train/validation split
    all_session_ids = df['session_id'].unique()
    train_ids, val_ids = train_test_split(all_session_ids, test_size=0.2, random_state=42)
    
    train_df = df[df['session_id'].isin(train_ids)]
    val_df = df[df['session_id'].isin(val_ids)]

    print(f"Training on {len(train_ids)} sessions, Validating on {len(val_ids)} sessions.")
    
    # Create datasets
    train_dataset = RobustNeuralRDE_Dataset(train_df, numerical_scaler, ohe_encoder, CONFIG['logsig_depth'])
    val_dataset = RobustNeuralRDE_Dataset(val_df, numerical_scaler, ohe_encoder, CONFIG['logsig_depth'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        collate_fn=robust_collate_fn, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        collate_fn=robust_collate_fn, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=True
    )

    # Calculate input dimensions
    path_dim = 1 + 2 + 2 + len(all_action_types)
    input_channels = signatory.logsignature_channels(path_dim, CONFIG['logsig_depth'])
    output_channels = 1

    print(f"Model input channels: {input_channels}")
    
    # Create model
    model = RobustNeuralRDE(input_channels, CONFIG["mlp_hidden_dims"], output_channels).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        factor=CONFIG["scheduler_factor"], 
        patience=CONFIG["scheduler_patience"], 
        verbose=True
    )

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print("\nStarting robust training...")
    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()
        
        train_loss = robust_train_loop(model, train_loader, optimizer, loss_fn, device)
        val_loss = robust_eval_loop(model, val_loader, loss_fn, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        end_time = time.time()
        epoch_mins = (end_time - start_time) / 60

        print(f"Epoch {epoch+1:02}/{CONFIG['epochs']:02} | "
              f"Time: {epoch_mins:.2f}m | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val. Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_robust_rde.pth')
            print(f"  -> New best validation loss. Model saved.")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Robust Neural RDE Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.yscale('log')  # Log scale for better visualization
    plt.savefig('robust_training_loss_plot.png')
    print("Saved training plot to 'robust_training_loss_plot.png'")

if __name__ == "__main__":
    main()
