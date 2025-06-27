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
from typing import Tuple, Dict, List
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')
sns.set_theme(style="whitegrid")

# --- Configuration & Hyperparameters ---
CONFIG = {
    "data_path": "/home/ubuntu/Shosyn-1.0/dataset/enriched_simulation_logs_500_new.csv",
    "batch_size": 64,  # Slightly smaller for multi-target stability
    "learning_rate": 1e-5,  # Conservative for multi-target learning
    "weight_decay": 1e-6,
    "epochs": 100,
    "logsig_depth": 2,
    "mlp_hidden_dims": [128, 64, 32],
    "num_workers": 4,
    "clip_value": 1.0,
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "frustration_weight": 0.7,  # Primary target weight
    "cognitive_weight": 0.3,    # Secondary target weight
}

class RobustStandardScaler:
    """A robust version of StandardScaler that never produces NaN values."""
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

# --- 1. Multi-Target Dataset Class ---
class MultiTargetNeuralRDE_Dataset(Dataset):
    """Dataset class that returns both frustration and cognitive load as targets."""
    
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
            time_deltas = np.maximum(time_deltas, 1e-6)
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
                path = np.vstack([path, path[-1:]])
            
            path_tensor = torch.tensor(path, dtype=torch.float32)
            
            # Robust log-signature computation
            try:
                logsignature = signatory.logsignature(path_tensor.unsqueeze(0), self.logsig_depth).squeeze(0)
                logsignature = torch.where(torch.isfinite(logsignature), logsignature, torch.zeros_like(logsignature))
            except Exception as e:
                print(f"Log-signature computation failed for session {session_id}: {e}")
                path_dim = path.shape[1]
                logsig_dim = signatory.logsignature_channels(path_dim, self.logsig_depth)
                logsignature = torch.zeros(logsig_dim, dtype=torch.float32)

            # MODIFICATION: Return both frustration and cognitive load as targets
            final_frustration = session_df['frustration_level'].iloc[-1]
            final_cognitive_load = session_df['cognitive_load'].iloc[-1]
            
            if not np.isfinite(final_frustration):
                final_frustration = 0.0
            if not np.isfinite(final_cognitive_load):
                final_cognitive_load = 0.1
            
            # Return both targets as a 2-element tensor
            targets = torch.tensor([final_frustration, final_cognitive_load], dtype=torch.float32)

            return logsignature, targets
            
        except Exception as e:
            print(f"Error processing session {idx}: {e}")
            # Return safe fallback values for both targets
            path_dim = 1 + 2 + 2 + len(self.ohe_encoder.categories_[0])
            logsig_dim = signatory.logsignature_channels(path_dim, self.logsig_depth)
            return torch.zeros(logsig_dim, dtype=torch.float32), torch.tensor([0.0, 0.1], dtype=torch.float32)

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
        return torch.zeros(batch_size, logsig_dim), torch.zeros(batch_size, 2)

# --- 2. Multi-Target Neural RDE Model ---
class MultiTargetNeuralRDE(nn.Module):
    """Neural RDE model that predicts both frustration and cognitive load."""
    
    def __init__(self, input_channels, hidden_dims, output_channels=2):
        super(MultiTargetNeuralRDE, self).__init__()
        
        # Shared feature extraction layers
        layers = []
        in_dim = input_channels
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())  # Bounded activation for stability
            layers.append(nn.Dropout(0.1))  # Regularization
            in_dim = h_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate heads for each target
        self.frustration_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )
        
        self.cognitive_load_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(), 
            nn.Linear(in_dim // 2, 1)
        )
        
        # Initialize weights to prevent initial instability
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, logsig):
        # Shared feature extraction
        shared_features = self.shared_layers(logsig)
        
        # Separate predictions
        frustration = self.frustration_head(shared_features)
        cognitive_load = self.cognitive_load_head(shared_features)
        
        # Combine outputs [batch_size, 2]
        output = torch.cat([frustration, cognitive_load], dim=1)
        
        # Ensure outputs are finite
        output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
        
        return output

# --- 3. Multi-Target Training Loops ---
def robust_train_loop(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_frustration_loss = 0
    total_cognitive_loss = 0
    valid_batches = 0
    
    for logsigs, targets in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        logsigs, targets = logsigs.to(device), targets.to(device)
        
        # Ensure inputs are finite
        logsigs = torch.where(torch.isfinite(logsigs), logsigs, torch.zeros_like(logsigs))
        targets = torch.where(torch.isfinite(targets), targets, torch.zeros_like(targets))
        
        predictions = model(logsigs)
        
        # MODIFICATION: Compute separate losses for each target
        frustration_loss = loss_fn(predictions[:, 0:1], targets[:, 0:1])
        cognitive_loss = loss_fn(predictions[:, 1:2], targets[:, 1:2])
        
        # Combined loss with weighting
        combined_loss = (CONFIG["frustration_weight"] * frustration_loss + 
                        CONFIG["cognitive_weight"] * cognitive_loss)
        
        if torch.isfinite(combined_loss):
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_value"])
            optimizer.step()
            
            total_loss += combined_loss.item()
            total_frustration_loss += frustration_loss.item()
            total_cognitive_loss += cognitive_loss.item()
            valid_batches += 1
        else:
            print("Warning: Non-finite loss detected, but continuing with next batch")
    
    return {
        'total_loss': total_loss / max(valid_batches, 1),
        'frustration_loss': total_frustration_loss / max(valid_batches, 1),
        'cognitive_loss': total_cognitive_loss / max(valid_batches, 1)
    }

def robust_eval_loop(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_frustration_loss = 0
    total_cognitive_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for logsigs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            logsigs, targets = logsigs.to(device), targets.to(device)
            
            logsigs = torch.where(torch.isfinite(logsigs), logsigs, torch.zeros_like(logsigs))
            targets = torch.where(torch.isfinite(targets), targets, torch.zeros_like(targets))
            
            predictions = model(logsigs)
            
            frustration_loss = loss_fn(predictions[:, 0:1], targets[:, 0:1])
            cognitive_loss = loss_fn(predictions[:, 1:2], targets[:, 1:2])
            combined_loss = (CONFIG["frustration_weight"] * frustration_loss + 
                           CONFIG["cognitive_weight"] * cognitive_loss)
            
            if torch.isfinite(combined_loss):
                total_loss += combined_loss.item()
                total_frustration_loss += frustration_loss.item()
                total_cognitive_loss += cognitive_loss.item()
                valid_batches += 1
    
    return {
        'total_loss': total_loss / max(valid_batches, 1),
        'frustration_loss': total_frustration_loss / max(valid_batches, 1),
        'cognitive_loss': total_cognitive_loss / max(valid_batches, 1)
    }

# --- 4. Main Training Function ---
def main():
    print("Starting Multi-Target Neural RDE Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("CUDA not available, running on CPU. Mixed precision will be disabled.")

    # Load and clean data
    df = pd.read_csv(CONFIG["data_path"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
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
    train_dataset = MultiTargetNeuralRDE_Dataset(train_df, numerical_scaler, ohe_encoder, CONFIG['logsig_depth'])
    val_dataset = MultiTargetNeuralRDE_Dataset(val_df, numerical_scaler, ohe_encoder, CONFIG['logsig_depth'])
    
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
    output_channels = 2  # MODIFICATION: Now predicting 2 targets

    print(f"Model input channels: {input_channels}")
    print(f"Model output channels: {output_channels}")
    
    # Create multi-target model
    model = MultiTargetNeuralRDE(input_channels, CONFIG["mlp_hidden_dims"], output_channels).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        factor=CONFIG["scheduler_factor"], 
        patience=CONFIG["scheduler_patience"], 
        verbose=True
    )

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_frustration_loss': [], 'val_frustration_loss': [],
        'train_cognitive_loss': [], 'val_cognitive_loss': []
    }

    print("\nStarting multi-target training...")
    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()
        
        train_metrics = robust_train_loop(model, train_loader, optimizer, loss_fn, device)
        val_metrics = robust_eval_loop(model, val_loader, loss_fn, device)
        
        scheduler.step(val_metrics['total_loss'])
        
        # Store all metrics
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_frustration_loss'].append(train_metrics['frustration_loss'])
        history['val_frustration_loss'].append(val_metrics['frustration_loss'])
        history['train_cognitive_loss'].append(train_metrics['cognitive_loss'])
        history['val_cognitive_loss'].append(val_metrics['cognitive_loss'])
        
        end_time = time.time()
        epoch_mins = (end_time - start_time) / 60

        print(f"Epoch {epoch+1:02}/{CONFIG['epochs']:02} | Time: {epoch_mins:.2f}m")
        print(f"  Total Loss - Train: {train_metrics['total_loss']:.6f} | Val: {val_metrics['total_loss']:.6f}")
        print(f"  Frustration - Train: {train_metrics['frustration_loss']:.6f} | Val: {val_metrics['frustration_loss']:.6f}")
        print(f"  Cognitive - Train: {train_metrics['cognitive_loss']:.6f} | Val: {val_metrics['cognitive_loss']:.6f}")
        
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(model.state_dict(), 'best_model_multitarget_rde.pth')
            print(f"  -> New best validation loss. Model saved.")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Enhanced plotting for multi-target training
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Total')
    plt.plot(history['val_loss'], label='Val Total')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_frustration_loss'], label='Train Frustration')
    plt.plot(history['val_frustration_loss'], label='Val Frustration')
    plt.title('Frustration Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_cognitive_loss'], label='Train Cognitive')
    plt.plot(history['val_cognitive_loss'], label='Val Cognitive')
    plt.title('Cognitive Load Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('multitarget_training_loss_plot.png', dpi=150, bbox_inches='tight')
    print("Saved multi-target training plot to 'multitarget_training_loss_plot.png'")

if __name__ == "__main__":
    main()
