import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchcde
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm

# Set a non-interactive backend for Matplotlib to work on servers without a display
import matplotlib
matplotlib.use('Agg')
sns.set_theme(style="whitegrid")

# --- Configuration & Hyperparameters ---
CONFIG = {
    "data_path": r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\Neo_Shosyn\Shosyn-1.0\dataset\enriched_simulation_logs_500.csv",
    "batch_size": 32,
    "learning_rate": 1e-4,  # MODIFICATION: Reduced learning rate for stability
    "weight_decay": 1e-5,
    "epochs": 50,
    "hidden_channels": 32,
    "cde_func_channels": 64,
    "cde_func_depth": 3,
    "readout_hidden_channels": 64,
    "num_workers": 0,
    "clip_value": 1.0
}

# --- 1. Robust Data Handling: Custom Dataset and Preprocessing ---
class SessionDataset(Dataset):
    def __init__(self, df, numerical_scaler, target_scaler, ohe_encoder):
        self.df = df
        self.numerical_scaler = numerical_scaler
        self.target_scaler = target_scaler
        self.ohe_encoder = ohe_encoder
        self.session_ids = df['session_id'].unique()

    def __len__(self):
        return len(self.session_ids)

    def __getitem__(self, idx):
        session_id = self.session_ids[idx]
        session_df = self.df[self.df['session_id'] == session_id].sort_values('timestamp')

        time_deltas = session_df['timestamp'].diff().dt.total_seconds().fillna(0).values
        time_deltas[time_deltas <= 0] = 1e-5

        psych_features = session_df[['frustration_level', 'cognitive_load']].values
        scaled_psych_features = self.numerical_scaler.transform(psych_features)
        
        scroll_features = session_df[['scroll_speed', 'scroll_depth']].fillna(0).values
        
        action_types = session_df['action_type'].values.reshape(-1, 1)
        action_ohe = self.ohe_encoder.transform(action_types)

        features = np.hstack([scaled_psych_features, scroll_features, action_ohe])
        
        final_frustration = session_df['frustration_level'].iloc[-1]
        target = self.target_scaler.transform(np.array([[final_frustration]]))

        X = np.hstack([time_deltas.reshape(-1, 1), features])

        return torch.tensor(X, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).view(-1)


def collate_fn(batch):
    sequences, targets = zip(*batch)
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    # MODIFICATION: Removed unused 'lengths' tensor from return
    return padded_sequences, torch.stack(targets)


# --- 2. Model Definition: The Neural CDE ---
class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels, depth):
        super(CDEFunc, self).__init__()
        # MODIFICATION: Use Tanh for numerical stability in the vector field
        layers = [nn.Linear(hidden_channels, CONFIG["cde_func_channels"]), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(CONFIG["cde_func_channels"], CONFIG["cde_func_channels"]), nn.Tanh()])
        layers.append(nn.Linear(CONFIG["cde_func_channels"], input_channels * hidden_channels))
        self.net = nn.Sequential(*layers)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

    def forward(self, t, z):
        return self.net(z).view(-1, self.hidden_channels, self.input_channels)

class NeuralCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NeuralCDE, self).__init__()
        self.initial = nn.Linear(input_channels, hidden_channels)
        self.func = CDEFunc(input_channels, hidden_channels, CONFIG["cde_func_depth"])
        self.readout = nn.Linear(hidden_channels, output_channels)

    def forward(self, x_padded):
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_padded)
        X = torchcde.CubicSpline(coeffs)
        z0 = self.initial(X.evaluate(X.interval[0]))
        z_final = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval, method='rk4', options={'step_size': 0.1})[:, 1]
        pred = self.readout(z_final)
        return pred

# --- 3. Training & Evaluation Loops ---
def train_loop(model, dataloader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad(set_to_none=True)
        sequences, targets = batch
        sequences, targets = sequences.to(device), targets.to(device)

        use_cuda = device.type == 'cuda'
        with torch.cuda.amp.autocast(enabled=use_cuda):
            predictions = model(sequences)
            loss = loss_fn(predictions, targets)

        if torch.isnan(loss):
            print("NaN loss detected! Skipping batch.")
            continue

        if use_cuda:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_value"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_value"])
            optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def eval_loop(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences, targets = batch
            sequences, targets = sequences.to(device), targets.to(device)
            predictions = model(sequences)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)


# --- 4. Main Orchestrator ---
def main():
    print("Starting Neural CDE Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("CUDA not available, running on CPU. Mixed precision will be disabled.")

    df = pd.read_csv(CONFIG["data_path"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Preparing scalers and encoders...")
    
    numerical_scaler = StandardScaler().fit(df[['frustration_level', 'cognitive_load']].values)
    target_scaler = StandardScaler().fit(df[['frustration_level']].values)
    
    all_action_types = df['action_type'].unique().reshape(-1, 1)
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(all_action_types)
    
    all_session_ids = df['session_id'].unique()
    train_ids, val_ids = train_test_split(all_session_ids, test_size=0.2, random_state=42)
    
    train_df = df[df['session_id'].isin(train_ids)]
    val_df = df[df['session_id'].isin(val_ids)]

    print(f"Training on {len(train_ids)} sessions, Validating on {len(val_ids)} sessions.")
    
    train_dataset = SessionDataset(train_df, numerical_scaler, target_scaler, ohe_encoder)
    val_dataset = SessionDataset(val_df, numerical_scaler, target_scaler, ohe_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=CONFIG["num_workers"], pin_memory=True)

    input_channels = 1 + 2 + 2 + len(all_action_types)
    output_channels = 1

    model = NeuralCDE(input_channels, CONFIG["hidden_channels"], output_channels).to(device)
    
    if torch.cuda.device_count() > 1 and use_cuda:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print("\nStarting training...")
    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()
        
        train_loss = train_loop(model, train_loader, optimizer, loss_fn, scaler, device)
        val_loss = eval_loop(model, val_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        end_time = time.time()
        epoch_mins = (end_time - start_time) / 60

        print(f"Epoch {epoch+1:02}/{CONFIG['epochs']:02} | "
              f"Time: {epoch_mins:.2f}m | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val. Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_cde.pth')
            print(f"  -> New best validation loss. Model saved to 'best_model_cde.pth'")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('training_loss_plot.png')
    print("Saved training plot to 'training_loss_plot.png'")

if __name__ == "__main__":
    main()
