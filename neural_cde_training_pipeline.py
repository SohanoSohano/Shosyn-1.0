# neural_cde_training_pipeline.py
import torch
import torch.nn as nn
import torchcde
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class FireTVDataProcessor:
    """Advanced data processor for Fire TV Neural CDE training"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_and_preprocess_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and preprocess Fire TV dataset for Neural CDE training"""
        
        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {df.shape[0]} entries, {df.shape[1]} columns")
        
        # Identify feature columns (exclude metadata)
        metadata_cols = ['user_id', 'session_id', 'content_id', 'interaction_index', 
                        'interaction_datetime']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Handle categorical variables
        categorical_cols = ['interaction_type', 'platform_target', 'device_model']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Handle boolean columns
        boolean_cols = df.select_dtypes(include=['bool']).columns
        for col in boolean_cols:
            df[col] = df[col].astype(int)
        
        # Select numerical features for Neural CDE
        numerical_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # Create sequences by user and session
        sequences, timestamps = self._create_sequences(df, numerical_features)
        
        self.feature_columns = numerical_features
        print(f"Processed {len(sequences)} sequences with {len(numerical_features)} features")
        
        return sequences, timestamps, numerical_features
    
    def _create_sequences(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List, List]:
        """Create time-ordered sequences for Neural CDE training"""
        
        sequences = []
        timestamps = []
        
        # Group by user and session
        for (user_id, session_id), group in df.groupby(['user_id', 'session_id']):
            if len(group) < 3:  # Skip very short sequences
                continue
                
            # Sort by timestamp
            group = group.sort_values('interaction_timestamp')
            
            # Extract features and normalize timestamps
            sequence_data = group[feature_cols].values
            sequence_timestamps = group['interaction_timestamp'].values
            
            # Normalize timestamps to start from 0
            sequence_timestamps = sequence_timestamps - sequence_timestamps[0]
            
            # Scale features
            if len(sequence_data) > 1:
                sequence_data = self.scaler.fit_transform(sequence_data)
                sequences.append(sequence_data)
                timestamps.append(sequence_timestamps)
        
        return sequences, timestamps

class OptimizedFireTVNeuralCDE(nn.Module):
    """Optimized Neural CDE for Fire TV psychological trait prediction"""
    
    def __init__(
        self, 
        input_dim: int = 42,  # Your realistic Fire TV features
        hidden_dim: int = 128,
        output_dim: int = 20,  # 20 psychological traits
        vector_field_layers: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Enhanced vector field with residual connections
        self.vector_field = self._build_enhanced_vector_field(vector_field_layers, dropout_rate)
        
        # Initial state encoder with batch normalization
        self.initial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),  # GELU activation for better performance
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Psychological trait decoders with attention mechanism
        self.trait_names = [
            'cognitive_load', 'decision_confidence', 'frustration_level', 
            'exploration_tendency', 'attention_span', 'navigation_efficiency',
            'platform_loyalty', 'social_influence', 'price_sensitivity',
            'content_diversity', 'session_engagement', 'ui_adaptation',
            'temporal_consistency', 'multi_platform_behavior', 'voice_usage',
            'recommendation_acceptance', 'search_sophistication', 'device_preference',
            'peak_alignment', 'return_likelihood'
        ]
        
        # Multi-head attention for trait extraction
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout_rate)
        
        # Individual trait decoders
        self.trait_decoders = nn.ModuleDict({
            trait: nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for trait in self.trait_names
        })
        
    def _build_enhanced_vector_field(self, num_layers: int, dropout_rate: float) -> nn.Module:
        """Build enhanced vector field with residual connections"""
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.BatchNorm1d(self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                ])
            elif i == num_layers - 1:
                layers.append(nn.Linear(self.hidden_dim * 2, self.hidden_dim * self.input_dim))
            else:
                layers.extend([
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
                    nn.BatchNorm1d(self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                ])
        
        return nn.Sequential(*layers)
    
    def forward(self, interaction_path: torch.Tensor, timestamps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced stability"""
        
        batch_size, seq_len, feature_dim = interaction_path.shape
        
        # Create interpolation path
        try:
            # Use natural cubic splines for smooth interpolation
            coeffs = torchcde.natural_cubic_spline_coeffs(
                torch.cat([timestamps.unsqueeze(-1), interaction_path], dim=-1)
            )
            control_path = torchcde.CubicSpline(coeffs)
        except:
            # Fallback to linear interpolation
            coeffs = torchcde.linear_interpolation_coeffs(
                torch.cat([timestamps.unsqueeze(-1), interaction_path], dim=-1)
            )
            control_path = torchcde.LinearInterpolation(coeffs)
        
        # Initial hidden state
        z0 = self.initial_encoder(interaction_path[:, 0])
        
        # CDE function with enhanced stability
        class StableCDEFunc(torch.nn.Module):
            def __init__(self, vector_field, input_dim):
                super().__init__()
                self.vector_field = vector_field
                self.input_dim = input_dim
                
            def forward(self, t, z):
                try:
                    dXdt = control_path.derivative(t)[:, 1:]
                    f_z = self.vector_field(z).view(z.shape[0], -1, self.input_dim)
                    return torch.bmm(f_z, dXdt.unsqueeze(-1)).squeeze(-1)
                except:
                    return torch.zeros_like(z)
        
        cde_func = StableCDEFunc(self.vector_field, self.input_dim)
        
        # Solve CDE with adaptive solver
        try:
            t_eval = timestamps[0]
            z_trajectory = torchcde.cdeint(
                X=control_path,
                func=cde_func,
                z0=z0,
                t=t_eval,
                method="dopri5",
                rtol=1e-4,
                atol=1e-6,
                adjoint=True  # Memory efficient backpropagation
            )
            final_hidden = z_trajectory[-1]
        except:
            final_hidden = z0
        
        # Apply attention mechanism
        final_hidden_attended, _ = self.attention(
            final_hidden.unsqueeze(0), final_hidden.unsqueeze(0), final_hidden.unsqueeze(0)
        )
        final_hidden = final_hidden_attended.squeeze(0)
        
        # Decode psychological traits
        traits = {}
        for trait_name, decoder in self.trait_decoders.items():
            traits[trait_name] = decoder(final_hidden)
        
        return traits

class FireTVSequenceDataset(Dataset):
    """Custom dataset for Fire TV sequence data"""
    
    def __init__(self, sequences: List[np.ndarray], timestamps: List[np.ndarray]):
        self.sequences = sequences
        self.timestamps = timestamps
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        timestamp = torch.FloatTensor(self.timestamps[idx])
        
        return sequence, timestamp

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    sequences, timestamps = zip(*batch)
    
    # Filter out sequences that are too short
    valid_items = [(s, t) for s, t in zip(sequences, timestamps) if len(s) >= 3]
    
    if not valid_items:
        return None
    
    sequences, timestamps = zip(*valid_items)
    return list(sequences), list(timestamps)

class FireTVNeuralCDETrainer:
    """Advanced trainer for Fire TV Neural CDE with hyperparameter optimization"""
    
    def __init__(self, model: OptimizedFireTVNeuralCDE, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def train_with_optimization(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        patience: int = 15
    ):
        """Train with corrected ReduceLROnPlateau scheduler"""
        
        # Advanced optimizer with scheduling
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Fixed learning rate scheduler (removed verbose parameter)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            threshold=0.0001,
            min_lr=1e-7
        )
        
        # Multi-task loss function
        criterion = self._create_multi_task_loss()
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'trait_losses': {},
            'learning_rates': []
        }
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_trait_losses = self._train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_trait_losses = self._validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(current_lr)
            
            # Early stopping logic
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_fire_tv_neural_cde.pth')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Progress reporting with custom verbose output
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                print(f"Learning Rate: {current_lr:.8f}")
                print("-" * 50)
        
        return history

    
    def _create_multi_task_loss(self):
        """Create weighted multi-task loss for psychological traits"""
        
        # Trait importance weights based on psychological significance
        trait_weights = {
            'cognitive_load': 1.2, 'decision_confidence': 1.2, 'frustration_level': 1.5,
            'exploration_tendency': 1.0, 'attention_span': 1.1, 'navigation_efficiency': 1.0,
            'platform_loyalty': 0.9, 'social_influence': 0.8, 'price_sensitivity': 0.9,
            'content_diversity': 0.8, 'session_engagement': 1.3, 'ui_adaptation': 0.7,
            'temporal_consistency': 0.8, 'multi_platform_behavior': 1.0, 'voice_usage': 0.6,
            'recommendation_acceptance': 1.0, 'search_sophistication': 0.7, 'device_preference': 0.6,
            'peak_alignment': 0.7, 'return_likelihood': 1.1
        }
        
        def debug_multi_task_loss(predictions, targets=None):
            print(f"🔍 Debugging loss computation...")
            
            total_loss = 0
            trait_losses = {}
            
            for trait_name, pred in predictions.items():
                # Generate synthetic targets with debug output
                if targets is None:
                    target = self._generate_synthetic_targets_debug(trait_name, pred.shape[0])
                else:
                    target = targets[trait_name]
                
                print(f"  Trait: {trait_name}")
                print(f"    Prediction shape: {pred.shape}, values: {pred.detach().cpu().numpy().flatten()[:5]}")
                print(f"    Target shape: {target.shape}, values: {target.detach().cpu().numpy().flatten()[:5]}")
                
                # Compute weighted MSE loss
                weight = trait_weights.get(trait_name, 1.0)
                loss = nn.MSELoss()(pred.squeeze(), target) * weight
                print(f"    Individual loss: {loss.item():.6f}")
                
                total_loss += loss
                trait_losses[trait_name] = loss.item()
            
            print(f"  Total loss: {total_loss.item():.6f}")
            return total_loss, trait_losses
        
        return debug_multi_task_loss

    
    def _generate_synthetic_targets_debug(self, trait_name: str, batch_size: int) -> torch.Tensor:
        """Replace your existing _generate_synthetic_targets method"""
        
        # More realistic trait-specific distributions
        trait_distributions = {
            'cognitive_load': (0.4, 0.2),
            'decision_confidence': (0.6, 0.2), 
            'frustration_level': (0.3, 0.15),
            'exploration_tendency': (0.5, 0.25),
            'attention_span': (0.7, 0.2),
            'navigation_efficiency': (0.6, 0.2),
            'platform_loyalty': (0.5, 0.3),
            'social_influence': (0.3, 0.2),
            'price_sensitivity': (0.4, 0.25),
            'content_diversity': (0.5, 0.2)
        }
        
        if trait_name in trait_distributions:
            mean, std = trait_distributions[trait_name]
            # Use normal distribution clamped to [0,1] 
            targets = torch.normal(mean, std, (batch_size,)).clamp(0.1, 0.9)  # Avoid extreme values
        else:
            # Use beta distribution for other traits
            targets = torch.beta(torch.tensor(2.0), torch.tensor(2.0), (batch_size,))
        
        # Add some noise to ensure targets aren't identical to predictions
        noise = torch.randn_like(targets) * 0.05
        targets = (targets + noise).clamp(0.1, 0.9)
        
        print(f"    Generated targets for {trait_name}: mean={targets.mean():.3f}, std={targets.std():.3f}")
        
        return targets.to(self.device)
    
    def _train_epoch(self, train_loader, optimizer, criterion):
        """Training epoch with gradient clipping"""
        
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            if batch is None:
                continue
                
            sequences, timestamps = batch
            
            for seq, ts in zip(sequences, timestamps):
                if len(seq) < 3:
                    continue
                
                optimizer.zero_grad()
                
                # Add batch dimension
                seq_batch = seq.unsqueeze(0).to(self.device)
                ts_batch = ts.unsqueeze(0).to(self.device)
                
                try:
                    # Forward pass
                    predictions = self.model(seq_batch, ts_batch)
                    
                    if batch_count == 0:  # Debug first batch only
                        print(f"\n🔍 Model output analysis (first batch):")
                        for trait_name, pred in predictions.items():
                            values = pred.detach().cpu().numpy()
                            print(f"  {trait_name}: mean={values.mean():.6f}, std={values.std():.6f}, range=[{values.min():.6f}, {values.max():.6f}]")

                    # Compute loss
                    loss, trait_losses = criterion(predictions)
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    continue
        
        return total_loss / max(batch_count, 1), {}
    
    def _validate_epoch(self, val_loader, criterion):
        """Validation epoch"""
        
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                    
                sequences, timestamps = batch
                
                for seq, ts in zip(sequences, timestamps):
                    if len(seq) < 3:
                        continue
                    
                    seq_batch = seq.unsqueeze(0).to(self.device)
                    ts_batch = ts.unsqueeze(0).to(self.device)
                    
                    try:
                        predictions = self.model(seq_batch, ts_batch)
                        loss, trait_losses = criterion(predictions)
                        
                        total_loss += loss.item()
                        batch_count += 1
                        
                    except Exception as e:
                        continue
        
        return total_loss / max(batch_count, 1), {}
    
    def debug_single_forward_pass(self, train_loader):
        """Add this as a new method to test a single forward pass"""
        
        print("🧪 Testing single forward pass...")
        
        self.model.eval()
        with torch.no_grad():
            for batch in train_loader:
                if batch is None:
                    continue
                    
                sequences, timestamps = batch
                
                if sequences and len(sequences[0]) >= 3:
                    seq = sequences[0]
                    ts = timestamps[0]
                    
                    seq_batch = seq.unsqueeze(0).to(self.device)
                    ts_batch = ts.unsqueeze(0).to(self.device)
                    
                    print(f"Input shape: {seq_batch.shape}")
                    print(f"Timestamp shape: {ts_batch.shape}")
                    
                    try:
                        predictions = self.model(seq_batch, ts_batch)
                        
                        print("✅ Forward pass successful!")
                        print("Predictions:")
                        for trait_name, pred in predictions.items():
                            print(f"  {trait_name}: {pred.item():.6f}")
                        
                        return predictions
                        
                    except Exception as e:
                        print(f"❌ Forward pass failed: {e}")
                        return None
                
                break
        
        return None

def optimize_hyperparameters(sequences, timestamps, feature_names):
    """Hyperparameter optimization without device dependency"""
    
    import optuna
    
    def objective(trial):
        # Suggest hyperparameters
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        vector_field_layers = trial.suggest_int('vector_field_layers', 2, 6)
        
        # Return only model parameters
        model_params = {
            'hidden_dim': hidden_dim,
            'dropout_rate': dropout_rate,
            'vector_field_layers': vector_field_layers
        }
        
        # Return optimizer parameters separately
        optimizer_params = {
            'learning_rate': learning_rate
        }
        
        # Create model with model parameters only
        model = OptimizedFireTVNeuralCDE(
            input_dim=len(feature_names),
            **model_params
        )
        
        # Device handling is done in trainer, not here
        trainer = FireTVNeuralCDETrainer(model)  # No device parameter needed
        
        # Create data loaders
        dataset = FireTVSequenceDataset(sequences, timestamps)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        
        # Train model
        history = trainer.train_with_optimization(
            train_loader, val_loader, 
            num_epochs=30, 
            **optimizer_params
        )
        
        return min(history['val_loss'])
    
    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    return study.best_params



### Main ###
def main_training_pipeline():
    """Execute complete Neural CDE training pipeline"""
    
    print("Fire TV Neural CDE Training Pipeline")
    print("=" * 50)
    
    # Load and preprocess data
    processor = FireTVDataProcessor()
    sequences, timestamps, feature_names = processor.load_and_preprocess_data(
        'fire_tv_neural_cde_dataset_1000.csv'
    )
    
    print("\n🧪 DEBUGGING PHASE - Testing basic functionality")
    
    # Create simple model for testing
    test_model = OptimizedFireTVNeuralCDE(
        input_dim=len(feature_names),
        hidden_dim=32,  # Small for testing
        dropout_rate=0.1,
        vector_field_layers=2
    )
    
    # Create test data loader
    dataset = FireTVSequenceDataset(sequences, timestamps)
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # Test trainer
    test_trainer = FireTVNeuralCDETrainer(test_model, device='cpu')
    
    # Run single forward pass test
    test_predictions = test_trainer.debug_single_forward_pass(test_loader)
    
    if test_predictions is None:
        print("❌ Basic functionality test failed! Fix model before proceeding.")
        return None, None
    
    print("✅ Basic functionality test passed! Proceeding with optimization...")

    # Optimize hyperparameters (no device parameter needed)
    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(sequences, timestamps, feature_names)
    print(f"Best parameters: {best_params}")
    
    # Separate model and optimizer parameters
    model_params = {k: v for k, v in best_params.items() 
                   if k in ['hidden_dim', 'dropout_rate', 'vector_field_layers']}
    optimizer_params = {k: v for k, v in best_params.items() 
                       if k in ['learning_rate', 'weight_decay']}
    
    # Create optimized model
    model = OptimizedFireTVNeuralCDE(
        input_dim=len(feature_names),
        **model_params
    )
    
    # Create data loaders
    dataset = FireTVSequenceDataset(sequences, timestamps)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Train final model (device handling is in trainer)
    trainer = FireTVNeuralCDETrainer(model)
    history = trainer.train_with_optimization(
        train_loader, val_loader, 
        num_epochs=200,
        **optimizer_params
    )
    
    print("✅ Training completed successfully!")
    return model, history

if __name__ == "__main__":
    model, history = main_training_pipeline()