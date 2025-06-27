import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import signatory
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import your robust classes (assuming they're in the same directory)
from train_nrde import RobustStandardScaler, RobustNeuralRDE_Dataset, RobustNeuralRDE, robust_collate_fn

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

def diagnose_dataset(data_path):
    """Comprehensive analysis of the raw dataset."""
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    df = pd.read_csv(data_path)
    print(f"Total events: {len(df)}")
    print(f"Total sessions: {df['session_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\n--- FRUSTRATION LEVEL ANALYSIS ---")
    frustration_stats = df['frustration_level'].describe()
    print(frustration_stats)
    
    print(f"\nUnique frustration levels: {df['frustration_level'].nunique()}")
    print(f"Zero frustration events: {(df['frustration_level'] == 0.0).sum()} / {len(df)} ({(df['frustration_level'] == 0.0).mean()*100:.1f}%)")
    
    print("\nTop 10 most common frustration levels:")
    print(df['frustration_level'].value_counts().head(10))
    
    print("\n--- SESSION-LEVEL FRUSTRATION ANALYSIS ---")
    session_final_frustration = df.groupby('session_id')['frustration_level'].last()
    print("Final frustration per session statistics:")
    print(session_final_frustration.describe())
    
    print(f"\nSessions ending with zero frustration: {(session_final_frustration == 0.0).sum()} / {len(session_final_frustration)} ({(session_final_frustration == 0.0).mean()*100:.1f}%)")
    
    print("\nTop 10 most common final frustration levels:")
    print(session_final_frustration.value_counts().head(10))
    
    print("\n--- OTHER FEATURES ANALYSIS ---")
    print("Cognitive load statistics:")
    print(df['cognitive_load'].describe())
    
    print(f"\nZero cognitive load events: {(df['cognitive_load'] == 0.0).sum()} / {len(df)} ({(df['cognitive_load'] == 0.0).mean()*100:.1f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Frustration level distribution
    axes[0,0].hist(df['frustration_level'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Distribution of All Frustration Levels')
    axes[0,0].set_xlabel('Frustration Level')
    axes[0,0].set_ylabel('Count')
    
    # Final frustration per session
    axes[0,1].hist(session_final_frustration, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[0,1].set_title('Distribution of Final Session Frustration')
    axes[0,1].set_xlabel('Final Frustration Level')
    axes[0,1].set_ylabel('Count')
    
    # Cognitive load distribution
    axes[1,0].hist(df['cognitive_load'], bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1,0].set_title('Distribution of Cognitive Load')
    axes[1,0].set_xlabel('Cognitive Load')
    axes[1,0].set_ylabel('Count')
    
    # Frustration vs Cognitive Load scatter
    sample_df = df.sample(min(1000, len(df)))  # Sample for performance
    axes[1,1].scatter(sample_df['cognitive_load'], sample_df['frustration_level'], alpha=0.5)
    axes[1,1].set_title('Frustration vs Cognitive Load')
    axes[1,1].set_xlabel('Cognitive Load')
    axes[1,1].set_ylabel('Frustration Level')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved dataset analysis plots to 'dataset_analysis.png'")
    
    return df, session_final_frustration

def diagnose_model_predictions(model_path, data_path):
    """Analyze what the trained model is actually predicting."""
    print("\n" + "="*60)
    print("MODEL PREDICTION ANALYSIS")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare data (same as training)
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Clean data
    numeric_columns = ['frustration_level', 'cognitive_load', 'scroll_speed', 'scroll_depth']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = np.where(np.isfinite(df[col]), df[col], 0)
    
    # Prepare scalers
    numerical_scaler = RobustStandardScaler()
    numerical_scaler.fit(df[['frustration_level', 'cognitive_load']].values)
    
    df['action_type'] = df['action_type'].fillna('unknown')
    all_action_types = df['action_type'].unique().reshape(-1, 1)
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(all_action_types)
    
    # Create test dataset
    all_session_ids = df['session_id'].unique()
    train_ids, val_ids = train_test_split(all_session_ids, test_size=0.2, random_state=42)
    val_df = df[df['session_id'].isin(val_ids)]
    
    val_dataset = RobustNeuralRDE_Dataset(val_df, numerical_scaler, ohe_encoder, logsig_depth=2)
    
    # Load model
    path_dim = 1 + 2 + 2 + len(all_action_types)
    input_channels = signatory.logsignature_channels(path_dim, 2)
    model = RobustNeuralRDE(input_channels, [64, 32], 1).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    # Collect predictions and targets
    predictions = []
    targets = []
    
    print("\nAnalyzing model predictions on validation set...")
    with torch.no_grad():
        for i in range(min(100, len(val_dataset))):  # Analyze first 100 samples
            try:
                logsig, target = val_dataset[i]
                pred = model(logsig.unsqueeze(0).to(device))
                
                predictions.append(pred.item())
                targets.append(target.item())
                
                if i < 10:  # Print first 10 for inspection
                    print(f"  Sample {i:2d}: Target={target.item():.6f}, Prediction={pred.item():.6f}, Error={abs(pred.item()-target.item()):.6f}")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    if not predictions:
        print("No valid predictions could be generated!")
        return
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    print(f"\n--- PREDICTION STATISTICS (n={len(predictions)}) ---")
    print(f"Predictions - Min: {predictions.min():.6f}, Max: {predictions.max():.6f}, Mean: {predictions.mean():.6f}, Std: {predictions.std():.6f}")
    print(f"Targets     - Min: {targets.min():.6f}, Max: {targets.max():.6f}, Mean: {targets.mean():.6f}, Std: {targets.std():.6f}")
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    print(f"\nModel Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.6f}")
    
    # Check if model is predicting constant values
    pred_unique = len(np.unique(np.round(predictions, 6)))
    target_unique = len(np.unique(np.round(targets, 6)))
    
    print(f"\nDiversity Analysis:")
    print(f"  Unique prediction values (rounded to 6 decimals): {pred_unique}")
    print(f"  Unique target values (rounded to 6 decimals): {target_unique}")
    
    if pred_unique < 5:
        print("  ⚠️  WARNING: Model is predicting very few unique values!")
        print("  This suggests the model has learned a trivial solution.")
    
    # Visualize predictions vs targets
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.hist(predictions, bins=30, alpha=0.7, label='Predictions', color='blue')
    plt.hist(targets, bins=30, alpha=0.7, label='Targets', color='red')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Distribution Comparison')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    errors = predictions - targets
    plt.hist(errors, bins=30, alpha=0.7, color='green')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Prediction Error Distribution')
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(predictions[:50], 'b-', label='Predictions', marker='o', markersize=3)
    plt.plot(targets[:50], 'r-', label='Targets', marker='s', markersize=3)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('First 50 Predictions vs Targets')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved model analysis plots to 'model_analysis.png'")
    
    return predictions, targets

def diagnose_data_pipeline():
    """Test the data pipeline for issues."""
    print("\n" + "="*60)
    print("DATA PIPELINE DIAGNOSIS")
    print("="*60)
    
    data_path = "/home/ubuntu/Shosyn-1.0/dataset/enriched_simulation_logs_500.csv"
    
    # Test raw data loading
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Successfully loaded {len(df)} events from {df['session_id'].nunique()} sessions")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Test data preprocessing
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        numeric_columns = ['frustration_level', 'cognitive_load', 'scroll_speed', 'scroll_depth']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        print("✓ Data preprocessing successful")
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        return
    
    # Test scaler
    try:
        scaler = RobustStandardScaler()
        scaler.fit(df[['frustration_level', 'cognitive_load']].values)
        print(f"✓ Scaler fit successful. Means: {scaler.mean_}, Scales: {scaler.scale_}")
    except Exception as e:
        print(f"✗ Error fitting scaler: {e}")
        return
    
    # Test dataset creation
    try:
        all_session_ids = df['session_id'].unique()
        test_df = df[df['session_id'].isin(all_session_ids[:5])]  # Test with 5 sessions
        
        df['action_type'] = df['action_type'].fillna('unknown')
        all_action_types = df['action_type'].unique().reshape(-1, 1)
        ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(all_action_types)
        
        dataset = RobustNeuralRDE_Dataset(test_df, scaler, ohe_encoder, 2)
        print(f"✓ Dataset creation successful with {len(dataset)} sessions")
        
        # Test first few samples
        for i in range(min(3, len(dataset))):
            logsig, target = dataset[i]
            print(f"  Sample {i}: LogSig shape={logsig.shape}, Target={target.item():.6f}")
            
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        return
    
    print("✓ Data pipeline diagnosis complete")

def main():
    """Run comprehensive diagnostics."""
    data_path = "/home/ubuntu/Shosyn-1.0/dataset/enriched_simulation_logs_500_new.csv"
    model_path = "best_model_robust_rde.pth"
    
    print("COMPREHENSIVE MODEL AND DATA DIAGNOSTICS")
    print("="*60)
    
    # 1. Analyze the raw dataset
    df, session_frustration = diagnose_dataset(data_path)
    
    # 2. Test the data pipeline
    diagnose_data_pipeline()
    
    # 3. Analyze model predictions (if model exists)
    if os.path.exists(model_path):
        predictions, targets = diagnose_model_predictions(model_path, data_path)
        
        # Final verdict
        print("\n" + "="*60)
        print("FINAL DIAGNOSIS")
        print("="*60)
        
        zero_targets = (targets == 0.0).sum()
        total_targets = len(targets)
        zero_percentage = (zero_targets / total_targets) * 100
        
        pred_std = np.std(predictions)
        target_std = np.std(targets)
        
        print(f"Target Analysis:")
        print(f"  - {zero_targets}/{total_targets} ({zero_percentage:.1f}%) targets are exactly 0.0")
        print(f"  - Target standard deviation: {target_std:.6f}")
        
        print(f"\nModel Analysis:")
        print(f"  - Prediction standard deviation: {pred_std:.6f}")
        print(f"  - Model MSE: {np.mean((predictions - targets) ** 2):.6f}")
        
        if zero_percentage > 80:
            print("\n🚨 CRITICAL ISSUE: Most targets are zero!")
            print("   This explains the suspiciously good performance.")
            print("   The model learned to predict ~0 for everything.")
            
        if pred_std < 0.001:
            print("\n🚨 CRITICAL ISSUE: Model predictions have very low variance!")
            print("   The model is predicting nearly constant values.")
            
        if target_std < 0.001:
            print("\n🚨 CRITICAL ISSUE: Targets have very low variance!")
            print("   The dataset lacks meaningful variation in the target variable.")
            
    else:
        print(f"\nModel file {model_path} not found. Skipping model analysis.")
    
    print(f"\nDiagnostic plots saved:")
    print(f"  - dataset_analysis.png: Raw data distributions")
    if os.path.exists(model_path):
        print(f"  - model_analysis.png: Model prediction analysis")

if __name__ == "__main__":
    import os
    main()
