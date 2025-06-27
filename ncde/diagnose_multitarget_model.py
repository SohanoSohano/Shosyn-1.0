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

# Import your robust classes
from train_multitarget_rde import MultiTargetNeuralRDE_Dataset, MultiTargetNeuralRDE, robust_collate_fn, RobustStandardScaler

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

def diagnose_multitarget_dataset(data_path):
    """Comprehensive analysis of the dataset for both targets."""
    print("="*60)
    print("MULTI-TARGET DATASET ANALYSIS")
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
    
    print("\n--- COGNITIVE LOAD ANALYSIS ---")
    cognitive_stats = df['cognitive_load'].describe()
    print(cognitive_stats)
    
    print(f"\nUnique cognitive load levels: {df['cognitive_load'].nunique()}")
    print(f"Zero cognitive load events: {(df['cognitive_load'] == 0.0).sum()} / {len(df)} ({(df['cognitive_load'] == 0.0).mean()*100:.1f}%)")
    
    print("\n--- SESSION-LEVEL ANALYSIS ---")
    session_final_frustration = df.groupby('session_id')['frustration_level'].last()
    session_final_cognitive = df.groupby('session_id')['cognitive_load'].last()
    
    print("Final frustration per session statistics:")
    print(session_final_frustration.describe())
    
    print("\nFinal cognitive load per session statistics:")
    print(session_final_cognitive.describe())
    
    # Correlation analysis
    correlation = np.corrcoef(session_final_frustration, session_final_cognitive)[0, 1]
    print(f"\nCorrelation between final frustration and cognitive load: {correlation:.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Frustration distribution
    axes[0,0].hist(df['frustration_level'], bins=50, alpha=0.7, edgecolor='black', color='red')
    axes[0,0].set_title('Distribution of All Frustration Levels')
    axes[0,0].set_xlabel('Frustration Level')
    axes[0,0].set_ylabel('Count')
    
    # Cognitive load distribution
    axes[0,1].hist(df['cognitive_load'], bins=50, alpha=0.7, edgecolor='black', color='blue')
    axes[0,1].set_title('Distribution of All Cognitive Load Levels')
    axes[0,1].set_xlabel('Cognitive Load')
    axes[0,1].set_ylabel('Count')
    
    # Correlation scatter plot
    axes[0,2].scatter(df['frustration_level'], df['cognitive_load'], alpha=0.5, s=1)
    axes[0,2].set_title('Frustration vs Cognitive Load')
    axes[0,2].set_xlabel('Frustration Level')
    axes[0,2].set_ylabel('Cognitive Load')
    
    # Final session values
    axes[1,0].hist(session_final_frustration, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1,0].set_title('Final Session Frustration')
    axes[1,0].set_xlabel('Final Frustration Level')
    axes[1,0].set_ylabel('Count')
    
    axes[1,1].hist(session_final_cognitive, bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[1,1].set_title('Final Session Cognitive Load')
    axes[1,1].set_xlabel('Final Cognitive Load')
    axes[1,1].set_ylabel('Count')
    
    # Session correlation
    axes[1,2].scatter(session_final_frustration, session_final_cognitive, alpha=0.6)
    axes[1,2].set_title(f'Session Final Values (r={correlation:.3f})')
    axes[1,2].set_xlabel('Final Frustration')
    axes[1,2].set_ylabel('Final Cognitive Load')
    
    plt.tight_layout()
    plt.savefig('multitarget_dataset_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved dataset analysis plots to 'multitarget_dataset_analysis.png'")
    
    return df, session_final_frustration, session_final_cognitive

def diagnose_multitarget_model_predictions(model_path, data_path):
    """Analyze what the trained multi-target model is predicting."""
    print("\n" + "="*60)
    print("MULTI-TARGET MODEL PREDICTION ANALYSIS")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare data
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
    
    val_dataset = MultiTargetNeuralRDE_Dataset(val_df, numerical_scaler, ohe_encoder, logsig_depth=3)
    
    # Load model
    path_dim = 1 + 2 + 2 + len(all_action_types)
    input_channels = signatory.logsignature_channels(path_dim, 3)
    model = MultiTargetNeuralRDE(input_channels, [128, 64], 2).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded multi-target model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    # Collect predictions and targets
    predictions_frustration = []
    predictions_cognitive = []
    targets_frustration = []
    targets_cognitive = []
    
    print("\nAnalyzing multi-target model predictions on validation set...")
    with torch.no_grad():
        for i in range(min(100, len(val_dataset))):
            try:
                logsig, target = val_dataset[i]
                pred = model(logsig.unsqueeze(0).to(device))
                
                pred_frustration = pred[0, 0].cpu().numpy()
                pred_cognitive = pred[0, 1].cpu().numpy()
                target_frustration = target[0].numpy()
                target_cognitive = target[1].numpy()
                
                predictions_frustration.append(pred_frustration)
                predictions_cognitive.append(pred_cognitive)
                targets_frustration.append(target_frustration)
                targets_cognitive.append(target_cognitive)
                
                if i < 10:  # Print first 10 for inspection
                    print(f"  Sample {i:2d}:")
                    print(f"    Frustration - Target: {target_frustration:.6f}, Prediction: {pred_frustration:.6f}, Error: {abs(pred_frustration-target_frustration):.6f}")
                    print(f"    Cognitive   - Target: {target_cognitive:.6f}, Prediction: {pred_cognitive:.6f}, Error: {abs(pred_cognitive-target_cognitive):.6f}")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    if not predictions_frustration:
        print("No valid predictions could be generated!")
        return
    
    predictions_frustration = np.array(predictions_frustration)
    predictions_cognitive = np.array(predictions_cognitive)
    targets_frustration = np.array(targets_frustration)
    targets_cognitive = np.array(targets_cognitive)
    
    print(f"\n--- FRUSTRATION PREDICTION STATISTICS (n={len(predictions_frustration)}) ---")
    print(f"Predictions - Min: {predictions_frustration.min():.6f}, Max: {predictions_frustration.max():.6f}, Mean: {predictions_frustration.mean():.6f}, Std: {predictions_frustration.std():.6f}")
    print(f"Targets     - Min: {targets_frustration.min():.6f}, Max: {targets_frustration.max():.6f}, Mean: {targets_frustration.mean():.6f}, Std: {targets_frustration.std():.6f}")
    
    print(f"\n--- COGNITIVE LOAD PREDICTION STATISTICS (n={len(predictions_cognitive)}) ---")
    print(f"Predictions - Min: {predictions_cognitive.min():.6f}, Max: {predictions_cognitive.max():.6f}, Mean: {predictions_cognitive.mean():.6f}, Std: {predictions_cognitive.std():.6f}")
    print(f"Targets     - Min: {targets_cognitive.min():.6f}, Max: {targets_cognitive.max():.6f}, Mean: {targets_cognitive.mean():.6f}, Std: {targets_cognitive.std():.6f}")
    
    # Calculate metrics for both targets
    mse_frustration = np.mean((predictions_frustration - targets_frustration) ** 2)
    mae_frustration = np.mean(np.abs(predictions_frustration - targets_frustration))
    
    mse_cognitive = np.mean((predictions_cognitive - targets_cognitive) ** 2)
    mae_cognitive = np.mean(np.abs(predictions_cognitive - targets_cognitive))
    
    print(f"\nFrustration Model Performance:")
    print(f"  MSE: {mse_frustration:.6f}")
    print(f"  MAE: {mae_frustration:.6f}")
    print(f"  RMSE: {np.sqrt(mse_frustration):.6f}")
    
    print(f"\nCognitive Load Model Performance:")
    print(f"  MSE: {mse_cognitive:.6f}")
    print(f"  MAE: {mae_cognitive:.6f}")
    print(f"  RMSE: {np.sqrt(mse_cognitive):.6f}")
    
    # Check prediction diversity
    pred_unique_frustration = len(np.unique(np.round(predictions_frustration, 6)))
    pred_unique_cognitive = len(np.unique(np.round(predictions_cognitive, 6)))
    target_unique_frustration = len(np.unique(np.round(targets_frustration, 6)))
    target_unique_cognitive = len(np.unique(np.round(targets_cognitive, 6)))
    
    print(f"\nDiversity Analysis:")
    print(f"  Frustration - Unique predictions: {pred_unique_frustration}, Unique targets: {target_unique_frustration}")
    print(f"  Cognitive Load - Unique predictions: {pred_unique_cognitive}, Unique targets: {target_unique_cognitive}")
    
    # Visualize multi-target predictions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Frustration predictions vs targets
    axes[0,0].scatter(targets_frustration, predictions_frustration, alpha=0.6, color='red')
    axes[0,0].plot([targets_frustration.min(), targets_frustration.max()], 
                   [targets_frustration.min(), targets_frustration.max()], 'k--', label='Perfect Prediction')
    axes[0,0].set_xlabel('True Frustration')
    axes[0,0].set_ylabel('Predicted Frustration')
    axes[0,0].set_title('Frustration: Predictions vs True Values')
    axes[0,0].legend()
    
    # Cognitive load predictions vs targets
    axes[0,1].scatter(targets_cognitive, predictions_cognitive, alpha=0.6, color='blue')
    axes[0,1].plot([targets_cognitive.min(), targets_cognitive.max()], 
                   [targets_cognitive.min(), targets_cognitive.max()], 'k--', label='Perfect Prediction')
    axes[0,1].set_xlabel('True Cognitive Load')
    axes[0,1].set_ylabel('Predicted Cognitive Load')
    axes[0,1].set_title('Cognitive Load: Predictions vs True Values')
    axes[0,1].legend()
    
    # Distribution comparison for frustration
    axes[0,2].hist(predictions_frustration, bins=30, alpha=0.7, label='Predictions', color='red')
    axes[0,2].hist(targets_frustration, bins=30, alpha=0.7, label='Targets', color='darkred')
    axes[0,2].set_xlabel('Frustration Value')
    axes[0,2].set_ylabel('Count')
    axes[0,2].set_title('Frustration Distribution Comparison')
    axes[0,2].legend()
    
    # Distribution comparison for cognitive load
    axes[1,0].hist(predictions_cognitive, bins=30, alpha=0.7, label='Predictions', color='blue')
    axes[1,0].hist(targets_cognitive, bins=30, alpha=0.7, label='Targets', color='darkblue')
    axes[1,0].set_xlabel('Cognitive Load Value')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Cognitive Load Distribution Comparison')
    axes[1,0].legend()
    
    # Error distributions
    errors_frustration = predictions_frustration - targets_frustration
    errors_cognitive = predictions_cognitive - targets_cognitive
    
    axes[1,1].hist(errors_frustration, bins=30, alpha=0.7, color='red')
    axes[1,1].set_xlabel('Prediction Error')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Frustration Prediction Error Distribution')
    axes[1,1].axvline(0, color='black', linestyle='--', label='Zero Error')
    axes[1,1].legend()
    
    axes[1,2].hist(errors_cognitive, bins=30, alpha=0.7, color='blue')
    axes[1,2].set_xlabel('Prediction Error')
    axes[1,2].set_ylabel('Count')
    axes[1,2].set_title('Cognitive Load Prediction Error Distribution')
    axes[1,2].axvline(0, color='black', linestyle='--', label='Zero Error')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig('multitarget_model_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved model analysis plots to 'multitarget_model_analysis.png'")
    
    return predictions_frustration, predictions_cognitive, targets_frustration, targets_cognitive

def main():
    """Run comprehensive multi-target diagnostics."""
    data_path = "enriched_simulation_logs_500_new.csv"
    model_path = "best_model_multitarget_rde.pth"
    
    print("COMPREHENSIVE MULTI-TARGET MODEL AND DATA DIAGNOSTICS")
    print("="*60)
    
    # 1. Analyze the dataset for both targets
    df, session_frustration, session_cognitive = diagnose_multitarget_dataset(data_path)
    
    # 2. Analyze model predictions (if model exists)
    if os.path.exists(model_path):
        pred_frust, pred_cog, target_frust, target_cog = diagnose_multitarget_model_predictions(model_path, data_path)
        
        # Final verdict
        print("\n" + "="*60)
        print("FINAL MULTI-TARGET DIAGNOSIS")
        print("="*60)
        
        # Target analysis
        zero_frustration = (target_frust == 0.0).sum()
        zero_cognitive = (target_cog == 0.0).sum()
        total_targets = len(target_frust)
        
        print(f"Target Analysis:")
        print(f"  - Frustration: {zero_frustration}/{total_targets} ({(zero_frustration/total_targets)*100:.1f}%) targets are exactly 0.0")
        print(f"  - Cognitive Load: {zero_cognitive}/{total_targets} ({(zero_cognitive/total_targets)*100:.1f}%) targets are exactly 0.0")
        print(f"  - Frustration std: {np.std(target_frust):.6f}")
        print(f"  - Cognitive Load std: {np.std(target_cog):.6f}")
        
        # Model analysis
        pred_frust_std = np.std(pred_frust)
        pred_cog_std = np.std(pred_cog)
        
        print(f"\nModel Analysis:")
        print(f"  - Frustration prediction std: {pred_frust_std:.6f}")
        print(f"  - Cognitive Load prediction std: {pred_cog_std:.6f}")
        print(f"  - Frustration RMSE: {np.sqrt(np.mean((pred_frust - target_frust) ** 2)):.6f}")
        print(f"  - Cognitive Load RMSE: {np.sqrt(np.mean((pred_cog - target_cog) ** 2)):.6f}")
        
        # Overall assessment
        if zero_frustration < total_targets * 0.1 and zero_cognitive < total_targets * 0.1:
            print("\n✅ EXCELLENT: Both targets have diverse, realistic distributions")
        else:
            print("\n⚠️ WARNING: Some targets may have limited diversity")
            
        if pred_frust_std > 0.01 and pred_cog_std > 0.01:
            print("✅ EXCELLENT: Model predictions show good diversity for both targets")
        else:
            print("⚠️ WARNING: Model predictions may be too conservative")
            
    else:
        print(f"\nModel file {model_path} not found. Skipping model analysis.")
    
    print(f"\nDiagnostic plots saved:")
    print(f"  - multitarget_dataset_analysis.png: Raw data distributions for both targets")
    if os.path.exists(model_path):
        print(f"  - multitarget_model_analysis.png: Model prediction analysis for both targets")

if __name__ == "__main__":
    import os
    main()
