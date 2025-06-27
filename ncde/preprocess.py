# preprocess.py
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib

# Load the base configuration to get the data path
from config import CONFIG

def preprocess_and_save():
    """
    Loads the raw enriched log, performs all feature engineering, and saves the
    processed data and scalers for efficient loading during training.
    """
    print(f"Loading raw data from {CONFIG['data_path']}...")
    df = pd.read_csv(CONFIG['data_path'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Preparing scalers and encoders...")
    
    # Fit scalers and encoders
    numerical_scaler = StandardScaler().fit(df[['frustration_level', 'cognitive_load']].values)
    target_scaler = StandardScaler().fit(df[['frustration_level']].values)
    all_action_types = df['action_type'].unique().reshape(-1, 1)
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(all_action_types)
    
    # Save the fitted scalers/encoders for later use (e.g., in production)
    joblib.dump(numerical_scaler, 'numerical_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    joblib.dump(ohe_encoder, 'ohe_encoder.pkl')
    print("Saved scalers and encoder to .pkl files.")

    # Split session IDs first
    all_session_ids = df['session_id'].unique()
    train_ids, val_ids = train_test_split(all_session_ids, test_size=0.2, random_state=42)

    processed_data = {'train': [], 'val': []}

    for split, ids in [('train', train_ids), ('val', val_ids)]:
        print(f"Processing {split} data...")
        for session_id in tqdm(ids):
            session_df = df[df['session_id'] == session_id].sort_values('timestamp')

            # Perform all feature engineering
            time_deltas = session_df['timestamp'].diff().dt.total_seconds().fillna(0).values
            time_deltas[time_deltas <= 0] = 1e-5

            psych_features = session_df[['frustration_level', 'cognitive_load']].values
            scaled_psych_features = numerical_scaler.transform(psych_features)
            
            scroll_features = session_df[['scroll_speed', 'scroll_depth']].fillna(0).values
            
            action_types = session_df['action_type'].values.reshape(-1, 1)
            action_ohe = ohe_encoder.transform(action_types)

            features = np.hstack([scaled_psych_features, scroll_features, action_ohe])
            
            final_frustration = session_df['frustration_level'].iloc[-1]
            target = target_scaler.transform(np.array([[final_frustration]]))

            X = np.hstack([time_deltas.reshape(-1, 1), features])

            # Append the final tensors to our list
            processed_data[split].append({
                'X': torch.tensor(X, dtype=torch.float32),
                'y': torch.tensor(target, dtype=torch.float32).view(-1)
            })

    # Save the processed data to a single efficient file
    torch.save(processed_data, CONFIG['processed_data_path'])
    print(f"Successfully preprocessed and saved data to {CONFIG['processed_data_path']}")


if __name__ == "__main__":
    preprocess_and_save()
