# scripts/inspect_csv_columns.py
import pandas as pd
import sys

# Update this path to be the absolute path to your dataset
# This ensures it runs from anywhere.
FILE_PATH = r'C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\fire_tv_neural_cde_transformer_instance_version\Shosyn-1.0\fire_tv_project\fire_tv_neural_cde_transformer\fire_tv_synthetic_dataset_v3_tmdb.csv'

try:
    print(f"Inspecting columns from: {FILE_PATH}")
    # We only need to read the first 5 rows to get the headers and data types
    df_sample = pd.read_csv(FILE_PATH, nrows=5)
    
    print("\n" + "="*50)
    print("--- AVAILABLE COLUMNS IN YOUR SYNTHETIC DATASET ---")
    print("="*50)
    for col in df_sample.columns:
        print(f"- {col}")
    print("="*50 + "\n")
    print("ACTION: Use this list to update 'synthetic_data_loader.py'.")

except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at '{FILE_PATH}'")
    print("   Please double-check the path in this script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
