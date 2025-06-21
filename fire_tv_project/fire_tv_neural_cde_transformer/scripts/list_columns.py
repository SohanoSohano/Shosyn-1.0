# scripts/list_columns.py
import pandas as pd

try:
    # Update this path if your file is located elsewhere
    file_path = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\fire_tv_production_dataset_parallel.csv"
    
    print(f"Inspecting columns from: {file_path}")
    
    # We only need to read the first row to get the headers
    df_headers = pd.read_csv(file_path, nrows=0)
    
    print("\n--- AVAILABLE COLUMNS ---")
    for col in df_headers.columns:
        print(f"- {col}")
    print("-------------------------\n")

except FileNotFoundError:
    print(f"ERROR: The file was not found at '{file_path}'")
    print("Please make sure the script is in the 'scripts' folder and the CSV is in the project root.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
