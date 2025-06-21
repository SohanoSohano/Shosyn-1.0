# scripts/diagnose_engagement_distribution.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

try:
    # Use the correct file path you provided.
    file_path = 'C:\\Users\\solos\\OneDrive\\Documents\\College\\Projects\\Advanced Behavioural Analysis for Content Recommendation\\fire_tv_production_dataset_parallel.csv'
    
    print(f"Loading data sample from: {file_path}")
    df_sample = pd.read_csv(file_path, usecols=['session_engagement_level'], nrows=1_000_000)
    print("Data loaded.")

    print("\n" + "="*50)
    print("--- DIAGNOSTIC REPORT: session_engagement_level ---")
    print("="*50)

    # --- 1. Statistical Summary ---
    print("\n[1] Statistical Summary:")
    # The .describe() method is the most powerful tool for this.
    # It will show us the min, max, mean, and key percentiles.
    stats = df_sample['session_engagement_level'].describe()
    print(stats)

    # --- 2. Check for Unique Values ---
    unique_values_count = df_sample['session_engagement_level'].nunique()
    print(f"\n[2] Number of Unique Values: {unique_values_count}")
    if unique_values_count < 10:
        print(f"   Unique Values: {df_sample['session_engagement_level'].unique()}")

    # --- 3. Generate Distribution Plot ---
    print("\n[3] Generating Distribution Plot...")
    plt.figure(figsize=(12, 7))
    sns.histplot(df_sample['session_engagement_level'], bins=50, kde=True)
    plt.title('Distribution of Session Engagement Level', fontsize=16)
    plt.xlabel('Session Engagement Level', fontsize=12)
    plt.ylabel('Frequency (Number of Interactions)', fontsize=12)
    
    output_filename = 'session_engagement_level_distribution.png'
    plt.savefig(output_filename)
    print(f"\n✅ Diagnostic plot saved to '{output_filename}'")
    print("   Please open this image file to visually inspect the data distribution.")

    print("\n" + "="*50)
    print("--- ANALYSIS COMPLETE ---")
    print("="*50)


except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at '{file_path}'")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
