import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import sys
from tqdm import tqdm
# MODIFICATION: Import the 'ast' module for safer, more flexible parsing
import ast

# --- Configuration ---
# Set a non-interactive backend for Matplotlib to work on servers without a display
import matplotlib
matplotlib.use('Agg')
sns.set_theme(style="whitegrid")

def load_dataset(file_path):
    """Loads and preprocesses the enriched dataset from a CSV file."""
    print(f"Loading dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: The dataset file was not found at '{file_path}'.")
        print("Please provide the correct path to your enriched log file.")
        return None
    
    # Parse JSON-like string columns back into Python objects
    for col in ['focused_item', 'sequence_context']:
        if col in df.columns:
            # MODIFICATION: Use ast.literal_eval instead of json.loads for robustness.
            # This correctly handles strings that might use single quotes.
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else {})
            
    # Extract persona name for grouping and analysis
    df['persona'] = df['user_id'].apply(lambda x: '_'.join(x.split('_')[1:-1]) if isinstance(x, str) else 'unknown')
    
    return df

def analyze_and_plot(df):
    """
    Performs a comprehensive analysis of the dataset, prints a report, 
    and generates plots.
    """
    if df is None:
        return

    # --- Test 1: Dataset Overview ---
    print("\n" + "="*25 + " 1. DATASET OVERVIEW " + "="*25)
    print(f"Total Events: {len(df)}")
    print(f"Unique Sessions: {df['session_id'].nunique()}")
    print(f"Unique Users: {df['user_id'].nunique()}")
    print(f"Unique Personas Sampled: {df['persona'].nunique()}")
    print(f"Date Range: {pd.to_datetime(df['timestamp']).min()} to {pd.to_datetime(df['timestamp']).max()}")

    # --- Test 2: Action Type Distribution ---
    print("\n" + "="*25 + " 2. ACTION TYPE DISTRIBUTION " + "="*25)
    action_counts = df['action_type'].value_counts()
    print(action_counts)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=action_counts.index, y=action_counts.values, palette="viridis")
    plt.title('Action Type Distribution Across All Sessions', fontsize=16)
    plt.xlabel('Action Type', fontsize=12)
    plt.ylabel('Total Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plot_1_action_distribution.png')
    print("Saved plot: plot_1_action_distribution.png")

    # --- Test 3: Session Length & Diversity ---
    print("\n" + "="*25 + " 3. SESSION LENGTH & DIVERSITY " + "="*25)
    session_lengths = df.groupby('session_id').size()
    print("Session Length Statistics:")
    print(session_lengths.describe())
    
    plt.figure(figsize=(12, 6))
    sns.histplot(session_lengths, bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Session Lengths (Events per Session)', fontsize=16)
    plt.xlabel('Number of Events', fontsize=12)
    plt.ylabel('Number of Sessions', fontsize=12)
    plt.tight_layout()
    plt.savefig('plot_2_session_lengths.png')
    print("Saved plot: plot_2_session_lengths.png")

    # --- Test 4: Screen Context Transitions ---
    print("\n" + "="*25 + " 4. SCREEN CONTEXT TRANSITIONS " + "="*25)
    df_sorted = df.sort_values(['session_id', 'timestamp'])
    df_sorted['next_screen_context'] = df_sorted.groupby('session_id')['screen_context'].shift(-1)
    transitions = df_sorted.groupby(['screen_context', 'next_screen_context']).size().unstack(fill_value=0)
    transition_prob = transitions.div(transitions.sum(axis=1), axis=0)
    print("Transition Probability Matrix:")
    print(transition_prob)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_prob, annot=True, fmt=".2f", cmap="Blues", linewidths=.5)
    plt.title('Screen Context Transition Probability Matrix', fontsize=16)
    plt.xlabel('Next Screen', fontsize=12)
    plt.ylabel('Current Screen', fontsize=12)
    plt.tight_layout()
    plt.savefig('plot_3_screen_transitions.png')
    print("Saved plot: plot_3_screen_transitions.png")

    # --- Test 5: Behavioral Pattern Analysis (Consecutive Clicks) ---
    print("\n" + "="*25 + " 5. BEHAVIORAL PATTERN ANALYSIS " + "="*25)
    df['consecutive_action_count'] = df['sequence_context'].apply(lambda x: x.get('consecutive_action_count', 0))
    click_consecutive = df[df['action_type'] == 'click']['consecutive_action_count']
    print("Consecutive 'click' action statistics:")
    print(click_consecutive.describe())
    if not click_consecutive.empty and click_consecutive.max() > 3:
        print("WARNING: Click loop detected! Max consecutive clicks > 3.")
    else:
        print("OK: No significant click loops detected (max consecutive clicks <= 3).")

    # --- Test 6: Persona-Driven Validation ---
    print("\n" + "="*25 + " 6. PERSONA-DRIVEN VALIDATION " + "="*25)
    avg_session_length = df.groupby('persona').apply(lambda x: x['session_id'].nunique() and len(x) / x['session_id'].nunique()).sort_values(ascending=False)
    print("Average Session Length per Persona (Top 5):")
    print(avg_session_length.head())
    
    avg_frustration = df.groupby('persona')['frustration_level'].mean().sort_values(ascending=False)
    print("\nAverage Frustration Level per Persona (Top 5):")
    print(avg_frustration.head())

    avg_cog_load = df.groupby('persona')['cognitive_load'].mean().sort_values(ascending=False)
    print("\nAverage Cognitive Load per Persona (Top 5):")
    print(avg_cog_load.head())

    plt.figure(figsize=(12, 6))
    avg_frustration.head(10).plot(kind='bar', color='salmon', alpha=0.7)
    plt.title('Top 10 Personas by Average Frustration Level', fontsize=16)
    plt.ylabel('Average Frustration', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('plot_4_persona_frustration.png')
    print("Saved plot: plot_4_persona_frustration.png")
    
    # --- Test 7: Derived State Analysis ---
    print("\n" + "="*25 + " 7. DERIVED STATE ANALYSIS " + "="*25)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['frustration_level'], color="red", label='Frustration', kde=True, stat="density", linewidth=0)
    sns.histplot(df['cognitive_load'], color="blue", label='Cognitive Load', kde=True, stat="density", linewidth=0)
    plt.title('Distribution of Derived Psychological States', fontsize=16)
    plt.xlabel('Level', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_5_derived_states_dist.png')
    print("Saved plot: plot_5_derived_states_dist.png")

    # --- Test 8: Scroll Behavior ---
    print("\n" + "="*25 + " 8. SCROLL BEHAVIOR ANALYSIS " + "="*25)
    scroll_df = df.dropna(subset=['scroll_speed', 'scroll_depth'])
    print(f"Total identified scroll events: {len(scroll_df)}")
    if not scroll_df.empty:
        print("Scroll statistics:")
        print(scroll_df[['scroll_speed', 'scroll_depth']].describe())
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=scroll_df, x='scroll_depth', y='scroll_speed', hue='persona', alpha=0.6, legend=False)
        plt.title('Scroll Speed vs. Scroll Depth', fontsize=16)
        plt.xlabel('Scroll Depth (Number of Items)', fontsize=12)
        plt.ylabel('Scroll Speed (Items per Second)', fontsize=12)
        plt.tight_layout()
        plt.savefig('plot_6_scroll_behavior.png')
        print("Saved plot: plot_6_scroll_behavior.png")
    else:
        print("No scroll events were identified in this dataset.")
        
    # --- Final Quality Score ---
    print("\n" + "="*25 + " 9. FINAL QUALITY SCORE " + "="*25)
    score = 0
    # Base score for valid file
    score += 10 if len(df) > 1000 else 5
    
    # Schema score
    required_cols = ['timestamp', 'session_id', 'user_id', 'action_type', 'frustration_level', 'cognitive_load']
    score += 20 if all(col in df.columns for col in required_cols) else 0
    
    # Diversity score
    score += 10 if df['session_id'].nunique() > 100 else 5
    score += 10 if df['persona'].nunique() >= 15 else 5
    
    # Behavioral Realism score
    action_dist = df['action_type'].value_counts(normalize=True)
    score += 15 if not action_dist.empty and all(action_dist < 0.7) else 5 # No single action dominates
    
    session_stats = df.groupby('session_id').size().describe()
    score += 15 if session_stats['mean'] > 20 and session_stats['std'] > 10 else 5 # Checks for non-trivial, varied sessions
    
    # Derived Attribute Quality score
    score += 10 if df['frustration_level'].mean() > 0.01 else 2 # Checks if frustration is being triggered
    score += 10 if df['cognitive_load'].mean() > 0.1 else 2 # Checks if cognitive load is dynamic
    
    print(f"Heuristic Dataset Quality Score: {score}/100")
    if score >= 80:
        print("Result: EXCELLENT. This dataset appears diverse, realistic, and ready for training.")
    elif score >= 60:
        print("Result: GOOD. The dataset is solid, but could benefit from more diversity or behavioral realism.")
    else:
        print("Result: NEEDS REVIEW. The dataset may have structural issues or lack sufficient realism.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_dataset.py <path_to_enriched_log_file.csv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    dataset = load_dataset(file_path)
    analyze_and_plot(dataset)

if __name__ == "__main__":
    main()
