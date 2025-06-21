# scripts/visualize_synthetic_dataset.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

def visualize_synthetic_data(file_path: str):
    """
    Generates a suite of visualizations to analyze and validate the synthetic
    Fire TV interaction dataset.
    """
    try:
        print(f"📈 Loading synthetic dataset from '{file_path}'...")
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded with {len(df):,} records and {len(df.columns)} columns.")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{file_path}'. Please provide a valid path.")
        sys.exit(1)

    # --- Create an 'archetype' column for visualization purposes ---
    # This reverse-engineers the archetype from the ground-truth traits
    # to validate our simulation logic.
    def assign_archetype(row):
        if row['frustration_level'] > 0.6:
            return 'Frustrated User'
        elif row['session_engagement_level'] > 0.8:
            return 'Power User'
        else:
            return 'Casual Viewer'
    df['archetype'] = df.apply(assign_archetype, axis=1)

    # --- 1. Archetype Distribution ---
    print("\n[1/4] Generating User Archetype Distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='archetype', data=df, palette='viridis', order=df['archetype'].value_counts().index)
    plt.title('Synthetic User Archetype Distribution', fontsize=16)
    plt.xlabel('User Archetype', fontsize=12)
    plt.ylabel('Number of Interactions', fontsize=12)
    plt.show()

    # --- 2. Key Psychological Traits by Archetype (Box Plots) ---
    print("[2/4] Generating Psychological Trait comparison plots...")
    psych_traits = [
        'session_engagement_level', 'frustration_level', 
        'exploration_tendency_score', 'navigation_efficiency_score',
        'cognitive_load_indicator', 'decision_confidence_score'
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Psychological Trait Comparison by User Archetype', fontsize=20)
    axes = axes.flatten()
    for i, trait in enumerate(psych_traits):
        sns.boxplot(x='archetype', y=trait, data=df, ax=axes[i], palette='mako')
        axes[i].set_title(trait)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Score')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # --- 3. Key Behavioral Features by Archetype (Violin Plots) ---
    print("[3/4] Generating Behavioral Feature comparison plots...")
    behavioral_features = [
        'scroll_speed', 'hover_duration', 
        'back_button_presses', 'dpad_up_count'
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Key Behavioral Feature Comparison by User Archetype', fontsize=20)
    axes = axes.flatten()
    for i, feature in enumerate(behavioral_features):
        sns.violinplot(x='archetype', y=feature, data=df, ax=axes[i], palette='rocket')
        axes[i].set_title(feature)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Value')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 4. Correlation Heatmap ---
    print("[4/4] Generating Correlation Heatmap...")
    correlation_features = psych_traits + ['dpad_down_count', 'back_button_presses', 'scroll_speed']
    corr_matrix = df[correlation_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix of Key Traits and Behaviors', fontsize=16)
    plt.show()

    print("\n🎉 Visualization script finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the synthetic Fire TV dataset.")
    parser.add_argument(
        "--file_path", 
        type=str, 
        default="fire_tv_synthetic_dataset.csv", 
        help="Path to the synthetic dataset CSV file."
    )
    args = parser.parse_args()
    
    visualize_synthetic_data(args.file_path)
