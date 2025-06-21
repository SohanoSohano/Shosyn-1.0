# scripts/analyze_tmdb_dataset.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os

class DatasetAnalyzer:
    """A tool to analyze and summarize the TMDb-integrated synthetic dataset."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self._load_data()
        if self.df is not None:
            self._prepare_data()

    def _load_data(self) -> pd.DataFrame | None:
        """Loads the dataset from the specified file path."""
        try:
            print(f"📈 Loading synthetic dataset from '{self.file_path}'...")
            df = pd.read_csv(self.file_path, low_memory=False)
            print(f"✅ Dataset loaded with {len(df):,} records and {len(df.columns)} columns.")
            return df
        except FileNotFoundError:
            print(f"❌ ERROR: File not found at '{self.file_path}'. Please provide a valid path.")
            return None

    def _prepare_data(self):
        """Creates the 'archetype' column needed for grouped analysis."""
        print("   Preparing data for analysis by assigning user archetypes...")
        def assign_archetype(row):
            if row['frustration_level'] > 0.6:
                return 'Frustrated User'
            elif row['session_engagement_level'] > 0.8:
                return 'Power User'
            else:
                return 'Casual Viewer'
        self.df['archetype'] = self.df.apply(assign_archetype, axis=1)

    def generate_summary_report(self):
        """Prints a high-level text summary of the dataset."""
        if self.df is None: return
        
        print("\n" + "="*50)
        print("--- DATASET SUMMARY REPORT ---")
        print("="*50)
        
        unique_users = self.df['user_id'].nunique()
        unique_content = self.df['content_id'].nunique()
        
        print(f"Total Records:        {len(self.df):,}")
        print(f"Unique Users:         {unique_users:,}")
        print(f"Unique Content Items: {unique_content:,} (from TMDb)")
        
        print("\n[1] User Archetype Distribution:")
        archetype_dist = self.df['archetype'].value_counts(normalize=True) * 100
        for name, pct in archetype_dist.items():
            print(f"  - {name:<15}: {pct:.2f}%")
        
        print("\n[2] Content Catalog Overview:")
        content_df = self.df.drop_duplicates(subset=['content_id'])
        print(f"  Genre Diversity:    {content_df['content_genre'].nunique()} unique genres")
        print(f"  Release Year Range: {int(content_df['release_year'].min())} - {int(content_df['release_year'].max())}")
        print(f"  TMDb Popularity:    Avg {content_df['tmdb_popularity'].mean():.2f}, Max {content_df['tmdb_popularity'].max():.2f}")
        print("="*50 + "\n")

    def visualize_psychological_profiles(self, save_path="psychological_profiles.png"):
        """Visualizes the distinct psychological traits of each archetype."""
        if self.df is None: return

        print(f"📊 Generating Psychological Profiles visualization... saving to '{save_path}'")
        traits_to_plot = ['session_engagement_level', 'frustration_level', 'exploration_tendency_score']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Psychological Trait Validation by User Archetype', fontsize=18, weight='bold')
        
        for i, trait in enumerate(traits_to_plot):
            sns.boxplot(x='archetype', y=trait, data=self.df, ax=axes[i], palette='viridis')
            axes[i].set_title(f'Distribution of\n"{trait}"', fontsize=14)
            axes[i].set_xlabel("User Archetype", fontsize=12)
            axes[i].set_ylabel("Score", fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
        plt.close()

    def visualize_genre_affinity(self, save_path="genre_affinity_heatmap.png"):
        """
        Visualizes the core enhancement: which users interact with which genres.
        This confirms the causal link needed for training a recommendation model.
        """
        if self.df is None: return
        
        print(f"📊 Generating Genre Affinity Heatmap... saving to '{save_path}'")
        affinity_data = self.df.groupby('archetype')['content_genre'].value_counts(normalize=True).unstack()
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(affinity_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Genre Interaction Affinity by User Archetype', fontsize=18, weight='bold')
        plt.xlabel('Content Genre', fontsize=12)
        plt.ylabel('User Archetype', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.savefig(save_path)
        plt.close()
        
    def run_full_analysis(self):
        """Executes all analysis and visualization steps."""
        if self.df is None:
            print("Analysis cannot proceed as data failed to load.")
            return
            
        self.generate_summary_report()
        self.visualize_psychological_profiles()
        self.visualize_genre_affinity()
        print("\n🎉 Analysis complete! Check the console report and the saved PNG files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and summarize the TMDb-integrated synthetic dataset.")
    parser.add_argument(
        "--file_path", 
        type=str, 
        default=r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\fire_tv_neural_cde_transformer_instance_version\Shosyn-1.0\fire_tv_project\fire_tv_neural_cde_transformer\fire_tv_synthetic_dataset_v3_tmdb.csv", 
        help="Path to the synthetic dataset CSV file."
    )
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer(file_path=args.file_path)
    analyzer.run_full_analysis()
