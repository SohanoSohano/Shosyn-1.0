# analyze_content_data.py
import pandas as pd
import numpy as np
from typing import Dict, List

def analyze_fire_tv_content_chunked(dataset_path: str, chunksize: int = 50000):
    """
    Memory-efficient analysis of Fire TV dataset using chunked reading
    """
    print("📊 Starting Memory-Efficient Content Data Analysis...")
    print(f"Reading dataset in chunks of {chunksize:,} rows")
    
    # Initialize tracking variables
    content_columns = ['content_id', 'content_type', 'content_genre', 'release_year']
    total_rows = 0
    unique_content_ids = set()
    unique_content_types = set()
    unique_content_genres = set()
    unique_release_years = set()
    
    # Sample data for inspection
    sample_data = {}
    
    try:
        # Read dataset in chunks to avoid memory issues
        chunk_iterator = pd.read_csv(dataset_path, chunksize=chunksize, low_memory=False)
        
        for chunk_num, chunk in enumerate(chunk_iterator):
            print(f"Processing chunk {chunk_num + 1}... (rows: {len(chunk):,})")
            
            total_rows += len(chunk)
            
            # Analyze content columns if they exist
            for col in content_columns:
                if col in chunk.columns:
                    if col == 'content_id':
                        unique_content_ids.update(chunk[col].dropna().astype(str))
                    elif col == 'content_type':
                        unique_content_types.update(chunk[col].dropna().astype(str))
                    elif col == 'content_genre':
                        unique_content_genres.update(chunk[col].dropna().astype(str))
                    elif col == 'release_year':
                        unique_release_years.update(chunk[col].dropna().astype(str))
                    
                    # Store sample data from first chunk
                    if chunk_num == 0 and col not in sample_data:
                        sample_values = chunk[col].dropna().unique()[:5]
                        sample_data[col] = sample_values.tolist()
            
            # Process only first few chunks for initial analysis
            if chunk_num >= 10:  # Analyze first 10 chunks (500K rows)
                print("Analyzed first 10 chunks for initial assessment...")
                break
                
    except Exception as e:
        print(f"Error during chunked reading: {e}")
        return None
    
    # Print analysis results
    print("\n" + "="*60)
    print("📊 CONTENT DATA ANALYSIS RESULTS:")
    print("="*60)
    print(f"Total rows analyzed: {total_rows:,}")
    print(f"Unique content IDs: {len(unique_content_ids):,}")
    print(f"Unique content types: {len(unique_content_types):,}")
    print(f"Unique content genres: {len(unique_content_genres):,}")
    print(f"Unique release years: {len(unique_release_years):,}")
    
    print("\n📋 SAMPLE DATA:")
    for col, samples in sample_data.items():
        print(f"{col}: {samples}")
    
    # Create summary for mapping
    content_summary = {
        'total_unique_content': len(unique_content_ids),
        'content_types': list(unique_content_types)[:10],  # First 10
        'content_genres': list(unique_content_genres)[:10],  # First 10
        'release_years': sorted(list(unique_release_years))[:10],  # First 10
        'sample_content_ids': list(unique_content_ids)[:20]  # First 20 for mapping
    }
    
    return content_summary

def create_sample_content_mapping(content_summary: Dict):
    """
    Create a sample content mapping based on analysis results
    """
    if not content_summary:
        print("No content summary available for mapping")
        return
    
    print("\n🎬 Creating Sample Content Mapping...")
    
    # Create sample mapping with realistic TMDb IDs
    sample_content_ids = content_summary.get('sample_content_ids', [])
    
    # Popular movie TMDb IDs for different genres
    popular_tmdb_ids = {
        'action': [550, 155, 27205, 603, 680],  # Fight Club, Dark Knight, Inception, Matrix, Pulp Fiction
        'comedy': [13, 11, 120, 862, 49026],    # Forrest Gump, Star Wars, LOTR, Toy Story, The Dark Knight Rises
        'drama': [238, 424, 389, 278, 240],     # Godfather, Schindler's List, 12 Angry Men, Shawshank, Godfather II
        'horror': [694, 346, 539, 1724, 4922],  # The Shining, Seven, Psycho, The Exorcist, Halloween
        'sci-fi': [19995, 157336, 62, 1726, 76341] # Avatar, Interstellar, 2001, Iron Man, Mad Max
    }
    
    mapping_data = []
    
    for i, content_id in enumerate(sample_content_ids[:100]):  # Limit to 100 for sample
        # Assign TMDb ID based on pattern or randomly
        genre_keys = list(popular_tmdb_ids.keys())
        selected_genre = genre_keys[i % len(genre_keys)]
        tmdb_ids = popular_tmdb_ids[selected_genre]
        selected_tmdb_id = tmdb_ids[i % len(tmdb_ids)]
        
        mapping_data.append({
            'content_id': content_id,
            'tmdb_id': selected_tmdb_id,
            'suggested_genre': selected_genre
        })
    
    # Save sample mapping
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv('sample_content_to_tmdb_mapping.csv', index=False)
    
    print(f"✅ Created sample mapping with {len(mapping_data)} entries")
    print("📁 Saved as: sample_content_to_tmdb_mapping.csv")
    print("\n💡 Next steps:")
    print("1. Review the sample mapping file")
    print("2. Manually enhance mappings for better accuracy")
    print("3. Use this file for TMDb-enhanced training")

def main():
    """Main function with error handling"""
    dataset_path = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\fire_tv_production_dataset_parallel.csv"
    
    try:
        # Analyze content with chunked reading
        content_summary = analyze_fire_tv_content_chunked(dataset_path, chunksize=50000)
        
        if content_summary:
            # Create sample mapping
            create_sample_content_mapping(content_summary)
            
            print("\n🎯 RECOMMENDATIONS:")
            print("- Your dataset is very large (51GB)")
            print("- Use the sample mapping to start training immediately")
            print("- Consider creating more accurate mappings for production use")
            print("- The chunked analysis approach will work for your training pipeline")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Ensure you have enough disk space")
        print("2. Try reducing chunksize to 10000 if still getting memory errors")
        print("3. Close other applications to free up memory")

if __name__ == "__main__":
    main()
