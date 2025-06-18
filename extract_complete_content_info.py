# extract_complete_content_info.py
import pandas as pd
import json
import ast

def extract_complete_content_info(dataset_path: str):
    """Extract all unique content with complete metadata for mapping"""
    
    print("📊 Extracting complete content information...")
    
    # Read dataset in chunks to get all unique content
    all_content = []
    chunk_size = 50000
    
    try:
        chunk_iterator = pd.read_csv(dataset_path, chunksize=chunk_size, low_memory=False)
        
        for chunk_num, chunk in enumerate(chunk_iterator):
            print(f"Processing chunk {chunk_num + 1}...")
            
            # Extract content info
            content_chunk = chunk[['content_id', 'content_type', 'content_genre', 'release_year']].drop_duplicates()
            all_content.append(content_chunk)
            
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None
    
    # Combine all chunks
    complete_content_df = pd.concat(all_content, ignore_index=True).drop_duplicates()
    
    print(f"✅ Extracted {len(complete_content_df)} unique content items")
    
    # Parse genre information
    complete_content_df['parsed_genres'] = complete_content_df['content_genre'].apply(parse_genre_string)
    complete_content_df['primary_genre'] = complete_content_df['parsed_genres'].apply(get_primary_genre)
    complete_content_df['genre_count'] = complete_content_df['parsed_genres'].apply(len)
    
    # Clean and standardize data
    complete_content_df['content_type'] = complete_content_df['content_type'].str.lower().str.strip()
    complete_content_df['release_year'] = pd.to_numeric(complete_content_df['release_year'], errors='coerce').fillna(2020).astype(int)
    
    # Save complete content info
    complete_content_df.to_csv('complete_content_info.csv', index=False)
    
    print("📁 Saved complete content info to: complete_content_info.csv")
    return complete_content_df

def parse_genre_string(genre_str):
    """Parse genre string like '["Sci-Fi", "Romance", "Thriller"]' into list"""
    try:
        if pd.isna(genre_str) or genre_str == '':
            return ['Unknown']
        
        # Handle JSON-like format
        if genre_str.startswith('[') and genre_str.endswith(']'):
            return ast.literal_eval(genre_str)
        else:
            # Handle comma-separated format
            return [g.strip().strip('"\'') for g in genre_str.split(',')]
    except:
        return ['Unknown']

def get_primary_genre(genre_list):
    """Get primary genre from list"""
    if not genre_list or len(genre_list) == 0:
        return 'Unknown'
    return genre_list[0]

# Run extraction
if __name__ == "__main__":
    dataset_path = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\fire_tv_production_dataset_parallel.csv"
    content_df = extract_complete_content_info(dataset_path)
