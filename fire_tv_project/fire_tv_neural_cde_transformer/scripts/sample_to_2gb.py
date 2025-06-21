import pandas as pd
import os

def sample_fire_tv_dataset_minimal(input_csv_path, output_csv_path, sample_fraction=0.04):
    """Minimal filtering to preserve data while sampling to ~2GB"""
    print(f"Starting to sample {input_csv_path} to approximately 2GB...")
    
    chunk_size = 10**6  # 1 million rows per chunk
    sampled_chunks = []
    chunk_count = 0
    total_original_rows = 0
    total_kept_rows = 0
    
    for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size):
        chunk_count += 1
        print(f"Processing chunk {chunk_count}...")
        
        original_size = len(chunk)
        total_original_rows += original_size
        
        # MINIMAL FILTERING - only remove obvious bad data
        # 1. Remove rows with missing user_id or content_id (critical for training)
        chunk = chunk.dropna(subset=['user_id', 'content_id'])
        
        # 2. Remove exact duplicates only
        chunk = chunk.drop_duplicates()
        
        filtered_size = len(chunk)
        total_kept_rows += filtered_size
        
        print(f"   Kept {filtered_size:,} out of {original_size:,} rows ({filtered_size/original_size*100:.1f}%)")
        
        # Sample the chunk
        if len(chunk) > 0:
            sampled_chunk = chunk.sample(frac=sample_fraction, random_state=42)
            sampled_chunks.append(sampled_chunk)
            print(f"   Sampled {len(sampled_chunk):,} rows")
        
        # Stop after reasonable amount for 2GB target
        if chunk_count >= 25:
            print(f"Processed {chunk_count} chunks, stopping to target ~2GB")
            break
    
    if not sampled_chunks:
        print("❌ No data survived sampling!")
        return None
    
    # Combine chunks
    print("Combining sampled chunks...")
    sampled_df = pd.concat(sampled_chunks, ignore_index=True)
    
    # Save result
    sampled_df.to_csv(output_csv_path, index=False)
    
    # Report results
    try:
        original_size_gb = os.path.getsize(input_csv_path) / (1024**3)
        new_size_gb = os.path.getsize(output_csv_path) / (1024**3)
        
        print(f"\n✅ Sampling complete!")
        print(f"   Original: {original_size_gb:.1f}GB")
        print(f"   Sampled: {new_size_gb:.1f}GB ({len(sampled_df):,} rows)")
        print(f"   Reduction: {(1-new_size_gb/original_size_gb)*100:.1f}%")
        print(f"   Users: {sampled_df['user_id'].nunique():,}")
        print(f"   Content: {sampled_df['content_id'].nunique():,}")
        
        return output_csv_path
    except Exception as e:
        print(f"Error: {e}")
        return output_csv_path

if __name__ == "__main__":
    input_path = "/home/ubuntu/fire_tv_data/fire_tv_production_dataset_parallel.csv"
    output_path = "/home/ubuntu/fire_tv_data/fire_tv_sampled_2gb.csv"
    
    sample_fire_tv_dataset_minimal(input_path, output_path, sample_fraction=0.04)
