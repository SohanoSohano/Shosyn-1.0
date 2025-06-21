import pandas as pd
import os

def simple_sample_dataset(input_csv_path, output_csv_path, sample_fraction=0.2):
    """Simple sampling with minimal filtering to preserve data"""
    print(f"Starting simple sampling of {input_csv_path}...")
    
    chunk_size = 1000000  # 1M rows per chunk
    sampled_chunks = []
    chunk_count = 0
    
    for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size):
        chunk_count += 1
        print(f"Processing chunk {chunk_count}...")
        
        original_size = len(chunk)
        
        # MINIMAL FILTERING - only remove obvious bad data
        # 1. Remove rows with missing user_id or content_id (critical for training)
        chunk = chunk.dropna(subset=['user_id', 'content_id'])
        
        # 2. Remove exact duplicates only
        chunk = chunk.drop_duplicates()
        
        filtered_size = len(chunk)
        print(f"   Kept {filtered_size:,} out of {original_size:,} rows ({filtered_size/original_size*100:.1f}%)")
        
        # Sample the chunk
        if len(chunk) > 0:
            sampled_chunk = chunk.sample(frac=sample_fraction, random_state=42)
            sampled_chunks.append(sampled_chunk)
            print(f"   Sampled {len(sampled_chunk):,} rows")
        
        # Stop after reasonable amount for 10GB target
        if chunk_count >= 25:
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
    output_path = "/home/ubuntu/fire_tv_data/fire_tv_sampled_10gb.csv"
    
    result = simple_sample_dataset(input_path, output_path, sample_fraction=0.2)
    
    if result:
        print(f"\n🎯 Ready to train with: {result}")
        print("\n🚀 Next steps:")
        print("1. Update training script to use sampled dataset")
        print("2. Add genre balancing and mixed precision")
        print("3. Start training - should be much faster!")
