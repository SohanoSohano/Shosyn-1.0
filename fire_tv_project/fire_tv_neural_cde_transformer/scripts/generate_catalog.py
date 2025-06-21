# scripts/generate_catalog.py

import pandas as pd
import numpy as np
import os

def generate_item_catalog_csv(output_path='data/item_catalog.csv', num_items=1000):
    """
    Generates a detailed item catalog CSV file for the recommendation service.
    This includes item_id, title, genre, and 15 psychological trait features.
    
    Args:
        output_path (str): The path where the CSV file will be saved.
        num_items (int): The number of dummy items to generate.
    """
    print("🔄 Generating item catalog...")
    
    # --- Step 1: Ensure the output directory exists ---
    # This prevents the "Cannot save file into a non-existent directory" error.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"   - Ensured directory '{output_dir}' exists.")

    # Use a fixed seed for reproducibility
    np.random.seed(42)  
    
    # --- Step 2: Define sample data ---
    # Sample genres commonly found in TMDb
    genres = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
        'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller',
        'War', 'Western', 'Biography'
    ]
    
    # --- Step 3: Generate item data ---
    # Generate dummy movie titles
    titles = [f'Movie_{i+1}' for i in range(num_items)]
    
    # Randomly assign genres (1-3 genres per movie)
    assigned_genres = [', '.join(np.random.choice(genres, size=np.random.randint(1, 4), replace=False)) for _ in range(num_items)]
    
    # Generate 15 psychological trait features per item.
    # These represent the "personality" of each movie.
    trait_features = np.random.uniform(-1, 1, size=(num_items, 15))
    
    # --- Step 4: Create a pandas DataFrame ---
    df = pd.DataFrame({
        'item_id': [f'movie_{i+1}' for i in range(num_items)],
        'title': titles,
        'genre': assigned_genres
    })
    
    # Add the 15 trait columns to the DataFrame
    for i in range(15):
        df[f'trait_{i+1}'] = trait_features[:, i]
        
    print(f"   - Generated {num_items} items with 15 trait features each.")
    
    # --- Step 5: Save the DataFrame to a CSV file ---
    df.to_csv(output_path, index=False)
    
    print(f"✅ Item catalog saved successfully to: {output_path}")
    
    return df

if __name__ == "__main__":
    # Execute the function when the script is run
    generate_item_catalog_csv()

