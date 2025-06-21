import torch
from collections import Counter
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.tmdb_integration import TMDbIntegration

# Initialize TMDb integration
TMDB_API_KEY = "c799fe85bcebb074eff49aa01dc6cdb0"
tmdb_integration = TMDbIntegration(TMDB_API_KEY)

# Define content_ids and content_mapping for testing
content_ids = [f"content_{i}" for i in range(10)]
content_mapping = {f"content_{i}": 550 + i for i in range(10)}

# Fetch TMDb data FIRST
print("Fetching TMDb data...")
tmdb_data = tmdb_integration.fetch_tmdb_data(content_ids, content_mapping)

def extract_genre_targets(tmdb_data):
    genre_list = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
        'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 
        'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
        'TV Movie', 'Thriller', 'War', 'Western', 'Biography'
    ]
    
    genre_targets = []
    all_genres = []
    
    for content_id, data in tmdb_data.items():
        content_genres = data.get('genres', [])
        all_genres.extend(content_genres)
        
        genre_vector = [1.0 if genre in content_genres else 0.0 for genre in genre_list]
        genre_targets.append(genre_vector)
        
        print(f"{content_id}: genres = {content_genres}")
        print(f"  Vector: {genre_vector}")
    
    print(f"\nGenre distribution: {Counter(all_genres)}")
    
    genre_tensor = torch.tensor(genre_targets, dtype=torch.float32)
    print(f"Genre targets shape: {genre_tensor.shape}")
    print(f"Genre targets sum per sample: {genre_tensor.sum(dim=1)}")
    
    return genre_tensor

# Now call the function with the defined tmdb_data
print("\nAnalyzing genre targets...")
genre_targets = extract_genre_targets(tmdb_data)

