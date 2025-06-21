import torch
import pandas as pd
from data.tmdb_integration import TMDbIntegration

# Initialize TMDb integration
TMDB_API_KEY = "c799fe85bcebb074eff49aa01dc6cdb0"
tmdb_integration = TMDbIntegration(TMDB_API_KEY)

# Test rating target generation
content_ids = [f"content_{i}" for i in range(10)]
content_mapping = {f"content_{i}": 550 + i for i in range(10)}

tmdb_data = tmdb_integration.fetch_tmdb_data(content_ids, content_mapping)

print("TMDb Data Sample:")
for content_id, data in list(tmdb_data.items())[:3]:
    print(f"{content_id}: {data}")

# Check rating extraction
def extract_rating_targets(tmdb_data):
    ratings = []
    for content_id, data in tmdb_data.items():
        rating = data.get('rating', 5.0) / 10.0
        ratings.append(rating)
        print(f"{content_id}: rating = {data.get('rating', 'MISSING')} -> normalized = {rating}")
    return torch.tensor(ratings, dtype=torch.float32)

rating_targets = extract_rating_targets(tmdb_data)
print(f"\nRating targets: {rating_targets}")
print(f"Rating targets stats: min={rating_targets.min()}, max={rating_targets.max()}, mean={rating_targets.mean()}")

