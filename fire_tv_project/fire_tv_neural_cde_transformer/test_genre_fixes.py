import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.tmdb_integration import TMDbIntegration

# Test the improved genre extraction
TMDB_API_KEY = "c799fe85bcebb074eff49aa01dc6cdb0"
tmdb_integration = TMDbIntegration(TMDB_API_KEY)

# Test with known movies
test_movies = {
    'content_0': 550,   # Fight Club
    'content_1': 13,    # Forrest Gump  
    'content_2': 680,   # Pulp Fiction
    'content_3': 155,   # The Dark Knight
    'content_4': 27205  # Inception
}

print("Testing improved genre extraction...")
for content_id, tmdb_id in test_movies.items():
    data = tmdb_integration._fetch_movie_details(tmdb_id)
    genres = data.get('genres', [])
    print(f"{content_id} (TMDb {tmdb_id}): {genres}")

print("\n✅ Genre extraction test completed!")

