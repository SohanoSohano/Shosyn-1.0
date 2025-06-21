# scripts/generate_synthetic_dataset.py (v3 - TMDb Integrated Hybrid Version)
import pandas as pd
import numpy as np
import random
import time
import requests
import json
import os
from tqdm import tqdm
import argparse
from datetime import datetime, timedelta

# --- CONFIGURATION ---
CONFIG = {
    "num_users": 5000,
    "sessions_per_user": (3, 6),
    "interactions_per_session": (20, 60),
    "output_filename": "fire_tv_synthetic_dataset_v3_tmdb.csv",
    "tmdb_cache_file": "tmdb_local_catalog.json",
    "tmdb_api_key": os.getenv("TMDB_API_KEY", None) # Best practice: use environment variables
}

# --- USER ARCHETYPES WITH GENRE PREFERENCES ---
# These archetypes will now select from a real catalog of TMDb movies.
USER_ARCHETYPES = {
    "Casual Viewer": { "proportion": 0.45, "preferred_genres": ['Comedy', 'Romance', 'Drama', 'Family'], "base_traits": { "session_engagement_level": (0.6, 0.1), "frustration_level": (0.3, 0.1), "exploration_tendency_score": (0.4, 0.15) }},
    "Power User": { "proportion": 0.45, "preferred_genres": ['Science Fiction', 'Thriller', 'Documentary', 'Horror', 'Action', 'Adventure'], "base_traits": { "session_engagement_level": (0.9, 0.05), "frustration_level": (0.1, 0.05), "exploration_tendency_score": (0.8, 0.1) }},
    "Frustrated User": { "proportion": 0.10, "preferred_genres": ['Comedy', 'Drama'], "base_traits": { "session_engagement_level": (0.4, 0.1), "frustration_level": (0.8, 0.1), "exploration_tendency_score": (0.2, 0.1) }}
}

class TMDbIntegratedDataGenerator:
    """Generates a dataset by simulating user behavior against a real TMDb movie catalog."""

    def __init__(self, config, archetypes):
        self.config = config
        self.archetypes = archetypes
        self.tmdb_api_key = self.config['tmdb_api_key']
        if not self.tmdb_api_key:
            raise ValueError("TMDB_API_KEY environment variable not set. Please set it to your API key.")
        
        # This is the core of the hybrid approach: load from cache or fetch from API
        self.content_catalog = self._load_or_fetch_catalog()

    def _fetch_and_cache_tmdb_catalog(self):
        """
        Performs a one-time fetch of movie data from TMDb and saves it to a local cache.
        This avoids hitting API rate limits during generation.
        """
        print("Local TMDb cache not found. Fetching data from TMDb API...")
        
        # Mapping TMDb genre IDs to names
        genre_map = {28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy', 80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family', 14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music', 9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction', 10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'}
        
        catalog = {name: [] for name in genre_map.values()}
        
        # Fetch multiple pages of popular movies to get a diverse set
        for page in tqdm(range(1, 21), desc="Fetching TMDb pages"): # 20 pages * 20 movies = 400 movies
            url = f"https://api.themoviedb.org/3/movie/popular?api_key={self.tmdb_api_key}&language=en-US&page={page}"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                movies = response.json().get('results', [])
                for movie in movies:
                    movie_genres = [genre_map.get(gid) for gid in movie['genre_ids'] if gid in genre_map]
                    if movie_genres:
                        # Add movie to each genre it belongs to
                        for genre in movie_genres:
                             catalog[genre].append({
                                'content_id': f"tmdb_{movie['id']}", # Use a prefix to denote real IDs
                                'title': movie['title'],
                                'content_genre': genre,
                                'release_year': int(movie['release_date'][:4]) if movie.get('release_date') else 2000,
                                'tmdb_popularity': movie['popularity'],
                                'tmdb_vote_average': movie['vote_average']
                            })
            except requests.exceptions.RequestException as e:
                print(f"Warning: Failed to fetch page {page}. Error: {e}")
            time.sleep(0.3) # Respect API rate limits

        with open(self.config['tmdb_cache_file'], 'w') as f:
            json.dump(catalog, f)
        print(f"✅ TMDb catalog cached locally to '{self.config['tmdb_cache_file']}'")
        return catalog

    def _load_or_fetch_catalog(self):
        """Loads the movie catalog from a local JSON file or fetches it if not present."""
        if os.path.exists(self.config['tmdb_cache_file']):
            print(f"Found local TMDb cache. Loading from '{self.config['tmdb_cache_file']}'...")
            with open(self.config['tmdb_cache_file'], 'r') as f:
                return json.load(f)
        else:
            return self._fetch_and_cache_tmdb_catalog()

    # The simulation logic for traits and behaviors remains the same
    def _simulate_session_traits(self, archetype_traits):
        session_traits = {}
        for trait, (mean, std) in archetype_traits.items():
            session_traits[trait] = np.clip(np.random.normal(mean, std), 0, 1)
        return session_traits

    def _simulate_behavioral_features(self, traits):
        # This function can remain the same or be enhanced further.
        # For brevity, it's kept as is from the previous version.
        features = {}
        dpad_nav = traits['session_engagement_level'] * 20 + traits['exploration_tendency_score'] * 15
        features.update({ 'dpad_up_count': int(abs(np.random.normal(dpad_nav / 2, 3))), 'dpad_down_count': int(abs(np.random.normal(dpad_nav / 2, 3))), 'dpad_left_count': int(abs(np.random.normal(dpad_nav / 4, 2))), 'dpad_right_count': int(abs(np.random.normal(dpad_nav / 4, 2))), 'back_button_presses': int(abs(np.random.normal(traits['frustration_level'] * 10, 2))), 'menu_revisits': int(abs(np.random.normal(traits['frustration_level'] * 5, 1))), 'scroll_speed': np.clip(np.random.normal(150 - traits['exploration_tendency_score'] * 50, 10), 50, 200), 'hover_duration': np.clip(np.random.normal(3 - traits['frustration_level'] * 2, 0.5), 0.5, 5), 'time_since_last_interaction': np.clip(np.random.exponential(15 - traits['session_engagement_level'] * 10), 1, 60)})
        return features

    def generate(self):
        print("Generating TMDb-backed synthetic dataset...")
        all_records = []
        user_archetypes = list(self.archetypes.keys())
        user_proportions = [self.archetypes[name]['proportion'] for name in user_archetypes]

        for user_idx in tqdm(range(self.config["num_users"]), desc="Simulating Users"):
            user_id = f"user_{user_idx}"
            archetype_name = np.random.choice(user_archetypes, p=user_proportions)
            archetype = self.archetypes[archetype_name]
            num_sessions = random.randint(*self.config["sessions_per_user"])
            
            for session_idx in range(num_sessions):
                session_id = f"{user_id}_session_{session_idx}"
                session_traits = self._simulate_session_traits(archetype['base_traits'])
                num_interactions = random.randint(*self.config["interactions_per_session"])
                timestamp = datetime.now() - timedelta(days=random.randint(1, 30))

                for _ in range(num_interactions):
                    timestamp += timedelta(seconds=random.randint(5, 120))
                    behavioral_features = self._simulate_behavioral_features(session_traits)
                    
                    # --- CAUSAL CONTENT SELECTION FROM REAL DATA ---
                    chosen_genre = random.choice(archetype['preferred_genres'])
                    if self.content_catalog.get(chosen_genre):
                        content = random.choice(self.content_catalog[chosen_genre])
                    else:
                        # Fallback to a random genre if preferred is unavailable
                        fallback_genre = random.choice(list(self.content_catalog.keys()))
                        content = random.choice(self.content_catalog[fallback_genre])
                    
                    record = { "user_id": user_id, "session_id": session_id, "interaction_timestamp": timestamp.isoformat() }
                    record.update(behavioral_features)
                    record.update(content) # This now contains real TMDb data
                    record.update(session_traits) # This adds the ground truth psychological traits
                    all_records.append(record)

        df = pd.DataFrame(all_records)
        print(f"\nGenerated a dataset with {len(df):,} records.")
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a TMDb-backed synthetic dataset for Fire TV.")
    parser.add_argument("--users", type=int, default=CONFIG["num_users"], help="Number of users.")
    parser.add_argument("--output", type=str, default=CONFIG["output_filename"], help="Output CSV filename.")
    parser.add_argument("--api_key", type=str, default=CONFIG["tmdb_api_key"], help="Your TMDb API key.")
    args = parser.parse_args()

    CONFIG["num_users"] = args.users
    CONFIG["output_filename"] = args.output
    # Prefer command-line key, then environment variable
    if args.api_key:
        CONFIG["tmdb_api_key"] = args.api_key

    start_time = time.time()
    generator = TMDbIntegratedDataGenerator(CONFIG, USER_ARCHETYPES)
    synthetic_df = generator.generate()

    print(f"Saving dataset to '{CONFIG['output_filename']}'...")
    synthetic_df.to_csv(CONFIG['output_filename'], index=False)
    
    end_time = time.time()
    print("\n" + "="*50)
    print("✅ TMDb-Integrated Synthetic Dataset Generation Complete!")
    print(f"   - File saved as: {CONFIG['output_filename']}")
    print(f"   - Total Records: {len(synthetic_df):,}")
    print(f"   - Time Taken:    {end_time - start_time:.2f} seconds")
    print("="*50)
