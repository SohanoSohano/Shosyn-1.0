# inference/data_enricher.py
import requests
from typing import Dict, Optional
import os  # Import os module to get environment variables

class TMDbDataEnricher:
    """
    A service to enrich movie recommendations with data from The Movie Database (TMDb).
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("TMDb API key is required. It was not provided during initialization.")
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        
        print(f"✅ TMDbDataEnricher initialized. Using API Key (last 4 chars): ...{self.api_key[-4:]}")

    def get_movie_details(self, movie_id: str) -> Optional[Dict]:
        """
        Fetches detailed movie information and streaming providers from TMDb.
        
        Args:
            movie_id: The movie ID (we assume it's the TMDb ID).
        
        Returns:
            A dictionary with enriched data, or None if the movie is not found.
        """
        # The movie ID from your catalog might be 'movie_63'. We need to extract the number.
        try:
            tmdb_id = int(movie_id.split('_')[-1])
        except (ValueError, IndexError):
            print(f"❌ ERROR: Could not parse TMDb ID from '{movie_id}'. Expected format 'movie_ID_NUMBER'.")
            return None

        # --- DEBUGGING START ---
        print(f"\n🚀 Enriching item: {movie_id} (Extracted TMDb ID: {tmdb_id})")
        # Construct the URL, but redact the full API key for logging
        url = f"{self.base_url}/movie/{tmdb_id}?api_key={self.api_key}&append_to_response=watch/providers"
        # Print the URL without the full key for security and clarity
        print(f"   - Requesting URL: {self.base_url}/movie/{tmdb_id}?api_key=...REDACTED...&append_to_response=watch/providers")
        # --- DEBUGGING END ---
        
        try:
            response = requests.get(url, timeout=10)
            
            # --- DEBUGGING START ---
            print(f"   - TMDb API Response Status Code: {response.status_code}")
            # --- DEBUGGING END ---

            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            data = response.json()

            # --- DEBUGGING START ---
            print(f"   - ✅ Successfully fetched data from TMDb for title: '{data.get('title', 'N/A')}'")
            # --- DEBUGGING END ---

            # --- Parse Streaming Providers ---
            # We'll focus on US streaming platforms for this example.
            # Check the structure: data -> watch/providers -> results -> US
            providers_us = data.get("watch/providers", {}).get("results", {}).get("US", {})
            flatrate_providers = providers_us.get("flatrate", [])  # 'flatrate' usually means subscription services
            
            streaming_platforms = [p['provider_name'] for p in flatrate_providers] if flatrate_providers else ["N/A"]

            # --- DEBUGGING START ---
            if "US" in data.get("watch/providers", {}).get("results", {}):
                print(f"   - Found streaming providers for US.")
                print(f"   - Flatrate providers: {streaming_platforms}")
            else:
                print(f"   - No US streaming providers found in TMDb response for {tmdb_id}.")
            # --- DEBUGGING END ---

            # --- Format the Enriched Data ---
            enriched_data = {
                "title": data.get("title", "Unknown Title"),
                "tagline": data.get("tagline"),
                "overview": data.get("overview"),
                "release_date": data.get("release_date"),
                "runtime": data.get("runtime"),
                "tmdb_rating": data.get("vote_average"),
                "poster_url": f"{self.image_base_url}{data['poster_path']}" if data.get("poster_path") else None,
                "backdrop_url": f"{self.image_base_url}{data['backdrop_path']}" if data.get("backdrop_path") else None,
                "streaming_on": streaming_platforms,  # Include streaming platforms
                "tmdb_id": tmdb_id  # Also include the actual tmdb_id for reference
            }
            return enriched_data

        except requests.exceptions.HTTPError as e:
            # --- DEBUGGING START ---
            print(f"   - ❌ HTTP ERROR fetching TMDb data for ID {tmdb_id}: {e}")
            if e.response.status_code == 401:
                print(f"   - CRITICAL: Status 401 Unauthorized. Your TMDb API key '{self.api_key[-4:]}' is likely invalid or missing API access.")
            elif e.response.status_code == 404:
                print(f"   - INFO: Status 404 Not Found. Movie with TMDb ID {tmdb_id} does not exist in TMDb's database.")
            elif e.response.status_code >= 500:
                print(f"   - ERROR: TMDb Server Error ({e.response.status_code}). Please try again later.")
            # --- DEBUGGING END ---
            return None
        except requests.exceptions.Timeout:
            print(f"   - ❌ REQUEST TIMEOUT for TMDb ID {tmdb_id}. Check network connection or increase timeout.")
            return None
        except requests.exceptions.ConnectionError:
            print(f"   - ❌ CONNECTION ERROR for TMDb ID {tmdb_id}. Check internet connection.")
            return None
        except ValueError as e:
            print(f"   - ❌ VALUE ERROR: {e}")
            return None
        except Exception as e:
            print(f"   - ❌ UNEXPECTED ERROR fetching data for TMDb ID {tmdb_id}: {type(e).__name__} - {e}")
            return None
