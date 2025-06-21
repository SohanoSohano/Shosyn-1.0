# data/tmdb_integration.py
import requests
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import json
from typing import Dict, List, Optional

# TMDb official genre mapping
TMDB_GENRE_MAPPING = {
    28: 'Action',
    12: 'Adventure', 
    16: 'Animation',
    35: 'Comedy',
    80: 'Crime',
    99: 'Documentary',
    18: 'Drama',
    10751: 'Family',
    14: 'Fantasy',
    36: 'History',
    27: 'Horror',
    10402: 'Music',
    9648: 'Mystery',
    10749: 'Romance',
    878: 'Science Fiction',
    10770: 'TV Movie',
    53: 'Thriller',
    10752: 'War',
    37: 'Western'
}

class TMDbIntegration:
    """
    Handles real TMDb data integration - more accurate and comprehensive than OMDb
    FIXED VERSION: Improved genre extraction and fallback strategies
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tmdb_cache = {}
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        
    def fetch_tmdb_data(self, content_ids: List[str], content_to_tmdb_map: Dict[str, int]) -> Dict:
        """
        Fetch comprehensive TMDb data for given content IDs
        content_to_tmdb_map: dict mapping your content_id to tmdb_id (integer)
        """
        tmdb_data = {}
        
        for content_id in content_ids:
            tmdb_id = content_to_tmdb_map.get(content_id)
            if not tmdb_id:
                tmdb_data[content_id] = self._create_default_data()
                continue
                
            if tmdb_id in self.tmdb_cache:
                tmdb_data[content_id] = self.tmdb_cache[tmdb_id]
                continue
            
            # Fetch movie details, credits, keywords, and reviews
            movie_data = self._fetch_movie_details(tmdb_id)
            credits_data = self._fetch_movie_credits(tmdb_id)
            keywords_data = self._fetch_movie_keywords(tmdb_id)
            reviews_data = self._fetch_movie_reviews(tmdb_id)
            
            if movie_data:
                # Combine all data sources with improved genre handling
                comprehensive_data = self._combine_tmdb_data(
                    movie_data, credits_data, keywords_data, reviews_data
                )
                tmdb_data[content_id] = comprehensive_data
                self.tmdb_cache[tmdb_id] = comprehensive_data
            else:
                tmdb_data[content_id] = self._create_default_data()
            
            # Respect rate limits - TMDb allows 40 requests per 10 seconds
            time.sleep(0.25)  # 4 requests per second to be safe
        
        return tmdb_data
    
    def _fetch_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch detailed movie information from TMDb with enhanced genre handling"""
        url = f"{self.base_url}/movie/{tmdb_id}"
        params = {
            'api_key': self.api_key,
            'language': 'en-US',
            'append_to_response': 'videos,images,external_ids'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Enhanced genre extraction with fallbacks
                genres = self._extract_genres_with_fallbacks(data, tmdb_id)
                data['processed_genres'] = genres
                
                return data
            else:
                print(f"TMDb API error for movie {tmdb_id}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching TMDb movie details for {tmdb_id}: {e}")
            return None
    
    def _extract_genres_with_fallbacks(self, movie_data: Dict, tmdb_id: int) -> List[str]:
        """Extract genres with multiple fallback strategies"""
        genres = []
        
        # Strategy 1: Use TMDb genres (primary)
        if 'genres' in movie_data and movie_data['genres']:
            genres = [genre['name'] for genre in movie_data['genres']]
            if genres and genres != ['Unknown']:
                return genres
        
        # Strategy 2: Fetch keywords and infer genres
        try:
            keywords_data = self._fetch_movie_keywords(tmdb_id)
            if keywords_data:
                inferred_genres = self._infer_genres_from_keywords(keywords_data.get('keywords', []))
                if inferred_genres:
                    return inferred_genres
        except:
            pass
        
        # Strategy 3: Use metadata heuristics
        metadata_genres = self._infer_genres_from_metadata(movie_data)
        if metadata_genres:
            return metadata_genres
        
        # Strategy 4: Use content-based inference
        content_genres = self._infer_genres_from_content(movie_data)
        if content_genres:
            return content_genres
        
        # Final fallback
        return ['Drama']
    
    def _infer_genres_from_keywords(self, keywords: List[Dict]) -> List[str]:
        """Infer genres from movie keywords"""
        if not keywords:
            return []
            
        keyword_to_genre = {
            # Action keywords
            'action': 'Action', 'fight': 'Action', 'martial arts': 'Action', 'superhero': 'Action',
            'chase': 'Action', 'explosion': 'Action', 'gun': 'Action', 'violence': 'Action',
            
            # Adventure keywords  
            'adventure': 'Adventure', 'journey': 'Adventure', 'quest': 'Adventure', 'exploration': 'Adventure',
            'treasure': 'Adventure', 'expedition': 'Adventure',
            
            # Comedy keywords
            'comedy': 'Comedy', 'humor': 'Comedy', 'funny': 'Comedy', 'satire': 'Comedy',
            'parody': 'Comedy', 'slapstick': 'Comedy', 'romantic comedy': 'Comedy',
            
            # Horror keywords
            'horror': 'Horror', 'scary': 'Horror', 'monster': 'Horror', 'ghost': 'Horror',
            'vampire': 'Horror', 'zombie': 'Horror', 'supernatural': 'Horror', 'slasher': 'Horror',
            
            # Romance keywords
            'romance': 'Romance', 'love': 'Romance', 'relationship': 'Romance', 'wedding': 'Romance',
            'romantic': 'Romance', 'love story': 'Romance',
            
            # Thriller keywords
            'thriller': 'Thriller', 'suspense': 'Thriller', 'mystery': 'Mystery', 'detective': 'Mystery',
            'investigation': 'Mystery', 'murder': 'Thriller', 'conspiracy': 'Thriller',
            
            # Crime keywords
            'crime': 'Crime', 'criminal': 'Crime', 'police': 'Crime', 'gangster': 'Crime',
            'heist': 'Crime', 'robbery': 'Crime', 'mafia': 'Crime',
            
            # War keywords
            'war': 'War', 'military': 'War', 'soldier': 'War', 'battle': 'War',
            'world war': 'War', 'combat': 'War',
            
            # Western keywords
            'western': 'Western', 'cowboy': 'Western', 'frontier': 'Western', 'saloon': 'Western',
            
            # Sci-Fi keywords
            'science fiction': 'Science Fiction', 'sci-fi': 'Science Fiction', 'space': 'Science Fiction',
            'alien': 'Science Fiction', 'future': 'Science Fiction', 'robot': 'Science Fiction',
            'time travel': 'Science Fiction', 'dystopia': 'Science Fiction',
            
            # Fantasy keywords
            'fantasy': 'Fantasy', 'magic': 'Fantasy', 'wizard': 'Fantasy', 'dragon': 'Fantasy',
            'fairy tale': 'Fantasy', 'mythology': 'Fantasy',
            
            # Family keywords
            'family': 'Family', 'children': 'Family', 'kids': 'Family', 'disney': 'Family',
            
            # Animation keywords
            'animation': 'Animation', 'animated': 'Animation', 'cartoon': 'Animation',
            
            # Music keywords
            'music': 'Music', 'musical': 'Music', 'singer': 'Music', 'band': 'Music',
            'concert': 'Music', 'song': 'Music'
        }
        
        inferred_genres = []
        keyword_names = [kw.get('name', '').lower() for kw in keywords]
        
        for keyword_name in keyword_names:
            for key_phrase, genre in keyword_to_genre.items():
                if key_phrase in keyword_name:
                    if genre not in inferred_genres:
                        inferred_genres.append(genre)
        
        return inferred_genres[:3]  # Limit to 3 genres
    
    def _infer_genres_from_metadata(self, data: Dict) -> List[str]:
        """Infer genres from movie metadata"""
        genres = []
        
        # Use runtime to infer some genres
        runtime = data.get('runtime', 0)
        if runtime:
            if runtime < 90:
                genres.append('Comedy')  # Short movies often comedies
            elif runtime > 150:
                genres.append('Drama')   # Long movies often dramas
        
        # Use release date for historical content
        release_date = data.get('release_date', '')
        if release_date:
            try:
                year = int(release_date[:4])
                if year < 1970:
                    genres.append('History')
                elif year < 1990:
                    genres.append('Drama')
            except:
                pass
        
        # Use adult flag
        if data.get('adult', False):
            genres.append('Drama')
        
        # Use popularity and vote average
        popularity = data.get('popularity', 0)
        vote_average = data.get('vote_average', 0)
        
        if popularity > 50 and vote_average > 7:
            genres.append('Action')  # Popular high-rated movies often action
        elif vote_average > 8:
            genres.append('Drama')   # High-rated movies often dramas
        
        return genres
    
    def _infer_genres_from_content(self, data: Dict) -> List[str]:
        """Infer genres from overview and title content"""
        genres = []
        
        # Analyze overview text
        overview = data.get('overview', '').lower()
        title = data.get('title', '').lower()
        
        content_text = f"{overview} {title}"
        
        # Content-based genre detection
        genre_keywords = {
            'Action': ['action', 'fight', 'battle', 'war', 'combat', 'violence', 'chase', 'explosion'],
            'Comedy': ['comedy', 'funny', 'humor', 'laugh', 'hilarious', 'comic', 'amusing'],
            'Horror': ['horror', 'scary', 'terror', 'fear', 'nightmare', 'haunted', 'evil', 'monster'],
            'Romance': ['love', 'romance', 'romantic', 'relationship', 'heart', 'passion', 'wedding'],
            'Thriller': ['thriller', 'suspense', 'mystery', 'dangerous', 'tension', 'chase', 'escape'],
            'Drama': ['drama', 'life', 'family', 'emotional', 'personal', 'struggle', 'story'],
            'Crime': ['crime', 'criminal', 'police', 'detective', 'murder', 'investigation', 'law'],
            'Adventure': ['adventure', 'journey', 'quest', 'explore', 'discover', 'travel', 'expedition'],
            'Science Fiction': ['future', 'space', 'alien', 'technology', 'robot', 'time', 'planet'],
            'Fantasy': ['magic', 'fantasy', 'wizard', 'dragon', 'supernatural', 'mystical', 'enchanted']
        }
        
        for genre, keywords in genre_keywords.items():
            if any(keyword in content_text for keyword in keywords):
                genres.append(genre)
        
        return genres[:2]  # Limit to 2 inferred genres
    
    def _fetch_movie_credits(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch cast and crew information"""
        url = f"{self.base_url}/movie/{tmdb_id}/credits"
        params = {'api_key': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def _fetch_movie_keywords(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch movie keywords for better content understanding"""
        url = f"{self.base_url}/movie/{tmdb_id}/keywords"
        params = {'api_key': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def _fetch_movie_reviews(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch movie reviews for sentiment analysis"""
        url = f"{self.base_url}/movie/{tmdb_id}/reviews"
        params = {'api_key': self.api_key, 'language': 'en-US'}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def _combine_tmdb_data(self, movie_data: Dict, credits_data: Dict, 
                          keywords_data: Dict, reviews_data: Dict) -> Dict:
        """Combine all TMDb data sources into comprehensive format with improved genre handling"""
        
        # Extract cast and crew
        cast = credits_data.get('cast', [])[:10]  # Top 10 cast members
        crew = credits_data.get('crew', [])
        directors = [person['name'] for person in crew if person['job'] == 'Director']
        writers = [person['name'] for person in crew if person['job'] in ['Writer', 'Screenplay']]
        
        # Extract keywords
        keywords = [kw['name'] for kw in keywords_data.get('keywords', [])]
        
        # Extract review sentiment (simplified)
        reviews = reviews_data.get('results', [])
        review_texts = [review['content'] for review in reviews[:5]]  # Top 5 reviews
        
        # Use processed genres (with fallbacks) instead of raw genres
        final_genres = movie_data.get('processed_genres', ['Drama'])
        
        return {
            'tmdb_id': movie_data['id'],
            'imdb_id': movie_data.get('external_ids', {}).get('imdb_id', ''),
            'title': movie_data.get('title', ''),
            'original_title': movie_data.get('original_title', ''),
            'overview': movie_data.get('overview', ''),
            'tagline': movie_data.get('tagline', ''),
            'rating': movie_data.get('vote_average', 0.0),
            'vote_count': movie_data.get('vote_count', 0),
            'popularity': movie_data.get('popularity', 0.0),
            'genres': final_genres,  # Use improved genre extraction
            'release_date': movie_data.get('release_date', ''),
            'runtime': movie_data.get('runtime', 0),
            'budget': movie_data.get('budget', 0),
            'revenue': movie_data.get('revenue', 0),
            'original_language': movie_data.get('original_language', ''),
            'production_countries': [country['name'] for country in movie_data.get('production_countries', [])],
            'production_companies': [company['name'] for company in movie_data.get('production_companies', [])],
            'cast': [{'name': actor['name'], 'character': actor.get('character', '')} for actor in cast],
            'directors': directors,
            'writers': writers,
            'keywords': keywords,
            'reviews': review_texts,
            'poster_path': movie_data.get('poster_path', ''),
            'backdrop_path': movie_data.get('backdrop_path', ''),
            'adult': movie_data.get('adult', False),
            'status': movie_data.get('status', ''),
            'homepage': movie_data.get('homepage', '')
        }
    
    def _create_default_data(self) -> Dict:
        """Create default data for content without TMDb mapping - with better genre diversity"""
        # Rotate through different default genres instead of always using 'Unknown'
        default_genres_pool = [
            ['Drama'], ['Comedy'], ['Action'], ['Thriller'], ['Romance'],
            ['Adventure'], ['Crime'], ['Horror'], ['Science Fiction'], ['Fantasy']
        ]
        
        import random
        selected_genres = random.choice(default_genres_pool)
        
        return {
            'tmdb_id': 0,
            'imdb_id': '',
            'title': 'Unknown',
            'original_title': 'Unknown',
            'overview': 'No overview available',
            'tagline': '',
            'rating': 5.0,
            'vote_count': 100,
            'popularity': 1.0,
            'genres': selected_genres,  # Diverse default genres
            'release_date': '2020-01-01',
            'runtime': 90,
            'budget': 0,
            'revenue': 0,
            'original_language': 'en',
            'production_countries': ['United States'],
            'production_companies': ['Unknown'],
            'cast': [],
            'directors': [],
            'writers': [],
            'keywords': [],
            'reviews': [],
            'poster_path': '',
            'backdrop_path': '',
            'adult': False,
            'status': 'Released',
            'homepage': ''
        }
    
    def create_tmdb_features(self, tmdb_data: Dict) -> torch.Tensor:
        """Convert comprehensive TMDb data into rich numerical features"""
        features = []
        
        for content_id, data in tmdb_data.items():
            feature_vector = [
                # Core metrics (7 features)
                data['rating'] / 10.0,  # Normalize rating (0-1)
                min(np.log(data['vote_count'] + 1) / 15.0, 1.0),  # Log-normalized votes
                min(data['popularity'] / 100.0, 1.0),  # Normalized popularity
                len(data['genres']) / 10.0,  # Genre diversity
                min(data['runtime'] / 200.0, 1.0),  # Normalized runtime
                min(len(data['cast']) / 20.0, 1.0),  # Cast size
                min(len(data['keywords']) / 50.0, 1.0),  # Keyword richness
                
                # Financial indicators (3 features)
                min(np.log(data['budget'] + 1) / 25.0, 1.0) if data['budget'] > 0 else 0.0,
                min(np.log(data['revenue'] + 1) / 25.0, 1.0) if data['revenue'] > 0 else 0.0,
                data['revenue'] / (data['budget'] + 1) if data['budget'] > 0 else 0.0,  # ROI
                
                # Content characteristics (5 features)
                1.0 if data['adult'] else 0.0,
                len(data['production_countries']) / 5.0,
                len(data['production_companies']) / 10.0,
                len(data['overview']) / 1000.0 if data['overview'] else 0.0,
                1.0 if data['tagline'] else 0.0,
            ]
            
            # Genre encoding (20 features) - more comprehensive than OMDb
            genre_list = [
                'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
                'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
                'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller',
                'War', 'Western', 'Biography'
            ]
            genre_encoding = [1.0 if genre in data['genres'] else 0.0 for genre in genre_list]
            feature_vector.extend(genre_encoding)
            
            # Language and country features (10 features)
            major_languages = ['en', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'zh', 'hi', 'ru']
            language_encoding = [1.0 if data['original_language'] == lang else 0.0 for lang in major_languages]
            feature_vector.extend(language_encoding)
            
            # Temporal features (5 features)
            if data['release_date']:
                try:
                    year = int(data['release_date'][:4])
                    decade = (year - 1900) // 10
                    feature_vector.extend([
                        (2025 - year) / 50.0,  # Age of content
                        1.0 if year >= 2020 else 0.0,  # Recent release
                        1.0 if year >= 2010 else 0.0,  # Modern era
                        1.0 if year >= 2000 else 0.0,  # Digital era
                        decade / 15.0  # Decade encoding
                    ])
                except:
                    feature_vector.extend([0.5, 0.0, 0.0, 0.0, 0.5])
            else:
                feature_vector.extend([0.5, 0.0, 0.0, 0.0, 0.5])
            
            # Ensure exactly 70 features (richer than OMDb's 50)
            while len(feature_vector) < 70:
                feature_vector.append(0.0)
            
            features.append(feature_vector[:70])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def create_content_embeddings(self, tmdb_data: Dict) -> torch.Tensor:
        """Create rich semantic embeddings from TMDb text data"""
        embeddings = []
        
        for content_id, data in tmdb_data.items():
            # Combine comprehensive text data
            text_parts = []
            
            if data['overview']:
                text_parts.append(f"Overview: {data['overview']}")
            
            if data['tagline']:
                text_parts.append(f"Tagline: {data['tagline']}")
            
            if data['genres']:
                text_parts.append(f"Genres: {', '.join(data['genres'])}")
            
            if data['cast']:
                cast_names = [actor['name'] for actor in data['cast'][:5]]
                text_parts.append(f"Cast: {', '.join(cast_names)}")
            
            if data['directors']:
                text_parts.append(f"Directors: {', '.join(data['directors'])}")
            
            if data['keywords']:
                text_parts.append(f"Keywords: {', '.join(data['keywords'][:10])}")
            
            if data['reviews']:
                # Include sentiment from reviews
                text_parts.append(f"Reviews: {' '.join(data['reviews'][:2])}")
            
            combined_text = ". ".join(text_parts) if text_parts else "No information available"
            
            # Generate embedding
            embedding = self.sentence_model.encode(combined_text)
            embeddings.append(torch.tensor(embedding, dtype=torch.float32))
        
        return torch.stack(embeddings)
    
    def search_tmdb_by_title(self, title: str) -> Optional[int]:
        """Search TMDb by movie title to get TMDb ID"""
        url = f"{self.base_url}/search/movie"
        params = {
            'api_key': self.api_key,
            'query': title,
            'language': 'en-US'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    return results[0]['id']  # Return first match
        except Exception as e:
            print(f"Error searching TMDb for '{title}': {e}")
        
        return None
    
    def save_cache(self, cache_file_path: str):
        """Save TMDb cache to file"""
        try:
            with open(cache_file_path, 'w') as f:
                json.dump(self.tmdb_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving TMDb cache: {e}")
    
    def load_cache(self, cache_file_path: str):
        """Load TMDb cache from file"""
        try:
            with open(cache_file_path, 'r') as f:
                self.tmdb_cache = json.load(f)
        except Exception as e:
            print(f"Error loading TMDb cache: {e}")

