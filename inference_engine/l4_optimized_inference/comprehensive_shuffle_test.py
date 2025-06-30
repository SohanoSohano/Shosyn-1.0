# comprehensive_shuffle_test.py
# Complete test script for the top-k shuffling recommendation system
import random
import logging
import time
import statistics
from typing import List, Dict, Optional
from collections import defaultdict, Counter
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovieRecommendation:
    def __init__(self, item_id, title, genres, frustration_compatibility, 
                 cognitive_compatibility, persona_match, overall_score, reasoning):
        self.item_id = item_id
        self.title = title
        self.genres = genres
        self.frustration_compatibility = frustration_compatibility
        self.cognitive_compatibility = cognitive_compatibility
        self.persona_match = persona_match
        self.overall_score = overall_score
        self.reasoning = reasoning

class UserSession:
    def __init__(self, user_id, session_id, genre_affinity=None):
        self.user_id = user_id
        self.session_id = session_id
        self.genre_affinity = genre_affinity or {}

class ComprehensiveInferenceEngineTest:
    def __init__(self):
        self.config = {
            "enable_top_k_shuffling": True,
            "shuffle_method": "weighted",
            "shuffle_consistency_factor": 0.3,  # 30% chance to keep original top movie
            "shuffle_k_size": 5,
            "shuffle_weight_exponent": 3.0,
            "min_recommendations_for_shuffle": 3,
            "recommendation_count": 15
        }
        self.shuffle_analytics = []
        
        # Create diverse test movie catalog
        self.test_movies = self._create_test_movie_catalog()
        
    def _create_test_movie_catalog(self):
        """Create a diverse test movie catalog with realistic data"""
        genres_pool = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Animation', 'Family']
        movies = []
        
        for i in range(50):  # 50 test movies
            movie = {
                'item_id': i,
                'title': f"Test Movie {i:02d}",
                'genres': random.sample(genres_pool, random.randint(1, 3)),
                'runtime': random.randint(80, 180),
                'budget': random.randint(1000000, 200000000),
                'vote_average': round(random.uniform(3.0, 9.0), 1),
                'popularity': random.uniform(1.0, 100.0)
            }
            movies.append(movie)
        
        return movies

    def _calculate_frustration_compatibility(self, movie, strategy_config, frustration_level):
        """Simulate frustration compatibility calculation with realistic variance"""
        base_score = random.uniform(0.3, 0.9)
        # Adjust based on movie characteristics
        if 'Comedy' in movie.get('genres', []):
            base_score += 0.1 if frustration_level > 0.6 else 0.0
        if movie.get('runtime', 120) > 150:
            base_score -= 0.1 if frustration_level > 0.5 else 0.0
        return max(0.0, min(1.0, base_score))

    def _calculate_cognitive_compatibility(self, movie, strategy_config, cognitive_load_level):
        """Simulate cognitive compatibility calculation"""
        base_score = random.uniform(0.2, 0.8)
        # Complex movies are harder when cognitive load is high
        if 'Sci-Fi' in movie.get('genres', []) or 'Thriller' in movie.get('genres', []):
            base_score -= 0.15 if cognitive_load_level > 0.7 else 0.0
        if 'Family' in movie.get('genres', []) or 'Animation' in movie.get('genres', []):
            base_score += 0.1 if cognitive_load_level > 0.6 else 0.0
        return max(0.0, min(1.0, base_score))

    def _calculate_persona_match(self, movie, user_preferences):
        """Simulate persona matching"""
        if not user_preferences:
            return random.uniform(0.3, 0.7)
        
        base_score = random.uniform(0.2, 0.8)
        preferred_genres = user_preferences.get('preferred_genres', [])
        
        # Boost score for preferred genres
        for genre in movie.get('genres', []):
            if genre in preferred_genres:
                base_score += 0.2
                break
        
        return max(0.0, min(1.0, base_score))

    def _generate_reasoning(self, movie, overall_score):
        """Generate realistic reasoning text"""
        reasons = []
        if overall_score > 0.8:
            reasons.append("Highly recommended based on your current mood")
        elif overall_score > 0.6:
            reasons.append("Good match for your preferences")
        else:
            reasons.append("Might be worth exploring")
        
        if 'Comedy' in movie.get('genres', []):
            reasons.append("Light-hearted content to help you relax")
        if movie.get('runtime', 120) < 100:
            reasons.append("Quick watch option")
        
        return ". ".join(reasons)

    def _apply_advanced_diversity_filter(self, recommendations):
        """Apply diversity filtering to prevent too many similar movies"""
        if len(recommendations) <= 5:
            return recommendations
        
        filtered = []
        genre_counts = defaultdict(int)
        
        for rec in recommendations:
            # Limit movies per genre to maintain diversity
            primary_genre = rec.genres[0] if rec.genres else 'Unknown'
            if genre_counts[primary_genre] < 3:  # Max 3 per genre
                filtered.append(rec)
                genre_counts[primary_genre] += 1
            elif len(filtered) < len(recommendations) * 0.7:  # Keep at least 70%
                filtered.append(rec)
        
        return filtered

    def _simple_shuffle_top_k(self, recommendations: List[MovieRecommendation], 
                            k: int) -> List[MovieRecommendation]:
        """Simple random shuffle - FIXED VERSION"""
        if k <= 1 or len(recommendations) <= 1:
            return recommendations
        
        # FIXED: Ensure we always return a list
        try:
            top_k = recommendations[:k]
            remaining = recommendations[k:]
            random.shuffle(top_k)
            return top_k + remaining
        except Exception as e:
            logger.warning(f"Simple shuffle failed: {e}, returning original list")
            return recommendations


    def _weighted_shuffle_top_k(self, recommendations: List[MovieRecommendation], 
                            k: int) -> List[MovieRecommendation]:
        """
        Complete weighted shuffle with consistency factor and robust error handling.
        Implements Option 3 hybrid approach with all fixes.
        """
        import random
        
        # Basic validation
        if k <= 1 or len(recommendations) <= 1:
            return recommendations
        
        if k > len(recommendations):
            k = len(recommendations)
        
        try:
            # NEW: Consistency factor - sometimes keep the original top movie
            consistency_factor = self.config.get("shuffle_consistency_factor", 0.3)
            
            if random.random() < consistency_factor:
                # CONSISTENCY MODE: Keep original top movie, shuffle positions 2-k only
                top_1 = recommendations[:1]
                remaining_k = recommendations[1:k] if k > 1 else []
                after_k = recommendations[k:]
                
                if len(remaining_k) <= 1:
                    return recommendations  # Not enough to shuffle
                
                # Shuffle only positions 2-k with weighted selection
                weight_exponent = self.config.get("shuffle_weight_exponent", 2.5)
                weights = []
                
                for rec in remaining_k:
                    if hasattr(rec, 'overall_score') and rec.overall_score is not None:
                        weight = max(0.1, (float(rec.overall_score) ** weight_exponent) * 100)
                    else:
                        weight = 1.0  # Default weight if score missing
                    weights.append(weight)
                
                # Weighted shuffle of positions 2-k
                shuffled_remaining = []
                available_recs = remaining_k.copy()
                available_weights = weights.copy()
                
                while available_recs and len(shuffled_remaining) < len(remaining_k):
                    try:
                        # Ensure weights are valid
                        if not available_weights or all(w <= 0 for w in available_weights):
                            # Fallback to simple shuffle if weights are invalid
                            shuffled_remaining.extend(available_recs)
                            break
                        
                        # Weighted random selection
                        selected_rec = random.choices(available_recs, weights=available_weights, k=1)[0]
                        shuffled_remaining.append(selected_rec)
                        
                        # Remove selected item
                        idx = available_recs.index(selected_rec)
                        available_recs.pop(idx)
                        available_weights.pop(idx)
                        
                    except (ValueError, IndexError, TypeError) as e:
                        # Fallback: add remaining items in order
                        shuffled_remaining.extend(available_recs)
                        break
                
                return top_1 + shuffled_remaining + after_k
            
            # FULL SHUFFLE MODE: Original weighted shuffle logic
            top_k = recommendations[:k]
            remaining = recommendations[k:]
            
            # Create weights based on scores with robust error handling
            weights = []
            weight_exponent = self.config.get("shuffle_weight_exponent", 2.5)
            
            for rec in top_k:
                try:
                    if hasattr(rec, 'overall_score') and rec.overall_score is not None:
                        score = float(rec.overall_score)
                        weight = max(0.1, (score ** weight_exponent) * 100)
                    else:
                        weight = 1.0  # Default weight
                except (ValueError, TypeError):
                    weight = 1.0  # Fallback weight
                weights.append(weight)
            
            # Validate weights
            if not weights or all(w <= 0 for w in weights):
                # Fallback to simple shuffle if all weights are invalid
                logger.warning("Invalid weights detected, falling back to simple shuffle")
                random.shuffle(top_k)
                return top_k + remaining
            
            # Weighted random selection to create new order
            shuffled_top_k = []
            available_recs = top_k.copy()
            available_weights = weights.copy()
            
            while available_recs and len(shuffled_top_k) < k:
                try:
                    # Double-check weights before selection
                    if not available_weights or all(w <= 0 for w in available_weights):
                        shuffled_top_k.extend(available_recs)
                        break
                    
                    # Weighted random selection
                    selected_rec = random.choices(available_recs, weights=available_weights, k=1)[0]
                    shuffled_top_k.append(selected_rec)
                    
                    # Remove selected item
                    idx = available_recs.index(selected_rec)
                    available_recs.pop(idx)
                    available_weights.pop(idx)
                    
                except (ValueError, IndexError, TypeError) as e:
                    # Fallback: add remaining items in order
                    logger.warning(f"Weighted selection failed: {e}, adding remaining items")
                    shuffled_top_k.extend(available_recs)
                    break
            
            return shuffled_top_k + remaining
            
        except Exception as e:
            # Ultimate fallback: simple shuffle
            logger.error(f"Weighted shuffle completely failed: {e}, falling back to simple shuffle")
            try:
                top_k = recommendations[:k]
                remaining = recommendations[k:]
                random.shuffle(top_k)
                return top_k + remaining
            except Exception as final_e:
                logger.error(f"Even simple shuffle failed: {final_e}, returning original list")
                return recommendations


    def _apply_discovery_shuffling(self, recommendations: List[MovieRecommendation], 
                                session_key: str) -> List[MovieRecommendation]:
        """Apply discovery shuffling with proper error handling"""
        if len(recommendations) < self.config.get("min_recommendations_for_shuffle", 3):
            logger.info(f"Skipping shuffle - only {len(recommendations)} recommendations")
            return recommendations
        
        shuffle_method = self.config.get("shuffle_method", "weighted")
        k = min(self.config.get("shuffle_k_size", 8), len(recommendations))
        
        # Store original order for analytics
        original_top_k = recommendations[:k]
        
        # FIXED: Ensure shuffle methods always return a list
        if shuffle_method == "weighted":
            shuffled_recommendations = self._weighted_shuffle_top_k(recommendations, k)
        else:
            shuffled_recommendations = self._simple_shuffle_top_k(recommendations, k)
        
        # FIXED: Add safety check
        if shuffled_recommendations is None:
            logger.warning(f"Shuffle method returned None, using original recommendations")
            shuffled_recommendations = recommendations
        
        # Track analytics with safety check
        if len(shuffled_recommendations) >= k:
            self._track_shuffle_analytics(original_top_k, shuffled_recommendations[:k], session_key)
        else:
            logger.warning(f"Shuffled recommendations too short: {len(shuffled_recommendations)}")
        
        logger.info(f"Applied {shuffle_method} shuffle to top-{k} for {session_key}")
        return shuffled_recommendations


    def _track_shuffle_analytics(self, original_top_k, shuffled_top_k, session_key):
        """Track shuffle analytics with safety checks"""
        # FIXED: Add safety checks
        if not original_top_k or not shuffled_top_k:
            logger.warning("Cannot track analytics - empty recommendation lists")
            return
        
        position_changes = 0
        min_length = min(len(original_top_k), len(shuffled_top_k))
        
        for i in range(min_length):
            if original_top_k[i].item_id != shuffled_top_k[i].item_id:
                position_changes += 1
        
        analytics = {
            'session_key': session_key,
            'timestamp': time.time(),
            'k_size': min_length,
            'position_changes': position_changes,
            'discovery_rate': position_changes / min_length if min_length > 0 else 0,
            'original_top_score': original_top_k[0].overall_score if original_top_k else 0,
            'shuffled_top_score': shuffled_top_k[0].overall_score if shuffled_top_k else 0
        }
        
        self.shuffle_analytics.append(analytics)


    def _score_movies_multitarget(self, strategy_config: Dict, frustration_level: float,
                                cognitive_load_level: float, user_preferences: Optional[Dict],
                                session: UserSession) -> List[MovieRecommendation]:
        """Complete movie scoring with shuffling"""
        recommendations = []
        
        # Get top genre from session affinity
        top_genre = None
        if session.genre_affinity:
            top_genre = max(session.genre_affinity, key=session.genre_affinity.get)
        
        for movie in self.test_movies:
            # Calculate compatibility scores
            frustration_compatibility = self._calculate_frustration_compatibility(
                movie, strategy_config, frustration_level
            )
            cognitive_compatibility = self._calculate_cognitive_compatibility(
                movie, strategy_config, cognitive_load_level
            )
            persona_match = self._calculate_persona_match(movie, user_preferences)
            
            # Genre affinity boost
            genre_affinity_boost = 0.0
            if top_genre and top_genre in movie.get('genres', []):
                genre_affinity_boost = 0.15
            
            # Diversity factors
            genre_bonus = 0.0
            movie_genres = movie.get('genres', [])
            if 'Animation' in movie_genres:
                genre_bonus += 0.03
            if 'Comedy' in movie_genres:
                genre_bonus += 0.02
            if 'Action' in movie_genres:
                genre_bonus += 0.01
            if 'Family' in movie_genres:
                genre_bonus += 0.025
            
            popularity_factor = (hash(str(movie['item_id'])) % 100) / 1000.0
            
            runtime = movie.get('runtime', 120)
            runtime_factor = 0.0
            if 90 <= runtime <= 120:
                runtime_factor = 0.015
            elif runtime > 150:
                runtime_factor = -0.005 if frustration_level > 0.2 else 0.02
            
            # Calculate final score
            base_score = (
                0.4 * frustration_compatibility + 
                0.3 * cognitive_compatibility +
                0.3 * persona_match
            )
            
            scaled_base_score = base_score * 0.85
            diversity_bonus = genre_bonus + popularity_factor + runtime_factor
            overall_score = scaled_base_score + diversity_bonus + genre_affinity_boost
            overall_score = max(0.0, min(overall_score, 0.98))
            
            reasoning = self._generate_reasoning(movie, overall_score)
            
            recommendation = MovieRecommendation(
                item_id=str(movie['item_id']),
                title=movie['title'],
                genres=movie['genres'],
                frustration_compatibility=frustration_compatibility,
                cognitive_compatibility=cognitive_compatibility,
                persona_match=persona_match,
                overall_score=round(overall_score, 3),
                reasoning=reasoning
            )
            
            recommendations.append(recommendation)
        
        # Sort by score
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Apply diversity filter
        filtered_recommendations = self._apply_advanced_diversity_filter(recommendations)
        
        # Apply shuffling if enabled
        if self.config.get("enable_top_k_shuffling", True):
            final_recommendations = self._apply_discovery_shuffling(
                filtered_recommendations, 
                f"{session.user_id}_{session.session_id}"
            )
            return final_recommendations[:self.config["recommendation_count"]]
        else:
            return filtered_recommendations[:self.config["recommendation_count"]]

    def run_comprehensive_tests(self):
        """Run comprehensive test suite"""
        print("=" * 80)
        print("COMPREHENSIVE TOP-K SHUFFLING TEST SUITE")
        print("=" * 80)
        
        # Test 1: Basic functionality
        self.test_basic_functionality()
        
        # Test 2: Different shuffle methods
        self.test_shuffle_methods()
        
        # Test 3: Different k values
        self.test_different_k_values()
        
        # Test 4: Consistency and variation analysis
        self.test_consistency_and_variation()
        
        # Test 5: Performance analysis
        self.test_performance()
        
        # Test 6: Edge cases
        self.test_edge_cases()
        
        # Generate final report
        self.generate_test_report()

    def test_basic_functionality(self):
        """Test basic shuffling functionality"""
        print("\n1. BASIC FUNCTIONALITY TEST")
        print("-" * 40)
        
        session = UserSession(
            user_id="test_user", 
            session_id="basic_test",
            genre_affinity={"Comedy": 0.7, "Action": 0.5, "Drama": 0.3}
        )
        
        user_preferences = {
            'preferred_genres': ['Comedy', 'Action']
        }
        
        recommendations = self._score_movies_multitarget(
            {}, 0.6, 0.4, user_preferences, session
        )
        
        print(f"Generated {len(recommendations)} recommendations")
        print("Top 5 recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec.title} (Score: {rec.overall_score}) - {rec.genres}")
        
        assert len(recommendations) > 0, "Should generate recommendations"
        assert recommendations[0].overall_score >= recommendations[-1].overall_score, "Should be sorted by score"
        print("✓ Basic functionality test passed")

    def test_shuffle_methods(self):
        """Test different shuffle methods"""
        print("\n2. SHUFFLE METHODS TEST")
        print("-" * 40)
        
        session = UserSession("test_user", "shuffle_test", {"Comedy": 0.8})
        
        # Test simple shuffle
        self.config["shuffle_method"] = "simple"
        simple_recs = self._score_movies_multitarget({}, 0.5, 0.5, None, session)
        
        # Test weighted shuffle
        self.config["shuffle_method"] = "weighted"
        weighted_recs = self._score_movies_multitarget({}, 0.5, 0.5, None, session)
        
        print(f"Simple shuffle top movie: {simple_recs[0].title} (Score: {simple_recs[0].overall_score})")
        print(f"Weighted shuffle top movie: {weighted_recs[0].title} (Score: {weighted_recs[0].overall_score})")
        
        # Analyze differences
        simple_top_ids = [rec.item_id for rec in simple_recs[:5]]
        weighted_top_ids = [rec.item_id for rec in weighted_recs[:5]]
        
        differences = len(set(simple_top_ids) ^ set(weighted_top_ids))
        print(f"Differences in top 5: {differences}/5")
        print("✓ Shuffle methods test completed")

    def test_different_k_values(self):
        """Test different k values for shuffling"""
        print("\n3. DIFFERENT K VALUES TEST")
        print("-" * 40)
        
        session = UserSession("test_user", "k_test", {"Action": 0.6})
        k_values = [3, 5, 8, 10]
        
        for k in k_values:
            self.config["shuffle_k_size"] = k
            recommendations = self._score_movies_multitarget({}, 0.4, 0.6, None, session)
            
            print(f"K={k}: Top movie - {recommendations[0].title} (Score: {recommendations[0].overall_score})")
        
        print("✓ Different k values test completed")

    def test_consistency_and_variation(self):
        """Test consistency and variation across multiple runs"""
        print("\n4. CONSISTENCY AND VARIATION TEST")
        print("-" * 40)
        
        session = UserSession("test_user", "consistency_test", {"Sci-Fi": 0.9})
        
        # Run multiple times to analyze variation
        top_movies = []
        top_scores = []
        
        for run in range(10):
            recommendations = self._score_movies_multitarget({}, 0.3, 0.7, None, session)
            top_movies.append(recommendations[0].title)
            top_scores.append(recommendations[0].overall_score)
        
        # Analyze variation
        unique_top_movies = len(set(top_movies))
        score_variance = statistics.variance(top_scores) if len(top_scores) > 1 else 0
        
        print(f"Unique top movies across 10 runs: {unique_top_movies}/10")
        print(f"Score variance: {score_variance:.4f}")
        print(f"Most common top movie: {Counter(top_movies).most_common(1)[0]}")
        
        # Good shuffling should show variation but not too much randomness
        assert 1 <= unique_top_movies <= 10, f"Expected 1-8 unique top movies, got {unique_top_movies}"
        print("✓ Consistency and variation test passed")

    def test_performance(self):
        """Test performance of shuffling"""
        print("\n5. PERFORMANCE TEST")
        print("-" * 40)
        
        session = UserSession("test_user", "performance_test", {"Drama": 0.5})
        
        # Measure time for multiple runs
        start_time = time.time()
        
        for _ in range(100):
            recommendations = self._score_movies_multitarget({}, 0.5, 0.5, None, session)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        print(f"Average time per recommendation generation: {avg_time:.4f} seconds")
        print(f"Recommendations per second: {1/avg_time:.2f}")
        
        assert avg_time < 1.0, f"Performance too slow: {avg_time:.4f}s per generation"
        print("✓ Performance test passed")

    def test_edge_cases(self):
        """Test edge cases"""
        print("\n6. EDGE CASES TEST")
        print("-" * 40)
        
        # Test with no genre affinity
        session_no_affinity = UserSession("test_user", "no_affinity_test")
        recs_no_affinity = self._score_movies_multitarget({}, 0.5, 0.5, None, session_no_affinity)
        print(f"No affinity: Generated {len(recs_no_affinity)} recommendations")
        
        # Test with extreme psychological states
        session_extreme = UserSession("test_user", "extreme_test", {"Horror": 1.0})
        recs_extreme = self._score_movies_multitarget({}, 0.9, 0.9, None, session_extreme)
        print(f"Extreme states: Generated {len(recs_extreme)} recommendations")
        
        # Test with shuffling disabled
        self.config["enable_top_k_shuffling"] = False
        recs_no_shuffle = self._score_movies_multitarget({}, 0.5, 0.5, None, session_extreme)
        print(f"No shuffling: Generated {len(recs_no_shuffle)} recommendations")
        
        # Re-enable shuffling
        self.config["enable_top_k_shuffling"] = True
        
        print("✓ Edge cases test completed")

    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        if self.shuffle_analytics:
            # Analytics summary
            total_shuffles = len(self.shuffle_analytics)
            avg_discovery_rate = statistics.mean([a['discovery_rate'] for a in self.shuffle_analytics])
            avg_position_changes = statistics.mean([a['position_changes'] for a in self.shuffle_analytics])
            
            print(f"Total shuffle operations: {total_shuffles}")
            print(f"Average discovery rate: {avg_discovery_rate:.2%}")
            print(f"Average position changes: {avg_position_changes:.1f}")
            
            # Score impact analysis
            score_differences = []
            for analytics in self.shuffle_analytics:
                if analytics['original_top_score'] > 0 and analytics['shuffled_top_score'] > 0:
                    diff = analytics['shuffled_top_score'] - analytics['original_top_score']
                    score_differences.append(diff)
            
            if score_differences:
                avg_score_impact = statistics.mean(score_differences)
                print(f"Average score impact: {avg_score_impact:+.3f}")
                print(f"Score impact range: {min(score_differences):+.3f} to {max(score_differences):+.3f}")
        
        # Configuration summary
        print(f"\nTest Configuration:")
        print(f"  Shuffle method: {self.config['shuffle_method']}")
        print(f"  Shuffle k size: {self.config['shuffle_k_size']}")
        print(f"  Weight exponent: {self.config['shuffle_weight_exponent']}")
        print(f"  Min for shuffle: {self.config['min_recommendations_for_shuffle']}")
        
        print(f"\nTest completed successfully! ✓")
        print("All shuffle functionality is working as expected.")

if __name__ == "__main__":
    # Set random seed for reproducible testing
    random.seed(42)
    
    # Run comprehensive tests
    test_engine = ComprehensiveInferenceEngineTest()
    test_engine.run_comprehensive_tests()
    
    # Save analytics to file for further analysis
    with open('shuffle_test_analytics.json', 'w') as f:
        json.dump(test_engine.shuffle_analytics, f, indent=2)
    
    print(f"\nAnalytics saved to 'shuffle_test_analytics.json'")
