# inference/performance_evaluation.py (Modified for Psychological Recommendation System)
import requests
import time
import psutil
import numpy as np
import json
import argparse
import sys
import os
from typing import List, Dict, Any

# --- Helper Functions for Metrics ---
def calculate_precision_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
    if not recommended_items or k == 0: return 0.0
    top_k_recs = recommended_items[:k]
    relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
    return relevant_in_top_k / k

def calculate_dcg_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
    dcg = 0.0
    for i, item_id in enumerate(recommended_items[:k]):
        if item_id in relevant_items:
            dcg += 1 / np.log2(i + 2)
    return dcg

def calculate_ndcg_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
    dcg = calculate_dcg_at_k(recommended_items, relevant_items, k)
    ideal_dcg = calculate_dcg_at_k(sorted(relevant_items, key=lambda x: x in recommended_items, reverse=True), relevant_items, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

# --- Main Evaluator Class ---
class PsychologicalRecommendationEvaluator:
    """
    Performance evaluator specifically designed for the psychological recommendation system.
    Tests behavioral analysis, psychological profiling, and content recommendations.
    """
    
    def __init__(self, api_url: str = "http://127.0.0.1:5000", server_pid: int = None):
        self.api_url = api_url
        self.server_process = psutil.Process(server_pid) if server_pid and psutil.pid_exists(server_pid) else None
        self.results = {}
        print(f"🧠 Initializing Psychological Recommendation Evaluator")
        print(f"   API URL: {self.api_url}")
        if self.server_process:
            print(f"   Monitoring server process with PID: {server_pid}")

    def _generate_behavioral_test_data(self):
        """
        Generate realistic behavioral test data that matches your model's input format.
        """
        # Power User: High engagement, low frustration, high exploration
        power_user = {
            "user_id": "power_user_test",
            "behavioral_history": [
                {
                    "dpad_up_count": 15, "dpad_down_count": 12, "dpad_left_count": 8, "dpad_right_count": 9,
                    "back_button_presses": 1, "menu_revisits": 0, "scroll_speed": 80, "hover_duration": 1.5,
                    "time_since_last_interaction": 2.0
                } for _ in range(10)  # Consistent power user behavior
            ],
            "expected_genres": ["Science Fiction", "Action", "Thriller"],
            "expected_profile": {"user_type": "Power User", "engagement_level": "High", "frustration_level": "Low"}
        }

        # Casual Viewer: Moderate engagement, low frustration, low exploration
        casual_user = {
            "user_id": "casual_user_test",
            "behavioral_history": [
                {
                    "dpad_up_count": 5, "dpad_down_count": 4, "dpad_left_count": 2, "dpad_right_count": 3,
                    "back_button_presses": 0, "menu_revisits": 1, "scroll_speed": 120, "hover_duration": 3.0,
                    "time_since_last_interaction": 8.0
                } for _ in range(5)  # Consistent casual behavior
            ],
            "expected_genres": ["Comedy", "Romance", "Family"],
            "expected_profile": {"user_type": "Casual Viewer", "engagement_level": "Medium", "frustration_level": "Low"}
        }

        # Frustrated User: High activity but with frustration indicators
        frustrated_user = {
            "user_id": "frustrated_user_test",
            "behavioral_history": [
                {
                    "dpad_up_count": 20, "dpad_down_count": 18, "dpad_left_count": 12, "dpad_right_count": 15,
                    "back_button_presses": 8, "menu_revisits": 5, "scroll_speed": 200, "hover_duration": 0.8,
                    "time_since_last_interaction": 1.0
                } for _ in range(8)  # Consistent frustrated behavior
            ],
            "expected_genres": ["Drama", "Comedy"],  # Simpler content when frustrated
            "expected_profile": {"user_type": "Frustrated User", "engagement_level": "High", "frustration_level": "High"}
        }

        # Adaptive User: Changes behavior over time
        adaptive_user = {
            "user_id": "adaptive_user_test",
            "initial_behavior": [
                {
                    "dpad_up_count": 3, "dpad_down_count": 2, "dpad_left_count": 1, "dpad_right_count": 1,
                    "back_button_presses": 0, "menu_revisits": 1, "scroll_speed": 150, "hover_duration": 4.0,
                    "time_since_last_interaction": 10.0
                } for _ in range(3)  # Initially casual
            ],
            "evolved_behavior": [
                {
                    "dpad_up_count": 18, "dpad_down_count": 15, "dpad_left_count": 10, "dpad_right_count": 12,
                    "back_button_presses": 2, "menu_revisits": 0, "scroll_speed": 70, "hover_duration": 1.2,
                    "time_since_last_interaction": 1.5
                } for _ in range(7)  # Becomes power user
            ],
            "expected_adaptation": True
        }

        return [power_user, casual_user, frustrated_user], adaptive_user

    def _make_api_call(self, endpoint: str, payload: Dict) -> tuple[Dict, float]:
        """Make API call to your recommendation system."""
        start_time = time.perf_counter()
        try:
            url = f"{self.api_url}/{endpoint}"
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json(), time.perf_counter() - start_time
        except requests.exceptions.RequestException as e:
            print(f"❌ API Request failed for {endpoint}: {e}")
            return None, -1.0
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error for {endpoint}: {e}")
            return None, -1.0

    def test_psychological_profiling_accuracy(self):
        """Test the accuracy of psychological trait prediction."""
        print("\n--- 1. Testing Psychological Profiling Accuracy ---")
        test_users, _ = self._generate_behavioral_test_data()
        
        profile_results = []
        
        for user_data in test_users:
            print(f"\n   Testing user: {user_data['user_id']}")
            
            # Create payload in the format your API expects
            payload = {
                "user_id": user_data["user_id"],
                "user_history": user_data["behavioral_history"],
                "top_k": 5
            }
            
            # Call your recommendation API
            api_response, latency = self._make_api_call("recommendations", payload)
            
            if api_response:
                # Extract psychological profile from response
                psychological_profile = api_response.get('psychological_profile', {})
                behavioral_summary = api_response.get('behavioral_summary', {})
                
                print(f"     Predicted Profile: {psychological_profile}")
                print(f"     Expected Profile: {user_data['expected_profile']}")
                
                # Evaluate psychological trait accuracy
                engagement_correct = self._evaluate_engagement_level(
                    psychological_profile.get('engagement_level', 0),
                    user_data['expected_profile']['engagement_level']
                )
                
                frustration_correct = self._evaluate_frustration_level(
                    psychological_profile.get('frustration_level', 0),
                    user_data['expected_profile']['frustration_level']
                )
                
                profile_results.append({
                    "user_id": user_data["user_id"],
                    "engagement_correct": engagement_correct,
                    "frustration_correct": frustration_correct,
                    "latency_ms": latency * 1000
                })
                
                print(f"     Engagement Assessment: {'✅' if engagement_correct else '❌'}")
                print(f"     Frustration Assessment: {'✅' if frustration_correct else '❌'}")
            else:
                print(f"     ❌ API call failed")
        
        self.results['psychological_profiling'] = profile_results

    def test_recommendation_relevance(self):
        """Test if recommendations match expected genres for each user type."""
        print("\n--- 2. Testing Recommendation Relevance ---")
        test_users, _ = self._generate_behavioral_test_data()
        
        recommendation_results = []
        
        for user_data in test_users:
            print(f"\n   Testing recommendations for: {user_data['user_id']}")
            
            payload = {
                "user_id": user_data["user_id"],
                "user_history": user_data["behavioral_history"],
                "top_k": 10
            }
            
            api_response, latency = self._make_api_call("recommendations", payload)
            
            if api_response and "recommendations" in api_response:
                recommendations = api_response["recommendations"]
                recommended_genres = [rec.get('content_genre', 'Unknown') for rec in recommendations]
                expected_genres = user_data["expected_genres"]
                
                print(f"     Recommended Genres: {recommended_genres[:5]}")
                print(f"     Expected Genres: {expected_genres}")
                
                # Calculate genre relevance
                genre_matches = sum(1 for genre in recommended_genres if genre in expected_genres)
                genre_relevance = genre_matches / len(recommendations) if recommendations else 0
                
                # Calculate average similarity score
                avg_similarity = np.mean([rec.get('similarity_score', 0) for rec in recommendations])
                
                recommendation_results.append({
                    "user_id": user_data["user_id"],
                    "genre_relevance": genre_relevance,
                    "avg_similarity_score": avg_similarity,
                    "total_recommendations": len(recommendations)
                })
                
                print(f"     Genre Relevance: {genre_relevance:.2%}")
                print(f"     Avg Similarity Score: {avg_similarity:.3f}")
            else:
                print(f"     ❌ No recommendations received")
        
        self.results['recommendation_relevance'] = recommendation_results

    def test_behavioral_adaptation(self):
        """Test if the model adapts to changing user behavior."""
        print("\n--- 3. Testing Behavioral Adaptation ---")
        _, adaptive_user = self._generate_behavioral_test_data()
        
        # Test initial behavior
        initial_payload = {
            "user_id": adaptive_user["user_id"],
            "user_history": adaptive_user["initial_behavior"],
            "top_k": 5
        }
        
        initial_response, _ = self._make_api_call("recommendations", initial_payload)
        
        # Test evolved behavior
        evolved_payload = {
            "user_id": adaptive_user["user_id"],
            "user_history": adaptive_user["evolved_behavior"],
            "top_k": 5
        }
        
        evolved_response, _ = self._make_api_call("recommendations", evolved_payload)
        
        adaptation_passed = False
        
        if initial_response and evolved_response:
            initial_profile = initial_response.get('psychological_profile', {})
            evolved_profile = evolved_response.get('psychological_profile', {})
            
            print(f"   Initial Profile: {initial_profile}")
            print(f"   Evolved Profile: {evolved_profile}")
            
            # Check if engagement level increased significantly
            initial_engagement = initial_profile.get('engagement_level', 0)
            evolved_engagement = evolved_profile.get('engagement_level', 0)
            
            engagement_increase = evolved_engagement - initial_engagement
            adaptation_passed = engagement_increase > 0.3  # Significant increase
            
            print(f"   Engagement Change: {initial_engagement:.3f} → {evolved_engagement:.3f} (Δ{engagement_increase:+.3f})")
            print(f"   Adaptation Test: {'✅ PASSED' if adaptation_passed else '❌ FAILED'}")
        
        self.results['behavioral_adaptation'] = {
            "passed": adaptation_passed,
            "initial_profile": initial_response.get('psychological_profile', {}) if initial_response else {},
            "evolved_profile": evolved_response.get('psychological_profile', {}) if evolved_response else {}
        }

    def test_system_performance(self):
        """Test system performance metrics."""
        print("\n--- 4. Testing System Performance ---")
        test_users, _ = self._generate_behavioral_test_data()
        
        latencies = []
        memory_usage = []
        
        for i in range(10):  # Run 10 performance tests
            user_data = test_users[i % len(test_users)]
            payload = {
                "user_id": f"perf_test_{i}",
                "user_history": user_data["behavioral_history"],
                "top_k": 10
            }
            
            if self.server_process:
                memory_before = self.server_process.memory_info().rss / (1024 * 1024)  # MB
            
            _, latency = self._make_api_call("recommendations", payload)
            
            if latency > 0:
                latencies.append(latency * 1000)  # Convert to ms
            
            if self.server_process:
                memory_after = self.server_process.memory_info().rss / (1024 * 1024)  # MB
                memory_usage.append(memory_after)
        
        self.results['system_performance'] = {
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "max_latency_ms": np.max(latencies) if latencies else 0,
            "avg_memory_mb": np.mean(memory_usage) if memory_usage else 0,
            "total_requests": len(latencies)
        }

    def _evaluate_engagement_level(self, predicted_value: float, expected_level: str) -> bool:
        """Evaluate if predicted engagement matches expected level."""
        if expected_level == "High":
            return predicted_value > 0.7
        elif expected_level == "Medium":
            return 0.3 <= predicted_value <= 0.7
        elif expected_level == "Low":
            return predicted_value < 0.3
        return False

    def _evaluate_frustration_level(self, predicted_value: float, expected_level: str) -> bool:
        """Evaluate if predicted frustration matches expected level."""
        if expected_level == "High":
            return predicted_value > 0.6
        elif expected_level == "Medium":
            return 0.3 <= predicted_value <= 0.6
        elif expected_level == "Low":
            return predicted_value < 0.3
        return False

    def run_full_evaluation(self):
        """Run the complete evaluation suite."""
        print("\n" + "🧠" * 20 + " PSYCHOLOGICAL RECOMMENDATION SYSTEM EVALUATION " + "🧠" * 20)
        
        self.test_psychological_profiling_accuracy()
        self.test_recommendation_relevance()
        self.test_behavioral_adaptation()
        
        if self.server_process:
            self.test_system_performance()
        
        self.generate_comprehensive_report()

    def generate_comprehensive_report(self):
        """Generate a comprehensive evaluation report."""
        print("\n" + "=" * 80)
        print("🎯 PSYCHOLOGICAL RECOMMENDATION SYSTEM - EVALUATION REPORT")
        print("=" * 80)

        # Psychological Profiling Results
        profiling_results = self.results.get('psychological_profiling', [])
        if profiling_results:
            engagement_accuracy = sum(1 for r in profiling_results if r['engagement_correct']) / len(profiling_results)
            frustration_accuracy = sum(1 for r in profiling_results if r['frustration_correct']) / len(profiling_results)
            avg_profiling_latency = np.mean([r['latency_ms'] for r in profiling_results])
            
            print(f"\n📊 1. PSYCHOLOGICAL PROFILING ACCURACY:")
            print(f"   Engagement Level Accuracy:  {engagement_accuracy:.1%}")
            print(f"   Frustration Level Accuracy: {frustration_accuracy:.1%}")
            print(f"   Overall Trait Accuracy:     {(engagement_accuracy + frustration_accuracy) / 2:.1%}")
            print(f"   Avg Profiling Latency:      {avg_profiling_latency:.2f} ms")

        # Recommendation Relevance Results
        relevance_results = self.results.get('recommendation_relevance', [])
        if relevance_results:
            avg_genre_relevance = np.mean([r['genre_relevance'] for r in relevance_results])
            avg_similarity_score = np.mean([r['avg_similarity_score'] for r in relevance_results])
            
            print(f"\n🎬 2. RECOMMENDATION RELEVANCE:")
            print(f"   Average Genre Relevance:    {avg_genre_relevance:.1%}")
            print(f"   Average Similarity Score:   {avg_similarity_score:.3f}")
            print(f"   Content Matching Quality:   {'✅ EXCELLENT' if avg_genre_relevance > 0.6 else '⚠️ NEEDS IMPROVEMENT'}")

        # Behavioral Adaptation Results
        adaptation_results = self.results.get('behavioral_adaptation', {})
        print(f"\n🔄 3. BEHAVIORAL ADAPTATION:")
        print(f"   Adaptation Test Result:     {'✅ PASSED' if adaptation_results.get('passed') else '❌ FAILED'}")
        
        if adaptation_results.get('passed'):
            print("   ✓ Model successfully adapts to changing user behavior")
        else:
            print("   ✗ Model may not be sensitive enough to behavioral changes")

        # System Performance Results
        performance_results = self.results.get('system_performance', {})
        if performance_results:
            print(f"\n⚡ 4. SYSTEM PERFORMANCE:")
            print(f"   Average Latency:           {performance_results.get('avg_latency_ms', 0):.2f} ms")
            print(f"   Maximum Latency:           {performance_results.get('max_latency_ms', 0):.2f} ms")
            print(f"   Average Memory Usage:      {performance_results.get('avg_memory_mb', 0):.2f} MB")
            print(f"   Performance Rating:        {'✅ EXCELLENT' if performance_results.get('avg_latency_ms', 0) < 500 else '⚠️ ACCEPTABLE' if performance_results.get('avg_latency_ms', 0) < 1000 else '❌ SLOW'}")

        # Overall Assessment
        print(f"\n🏆 OVERALL SYSTEM ASSESSMENT:")
        
        # Calculate overall score
        scores = []
        if profiling_results:
            trait_score = (engagement_accuracy + frustration_accuracy) / 2
            scores.append(trait_score)
        
        if relevance_results:
            relevance_score = avg_genre_relevance
            scores.append(relevance_score)
        
        if adaptation_results:
            adaptation_score = 1.0 if adaptation_results.get('passed') else 0.0
            scores.append(adaptation_score)
        
        overall_score = np.mean(scores) if scores else 0
        
        print(f"   Overall Performance Score: {overall_score:.1%}")
        
        if overall_score >= 0.8:
            print("   🎉 EXCELLENT: System ready for production deployment!")
        elif overall_score >= 0.6:
            print("   👍 GOOD: System performing well with minor improvements needed")
        elif overall_score >= 0.4:
            print("   ⚠️ FAIR: System needs significant improvements before deployment")
        else:
            print("   ❌ POOR: System requires major fixes and retraining")

        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Psychological Recommendation System Performance")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5000", help="API base URL")
    parser.add_argument("--pid", type=int, help="Server process ID for monitoring")
    
    args = parser.parse_args()
    
    if args.pid and not psutil.pid_exists(args.pid):
        print(f"Error: Process with PID {args.pid} does not exist.")
        sys.exit(1)
    
    evaluator = PsychologicalRecommendationEvaluator(api_url=args.url, server_pid=args.pid)
    evaluator.run_full_evaluation()
