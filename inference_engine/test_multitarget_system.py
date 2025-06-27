import requests
import json
import time
from datetime import datetime, timedelta

def generate_realistic_multitarget_test_log():
    """Generate a realistic test log for multi-target testing."""
    
    base_time = datetime.now()
    
    # Realistic session showing both frustration and cognitive load evolution
    realistic_log = [
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_multitarget_001", 
            "action_type": "session_start",
            "timestamp": base_time.isoformat(),
            "screen_context": "Home",
            "focused_item": json.dumps({
                "item_id": "home",
                "title": "Home Screen", 
                "genres": []
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 0.0,
                "consecutive_action_count": 1
            }),
            "frustration_level": 0.05,  # Starting low
            "cognitive_load": 0.15      # Starting with some cognitive load
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_multitarget_001",
            "action_type": "dpad_right", 
            "timestamp": (base_time + timedelta(seconds=2)).isoformat(),
            "screen_context": "Home",
            "focused_item": json.dumps({
                "item_id": "movie_123",
                "title": "Complex Sci-Fi Movie",
                "genres": ["Sci-Fi", "Drama"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 2.0,
                "consecutive_action_count": 1
            }),
            "frustration_level": 0.08,
            "cognitive_load": 0.25      # Increased due to complex content
        },
        {
            "user_id": "user_stressed_professional_1", 
            "session_id": "session_multitarget_001",
            "action_type": "dpad_right",
            "timestamp": (base_time + timedelta(seconds=3)).isoformat(),
            "screen_context": "Home",
            "focused_item": json.dumps({
                "item_id": "movie_124", 
                "title": "Documentary Series",
                "genres": ["Documentary"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 1.0,
                "consecutive_action_count": 2
            }),
            "frustration_level": 0.12,
            "cognitive_load": 0.45      # High cognitive load for documentary
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_multitarget_001", 
            "action_type": "back",
            "timestamp": (base_time + timedelta(seconds=7)).isoformat(),
            "screen_context": "Home",
            "focused_item": json.dumps({
                "item_id": "movie_123",
                "title": "Complex Sci-Fi Movie",
                "genres": ["Sci-Fi", "Drama"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 4.0,
                "consecutive_action_count": 1
            }),
            "frustration_level": 0.25,  # Frustration building
            "cognitive_load": 0.55      # High cognitive load from complex content
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_multitarget_001",
            "action_type": "dpad_down",
            "timestamp": (base_time + timedelta(seconds=9)).isoformat(),
            "screen_context": "Home",
            "focused_item": json.dumps({
                "item_id": "movie_200",
                "title": "Light Comedy",
                "genres": ["Comedy"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 2.0,
                "consecutive_action_count": 1
            }),
            "frustration_level": 0.20,  # Slight relief seeing comedy
            "cognitive_load": 0.30      # Lower cognitive load for simple content
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_multitarget_001",
            "action_type": "click",
            "timestamp": (base_time + timedelta(seconds=12)).isoformat(),
            "screen_context": "Detail_Page",
            "focused_item": json.dumps({
                "item_id": "movie_200",
                "title": "Light Comedy", 
                "genres": ["Comedy"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 3.0,
                "consecutive_action_count": 1
            }),
            "click_type": "more_info",
            "frustration_level": 0.15,  # Decision made, frustration reducing
            "cognitive_load": 0.20      # Low cognitive load for simple content
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_multitarget_001",
            "action_type": "dpad_right",
            "timestamp": (base_time + timedelta(seconds=15)).isoformat(),
            "screen_context": "Detail_Page",
            "focused_item": json.dumps({
                "item_id": "movie_201",
                "title": "Action Thriller",
                "genres": ["Action", "Thriller"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 3.0,
                "consecutive_action_count": 1
            }),
            "frustration_level": 0.18,
            "cognitive_load": 0.35      # Medium cognitive load for action content
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_multitarget_001",
            "action_type": "click",
            "timestamp": (base_time + timedelta(seconds=18)).isoformat(),
            "screen_context": "Detail_Page",
            "focused_item": json.dumps({
                "item_id": "movie_200",
                "title": "Light Comedy", 
                "genres": ["Comedy"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 3.0,
                "consecutive_action_count": 1
            }),
            "click_type": "play",
            "frustration_level": 0.10,  # Final decision made, frustration low
            "cognitive_load": 0.15      # Very low cognitive load for chosen content
        }
    ]
    
    return realistic_log

class MultiTargetRecommendationTester:
    """Comprehensive tester for the multi-target recommendation system."""
    
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        
    def test_health(self):
        """Test if the multi-target API is running."""
        try:
            response = requests.get(f"{self.api_url}/health")
            health_data = response.json()
            print(f"🏥 Health Check: {health_data.get('status', 'unknown')}")
            print(f"🎯 Model Type: {health_data.get('model_type', 'unknown')}")
            print(f"📊 Targets: {', '.join(health_data.get('targets', []))}")
            print(f"📈 Current Stats: {health_data.get('stats', {})}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def run_multitarget_test(self):
        """Run a complete test with realistic multi-target behavior."""
        
        test_events = generate_realistic_multitarget_test_log()
        
        user_id = test_events[0]['user_id']
        session_id = test_events[0]['session_id']
        
        print(f"\n🎬 Testing Multi-Target Recommendation System")
        print(f"👤 User: {user_id}")
        print(f"📱 Session: {session_id}")
        print("="*80)
        
        # Track predictions over time
        prediction_history = []
        
        # Send events one by one and track both psychological dimensions
        for i, event in enumerate(test_events):
            print(f"\n📱 Event {i+1}: {event['action_type']}")
            print(f"   ⏰ Timestamp: {event['timestamp']}")
            print(f"   🎯 Focused Item: {json.loads(event['focused_item'])['title']}")
            print(f"   😤 Actual Frustration: {event.get('frustration_level', 'N/A'):.3f}")
            print(f"   🧠 Actual Cognitive Load: {event.get('cognitive_load', 'N/A'):.3f}")
            
            # Send event to API
            response = requests.post(
                f"{self.api_url}/update_session",
                json={
                    "user_id": user_id,
                    "session_id": session_id,
                    "event": event
                }
            )
            
            result = response.json()
            print(f"   ✅ API Response: {result.get('status', 'unknown')}")
            
            if 'predicted_frustration' in result and 'predicted_cognitive_load' in result:
                pred_frustration = result['predicted_frustration']
                pred_cognitive = result['predicted_cognitive_load']
                actual_frustration = event.get('frustration_level', 0)
                actual_cognitive = event.get('cognitive_load', 0)
                
                print(f"   🤖 Predicted Frustration: {pred_frustration:.3f}")
                print(f"   🤖 Predicted Cognitive Load: {pred_cognitive:.3f}")
                print(f"   📊 Frustration Error: {abs(pred_frustration - actual_frustration):.3f}")
                print(f"   📊 Cognitive Error: {abs(pred_cognitive - actual_cognitive):.3f}")
                
                # Store prediction for analysis
                prediction_history.append({
                    'event': i+1,
                    'pred_frustration': pred_frustration,
                    'actual_frustration': actual_frustration,
                    'pred_cognitive': pred_cognitive,
                    'actual_cognitive': actual_cognitive
                })
                
                # Check if recommendations are needed
                if result.get('recommendations_needed', False):
                    print(f"   ⚠️  High psychological load detected - recommendations needed!")
            
            time.sleep(0.5)  # Simulate realistic timing
        
        # Analyze prediction accuracy
        self._analyze_prediction_accuracy(prediction_history)
        
        # Get final psychological state
        print(f"\n🧠 Getting Current Psychological State...")
        print("="*80)
        
        psych_response = requests.post(
            f"{self.api_url}/predict_psychological_state",
            json={
                "user_id": user_id,
                "session_id": session_id
            }
        )
        
        if psych_response.status_code == 200:
            psych_data = psych_response.json()
            print(f"✅ Current Psychological State:")
            print(f"   😤 Frustration Level: {psych_data.get('predicted_frustration', 0):.3f}")
            print(f"   🧠 Cognitive Load: {psych_data.get('predicted_cognitive_load', 0):.3f}")
            print(f"   📊 Events Processed: {psych_data.get('event_count', 0)}")
        else:
            print(f"❌ Failed to get psychological state: {psych_response.json()}")
        
        # Get final recommendations based on both dimensions
        print(f"\n🎯 Getting Multi-Target Recommendations...")
        print("="*80)
        
        user_preferences = {
            "preferred_genres": ["Comedy", "Family", "Animation"],
            "user_persona": "stressed_professional"
        }
        
        rec_response = requests.post(
            f"{self.api_url}/get_recommendations",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "user_preferences": user_preferences
            }
        )
        
        recommendations = rec_response.json()
        
        if recommendations.get('status') == 'success':
            print(f"✅ Generated {recommendations['count']} multi-target recommendations")
            
            for i, rec in enumerate(recommendations['recommendations'][:5]):
                print(f"\n🎬 Recommendation {i+1}:")
                print(f"   🎭 Title: {rec['title']}")
                print(f"   🎨 Genres: {', '.join(rec['genres'])}")
                print(f"   😤 Frustration Compatibility: {rec['frustration_compatibility']:.2f}")
                print(f"   🧠 Cognitive Compatibility: {rec['cognitive_compatibility']:.2f}")
                print(f"   👤 Persona Match: {rec['persona_match']:.2f}")
                print(f"   ⭐ Overall Score: {rec['overall_score']:.2f}")
                print(f"   💡 Reasoning: {rec['reasoning']}")
        else:
            print(f"❌ Recommendation failed: {recommendations}")
        
        # Get final session statistics
        stats_response = requests.get(f"{self.api_url}/session_stats")
        stats = stats_response.json()
        print(f"\n📊 Final Session Statistics:")
        print(f"   🔄 Active Sessions: {stats.get('active_sessions', 0)}")
        print(f"   📝 Total Events: {stats.get('total_events', 0)}")
        print(f"   😤 Average Frustration: {stats.get('avg_frustration', 0):.3f}")
        print(f"   🧠 Average Cognitive Load: {stats.get('avg_cognitive_load', 0):.3f}")
    
    def _analyze_prediction_accuracy(self, prediction_history):
        """Analyze the accuracy of multi-target predictions."""
        if not prediction_history:
            return
        
        print(f"\n📊 Multi-Target Prediction Analysis")
        print("="*80)
        
        # Calculate errors
        frustration_errors = [abs(p['pred_frustration'] - p['actual_frustration']) for p in prediction_history]
        cognitive_errors = [abs(p['pred_cognitive'] - p['actual_cognitive']) for p in prediction_history]
        
        # Calculate metrics
        avg_frustration_error = sum(frustration_errors) / len(frustration_errors)
        avg_cognitive_error = sum(cognitive_errors) / len(cognitive_errors)
        max_frustration_error = max(frustration_errors)
        max_cognitive_error = max(cognitive_errors)
        
        print(f"🎯 Frustration Prediction Accuracy:")
        print(f"   📊 Average Error: {avg_frustration_error:.3f}")
        print(f"   📊 Maximum Error: {max_frustration_error:.3f}")
        print(f"   📊 Error Rate: {(avg_frustration_error / 1.0) * 100:.1f}%")
        
        print(f"\n🧠 Cognitive Load Prediction Accuracy:")
        print(f"   📊 Average Error: {avg_cognitive_error:.3f}")
        print(f"   📊 Maximum Error: {max_cognitive_error:.3f}")
        print(f"   📊 Error Rate: {(avg_cognitive_error / 1.0) * 100:.1f}%")
        
        # Overall assessment
        overall_accuracy = 1.0 - ((avg_frustration_error + avg_cognitive_error) / 2.0)
        print(f"\n⭐ Overall Multi-Target Accuracy: {overall_accuracy * 100:.1f}%")
        
        if overall_accuracy > 0.8:
            print("✅ Excellent prediction accuracy!")
        elif overall_accuracy > 0.6:
            print("✅ Good prediction accuracy!")
        elif overall_accuracy > 0.4:
            print("⚠️ Moderate prediction accuracy - consider model tuning")
        else:
            print("❌ Poor prediction accuracy - model needs improvement")
    
    def run_stress_test(self):
        """Run a stress test with multiple concurrent sessions."""
        print(f"\n🔥 Running Multi-Target Stress Test...")
        print("="*80)
        
        # Create multiple test sessions
        sessions = []
        for i in range(5):
            session_data = {
                "user_id": f"stress_test_user_{i}",
                "session_id": f"stress_session_{i}",
                "events": generate_realistic_multitarget_test_log()
            }
            sessions.append(session_data)
        
        # Process all sessions concurrently
        import threading
        
        def process_session(session_data):
            for event in session_data["events"]:
                event["user_id"] = session_data["user_id"]
                event["session_id"] = session_data["session_id"]
                
                try:
                    response = requests.post(
                        f"{self.api_url}/update_session",
                        json={
                            "user_id": session_data["user_id"],
                            "session_id": session_data["session_id"],
                            "event": event
                        },
                        timeout=5
                    )
                    if response.status_code != 200:
                        print(f"❌ Error in session {session_data['session_id']}: {response.status_code}")
                except Exception as e:
                    print(f"❌ Exception in session {session_data['session_id']}: {e}")
                
                time.sleep(0.1)  # Small delay between events
        
        # Start all sessions
        threads = []
        start_time = time.time()
        
        for session_data in sessions:
            thread = threading.Thread(target=process_session, args=(session_data,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Get final stats
        stats_response = requests.get(f"{self.api_url}/session_stats")
        stats = stats_response.json()
        
        print(f"✅ Stress test completed in {end_time - start_time:.2f} seconds")
        print(f"📊 Final Stats:")
        print(f"   🔄 Active Sessions: {stats.get('active_sessions', 0)}")
        print(f"   📝 Total Events: {stats.get('total_events', 0)}")
        print(f"   😤 Average Frustration: {stats.get('avg_frustration', 0):.3f}")
        print(f"   🧠 Average Cognitive Load: {stats.get('avg_cognitive_load', 0):.3f}")

def main():
    """Run the complete multi-target test suite."""
    
    print("🔧 Generating realistic multi-target test data...")
    
    # Initialize tester
    tester = MultiTargetRecommendationTester()
    
    # Check if API is running
    if not tester.test_health():
        print("❌ Multi-target API is not running. Please start the server first:")
        print("   python api_server_multitarget.py")
        return
    
    print("\n" + "="*80)
    print("🚀 MULTI-TARGET NEURAL RDE TESTING SUITE")
    print("="*80)
    
    # Run main test
    tester.run_multitarget_test()
    
    # Run stress test
    tester.run_stress_test()
    
    print(f"\n✅ Multi-target testing completed successfully!")
    print("🎯 Your system can now predict both frustration and cognitive load")
    print("🎬 Recommendations are optimized for both psychological dimensions")

if __name__ == "__main__":
    main()
