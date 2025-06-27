# test_recommendation_system.py
import requests
import json
import time
from datetime import datetime, timedelta

class RecommendationTester:
    """Comprehensive tester for the recommendation system."""
    
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        
    def test_health(self):
        """Test if the API is running."""
        try:
            response = requests.get(f"{self.api_url}/health")
            print(f"Health Check: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def run_realistic_test(self):
        """Run a complete test with realistic Fire TV user behavior."""
        
        # Load realistic test log
        with open('realistic_test_log.json', 'r') as f:
            test_events = json.load(f)
        
        user_id = test_events[0]['user_id']
        session_id = test_events[0]['session_id']
        
        print(f"\n🎬 Testing Recommendation System")
        print(f"User: {user_id}")
        print(f"Session: {session_id}")
        print("="*60)
        
        # Send events one by one and track frustration prediction
        for i, event in enumerate(test_events):
            print(f"\n📱 Event {i+1}: {event['action_type']}")
            print(f"   Timestamp: {event['timestamp']}")
            print(f"   Actual Frustration: {event.get('frustration_level', 'N/A')}")
            
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
            print(f"   API Response: {result.get('status', 'unknown')}")
            
            if 'predicted_frustration' in result:
                predicted = result['predicted_frustration']
                actual = event.get('frustration_level', 0)
                print(f"   🧠 Predicted Frustration: {predicted:.3f}")
                print(f"   📊 Actual Frustration: {actual:.3f}")
                print(f"   📈 Prediction Error: {abs(predicted - actual):.3f}")
                
                # Check if recommendations are needed
                if result.get('recommendations_needed', False):
                    print(f"   ⚠️  High frustration detected - recommendations needed!")
            
            time.sleep(0.5)  # Simulate realistic timing
        
        # Get final recommendations
        print(f"\n🎯 Getting Final Recommendations...")
        print("="*60)
        
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
            print(f"✅ Generated {recommendations['count']} recommendations")
            
            for i, rec in enumerate(recommendations['recommendations'][:5]):
                print(f"\n🎬 Recommendation {i+1}:")
                print(f"   Title: {rec['title']}")
                print(f"   Genres: {', '.join(rec['genres'])}")
                print(f"   Frustration Compatibility: {rec['frustration_compatibility']:.2f}")
                print(f"   Persona Match: {rec['persona_match']:.2f}")
                print(f"   Overall Score: {rec['overall_score']:.2f}")
                print(f"   Reasoning: {rec['reasoning']}")
        else:
            print(f"❌ Recommendation failed: {recommendations}")
        
        # Get session statistics
        stats_response = requests.get(f"{self.api_url}/session_stats")
        stats = stats_response.json()
        print(f"\n📊 Session Statistics:")
        print(f"   Active Sessions: {stats.get('active_sessions', 0)}")
        print(f"   Total Events: {stats.get('total_events', 0)}")
        print(f"   Average Frustration: {stats.get('avg_frustration', 0):.3f}")

def main():
    """Run the complete test suite."""
    
    # Generate realistic test data
    print("🔧 Generating realistic test data...")
    exec(open('realistic_test_log.py').read())
    
    # Initialize tester
    tester = RecommendationTester()
    
    # Check if API is running
    if not tester.test_health():
        print("❌ API is not running. Please start the server first:")
        print("   python api_server.py")
        return
    
    # Run realistic test
    tester.run_realistic_test()
    
    print(f"\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()
