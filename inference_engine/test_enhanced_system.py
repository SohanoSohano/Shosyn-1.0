import requests
import json
import time
from datetime import datetime, timedelta

def test_enhanced_psychological_tracking():
    """Test the enhanced psychological tracking system."""
    
    api_url = "http://localhost:5000"
    user_id = "enhanced_test_user"
    session_id = "enhanced_session_001"
    
    # Simulate a user journey with increasing then decreasing frustration
    events = [
        {"action_type": "session_start", "frustration_level": 0.05, "cognitive_load": 0.1},
        {"action_type": "dpad_right", "frustration_level": 0.10, "cognitive_load": 0.15},   # Clear increase
        {"action_type": "dpad_right", "frustration_level": 0.20, "cognitive_load": 0.30},   # Clear increase
        {"action_type": "back", "frustration_level": 0.35, "cognitive_load": 0.45},         # Peak stress
        {"action_type": "back", "frustration_level": 0.40, "cognitive_load": 0.50},         # Still high
        {"action_type": "dpad_down", "frustration_level": 0.25, "cognitive_load": 0.30},    # Clear decrease
        {"action_type": "click", "frustration_level": 0.10, "cognitive_load": 0.15},        # Recovery
    ]
    
    print("🧠 Testing Enhanced Psychological Tracking")
    print("="*60)
    
    for i, event_data in enumerate(events):
        event_data["timestamp"] = datetime.now().isoformat()
        
        event = {
            "user_id": user_id,
            "session_id": session_id,
            "event": event_data
        }
        
        response = requests.post(f"{api_url}/update_session", json=event)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📱 Event {i+1}: {event_data['action_type']}")
            print(f"   Actual: F={event_data['frustration_level']:.3f}, C={event_data['cognitive_load']:.3f}")
            
            if 'psychological_trends' in result:
                trends = result['psychological_trends']
                print(f"   Predicted: F={result['predicted_frustration']:.3f}, C={result['predicted_cognitive_load']:.3f}")
                print(f"   Trends: Frustration={trends['frustration_trend']}, Cognitive={trends['cognitive_trend']}")
                print(f"   Recovery Phase: {trends['recovery_phase']}")
                print(f"   Session Duration: {trends['session_duration']:.1f}s")
        
        time.sleep(1)  # Simulate realistic timing
    
    # Test recommendations with enhanced reasoning
    print(f"\n🎯 Testing Enhanced Recommendations...")
    rec_response = requests.post(f"{api_url}/get_recommendations", json={
        "user_id": user_id,
        "session_id": session_id,
        "user_preferences": {"preferred_genres": ["Comedy", "Animation"]}
    })
    
    if rec_response.status_code == 200:
        recs = rec_response.json()
        print(f"✅ Generated {recs['count']} enhanced recommendations")
        
        for i, rec in enumerate(recs['recommendations'][:3]):
            print(f"\n🎬 Enhanced Recommendation {i+1}:")
            print(f"   Title: {rec['title']}")
            print(f"   Score: {rec['overall_score']:.3f}")
            print(f"   Enhanced Reasoning: {rec['reasoning']}")

if __name__ == "__main__":
    test_enhanced_psychological_tracking()
