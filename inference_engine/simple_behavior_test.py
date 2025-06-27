# simple_behavior_test.py
import requests
import json
import time
from datetime import datetime

def test_fire_tv_behaviors():
    """Test realistic Fire TV user behaviors on Windows."""
    
    api_url = "http://localhost:5000"
    
    print("🎬 Testing Fire TV User Behaviors on Windows")
    print("="*60)
    
    # Test health first
    try:
        health = requests.get(f"{api_url}/health")
        print(f"✅ API Health: {health.json()['status']}")
    except:
        print("❌ API not running. Start api_server_multitarget.py first")
        return
    
    # Scenario 1: Frustrated User
    print("\n📱 Scenario 1: Frustrated User Journey")
    user_id = "frustrated_windows_user"
    session_id = "session_001"
    
    behaviors = [
        {"action": "session_start", "frustration": 0.1, "cognitive": 0.2, "desc": "User starts browsing"},
        {"action": "dpad_right", "frustration": 0.25, "cognitive": 0.35, "desc": "Browsing complex content"},
        {"action": "dpad_right", "frustration": 0.4, "cognitive": 0.5, "desc": "Still searching"},
        {"action": "back", "frustration": 0.6, "cognitive": 0.6, "desc": "Getting frustrated"},
        {"action": "dpad_down", "frustration": 0.45, "cognitive": 0.4, "desc": "Trying different row"},
        {"action": "click", "frustration": 0.2, "cognitive": 0.2, "desc": "Found simple content"}
    ]
    
    for i, behavior in enumerate(behaviors):
        print(f"\n  Step {i+1}: {behavior['desc']}")
        print(f"    Action: {behavior['action']}")
        print(f"    Expected - Frustration: {behavior['frustration']:.1f}, Cognitive: {behavior['cognitive']:.1f}")
        
        event = {
            "user_id": user_id,
            "session_id": session_id,
            "event": {
                "action_type": behavior['action'],
                "timestamp": datetime.now().isoformat(),
                "frustration_level": behavior['frustration'],
                "cognitive_load": behavior['cognitive'],
                "screen_context": "Home",
                "focused_item": json.dumps({
                    "item_id": f"item_{i}",
                    "title": f"Content {i}",
                    "genres": ["Action", "Drama"]
                })
            }
        }
        
        try:
            response = requests.post(f"{api_url}/update_session", json=event)
            if response.status_code == 200:
                result = response.json()
                if 'predicted_frustration' in result:
                    pred_f = result['predicted_frustration']
                    pred_c = result['predicted_cognitive_load']
                    print(f"    Predicted - Frustration: {pred_f:.3f}, Cognitive: {pred_c:.3f}")
                    
                    # Calculate accuracy
                    f_error = abs(pred_f - behavior['frustration'])
                    c_error = abs(pred_c - behavior['cognitive'])
                    print(f"    Accuracy - Frustration Error: {f_error:.3f}, Cognitive Error: {c_error:.3f}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        time.sleep(0.5)
    
    # Get recommendations
    print(f"\n🎯 Getting Recommendations...")
    rec_request = {
        "user_id": user_id,
        "session_id": session_id,
        "user_preferences": {"preferred_genres": ["Comedy", "Family"]}
    }
    
    try:
        rec_response = requests.post(f"{api_url}/get_recommendations", json=rec_request)
        if rec_response.status_code == 200:
            recs = rec_response.json()
            print(f"✅ Generated {recs['count']} recommendations")
            
            for i, rec in enumerate(recs['recommendations'][:3]):
                print(f"  {i+1}. {rec['title']} (Score: {rec['overall_score']:.2f})")
                print(f"     Frustration Compatibility: {rec['frustration_compatibility']:.2f}")
                print(f"     Cognitive Compatibility: {rec['cognitive_compatibility']:.2f}")
                print(f"     Reasoning: {rec['reasoning']}")
    except Exception as e:
        print(f"❌ Recommendation error: {e}")

if __name__ == "__main__":
    test_fire_tv_behaviors()
