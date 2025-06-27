# test_improvements.py
import requests
import json
from datetime import datetime, timedelta

def test_improved_predictions():
    """Test the improved prediction system."""
    api_url = "http://localhost:5000"
    
    # High frustration scenario
    events = [
        {"action_type": "session_start", "frustration_level": 0.1, "cognitive_load": 0.2},
        {"action_type": "dpad_right", "frustration_level": 0.3, "cognitive_load": 0.4},
        {"action_type": "back", "frustration_level": 0.6, "cognitive_load": 0.7},
        {"action_type": "back", "frustration_level": 0.8, "cognitive_load": 0.8}
    ]
    
    user_id = "test_improved_user"
    session_id = "test_improved_session"
    
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
            if 'predicted_frustration' in result:
                print(f"Event {i+1}: Predicted F={result['predicted_frustration']:.3f}, C={result['predicted_cognitive_load']:.3f}")
                print(f"         Expected F={event_data['frustration_level']:.3f}, C={event_data['cognitive_load']:.3f}")

if __name__ == "__main__":
    test_improved_predictions()
