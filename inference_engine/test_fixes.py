# test_fixes.py
import requests
import json
import time
from datetime import datetime, timedelta

def test_all_fixes():
    """Test all three fixes comprehensively."""
    
    api_url = "http://localhost:5000"
    user_id = "fix_test_user"
    session_id = "fix_test_session"
    
    # Test pattern: clear increase then clear decrease (should trigger recovery)
    events = [
        {"action_type": "session_start", "frustration_level": 0.05, "cognitive_load": 0.1},
        {"action_type": "dpad_right", "frustration_level": 0.15, "cognitive_load": 0.2},
        {"action_type": "back", "frustration_level": 0.35, "cognitive_load": 0.4},      # Peak
        {"action_type": "back", "frustration_level": 0.40, "cognitive_load": 0.45},     # Still high
        {"action_type": "dpad_down", "frustration_level": 0.25, "cognitive_load": 0.3}, # Recovery start
        {"action_type": "click", "frustration_level": 0.10, "cognitive_load": 0.15},    # Clear recovery
    ]
    
    print("🔧 Testing All Fixes")
    print("="*60)
    
    for i, event_data in enumerate(events):
        event_data["timestamp"] = datetime.now().isoformat()
        
        response = requests.post(f"{api_url}/update_session", json={
            "user_id": user_id,
            "session_id": session_id,
            "event": event_data
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📱 Event {i+1}: {event_data['action_type']}")
            print(f"   Actual: F={event_data['frustration_level']:.3f}, C={event_data['cognitive_load']:.3f}")
            
            if 'psychological_trends' in result:
                trends = result['psychological_trends']
                print(f"   Predicted: F={result['predicted_frustration']:.3f}, C={result['predicted_cognitive_load']:.3f}")
                print(f"   Trends: F={trends['frustration_trend']}, C={trends['cognitive_trend']}")
                print(f"   🎯 Recovery Phase: {trends['recovery_phase']}")
                
                # Verify fixes
                if i >= 4 and trends['recovery_phase']:
                    print("   ✅ RECOVERY DETECTION WORKING!")
                if i == 5 and trends['frustration_trend'] == 'decreasing':
                    print("   ✅ TREND DETECTION FIXED!")
                if result['predicted_frustration'] > 0.15:
                    print("   ✅ PREDICTION SCALING IMPROVED!")

if __name__ == "__main__":
    test_all_fixes()
