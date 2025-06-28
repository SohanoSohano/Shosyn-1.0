# quick_verification_test.py
import requests

def verify_enhancements():
    response = requests.post("http://localhost:5000/update_session", json={
        "user_id": "verify_user",
        "session_id": "verify_session",
        "event": {
            "action_type": "back",
            "timestamp": "2025-06-28T15:00:00Z",
            "frustration_level": 0.8,
            "cognitive_load": 0.9
        }
    })
    
    if response.status_code == 200:
        result = response.json()
        pred_f = result.get('predicted_frustration', 0)
        print(f"High stress prediction: {pred_f:.3f}")
        
        # Should be much higher than 0.39 if enhancements are working
        if pred_f > 0.5:
            print("✅ Enhancements appear to be working!")
        else:
            print("❌ Enhancements not implemented yet")

verify_enhancements()
