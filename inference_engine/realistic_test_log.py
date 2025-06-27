# realistic_test_log.py
import json
from datetime import datetime, timedelta

def generate_realistic_test_log():
    """Generate a realistic Fire TV user session log for testing."""
    
    base_time = datetime.now()
    
    # Realistic Fire TV session - user browsing and getting frustrated
    realistic_log = [
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_test_001", 
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
            "cognitive_load": 0.1
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_test_001",
            "action_type": "dpad_right", 
            "timestamp": (base_time + timedelta(seconds=2)).isoformat(),
            "screen_context": "Home",
            "focused_item": json.dumps({
                "item_id": "movie_123",
                "title": "Action Movie",
                "genres": ["Action", "Thriller"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 2.0,
                "consecutive_action_count": 1
            }),
            "frustration_level": 0.08,
            "cognitive_load": 0.15
        },
        {
            "user_id": "user_stressed_professional_1", 
            "session_id": "session_test_001",
            "action_type": "dpad_right",
            "timestamp": (base_time + timedelta(seconds=3)).isoformat(),
            "screen_context": "Home",
            "focused_item": json.dumps({
                "item_id": "movie_124", 
                "title": "Horror Film",
                "genres": ["Horror", "Thriller"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 1.0,
                "consecutive_action_count": 2
            }),
            "frustration_level": 0.12,
            "cognitive_load": 0.18
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_test_001", 
            "action_type": "dpad_right",
            "timestamp": (base_time + timedelta(seconds=4)).isoformat(),
            "screen_context": "Home",
            "focused_item": json.dumps({
                "item_id": "movie_125",
                "title": "Another Action Movie", 
                "genres": ["Action", "Adventure"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 1.0,
                "consecutive_action_count": 3
            }),
            "frustration_level": 0.18,  # Frustration building
            "cognitive_load": 0.22
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_test_001",
            "action_type": "back",
            "timestamp": (base_time + timedelta(seconds=7)).isoformat(),
            "screen_context": "Home", 
            "focused_item": json.dumps({
                "item_id": "movie_123",
                "title": "Action Movie",
                "genres": ["Action", "Thriller"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 3.0,
                "consecutive_action_count": 1
            }),
            "frustration_level": 0.25,  # Getting frustrated
            "cognitive_load": 0.28
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_test_001",
            "action_type": "dpad_down",
            "timestamp": (base_time + timedelta(seconds=9)).isoformat(),
            "screen_context": "Home",
            "focused_item": json.dumps({
                "item_id": "movie_200",
                "title": "Comedy Special",
                "genres": ["Comedy"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 2.0,
                "consecutive_action_count": 1
            }),
            "frustration_level": 0.22,  # Slight relief seeing comedy
            "cognitive_load": 0.25
        },
        {
            "user_id": "user_stressed_professional_1",
            "session_id": "session_test_001",
            "action_type": "click",
            "timestamp": (base_time + timedelta(seconds=12)).isoformat(),
            "screen_context": "Detail_Page",
            "focused_item": json.dumps({
                "item_id": "movie_200",
                "title": "Comedy Special", 
                "genres": ["Comedy"]
            }),
            "sequence_context": json.dumps({
                "time_since_last_action": 3.0,
                "consecutive_action_count": 1
            }),
            "click_type": "more_info",
            "frustration_level": 0.18,  # Decision made, frustration reducing
            "cognitive_load": 0.20
        }
    ]
    
    return realistic_log

# Generate and save test log
test_log = generate_realistic_test_log()
with open('realistic_test_log.json', 'w') as f:
    json.dump(test_log, f, indent=2)

print("Generated realistic test log with", len(test_log), "events")
print("User journey: session_start → browsing action content → getting frustrated → finding comedy → clicking for more info")
