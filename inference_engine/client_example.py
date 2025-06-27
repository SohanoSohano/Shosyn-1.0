import requests
import json
import time

class RecommendationClient:
    """Client for interacting with the recommendation API."""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
    
    def update_session(self, user_id: str, session_id: str, event: dict):
        """Send user event to update session."""
        response = requests.post(
            f"{self.api_url}/update_session",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "event": event
            }
        )
        return response.json()
    
    def get_recommendations(self, user_id: str, session_id: str, user_preferences: dict = None):
        """Get movie recommendations."""
        response = requests.post(
            f"{self.api_url}/get_recommendations",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "user_preferences": user_preferences
            }
        )
        return response.json()

# Example usage
if __name__ == "__main__":
    client = RecommendationClient()
    
    # Simulate user session
    user_id = "user_123"
    session_id = "session_456"
    
    # Send some events
    events = [
        {"action_type": "session_start", "timestamp": time.time()},
        {"action_type": "dpad_right", "timestamp": time.time() + 1},
        {"action_type": "click", "timestamp": time.time() + 2},
        {"action_type": "back", "timestamp": time.time() + 3}
    ]
    
    for event in events:
        result = client.update_session(user_id, session_id, event)
        print(f"Event result: {result}")
        time.sleep(1)
    
    # Get recommendations
    user_preferences = {
        "preferred_genres": ["Action", "Comedy", "Sci-Fi"]
    }
    
    recommendations = client.get_recommendations(user_id, session_id, user_preferences)
    print(f"\nRecommendations: {json.dumps(recommendations, indent=2)}")
