import requests
import json

url = "http://127.0.0.1:5001/recommendations"

# Example user history (adjust as per your model's expected input feature_dim)
# Ensure the number of features matches your model's input_dim (18)
user_data = {
    "user_id": "test_user_from_python",
    "history": [
        {"feature_1": 0.1, "feature_2": 0.2, "feature_3": 0.3, "feature_4": 0.4, "feature_5": 0.5, "feature_6": 0.6, "feature_7": 0.7, "feature_8": 0.8, "feature_9": 0.9, "feature_10": 1.0, "feature_11": 1.1, "feature_12": 1.2, "feature_13": 1.3, "feature_14": 1.4, "feature_15": 1.5, "feature_16": 1.6, "feature_17": 1.7, "feature_18": 1.8},
        {"feature_1": 0.2, "feature_2": 0.3, "feature_3": 0.4, "feature_4": 0.5, "feature_5": 0.6, "feature_6": 0.7, "feature_7": 0.8, "feature_8": 0.9, "feature_9": 1.0, "feature_10": 1.1, "feature_11": 1.2, "feature_12": 1.3, "feature_14": 1.5, "feature_15": 1.6, "feature_16": 1.7, "feature_17": 1.8, "feature_18": 1.9}
    ]
}

headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, data=json.dumps(user_data), headers=headers)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status code: {e.response.status_code}")
        print(f"Response body: {e.response.text}")

