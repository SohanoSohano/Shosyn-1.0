# Test the enhanced recommendations
import requests
import json

# Test data
test_data = {
    "user_id": "test_user_enhanced",
    "history": [
        {"feature_1": 0.1, "feature_2": 0.2, "feature_3": 0.3, "feature_4": 0.4, "feature_5": 0.5, 
         "feature_6": 0.6, "feature_7": 0.7, "feature_8": 0.8, "feature_9": 0.9, "feature_10": 1.0, 
         "feature_11": 1.1, "feature_12": 1.2, "feature_13": 1.3, "feature_14": 1.4, "feature_15": 1.5, 
         "feature_16": 1.6, "feature_17": 1.7, "feature_18": 1.8}
    ],
    "top_k": 5
}

response = requests.post("http://127.0.0.1:5001/recommendations/enhanced", json=test_data)
enhanced_result = response.json()

print("Enhanced Recommendations:")
print(json.dumps(enhanced_result, indent=2))
