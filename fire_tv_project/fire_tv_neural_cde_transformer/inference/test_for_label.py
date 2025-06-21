# Test script to verify complete data
import requests
import json

response = requests.post("http://127.0.0.1:5001/recommendations/enhanced", json={
    "user_id": "test_complete_traits",
    "history": [{"feature_1": 0.1, "feature_2": 0.2, "feature_3": 0.3, "feature_4": 0.4, "feature_5": 0.5, 
                 "feature_6": 0.6, "feature_7": 0.7, "feature_8": 0.8, "feature_9": 0.9, "feature_10": 1.0, 
                 "feature_11": 1.1, "feature_12": 1.2, "feature_13": 1.3, "feature_14": 1.4, "feature_15": 1.5, 
                 "feature_16": 1.6, "feature_17": 1.7, "feature_18": 1.8}],
    "top_k": 3
})

data = response.json()

# Check each recommendation for complete traits
for i, rec in enumerate(data["recommendations"]):
    trait_count = len([key for key in rec.keys() if key.startswith("trait_")])
    print(f"Recommendation {i+1} ({rec['item_id']}): {trait_count} traits")
    
    # Verify all 15 traits are present
    missing_traits = []
    for trait_num in range(1, 16):
        if f"trait_{trait_num}" not in rec:
            missing_traits.append(trait_num)
    
    if missing_traits:
        print(f"  ❌ Missing traits: {missing_traits}")
    else:
        print(f"  ✅ All 15 traits present")
