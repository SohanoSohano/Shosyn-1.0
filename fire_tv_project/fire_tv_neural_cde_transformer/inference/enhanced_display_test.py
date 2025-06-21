# inference/enhanced_display_test.py
import requests
import json
import sys

def test_enhanced_recommendations():
    """Test enhanced recommendations with improved display formatting"""
    
    # Test data
    test_data = {
        "user_id": "Hanzo",
        "history": [
            {
                "feature_1": 0.1, "feature_2": 0.2, "feature_3": 0.3, "feature_4": 0.4, "feature_5": 0.5,
                "feature_6": 0.6, "feature_7": 0.7, "feature_8": 0.8, "feature_9": 0.9, "feature_10": 1.0,
                "feature_11": 1.1, "feature_12": 1.2, "feature_13": 1.3, "feature_14": 1.4, "feature_15": 1.5,
                "feature_16": 1.6, "feature_17": 1.7, "feature_18": 1.8
            }
        ],
        "top_k": 3
    }
    
    print("Testing Enhanced Recommendations Display...")
    print("="*80)
    
    try:
        # Make the API request
        response = requests.post(
            "http://127.0.0.1:5001/recommendations/enhanced", 
            json=test_data,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Step 2: Verify complete data structure
        verify_complete_data(data)
        
        # Step 3: Display with improved formatting
        display_enhanced_recommendations(data)
        
        # Step 4: Save formatted output to file
        save_formatted_output(data)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def verify_complete_data(data):
    """Step 2: Verify all recommendations have complete trait data"""
    print("\nVERIFYING DATA COMPLETENESS")
    print("-" * 50)
    
    recommendations = data.get("recommendations", [])
    print(f"Total recommendations received: {len(recommendations)}")
    
    for i, rec in enumerate(recommendations, 1):
        # Count traits
        trait_count = len([key for key in rec.keys() if key.startswith("trait_")])
        
        # Check for missing traits
        missing_traits = []
        for trait_num in range(1, 16):
            trait_key = f"trait_{trait_num}"
            if trait_key not in rec:
                missing_traits.append(trait_num)
        
        # Report status
        status = "✅ COMPLETE" if not missing_traits else f"❌ MISSING {len(missing_traits)} traits"
        print(f"  Recommendation {i} ({rec.get('item_id', 'Unknown')}): {trait_count}/15 traits - {status}")
        
        if missing_traits:
            print(f"    Missing traits: {missing_traits}")

def display_enhanced_recommendations(data):
    """Step 3: Display with improved formatting, including movie genres."""
    print("\n ENHANCED RECOMMENDATIONS DISPLAY")
    print("=" * 80)
    
    # Display user profile summary
    if "fire_tv_profile" in data:
        profile = data["fire_tv_profile"]
        print(f"\n USER PROFILE: {data.get('user_id', 'Unknown')}")
        print(f"   User Type: {profile.get('user_type', 'Unknown')}")
        print(f"   Summary: {data.get('personalization_summary', 'No summary available')}")
    
    # Display each recommendation with clean formatting
    recommendations = data.get("recommendations", [])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{'🎬' * 3} RECOMMENDATION {i} {'🎬' * 3}")
        
        if "enriched_data" in rec and rec["enriched_data"]:
            enriched = rec["enriched_data"]
            print(f"Title: {enriched.get('title', 'N/A')}")
            
            ### MODIFICATION START ###
            # Display genres from the enriched data
            genres = enriched.get('genres', [])  # Expecting a list of genres
            if genres:
                print(f"Genres: {', '.join(genres)}")
            else:
                # Fallback to the original genre field if enriched one is empty
                print(f"Genre: {rec.get('genre', 'N/A')}")
            ### MODIFICATION END ###

            print(f"Tagline: {enriched.get('tagline', 'N/A')}")
            print(f"Overview: {enriched.get('overview', 'N/A')[:100]}...")
            print(f"Release Date: {enriched.get('release_date', 'N/A')}")
            print(f"TMDb Rating: {enriched.get('tmdb_rating', 'N/A')}")
            print(f"Streaming On: {', '.join(enriched.get('streaming_on', ['N/A']))}")
        else:
            # Fallback for items without enriched data
            print(f"Title: {rec.get('title', 'Unknown')}")
            print(f"Item ID: {rec.get('item_id', 'Unknown')}")
            print(f"Genre: {rec.get('genre', 'Unknown')}")
            
        print(f"Item ID: {rec.get('item_id', 'Unknown')}")
        print(f"Score: {rec.get('score', 0):.4f}")
        
        if "score_interpretation" in rec:
            score_info = rec["score_interpretation"]
            print(f"Rating: {score_info.get('band', 'Unknown')} ({score_info.get('description', 'No description')})")
        
        if "named_psychological_traits" in rec:
            print(f"\nPSYCHOLOGICAL TRAITS (Named):")
            sorted_traits = sorted(rec["named_psychological_traits"].items(), key=lambda item: item[0])
            for j in range(0, len(sorted_traits), 3):
                row_traits = sorted_traits[j:j+3]
                display_row = []
                for name, value in row_traits:
                    display_row.append(f"{name}: {value:6.3f}")
                print(f"   {' | '.join(display_row)}")
        
        print("-" * 60)


def save_formatted_output(data):
    """Step 4: Save beautifully formatted JSON to file"""
    print("\nSAVING FORMATTED OUTPUT")
    print("-" * 30)
    
    # Use json.dumps with proper formatting as shown in search results
    formatted_json = json.dumps(data, indent=4, separators=(',', ': '), sort_keys=False)
    
    # Save to file
    filename = "enhanced_recommendations_formatted.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(formatted_json)
        print(f"✅ Formatted output saved to: {filename}")
        print(f"   File size: {len(formatted_json):,} characters")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")
    
    # Also save a compact version for comparison
    compact_json = json.dumps(data, separators=(',', ':'))
    compact_filename = "enhanced_recommendations_compact.json"
    try:
        with open(compact_filename, 'w', encoding='utf-8') as f:
            f.write(compact_json)
        print(f"✅ Compact output saved to: {compact_filename}")
        print(f"   File size: {len(compact_json):,} characters")
        print(f"   Space saved: {len(formatted_json) - len(compact_json):,} characters ({((len(formatted_json) - len(compact_json)) / len(formatted_json) * 100):.1f}%)")
    except Exception as e:
        print(f"❌ Failed to save compact file: {e}")

if __name__ == "__main__":
    success = test_enhanced_recommendations()
    if success:
        print("\nEnhanced display test completed successfully!")
    else:
        print("\nEnhanced display test failed!")
        sys.exit(1)
