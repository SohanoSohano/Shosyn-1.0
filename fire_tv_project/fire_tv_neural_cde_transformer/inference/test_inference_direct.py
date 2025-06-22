# test_inference_direct.py (TRIAL 2)
import sys
import os
from pathlib import Path

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # Go up two levels from inference/test_inference_direct.py
sys.path.insert(0, str(project_root))

from inference.psychological_inference_engine import PsychologicalInferenceEngine

def test_direct_inference():
    """Test the inference engine directly without middleware."""
    
    print("🧪 Testing Direct Inference Pipeline")
    print("=" * 50)
    
    try:
        # Initialize the inference engine
        print("1. Loading model and creating movie profiles...")
        engine = PsychologicalInferenceEngine(
            model_path=r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\best_performance_model.pth",
            tmdb_cache_path=r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\fire_tv_neural_cde_transformer_instance_version\Shosyn-1.0\fire_tv_project\fire_tv_neural_cde_transformer\tmdb_local_catalog.json",
            device='auto'
        )
        
        # Test different user behavior patterns
        test_users = {
            "Power User": {
                'dpad_up_count': 15, 'dpad_down_count': 12, 'dpad_left_count': 8, 'dpad_right_count': 9,
                'back_button_presses': 1, 'menu_revisits': 0, 'scroll_speed': 80, 'hover_duration': 1.5,
                'time_since_last_interaction': 2.0
            },
            "Casual Viewer": {
                'dpad_up_count': 5, 'dpad_down_count': 4, 'dpad_left_count': 2, 'dpad_right_count': 3,
                'back_button_presses': 0, 'menu_revisits': 1, 'scroll_speed': 120, 'hover_duration': 3.0,
                'time_since_last_interaction': 8.0
            },
            "Frustrated User": {
                'dpad_up_count': 20, 'dpad_down_count': 18, 'dpad_left_count': 12, 'dpad_right_count': 15,
                'back_button_presses': 8, 'menu_revisits': 5, 'scroll_speed': 200, 'hover_duration': 0.8,
                'time_since_last_interaction': 1.0
            }
        }
        
        for user_type, behavior in test_users.items():
            print(f"\n2. Testing {user_type}:")
            print(f"   Behavior: {behavior}")
            
            # Get psychological profile
            psych_vector = engine.predict_user_psychology(behavior)
            print(f"   Psychological Vector: {psych_vector.cpu().numpy().flatten()}")
            
            # Get recommendations
            recommendations = engine.get_recommendations(psych_vector, top_k=5)
            
            print(f"   Top 5 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"     {i}. {rec['title']} ({rec['content_genre']}) - Score: {rec['similarity_score']:.3f}")
        
        print("\n✅ Direct inference test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_direct_inference()
