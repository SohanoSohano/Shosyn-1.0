# inference/main_inference_script.py (TRIAL 2)
import sys
import os
import pandas as pd
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.recommendation_pipeline import RecommendationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import your Firebase data fetching functions
# (Assuming these are in a separate module based on your code)
def fetch_data_from_firebase_rest(node_path):
    """Your existing Firebase fetching function"""
    # ... (your existing implementation)
    pass

def load_content_metadata(filepath):
    """Your existing metadata loading function"""
    # ... (your existing implementation)
    pass

def process_logs_to_dataframe(logs_dict):
    """Your existing log processing function"""
    # ... (your existing implementation)
    pass

def main():
    """Main execution function for the inference pipeline."""
    try:
        logger.info("🚀 Starting FireTV Recommendation Inference Pipeline")
        
        # Configuration
        FIREBASE_REST_URL = "https://firetv-project-ba2ad-default-rtdb.asia-southeast1.firebasedatabase.app"
        DATABASE_NODES = {
            "movements": "logs/movement.json",
            "clicks": "logs/clicks.json",
            "hovers": "logs/hovers.json",
            "hover_durations": "logs/hover_durations.json"
        }
        CONTENT_METADATA_FILE = 'netflix_prime_content_deduped.json'
        MODEL_PATH = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\best_performance_model.pth"
        TMDB_CACHE_PATH = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\fire_tv_neural_cde_transformer_instance_version\Shosyn-1.0\fire_tv_project\fire_tv_neural_cde_transformer\tmdb_local_catalog.json"
        
        # Initialize the recommendation pipeline
        logger.info("Initializing recommendation pipeline...")
        pipeline = RecommendationPipeline(
            model_path=MODEL_PATH,
            tmdb_cache_path=TMDB_CACHE_PATH,
            device='auto'
        )
        
        # Perform health check
        health_status = pipeline.health_check()
        logger.info(f"Pipeline health check: {health_status['status']}")
        
        if health_status['status'] != 'healthy':
            logger.error(f"Pipeline unhealthy: {health_status.get('error', 'Unknown error')}")
            return
        
        # Fetch data from Firebase
        logger.info("Fetching data from Firebase...")
        all_logs = {}
        for node, path in DATABASE_NODES.items():
            try:
                data = fetch_data_from_firebase_rest(path)
                all_logs[node] = data
                logger.info(f"Fetched {len(data) if data else 0} records from {node}")
            except Exception as e:
                logger.error(f"Failed to fetch {node}: {e}")
                all_logs[node] = []
        
        # Process logs into DataFrame
        logger.info("Processing logs into structured format...")
        master_df = process_logs_to_dataframe(all_logs)
        
        if master_df.empty:
            logger.warning("No interaction data found. Exiting.")
            return
        
        logger.info(f"Processed {len(master_df)} total interactions")
        
        # Generate recommendations for each user
        unique_users = master_df['user_id'].unique()
        logger.info(f"Generating recommendations for {len(unique_users)} users...")
        
        all_recommendations = {}
        
        for i, user_id in enumerate(unique_users, 1):
            try:
                logger.info(f"Processing user {i}/{len(unique_users)}: {user_id}")
                
                # Get user's interaction data
                user_df = master_df[master_df['user_id'] == user_id].copy()
                
                # Generate recommendations
                recommendations = pipeline.recommend_for_user(
                    user_df=user_df,
                    user_id=user_id,
                    top_k=10,
                    use_sessions=True
                )
                
                all_recommendations[user_id] = recommendations
                
                # Display results for this user
                display_user_recommendations(user_id, recommendations)
                
            except Exception as e:
                logger.error(f"Failed to process user {user_id}: {e}")
                continue
        
        # Save results
        output_file = f"recommendations_output_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_recommendations, f, indent=2, default=str)
        
        logger.info(f"✅ Recommendations saved to {output_file}")
        logger.info(f"🎉 Pipeline completed successfully for {len(all_recommendations)} users")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

def display_user_recommendations(user_id: str, recommendations: dict):
    """Display recommendations for a user in a formatted way."""
    print("\n" + "="*80)
    print(f"🎬 RECOMMENDATIONS FOR USER: {user_id}")
    print("="*80)
    
    if 'error' in recommendations:
        print(f"❌ Error: {recommendations['error']}")
        return
    
    # Display psychological profile
    profile = recommendations.get('psychological_profile', {})
    print(f"\n🧠 PSYCHOLOGICAL PROFILE:")
    print(f"  • Engagement Level:     {profile.get('engagement_level', 0):.3f}")
    print(f"  • Frustration Level:    {profile.get('frustration_level', 0):.3f}")
    print(f"  • Exploration Tendency: {profile.get('exploration_tendency', 0):.3f}")
    
    # Display behavioral summary
    behavior = recommendations.get('behavioral_summary', {})
    print(f"\n📊 BEHAVIORAL SUMMARY:")
    print(f"  • D-Pad Usage:     ↑{behavior.get('dpad_up_count', 0)} ↓{behavior.get('dpad_down_count', 0)} ←{behavior.get('dpad_left_count', 0)} →{behavior.get('dpad_right_count', 0)}")
    print(f"  • Back Presses:    {behavior.get('back_button_presses', 0)}")
    print(f"  • Scroll Speed:    {behavior.get('scroll_speed', 0):.1f} presses/sec")
    print(f"  • Hover Duration:  {behavior.get('hover_duration', 0):.1f} seconds")
    
    # Display recommendations
    recs = recommendations.get('recommendations', [])
    print(f"\n🎯 TOP {len(recs)} RECOMMENDATIONS:")
    
    for i, rec in enumerate(recs, 1):
        print(f"\n  {i}. {rec.get('title', 'Unknown Title')}")
        print(f"     Genre: {rec.get('content_genre', 'Unknown')} | Year: {rec.get('release_year', 'N/A')}")
        print(f"     Similarity: {rec.get('similarity_score', 0):.3f} | Confidence: {rec.get('confidence_level', 'Unknown')}")
        print(f"     TMDb Rating: {rec.get('tmdb_vote_average', 0):.1f} | Popularity: {rec.get('tmdb_popularity', 0):.1f}")
    
    # Display metadata
    metadata = recommendations.get('metadata', {})
    print(f"\n📈 PROCESSING METADATA:")
    print(f"  • Total Interactions: {metadata.get('total_interactions', 0)}")
    print(f"  • Model Confidence:   {metadata.get('model_confidence', 'Unknown')}")
    print(f"  • Processing Method:  {metadata.get('processing_method', 'Unknown')}")

if __name__ == "__main__":
    main()
