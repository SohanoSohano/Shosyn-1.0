# inference/api_server.py
from flask import Flask, request, jsonify
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.recommendation_pipeline import RecommendationPipeline
import pandas as pd

app = Flask(__name__)

# Initialize your pipeline with your specific paths
pipeline = RecommendationPipeline(
    model_path=r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\best_performance_model.pth",
    tmdb_cache_path=r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\fire_tv_neural_cde_transformer_instance_version\Shosyn-1.0\fire_tv_project\fire_tv_neural_cde_transformer\tmdb_local_catalog.json",
    device='auto'
)

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        user_history = data.get('user_history', [])
        top_k = data.get('top_k', 10)
        
        # Convert behavioral history to DataFrame format
        user_df = pd.DataFrame(user_history)
        
        # Generate recommendations
        recommendations = pipeline.recommend_for_user(
            user_df=user_df,
            user_id=user_id,
            top_k=top_k
        )
        
        return jsonify(recommendations)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    health = pipeline.health_check()
    return jsonify(health)

if __name__ == '__main__':
    print("🚀 Starting Psychological Recommendation API Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
