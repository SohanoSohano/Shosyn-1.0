# inference/app.py (lines 1-25)
import traceback
import sys
import os

# --- CRITICAL PATH SETUP (MUST BE FIRST) ---
# Get the absolute path of the directory containing app.py (i.e., 'inference')
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up to reach the project root ('fire_tv_neural_cde_transformer')
project_root = os.path.dirname(current_dir)

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Appended project root to sys.path: {project_root}")

# Add the inference directory to sys.path for local imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"Appended inference dir to sys.path: {current_dir}")
# --- END CRITICAL PATH SETUP ---

# Now import Flask and other modules
from flask import Flask, request, jsonify

# Import your enhanced recommender (this should now work)
from enhanced_recommender import EnhancedFireTVRecommendationService

# Now that the path is set, all local imports should work
from flask import Flask, request, jsonify
from recommender import RecommendationService # This import will now succeed
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('error.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# --- Configuration ---
# Use os.path.join for robust path construction
MODEL_PATH = os.path.join(project_root, "models", "tmdb_enhanced_hybrid_model_cuda.pth")
ITEM_CATALOG_PATH = os.path.join(project_root, "data", "item_catalog.csv")
# --- Initialize Flask App and Recommendation Service ---
print("Starting Recommendation API...")
app = Flask(__name__)
app.logger.addHandler(handler)
# Initialize the service ONCE when the app starts
recommender = EnhancedFireTVRecommendationService(model_path=MODEL_PATH, item_catalog_path=ITEM_CATALOG_PATH)
print("✅ API is live and ready for requests.")

@app.route('/recommendations/enhanced', methods=['POST'])
def enhanced_recommend():
    """Enhanced API endpoint with full interpretability"""
    data = request.get_json()
    if not data or 'user_id' not in data or 'history' not in data:
        return jsonify({"error": "Missing 'user_id' or 'history' in request"}), 400
    
    user_id = data['user_id']
    user_history = data['history']
    top_k = data.get('top_k', 10)
    
    try:
        enhanced_recommendations = recommender.get_enhanced_recommendations(user_id, user_history, top_k)
        return jsonify(enhanced_recommendations)
    except Exception as e:
        print(f"Error during enhanced recommendation: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Failed to generate enhanced recommendations", 
            "details": str(e)
        }), 500

if __name__ == "__main__":
    # For production, use a proper WSGI server like Gunicorn or uWSGI
    app.run(host='0.0.0.0', port=5001, debug=True)
