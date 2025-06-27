from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from inference_engine import NeuralRDEInferenceEngine
import threading
import time

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize inference engine
inference_engine = NeuralRDEInferenceEngine(
    model_path="best_model_robust_rde.pth",
    movie_catalog_path="tmdb_5000_movies.csv"
)

# Background task for session cleanup
def cleanup_sessions():
    while True:
        time.sleep(300)  # Every 5 minutes
        inference_engine.cleanup_expired_sessions()

cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "stats": inference_engine.get_session_stats()
    })

@app.route('/update_session', methods=['POST'])
def update_session():
    """Update user session with new event."""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['user_id', 'session_id', 'event']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Update session
        result = inference_engine.update_session(
            user_id=data['user_id'],
            session_id=data['session_id'],
            event=data['event']
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Session update error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get movie recommendations for user."""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['user_id', 'session_id']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Get recommendations
        recommendations = inference_engine.get_recommendations(
            user_id=data['user_id'],
            session_id=data['session_id'],
            user_preferences=data.get('user_preferences')
        )
        
        # Convert to JSON-serializable format
        recommendations_json = [
            {
                "item_id": rec.item_id,
                "title": rec.title,
                "genres": rec.genres,
                "frustration_compatibility": rec.frustration_compatibility,
                "persona_match": rec.persona_match,
                "overall_score": rec.overall_score,
                "reasoning": rec.reasoning
            }
            for rec in recommendations
        ]
        
        return jsonify({
            "status": "success",
            "recommendations": recommendations_json,
            "count": len(recommendations_json)
        })
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/session_stats', methods=['GET'])
def session_stats():
    """Get session statistics."""
    return jsonify(inference_engine.get_session_stats())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
