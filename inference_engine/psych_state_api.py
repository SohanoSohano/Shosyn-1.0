# psych_state_api.py

from __future__ import annotations

import logging

from flask import Flask, jsonify, request
from flask_cors import CORS

from inference_engine_multitarget import MultiTargetInferenceEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for development


MODEL_PATH = "../ncde/best_model_multitarget_rde.pth"
CATALOG_PATH = "../dataset/tmdb_5000_movies.csv"

DEFAULT_USER_ID = "USER_ABC_123"
GLOBAL_SESSION_ID = "global"

# Initialize a single instance of the inference engine.
# It will hold user session states in memory.
try:
    inference_engine = MultiTargetInferenceEngine(
        model_path=MODEL_PATH,
        movie_catalog_path=CATALOG_PATH
    )
    model_loaded = True
except Exception as e:
    logger.error(f"FATAL: Could not load inference engine model. Error: {e}")
    model_loaded = False

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify the server is running."""
    if not model_loaded:
        return jsonify({
            "status": "unhealthy",
            "reason": "Inference engine model failed to load."
        }), 500
        
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "message": "Psychological State API is online."
    })

@app.route("/psych-state", methods=["POST"])
def update_psych_state():
    """
    Receives a single user interaction event and returns the updated
    predicted frustration and cognitive load.
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # 1. Get the single event from the Fire TV app's request body.
        event_data = request.get_json()
        if not event_data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # 2. Get user_id from query params or use default.
        user_id = request.args.get("user_id", DEFAULT_USER_ID)

        # 3. Process the single event through the inference engine.
        # The engine updates the user's session state internally.
        result = inference_engine.update_session(
            user_id=user_id,
            session_id=GLOBAL_SESSION_ID,
            event=event_data,
        )

        # 4. Extract the predicted values from the engine's response.
        if result.get("status") == "success":
            frustration = result.get("predicted_frustration", 0.0)
            cognitive_load = result.get("predicted_cognitive_load", 0.1)

            # 5. Send the two key values back to the front-end.
            return jsonify({
                "frustration": frustration,
                "cognitive_load": cognitive_load
            }), 200
        elif result.get("status") == "insufficient_data":
            # Not enough events yet to make a prediction.
            return jsonify({
                "frustration": 0.0,
                "cognitive_load": 0.1,
                "status": "insufficient_data"
            }), 202 # Accepted, but not processed
        else:
            # Handle other potential errors from the engine
            return jsonify({"error": result.get("message", "Prediction failed")}), 500

    except Exception as exc:
        logger.exception("Failed to process psychological state update")
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    if not model_loaded:
        logger.error("Server cannot start because the model failed to load.")
    else:
        # Use a production-ready WSGI server like Gunicorn or Waitress
        # in a real deployment instead of app.run().
        logger.info("🚀 Starting Psychological State API on 0.0.0.0:5001 …")
        app.run(host="0.0.0.0", port=5001, debug=False)
