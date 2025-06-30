# File: ultra_fast_psych_api.py

from __future__ import annotations

import logging
import time
from flask import Flask, jsonify, request
from flask_cors import CORS

from optimized_inference_engine_l4 import MultiTargetInferenceEngine

# ───────────────────────────── Logging ──────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ───────────────────────────── Flask app ────────────────────────────
app = Flask(__name__)
CORS(app)

# ───────────────────────── Configuration ────────────────────────────
MODEL_PATH = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\Neo_Shosyn\Shosyn-1.0\ncde\best_model_multitarget_rde.pth"  # Adjust path as needed
CATALOG_PATH = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\Neo_Shosyn\Shosyn-1.0\dataset\tmdb_5000_movies.csv"       # Adjust path as needed

DEFAULT_USER_ID = "Hanzo"
GLOBAL_SESSION_ID = "global"

# Initialize L4 optimized inference engine
try:
    inference_engine = MultiTargetInferenceEngine(
        model_path=MODEL_PATH,
        movie_catalog_path=CATALOG_PATH,
        batch_size=32,      # Optimize for L4
        max_wait_ms=5       # Ultra-low latency
    )
    model_loaded = True
    logger.info("🚀 L4 Optimized Inference Engine loaded successfully")
except Exception as e:
    logger.error(f"FATAL: Could not load L4 inference engine. Error: {e}")
    model_loaded = False

# ─────────────────────────── API Endpoints ──────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Enhanced health check with L4 GPU status"""
    if not model_loaded:
        return jsonify({
            "status": "unhealthy",
            "reason": "L4 inference engine model failed to load."
        }), 500
    
    stats = inference_engine.get_session_stats()
    
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "gpu_optimized": True,
        "device": stats.get("device", "unknown"),
        "batch_size": stats.get("batch_size", 0),
        "active_users": stats.get("total_users", 0),
        "pending_requests": stats.get("pending_requests", 0),
        "message": "L4 GPU Psychological State API is online and optimized"
    })

@app.route("/psych-state-ultra", methods=["POST"])
def ultra_fast_psych_state():
    """
    Ultra-fast psychological state prediction optimized for L4 GPU.
    Target: sub-10ms response times.
    """
    if not model_loaded:
        return jsonify({"error": "L4 inference engine not loaded"}), 503

    start_time = time.time()
    
    try:
        # Get event data
        event_data = request.get_json()
        if not event_data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        user_id = request.args.get("user_id", DEFAULT_USER_ID)
        
        # Add timestamp if not present
        if "timestamp" not in event_data:
            event_data["timestamp"] = time.time()
        
        # Process through L4 optimized engine
        result = inference_engine.update_session(
            user_id=user_id,
            session_id=GLOBAL_SESSION_ID,
            event=event_data,
        )
        
        # Calculate total API response time
        total_time_ms = (time.time() - start_time) * 1000
        
        if result.get("status") == "success":
            return jsonify({
                "frustration": result.get("frustration", 0.0),
                "cognitive_load": result.get("cognitive_load", 0.1),
                "inference_time_ms": result.get("inference_time_ms", 0),
                "total_time_ms": total_time_ms,
                "status": "success"
            }), 200
        elif result.get("status") == "error":
            return jsonify({
                "frustration": 0.0,
                "cognitive_load": 0.1,
                "total_time_ms": total_time_ms,
                "status": "error",
                "message": result.get("message", "Prediction failed")
            }), 500
        else:
            # Fallback for any other status
            return jsonify({
                "frustration": 0.0,
                "cognitive_load": 0.1,
                "total_time_ms": total_time_ms,
                "status": "insufficient_data"
            }), 202

    except Exception as exc:
        total_time_ms = (time.time() - start_time) * 1000
        logger.exception("Ultra-fast psychological state prediction failed")
        return jsonify({
            "error": str(exc),
            "total_time_ms": total_time_ms
        }), 500

@app.route("/psych-state-batch", methods=["POST"])
def batch_psych_state():
    """
    Batch processing endpoint for multiple events.
    Optimized for maximum L4 GPU utilization.
    """
    if not model_loaded:
        return jsonify({"error": "L4 inference engine not loaded"}), 503

    start_time = time.time()
    
    try:
        # Get batch of events
        events_batch = request.get_json()
        if not events_batch or not isinstance(events_batch, list):
            return jsonify({"error": "Expected array of events"}), 400

        user_id = request.args.get("user_id", DEFAULT_USER_ID)
        
        # Process each event in the batch
        results = []
        for event_data in events_batch:
            if "timestamp" not in event_data:
                event_data["timestamp"] = time.time()
            
            result = inference_engine.update_session(
                user_id=user_id,
                session_id=GLOBAL_SESSION_ID,
                event=event_data,
            )
            
            results.append({
                "frustration": result.get("frustration", 0.0),
                "cognitive_load": result.get("cognitive_load", 0.1),
                "status": result.get("status", "unknown")
            })
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return jsonify({
            "results": results,
            "batch_size": len(events_batch),
            "total_time_ms": total_time_ms,
            "avg_time_per_event_ms": total_time_ms / len(events_batch)
        }), 200

    except Exception as exc:
        total_time_ms = (time.time() - start_time) * 1000
        logger.exception("Batch psychological state prediction failed")
        return jsonify({
            "error": str(exc),
            "total_time_ms": total_time_ms
        }), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get detailed statistics about the L4 inference engine"""
    if not model_loaded:
        return jsonify({"error": "L4 inference engine not loaded"}), 503
    
    stats = inference_engine.get_session_stats()
    return jsonify(stats), 200

# ─────────────────────────────── Main ───────────────────────────────
if __name__ == "__main__":
    if not model_loaded:
        logger.error("Server cannot start because the L4 model failed to load.")
        exit(1)
    else:
        logger.info("🚀 Starting Ultra-Fast L4 GPU Psychological State API on 0.0.0.0:5001")
        app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
