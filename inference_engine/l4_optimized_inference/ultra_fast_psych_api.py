from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from inference_engine_multitarget import MultiTargetInferenceEngine

# ───────────────────────────── Logging ──────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ───────────────────────────── Flask app ────────────────────────────
app = Flask(__name__)
CORS(app)

# ───────────────────────── Configuration ────────────────────────────
FIREBASE_DB = (
    "https://firetv-project-ba2ad-default-rtdb.asia-southeast1."
    "firebasedatabase.app/interactions.json"
)
LATEST_PATH = Path("latest_recommendation.json")

DEFAULT_USER_ID = "USER_ABC_123"        # change if you have more users
GLOBAL_SESSION_ID = "global"            # single logical session key

# Initialize inference engine
try:
    inference_engine = MultiTargetInferenceEngine(
        model_path="../ncde/best_model_multitarget_rde.pth",
        movie_catalog_path="../dataset/tmdb_5000_movies.csv"
    )
    model_loaded = True
    logger.info("🚀 Inference Engine loaded successfully")
except Exception as e:
    logger.error(f"FATAL: Could not load inference engine. Error: {e}")
    model_loaded = False

# ─────────────────── Background cleanup for old sessions ────────────
def _cleanup_loop():
    while True:
        time.sleep(300)                 # every 5 min
        inference_engine.cleanup_expired_sessions()

threading.Thread(target=_cleanup_loop, daemon=True).start()

# ───────────────────────── Helper functions ─────────────────────────
def _pull_events_from_firebase(user_id: str) -> list[dict]:
    """
    Download the full interactions blob (few MB tops) and
    keep only the rows for the requested user-id.
    This sidesteps Firebase's strict indexed-query rules.
    """
    r = requests.get(FIREBASE_DB, timeout=10)
    r.raise_for_status()

    raw: dict[str, dict] = r.json() or {}
    events = [v for v in raw.values() if v.get("user_id") == user_id]
    events.sort(key=lambda e: e.get("timestamp", ""))
    return events

def _serialize_recommendations(recs):
    return [
        {
            "item_id":                 r.item_id,
            "title":                   r.title,
            "genres":                  r.genres,
            "frustration_compatibility": r.frustration_compatibility,
            "cognitive_compatibility":   r.cognitive_compatibility,
            "persona_match":             r.persona_match,
            "overall_score":             r.overall_score,
            "reasoning":                 r.reasoning,
        }
        for r in recs
    ]

# ─────────────────────────── API endpoints ──────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "engine_targets": ["frustration", "cognitive_load"],
        "stats": inference_engine.get_session_stats() if model_loaded else {},
    })

@app.route("/recommendation", methods=["GET"])
def generate_recommendation():
    """
    1. Download all Firebase events for the (single) user.
    2. Stream them through the Neural-RDE engine.
    3. Dump Leanback-friendly JSON to disk (and return 200).
    """
    if not model_loaded:
        return jsonify({"error": "Inference engine not loaded"}), 503

    try:
        user_id = request.args.get("user_id", DEFAULT_USER_ID)

        events = _pull_events_from_firebase(user_id)
        if not events:
            return jsonify({"error": f"No events found for user={user_id}"}), 404

        # Push every event into the engine.
        for ev in events:
            inference_engine.update_session(
                user_id=user_id,
                session_id=GLOBAL_SESSION_ID,   # keep one logical session
                event=ev,
            )

        recs = inference_engine.get_recommendations(
            user_id=user_id,
            session_id=GLOBAL_SESSION_ID,
            user_preferences={},                # extend later if needed
        )

        rec_json = {"recommendation": _serialize_recommendations(recs)}
        LATEST_PATH.write_text(json.dumps(rec_json, indent=2))
        logger.info("Generated %d recommendations for %s", len(recs), user_id)
        return jsonify(rec_json), 200

    except Exception as exc:
        logger.exception("Recommendation generation failed")
        return jsonify({"error": str(exc)}), 500

@app.route("/api/recommendation/file", methods=["GET"])
def serve_latest_recommendation():
    if LATEST_PATH.exists():
        return send_file(LATEST_PATH, mimetype="application/json")
    return jsonify({"error": "No recommendation file generated yet"}), 404

# ─────────────────────────────── Main ───────────────────────────────
if __name__ == "__main__":
    logger.info("🚀 Starting Flask server on 0.0.0.0:5000 …")
    app.run(host="0.0.0.0", port=5000, debug=False)
