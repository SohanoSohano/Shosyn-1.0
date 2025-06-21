# inference/performance_evaluation.py (Corrected and Enhanced Version)
import requests
import time
import psutil
import numpy as np
import json
import argparse
import sys
import os
from typing import List, Dict, Any

# --- Helper Functions for Metrics (Unchanged) ---
# ... (The helper functions calculate_precision_at_k, calculate_dcg_at_k, calculate_ndcg_at_k remain the same) ...
def calculate_precision_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
    if not recommended_items or k == 0: return 0.0
    top_k_recs = recommended_items[:k]
    relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
    return relevant_in_top_k / k
def calculate_dcg_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
    dcg = 0.0
    for i, item_id in enumerate(recommended_items[:k]):
        if item_id in relevant_items:
            dcg += 1 / np.log2(i + 2)
    return dcg
def calculate_ndcg_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
    dcg = calculate_dcg_at_k(recommended_items, relevant_items, k)
    ideal_dcg = calculate_dcg_at_k(sorted(relevant_items, key=lambda x: x in recommended_items, reverse=True), relevant_items, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

# --- Main Evaluator Class ---
class InferencePerformanceEvaluator:
    # ... (__init__ and _make_api_call remain the same) ...
    def __init__(self, api_url: str, server_pid: int = None):
        self.api_url = f"{api_url}/recommendations/enhanced"
        self.server_process = psutil.Process(server_pid) if server_pid and psutil.pid_exists(server_pid) else None
        self.results = {}
        print(f"📈 Initializing evaluator for API at {self.api_url}")
        if self.server_process:
            print(f"   Monitoring server process with PID: {server_pid}")
        else:
            print("   Warning: Server PID not provided. System metrics will be limited.")

    def _generate_mock_data(self):
        """
        --- MODIFIED & ENHANCED ---
        1. CORRECTED `ground_truth_liked_items` to use 'movie_XXX' format.
        2. ENHANCED feature ranges to make user profiles more distinct.
        """
        # Power User: High values for interaction features (e.g., clicks, scrolls)
        power_user = {
            "user_id": "PowerUser_01",
            "history": [{"feature_" + str(i): np.random.uniform(0.8, 2.0) for i in range(1, 19)} for _ in range(15)],
            "ground_truth_liked_items": ["movie_101", "movie_205", "movie_303", "movie_407"], # CORRECTED FORMAT
            "ground_truth_profile": { "user_type": "Power User" }
        }

        # Casual User: Low values for interaction features
        casual_user = {
            "user_id": "CasualViewer_02",
            "history": [{"feature_" + str(i): np.random.uniform(0.0, 0.5) for i in range(1, 19)} for _ in range(5)],
            "ground_truth_liked_items": ["movie_501", "movie_502"], # CORRECTED FORMAT
            "ground_truth_profile": { "user_type": "Casual Viewer" }
        }
        
        # Adaptive User: Transitions from casual to power user behavior
        adaptive_user = {
            "user_id": "AdaptiveUser_03",
            "initial_history": [{"feature_" + str(i): np.random.uniform(0.0, 0.5) for i in range(1, 19)} for _ in range(5)], # Distinctly casual
            "changed_history": [{"feature_" + str(i): np.random.uniform(0.8, 2.0) for i in range(1, 19)} for _ in range(5)], # Distinctly power user
            "ground_truth_profile_final": { "user_type": "Power User" }
        }

        return [power_user, casual_user], adaptive_user

    def _make_api_call(self, payload: Dict) -> (Dict, float):
        start_time = time.perf_counter()
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json(), time.perf_counter() - start_time
        except requests.exceptions.RequestException as e:
            print(f"❌ API Request failed for user {payload.get('user_id')}: {e}")
            return None, -1.0
        except json.JSONDecodeError:
            print(f"❌ Failed to decode JSON from response for user {payload.get('user_id')}. Response text: {response.text}")
            return None, -1.0

    # The rest of the test methods (`test_recommendation_quality`, `test_psychological_modeling`, etc.)
    # can remain exactly as they were in the previous "fully modified" version. The changes above
    # are the only ones needed to fix the identified issues. For completeness, the full, corrected
    # class methods are included below.

    def test_recommendation_quality(self, k: int = 5):
        print("\n--- 2. Testing Recommendation Quality (Model Performance) ---")
        mock_users, _ = self._generate_mock_data()
        precisions, ndcgs = [], []
        print("   Running diagnostic checks on item matching...")
        for user_data in mock_users:
            payload = {"user_id": user_data["user_id"], "history": user_data["history"], "top_k": k}
            api_response, _ = self._make_api_call(payload)
            if api_response and "recommendations" in api_response:
                recommended_ids = [rec.get('item_id', 'MISSING_ID') for rec in api_response["recommendations"]]
                relevant_ids = user_data["ground_truth_liked_items"]
                print(f"\n--- DEBUG: Item ID Matching for User: {user_data['user_id']} ---")
                print(f"  RECOMMENDED IDs:  {recommended_ids}")
                print(f"  GROUND TRUTH IDs: {relevant_ids}")
                intersection = set(recommended_ids) & set(relevant_ids)
                print(f"  Intersection ({len(intersection)} items): {intersection if intersection else 'NONE'}")
                print("----------------------------------------------------------")
                precisions.append(calculate_precision_at_k(recommended_ids, relevant_ids, k))
                ndcgs.append(calculate_ndcg_at_k(recommended_ids, relevant_ids, k))
        self.results['recommendation_quality'] = {"avg_precision_at_k": np.mean(precisions), "avg_ndcg_at_k": np.mean(ndcgs)}

    def test_psychological_modeling(self):
        print("\n--- 3. Testing Psychological Modeling (Model Performance) ---")
        mock_users, adaptive_user = self._generate_mock_data()
        profile_accuracy_results = []
        print("   Testing trait prediction accuracy...")
        for user_data in mock_users:
            payload = {"user_id": user_data["user_id"], "history": user_data["history"], "top_k": 3}
            api_response, _ = self._make_api_call(payload)
            if api_response and "fire_tv_profile" in api_response:
                predicted_profile = api_response["fire_tv_profile"]
                expected_profile = user_data["ground_truth_profile"]
                is_correct = predicted_profile.get("user_type") == expected_profile.get("user_type")
                profile_accuracy_results.append({"user_id": user_data["user_id"], "predicted": predicted_profile.get("user_type"), "expected": expected_profile.get("user_type"), "correct": is_correct})
        print("   Testing model's adaptation...")
        payload_initial = {"user_id": adaptive_user["user_id"], "history": adaptive_user["initial_history"]}
        res_initial, _ = self._make_api_call(payload_initial)
        payload_final = {"user_id": adaptive_user["user_id"], "history": adaptive_user["changed_history"]}
        res_final, _ = self._make_api_call(payload_final)
        adaptation_details = {"passed": False}
        if res_initial and res_final and "fire_tv_profile" in res_initial and "fire_tv_profile" in res_final:
            profile_initial, profile_final = res_initial["fire_tv_profile"], res_final["fire_tv_profile"]
            adaptation_details.update({"initial_profile": profile_initial.get('user_type'), "final_profile": profile_final.get('user_type'), "expected_final": adaptive_user['ground_truth_profile_final']['user_type']})
            print("\n--- DEBUG: Adaptation Test Details ---")
            print(f"  Profile from Initial (Casual) History: User Type = {adaptation_details['initial_profile']}")
            print(f"  Profile from Changed (Power) History:  User Type = {adaptation_details['final_profile']}")
            print(f"  Expected Final Profile:                User Type = {adaptation_details['expected_final']}")
            print("--------------------------------------\n")
            if (adaptation_details["initial_profile"] != adaptation_details["final_profile"] and adaptation_details["final_profile"] == adaptation_details["expected_final"]):
                adaptation_details["passed"] = True
        self.results['psychological_modeling'] = {"profile_accuracy_results": profile_accuracy_results, "adaptation_details": adaptation_details}

    def run_full_suite(self):
        print("\n" + "#"*20 + " RUNNING FULL EVALUATION SUITE " + "#"*20)
        self.test_recommendation_quality()
        self.test_psychological_modeling()
        if self.server_process: self.test_system_performance()
        self.generate_report()

    def test_system_performance(self, num_requests: int = 10):
        print("\n--- 4. Testing System Performance (Latency & Throughput) ---")
        if not self.server_process: return
        latencies = []
        mock_users, _ = self._generate_mock_data()
        for i in range(num_requests):
            user_data = mock_users[i % len(mock_users)]
            payload = {"user_id": user_data["user_id"], "history": user_data["history"], "top_k": 5}
            _, latency = self._make_api_call(payload)
            if latency != -1.0: latencies.append(latency)
        self.results['system_performance'] = {"avg_latency_ms": np.mean(latencies) * 1000 if latencies else 0}

    def generate_report(self):
        print("\n" + "="*25 + " INFERENCE PERFORMANCE REPORT " + "="*25)
        rec_qual = self.results.get('recommendation_quality', {})
        print("\n--- 1. Recommendation Quality (CRITICAL) ---")
        print(f"  Precision@5:          {rec_qual.get('avg_precision_at_k', 0):.4f}")
        print(f"  NDCG@5:               {rec_qual.get('avg_ndcg_at_k', 0):.4f}")
        psy_model = self.results.get('psychological_modeling', {})
        accuracy_results = psy_model.get('profile_accuracy_results', [])
        adaptation_details = psy_model.get('adaptation_details', {})
        print("\n--- 2. Psychological Modeling (CRITICAL) ---")
        if accuracy_results:
            correct_count = sum(1 for r in accuracy_results if r['correct'])
            print(f"  Trait Prediction Acc: {correct_count / len(accuracy_results):.2%} ({correct_count}/{len(accuracy_results)} correct)")
        print(f"  Adaptation Test:      {'✅ PASSED' if adaptation_details.get('passed') else '❌ FAILED'}")
        if any(not r['correct'] for r in accuracy_results) or not adaptation_details.get('passed'):
            print("\n--- 2a. Deep Dive: Psychological Modeling Diagnostics ---")
            for res in accuracy_results:
                if not res['correct']: print(f"  [Profile Mismatch] User '{res['user_id']}': Predicted '{res['predicted']}', Expected '{res['expected']}'")
            if not adaptation_details.get('passed'): print(f"  [Adaptation Failure] Initial: '{adaptation_details.get('initial_profile')}', Final: '{adaptation_details.get('final_profile')}', Expected: '{adaptation_details.get('expected_final')}'")
        sys_perf = self.results.get('system_performance', {})
        print("\n--- 3. System Performance (Secondary) ---")
        print(f"  Average Latency:      {sys_perf.get('avg_latency_ms', 0):.2f} ms")
        if self.server_process: print(f"  Final Memory/CPU:     {self.server_process.memory_info().rss / (1024 * 1024):.2f} MB / {self.server_process.cpu_percent()}%")
        print("\n" + "="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run performance evaluation for the Fire TV Recommendation API.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5001", help="The base URL of the inference API.")
    parser.add_argument("--pid", type=int, help="The Process ID (PID) of the running API server.")
    args = parser.parse_args()
    if args.pid and not psutil.pid_exists(args.pid):
        print(f"Error: Process with PID {args.pid} does not exist.")
        sys.exit(1)
    evaluator = InferencePerformanceEvaluator(api_url=args.url, server_pid=args.pid)
    evaluator.run_full_suite()

