# inference/performance_evaluation.py
import requests
import time
import psutil
import numpy as np
import json
import argparse
import sys
import os
from typing import List, Dict, Any

# --- Helper Functions for Metrics ---

def calculate_precision_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
    """Calculates Precision@K: Proportion of relevant items in the top-K recommendations."""
    if not recommended_items or k == 0:
        return 0.0
    
    top_k_recs = recommended_items[:k]
    relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
    return relevant_in_top_k / k

def calculate_dcg_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
    """Calculates Discounted Cumulative Gain (DCG)@K."""
    dcg = 0.0
    for i, item_id in enumerate(recommended_items[:k]):
        if item_id in relevant_items:
            # Relevance is 1 if recommended item is in the ground truth, else 0
            relevance = 1
            dcg += relevance / np.log2(i + 2)  # Log base 2, index i+2 to avoid log(1)=0
    return dcg

def calculate_ndcg_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain (NDCG)@K."""
    dcg = calculate_dcg_at_k(recommended_items, relevant_items, k)
    # Ideal DCG (IDCG) is calculated by having all relevant items at the top
    ideal_dcg = calculate_dcg_at_k(sorted(relevant_items, key=lambda x: x in recommended_items, reverse=True), relevant_items, k)
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# --- Main Evaluator Class ---

class InferencePerformanceEvaluator:
    """A comprehensive suite to test the Fire TV recommendation inference pipeline."""

    def __init__(self, api_url: str, server_pid: int = None):
        self.api_url = f"{api_url}/recommendations/enhanced"
        self.server_process = psutil.Process(server_pid) if server_pid else None
        self.results = {}
        print(f"📈 Initializing evaluator for API at {self.api_url}")
        if self.server_process:
            print(f"   Monitoring server process with PID: {server_pid}")
        else:
            print("   Warning: Server PID not provided. Memory/CPU metrics will be skipped.")

    def _generate_mock_data(self):
        """Generates mock user data for different profiles to test against."""
        # A user who is highly engaged and explores diverse content
        power_user = {
            "user_id": "PowerUser_01",
            "history": [{"feature_" + str(i): np.random.uniform(0.7, 1.5) for i in range(1, 19)} for _ in range(15)],
            "ground_truth_liked_items": ["item_101", "item_205", "item_303", "item_407"],
            "ground_truth_profile": { "user_type": "Power User", "engagement_pattern": "Highly Engaged"}
        }

        # A user who is more casual and has clear preferences
        casual_user = {
            "user_id": "CasualViewer_02",
            "history": [{"feature_" + str(i): np.random.uniform(0.1, 0.6) for i in range(1, 19)} for _ in range(5)],
            "ground_truth_liked_items": ["item_501", "item_502"],
            "ground_truth_profile": { "user_type": "Casual Viewer", "engagement_pattern": "Focused" }
        }
        
        # A user whose behavior changes, to test adaptation
        adaptive_user = {
            "user_id": "AdaptiveUser_03",
            "initial_history": [{"feature_" + str(i): np.random.uniform(0.1, 0.5) for i in range(1, 19)} for _ in range(5)], # Initially casual
            "changed_history": [{"feature_" + str(i): np.random.uniform(0.8, 1.8) for i in range(1, 19)} for _ in range(5)], # Becomes highly engaged
            "ground_truth_profile_initial": { "user_type": "Casual Viewer" },
            "ground_truth_profile_final": { "user_type": "Power User" }
        }

        return [power_user, casual_user], adaptive_user

    def _make_api_call(self, payload: Dict) -> (Dict, float):
        """Makes a single API call and measures latency."""
        start_time = time.perf_counter()
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            latency = time.perf_counter() - start_time
            return response.json(), latency
        except requests.exceptions.RequestException as e:
            print(f"❌ API Request failed: {e}")
            return None, -1.0

    def test_system_performance(self, num_requests: int = 20):
        """1. Measures latency, throughput, and memory/CPU usage."""
        print("\n--- 1. Testing System Performance ---")
        latencies = []
        
        mock_users, _ = self._generate_mock_data()
        initial_mem = self.server_process.memory_info().rss / (1024 * 1024) if self.server_process else 0
        
        total_start_time = time.perf_counter()
        for i in range(num_requests):
            user_data = mock_users[i % len(mock_users)]
            payload = {"user_id": user_data["user_id"], "history": user_data["history"], "top_k": 5}
            _, latency = self._make_api_call(payload)
            if latency != -1.0:
                latencies.append(latency)
        total_duration = time.perf_counter() - total_start_time
        
        final_mem = self.server_process.memory_info().rss / (1024 * 1024) if self.server_process else 0

        self.results['system_performance'] = {
            "avg_latency_ms": np.mean(latencies) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "throughput_rps": num_requests / total_duration if total_duration > 0 else float('inf'),
            "memory_usage_mb": final_mem,
            "memory_increase_mb": final_mem - initial_mem if initial_mem > 0 else "N/A",
            "cpu_percent": self.server_process.cpu_percent() if self.server_process else "N/A"
        }

    def test_recommendation_quality(self, k: int = 5):
        """2. Evaluates Precision@K, NDCG@K, and simulates CTR and Discovery Rate."""
        print("\n--- 2. Testing Recommendation Quality ---")
        mock_users, _ = self._generate_mock_data()
        precisions, ndcgs, ctrs, conversions, discoveries = [], [], [], [], []

        for user_data in mock_users:
            payload = {"user_id": user_data["user_id"], "history": user_data["history"], "top_k": k}
            api_response, _ = self._make_api_call(payload)
            
            if api_response and "recommendations" in api_response:
                recommended_ids = [rec.get('item_id', rec.get('title')) for rec in api_response["recommendations"]]
                relevant_ids = user_data["ground_truth_liked_items"]
                
                # Precision and NDCG
                precisions.append(calculate_precision_at_k(recommended_ids, relevant_ids, k))
                ndcgs.append(calculate_ndcg_at_k(recommended_ids, relevant_ids, k))
                
                # Simulate CTR/Conversion
                clicks = len(set(recommended_ids) & set(relevant_ids))
                ctrs.append(clicks / k)
                conversions.append(clicks * 0.5 / k) # Assume 50% of clicks convert

                # Discovery Rate
                history_items = {item.get('item_id', '') for item in user_data['history']}
                new_items = len(set(recommended_ids) - history_items)
                discoveries.append(new_items / k)

        self.results['recommendation_quality'] = {
            "avg_precision_at_k": np.mean(precisions),
            "avg_ndcg_at_k": np.mean(ndcgs),
            "simulated_ctr": np.mean(ctrs),
            "simulated_conversion_rate": np.mean(conversions),
            "avg_discovery_rate": np.mean(discoveries)
        }
        
    def test_psychological_modeling(self):
        """3. Tests Trait Prediction Accuracy and Adaptation Speed."""
        print("\n--- 3. Testing Psychological Modeling ---")
        mock_users, adaptive_user = self._generate_mock_data()
        correct_profiles = 0
        
        # Test accuracy
        for user_data in mock_users:
            payload = {"user_id": user_data["user_id"], "history": user_data["history"], "top_k": 3}
            api_response, _ = self._make_api_call(payload)
            if api_response:
                predicted_profile = api_response["fire_tv_profile"]
                if predicted_profile["user_type"] == user_data["ground_truth_profile"]["user_type"]:
                    correct_profiles += 1
        
        # Test adaptation
        payload_initial = {"user_id": adaptive_user["user_id"], "history": adaptive_user["initial_history"]}
        res_initial, _ = self._make_api_call(payload_initial)
        
        payload_final = {"user_id": adaptive_user["user_id"], "history": adaptive_user["changed_history"]}
        res_final, _ = self._make_api_call(payload_final)
        
        adaptation_success = False
        if res_initial and res_final:
            type_initial = res_initial["fire_tv_profile"]["user_type"]
            type_final = res_final["fire_tv_profile"]["user_type"]
            if type_initial != type_final and type_final == adaptive_user["ground_truth_profile_final"]["user_type"]:
                adaptation_success = True

        self.results['psychological_modeling'] = {
            "trait_prediction_accuracy": correct_profiles / len(mock_users),
            "adaptation_test_passed": adaptation_success
        }
    
    def run_evaluation(self):
        """Executes all performance tests."""
        self.test_system_performance()
        self.test_recommendation_quality()
        self.test_psychological_modeling()
        self.generate_report()

    def generate_report(self):
        """Prints a final, formatted report of all test results."""
        print("\n" + "="*25 + " INFERENCE PERFORMANCE REPORT " + "="*25)
        
        # System Performance
        sys_perf = self.results.get('system_performance', {})
        print("\n--- 1. System Performance Metrics ---")
        print(f"  Average Latency:      {sys_perf.get('avg_latency_ms', 0):.2f} ms")
        print(f"  P95 Latency:          {sys_perf.get('p95_latency_ms', 0):.2f} ms")
        print(f"  Throughput:           {sys_perf.get('throughput_rps', 0):.2f} requests/sec")
        print(f"  Final Memory Usage:   {sys_perf.get('memory_usage_mb', 0):.2f} MB")
        print(f"  Memory Increase:      {sys_perf.get('memory_increase_mb', 'N/A')}" + (" MB" if isinstance(sys_perf.get('memory_increase_mb'), float) else ""))
        print(f"  CPU Usage:            {sys_perf.get('cpu_percent', 'N/A')}%")

        # Recommendation Quality
        rec_qual = self.results.get('recommendation_quality', {})
        print("\n--- 2. Recommendation Quality Metrics (Simulated) ---")
        print(f"  Precision@5:          {rec_qual.get('avg_precision_at_k', 0):.4f}")
        print(f"  NDCG@5:               {rec_qual.get('avg_ndcg_at_k', 0):.4f}")
        print(f"  Simulated CTR:        {rec_qual.get('simulated_ctr', 0):.2%}")
        print(f"  Simulated Conversion: {rec_qual.get('simulated_conversion_rate', 0):.2%}")
        print(f"  Discovery Rate:       {rec_qual.get('avg_discovery_rate', 0):.2%}")

        # Psychological Modeling
        psy_model = self.results.get('psychological_modeling', {})
        print("\n--- 3. Psychological Modeling Metrics ---")
        print(f"  Trait Prediction Acc: {psy_model.get('trait_prediction_accuracy', 0):.2%}")
        print(f"  Adaptation Test:      {'✅ PASSED' if psy_model.get('adaptation_test_passed') else '❌ FAILED'}")
        
        # User Experience (Guidance)
        print("\n--- 4. User Experience (Long-Term Measurement Guidance) ---")
        print("  - User Retention:     Track user cohorts over 30/60/90 days after feature rollout.")
        print("  - Session Duration:   Log and compare average session length via analytics.")
        print("  - User Satisfaction:  Implement in-app thumbs up/down feedback on recommendations.")
        
        print("\n" + "="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run performance evaluation for the Fire TV Recommendation API.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5001", help="The base URL of the inference API.")
    parser.add_argument("--pid", type=int, help="The Process ID (PID) of the running API server to monitor memory/CPU.")
    
    args = parser.parse_args()
    
    if args.pid is None:
        print("Warning: --pid not provided. To monitor memory and CPU, find the server's PID (e.g., using 'ps aux | grep app.py') and pass it with --pid <PID>.")

    evaluator = InferencePerformanceEvaluator(api_url=args.url, server_pid=args.pid)
    evaluator.run_evaluation()

