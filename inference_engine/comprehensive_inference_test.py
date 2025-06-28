# comprehensive_inference_test.py
import requests
import json
import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import threading
import statistics
from dataclasses import dataclass
import numpy as np


@dataclass
class TestResult:
    test_name: str
    passed: bool
    details: Dict
    execution_time: float
    error_message: str = ""

class ComprehensiveInferenceEngineTest:
    """Extensive testing suite for multi-target Neural RDE inference engine."""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.test_results: List[TestResult] = []
        self.session_counter = 0
        self.user_counter = 0
        
        # Test data patterns
        self.stress_patterns = self._create_stress_patterns()
        self.recovery_patterns = self._create_recovery_patterns()
        self.exploration_patterns = self._create_exploration_patterns()
        
        # Performance tracking
        self.response_times = []
        self.prediction_accuracies = []
        self.recommendation_qualities = []
        
    def _create_stress_patterns(self) -> List[Dict]:
        """Create realistic stress-inducing user behavior patterns."""
        return [
            {
                "name": "frustrated_browser",
                "description": "User gets increasingly frustrated with navigation",
                "events": [
                    {"action_type": "session_start", "frustration_level": 0.05, "cognitive_load": 0.1},
                    {"action_type": "dpad_right", "frustration_level": 0.1, "cognitive_load": 0.15},
                    {"action_type": "dpad_right", "frustration_level": 0.2, "cognitive_load": 0.25},
                    {"action_type": "back", "frustration_level": 0.35, "cognitive_load": 0.4},
                    {"action_type": "back", "frustration_level": 0.5, "cognitive_load": 0.55},
                    {"action_type": "back", "frustration_level": 0.65, "cognitive_load": 0.7},
                    {"action_type": "dpad_down", "frustration_level": 0.45, "cognitive_load": 0.5},
                    {"action_type": "click", "frustration_level": 0.25, "cognitive_load": 0.3}
                ]
            },
            {
                "name": "overwhelmed_user",
                "description": "User overwhelmed by complex interface",
                "events": [
                    {"action_type": "session_start", "frustration_level": 0.1, "cognitive_load": 0.2},
                    {"action_type": "dpad_right", "frustration_level": 0.15, "cognitive_load": 0.35},
                    {"action_type": "dpad_right", "frustration_level": 0.25, "cognitive_load": 0.5},
                    {"action_type": "dpad_right", "frustration_level": 0.4, "cognitive_load": 0.65},
                    {"action_type": "back", "frustration_level": 0.6, "cognitive_load": 0.8},
                    {"action_type": "back", "frustration_level": 0.7, "cognitive_load": 0.85},
                    {"action_type": "dpad_down", "frustration_level": 0.5, "cognitive_load": 0.6},
                    {"action_type": "click", "frustration_level": 0.2, "cognitive_load": 0.25}
                ]
            }
        ]
    
    def _create_recovery_patterns(self) -> List[Dict]:
        """Create recovery behavior patterns."""
        return [
            {
                "name": "gradual_recovery",
                "description": "User gradually recovers from high stress",
                "events": [
                    {"action_type": "session_start", "frustration_level": 0.8, "cognitive_load": 0.7},
                    {"action_type": "dpad_down", "frustration_level": 0.65, "cognitive_load": 0.55},
                    {"action_type": "dpad_right", "frustration_level": 0.5, "cognitive_load": 0.4},
                    {"action_type": "click", "frustration_level": 0.35, "cognitive_load": 0.25},
                    {"action_type": "dpad_right", "frustration_level": 0.2, "cognitive_load": 0.15},
                    {"action_type": "click", "frustration_level": 0.1, "cognitive_load": 0.1}
                ]
            }
        ]
    
    def _create_exploration_patterns(self) -> List[Dict]:
        """Create normal exploration patterns."""
        return [
            {
                "name": "casual_browser",
                "description": "Relaxed user browsing casually",
                "events": [
                    {"action_type": "session_start", "frustration_level": 0.02, "cognitive_load": 0.05},
                    {"action_type": "dpad_right", "frustration_level": 0.03, "cognitive_load": 0.08},
                    {"action_type": "dpad_right", "frustration_level": 0.04, "cognitive_load": 0.1},
                    {"action_type": "click", "frustration_level": 0.02, "cognitive_load": 0.06},
                    {"action_type": "dpad_right", "frustration_level": 0.03, "cognitive_load": 0.07},
                    {"action_type": "click", "frustration_level": 0.01, "cognitive_load": 0.05}
                ]
            }
        ]
    
    def run_comprehensive_test(self) -> Dict:
        """Run the complete test suite."""
        print("🚀 COMPREHENSIVE MULTI-TARGET INFERENCE ENGINE TEST")
        print("="*80)
        
        start_time = time.time()
        
        # Test 1: API Health and Basic Functionality
        self._test_api_health()
        
        # Test 2: Single User Journey Tests
        self._test_stress_patterns()
        self._test_recovery_patterns()
        self._test_exploration_patterns()
        
        # Test 3: Prediction Accuracy Tests
        self._test_prediction_accuracy()
        
        # Test 4: Trend Detection Tests
        self._test_trend_detection()
        
        # Test 5: Recovery Detection Tests
        self._test_recovery_detection()
        
        # Test 6: Recommendation Quality Tests
        self._test_recommendation_quality()
        self._test_recommendation_diversity()
        self._test_recommendation_appropriateness()
        
        # Test 7: Performance and Scalability Tests
        self._test_concurrent_sessions()
        self._test_response_times()
        self._test_memory_usage()
        
        # Test 8: Edge Cases and Error Handling
        self._test_edge_cases()
        self._test_error_handling()
        
        # Test 9: Continuous Activity Simulation
        self._test_continuous_activity()
        
        # Test 10: Long Session Tests
        self._test_long_sessions()
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        return self._generate_test_report(total_time)
    
    def _test_api_health(self):
        """Test basic API health and connectivity."""
        print("\n🏥 Testing API Health and Basic Functionality")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Health check
            response = requests.get(f"{self.api_url}/health", timeout=5)
            health_passed = response.status_code == 200
            
            if health_passed:
                health_data = response.json()
                print(f"✅ API Health: {health_data.get('status')}")
                print(f"   Model Type: {health_data.get('model_type')}")
                print(f"   Targets: {health_data.get('targets')}")
            else:
                print(f"❌ API Health Check Failed: {response.status_code}")
            
            # Session stats
            stats_response = requests.get(f"{self.api_url}/session_stats", timeout=5)
            stats_passed = stats_response.status_code == 200
            
            if stats_passed:
                stats = stats_response.json()
                print(f"✅ Session Stats: {stats}")
            
            self.test_results.append(TestResult(
                test_name="API Health",
                passed=health_passed and stats_passed,
                details={"health_status": health_passed, "stats_status": stats_passed},
                execution_time=time.time() - start_time
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="API Health",
                passed=False,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))
    
    def _test_stress_patterns(self):
        """Test stress-inducing behavior patterns."""
        print("\n😤 Testing Stress Patterns")
        print("-" * 60)
        
        for pattern in self.stress_patterns:
            start_time = time.time()
            user_id, session_id = self._get_unique_ids()
            
            print(f"\n🎬 Testing: {pattern['name']}")
            print(f"   Description: {pattern['description']}")
            
            predictions = []
            trends = []
            recovery_phases = []
            
            for i, event in enumerate(pattern['events']):
                result = self._send_event(user_id, session_id, event)
                
                if result and result.get('status') == 'success':
                    if 'predicted_frustration' in result:
                        predictions.append({
                            'event': i,
                            'predicted_frustration': result['predicted_frustration'],
                            'predicted_cognitive': result['predicted_cognitive_load'],
                            'actual_frustration': event['frustration_level'],
                            'actual_cognitive': event['cognitive_load']
                        })
                    
                    if 'psychological_trends' in result:
                        trends.append(result['psychological_trends'])
                        recovery_phases.append(result['psychological_trends'].get('recovery_phase', False))
            
            # Analyze stress pattern results
            stress_detected = any(p['predicted_frustration'] > 0.3 for p in predictions)
            trend_changes = len(set(t.get('frustration_trend') for t in trends)) > 1
            
            self.test_results.append(TestResult(
                test_name=f"Stress Pattern: {pattern['name']}",
                passed=stress_detected and trend_changes,
                details={
                    "stress_detected": stress_detected,
                    "trend_changes": trend_changes,
                    "predictions": predictions,
                    "recovery_detected": any(recovery_phases)
                },
                execution_time=time.time() - start_time
            ))
            
            print(f"   ✅ Stress Detected: {stress_detected}")
            print(f"   ✅ Trend Changes: {trend_changes}")
            print(f"   📊 Predictions: {len(predictions)}")
    
    def _test_recovery_patterns(self):
        """Test recovery behavior patterns."""
        print("\n🔄 Testing Recovery Patterns")
        print("-" * 60)
        
        for pattern in self.recovery_patterns:
            start_time = time.time()
            user_id, session_id = self._get_unique_ids()
            
            print(f"\n🎬 Testing: {pattern['name']}")
            
            recovery_detected = False
            decreasing_trends = 0
            
            for i, event in enumerate(pattern['events']):
                result = self._send_event(user_id, session_id, event)
                
                if result and 'psychological_trends' in result:
                    trends = result['psychological_trends']
                    if trends.get('recovery_phase'):
                        recovery_detected = True
                    if trends.get('frustration_trend') == 'decreasing':
                        decreasing_trends += 1
            
            self.test_results.append(TestResult(
                test_name=f"Recovery Pattern: {pattern['name']}",
                passed=recovery_detected and decreasing_trends >= 2,
                details={
                    "recovery_detected": recovery_detected,
                    "decreasing_trends": decreasing_trends
                },
                execution_time=time.time() - start_time
            ))
            
            print(f"   ✅ Recovery Detected: {recovery_detected}")
            print(f"   📉 Decreasing Trends: {decreasing_trends}")
    
    def _test_prediction_accuracy(self):
        """Test prediction accuracy across different scenarios."""
        print("\n🎯 Testing Prediction Accuracy")
        print("-" * 60)
        
        start_time = time.time()
        user_id, session_id = self._get_unique_ids()
        
        # Test with known values
        test_cases = [
            {"frustration_level": 0.1, "cognitive_load": 0.15, "expected_range": (0.05, 0.25)},
            {"frustration_level": 0.5, "cognitive_load": 0.6, "expected_range": (0.3, 0.8)},
            {"frustration_level": 0.8, "cognitive_load": 0.9, "expected_range": (0.5, 1.0)},
        ]
        
        accurate_predictions = 0
        total_predictions = 0
        
        for i, case in enumerate(test_cases):
            event = {
                "action_type": "dpad_right" if i % 2 == 0 else "back",
                "timestamp": datetime.now().isoformat(),
                "frustration_level": case["frustration_level"],
                "cognitive_load": case["cognitive_load"]
            }
            
            result = self._send_event(user_id, session_id, event)
            
            if result and 'predicted_frustration' in result:
                pred_f = result['predicted_frustration']
                pred_c = result['predicted_cognitive_load']
                
                # Check if predictions are in reasonable range
                f_in_range = case["expected_range"][0] <= pred_f <= case["expected_range"][1]
                c_in_range = case["expected_range"][0] <= pred_c <= case["expected_range"][1]
                
                if f_in_range and c_in_range:
                    accurate_predictions += 1
                
                total_predictions += 1
                
                print(f"   Test {i+1}: Actual F={case['frustration_level']:.1f}, Predicted F={pred_f:.3f} {'✅' if f_in_range else '❌'}")
                print(f"           Actual C={case['cognitive_load']:.1f}, Predicted C={pred_c:.3f} {'✅' if c_in_range else '❌'}")
        
        accuracy = accurate_predictions / total_predictions if total_predictions > 0 else 0
        
        self.test_results.append(TestResult(
            test_name="Prediction Accuracy",
            passed=accuracy >= 0.6,  # 60% accuracy threshold
            details={
                "accuracy": accuracy,
                "accurate_predictions": accurate_predictions,
                "total_predictions": total_predictions
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"   📊 Overall Accuracy: {accuracy:.1%}")
    
    def _test_recommendation_quality(self):
        """Test recommendation quality and appropriateness."""
        print("\n🎬 Testing Recommendation Quality")
        print("-" * 60)
        
        start_time = time.time()
        
        # Test recommendations for different psychological states
        test_scenarios = [
            {
                "name": "High Stress User",
                "events": [
                    {"action_type": "session_start", "frustration_level": 0.1, "cognitive_load": 0.1},
                    {"action_type": "back", "frustration_level": 0.6, "cognitive_load": 0.7},
                    {"action_type": "back", "frustration_level": 0.8, "cognitive_load": 0.8}
                ],
                "expected_genres": ["Comedy", "Animation", "Family"],
                "avoid_genres": ["Horror", "Thriller"]
            },
            {
                "name": "Relaxed User",
                "events": [
                    {"action_type": "session_start", "frustration_level": 0.02, "cognitive_load": 0.05},
                    {"action_type": "dpad_right", "frustration_level": 0.03, "cognitive_load": 0.06},
                    {"action_type": "click", "frustration_level": 0.02, "cognitive_load": 0.05}
                ],
                "expected_genres": ["Action", "Adventure", "Drama"],
                "avoid_genres": []
            }
        ]
        
        recommendation_quality_scores = []
        
        for scenario in test_scenarios:
            user_id, session_id = self._get_unique_ids()
            
            print(f"\n🎭 Testing: {scenario['name']}")
            
            # Send events to establish psychological state
            for event in scenario['events']:
                event["timestamp"] = datetime.now().isoformat()
                self._send_event(user_id, session_id, event)
                time.sleep(0.1)
            
            # Get recommendations
            rec_response = requests.post(f"{self.api_url}/get_recommendations", json={
                "user_id": user_id,
                "session_id": session_id,
                "user_preferences": {"preferred_genres": ["Comedy", "Action"]}
            })
            
            if rec_response.status_code == 200:
                recs = rec_response.json()
                recommendations = recs.get('recommendations', [])
                
                # Analyze recommendation quality
                quality_score = self._analyze_recommendation_quality(
                    recommendations, scenario['expected_genres'], scenario['avoid_genres']
                )
                
                recommendation_quality_scores.append(quality_score)
                
                print(f"   📊 Quality Score: {quality_score:.2f}")
                print(f"   🎬 Recommendations: {len(recommendations)}")
                
                # Show top 3 recommendations
                for i, rec in enumerate(recommendations[:3]):
                    print(f"      {i+1}. {rec.get('title', 'Unknown')} (Score: {rec.get('overall_score', 0):.3f})")
                    print(f"         Genres: {rec.get('genres', [])}")
                    print(f"         Reasoning: {rec.get('reasoning', 'No reasoning')[:100]}...")
        
        avg_quality = statistics.mean(recommendation_quality_scores) if recommendation_quality_scores else 0
        
        self.test_results.append(TestResult(
            test_name="Recommendation Quality",
            passed=avg_quality >= 0.7,
            details={
                "average_quality": avg_quality,
                "quality_scores": recommendation_quality_scores
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"\n   📊 Average Quality Score: {avg_quality:.2f}")
    
    def _test_concurrent_sessions(self):
        """Test handling of multiple concurrent sessions."""
        print("\n🔄 Testing Concurrent Sessions")
        print("-" * 60)
        
        start_time = time.time()
        
        def create_concurrent_session(session_num):
            user_id = f"concurrent_user_{session_num}"
            session_id = f"concurrent_session_{session_num}"
            
            events = [
                {"action_type": "session_start", "frustration_level": 0.1, "cognitive_load": 0.1},
                {"action_type": "dpad_right", "frustration_level": 0.2, "cognitive_load": 0.2},
                {"action_type": "click", "frustration_level": 0.15, "cognitive_load": 0.15}
            ]
            
            success_count = 0
            for event in events:
                event["timestamp"] = datetime.now().isoformat()
                result = self._send_event(user_id, session_id, event)
                if result and result.get('status') == 'success':
                    success_count += 1
                time.sleep(0.1)
            
            return success_count == len(events)
        
        # Create 10 concurrent sessions
        threads = []
        results = []
        
        for i in range(10):
            thread = threading.Thread(target=lambda i=i: results.append(create_concurrent_session(i)))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        success_rate = sum(results) / len(results) if results else 0
        
        self.test_results.append(TestResult(
            test_name="Concurrent Sessions",
            passed=success_rate >= 0.8,
            details={
                "success_rate": success_rate,
                "successful_sessions": sum(results),
                "total_sessions": len(results)
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"   📊 Success Rate: {success_rate:.1%}")
        print(f"   ✅ Successful Sessions: {sum(results)}/{len(results)}")
    
    def _test_continuous_activity(self):
        """Test continuous user activity simulation."""
        print("\n🔄 Testing Continuous Activity Simulation")
        print("-" * 60)
        
        start_time = time.time()
        duration = 60  # 1 minute of continuous activity
        
        user_id, session_id = self._get_unique_ids()
        
        events_sent = 0
        successful_predictions = 0
        recommendations_generated = 0
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Generate random realistic event
            action_types = ["dpad_right", "dpad_down", "click", "back"]
            action = random.choice(action_types)
            
            # Generate realistic psychological values based on action
            if action == "back":
                frustration = random.uniform(0.3, 0.8)
                cognitive = random.uniform(0.4, 0.7)
            else:
                frustration = random.uniform(0.05, 0.3)
                cognitive = random.uniform(0.1, 0.4)
            
            event = {
                "action_type": action,
                "timestamp": datetime.now().isoformat(),
                "frustration_level": frustration,
                "cognitive_load": cognitive
            }
            
            result = self._send_event(user_id, session_id, event)
            events_sent += 1
            
            if result and 'predicted_frustration' in result:
                successful_predictions += 1
            
            # Occasionally test recommendations
            if events_sent % 5 == 0:
                rec_response = requests.post(f"{self.api_url}/get_recommendations", json={
                    "user_id": user_id,
                    "session_id": session_id
                })
                if rec_response.status_code == 200:
                    recommendations_generated += 1
            
            time.sleep(0.5)  # 500ms between events
        
        prediction_rate = successful_predictions / events_sent if events_sent > 0 else 0
        
        self.test_results.append(TestResult(
            test_name="Continuous Activity",
            passed=prediction_rate >= 0.8 and recommendations_generated >= 5,
            details={
                "events_sent": events_sent,
                "successful_predictions": successful_predictions,
                "prediction_rate": prediction_rate,
                "recommendations_generated": recommendations_generated,
                "duration": duration
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"   📊 Events Sent: {events_sent}")
        print(f"   ✅ Prediction Rate: {prediction_rate:.1%}")
        print(f"   🎬 Recommendations Generated: {recommendations_generated}")
    
    def _test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        print("\n⚠️  Testing Edge Cases")
        print("-" * 60)
        
        start_time = time.time()
        edge_cases_passed = 0
        total_edge_cases = 0
        
        # Test 1: Empty session recommendations
        try:
            rec_response = requests.post(f"{self.api_url}/get_recommendations", json={
                "user_id": "nonexistent_user",
                "session_id": "nonexistent_session"
            })
            if rec_response.status_code == 200:
                edge_cases_passed += 1
                print("   ✅ Empty session recommendations handled")
            total_edge_cases += 1
        except:
            pass
        
        # Test 2: Invalid event data
        try:
            invalid_event = {
                "user_id": "edge_test_user",
                "session_id": "edge_test_session",
                "event": {
                    "action_type": "invalid_action",
                    "timestamp": "invalid_timestamp",
                    "frustration_level": "not_a_number",
                    "cognitive_load": 999
                }
            }
            response = requests.post(f"{self.api_url}/update_session", json=invalid_event)
            if response.status_code in [200, 400]:  # Either handled gracefully or proper error
                edge_cases_passed += 1
                print("   ✅ Invalid event data handled")
            total_edge_cases += 1
        except:
            pass
        
        # Test 3: Extreme psychological values
        try:
            user_id, session_id = self._get_unique_ids()
            extreme_event = {
                "action_type": "click",
                "timestamp": datetime.now().isoformat(),
                "frustration_level": 1.5,  # Above normal range
                "cognitive_load": -0.5     # Below normal range
            }
            result = self._send_event(user_id, session_id, extreme_event)
            if result:
                edge_cases_passed += 1
                print("   ✅ Extreme psychological values handled")
            total_edge_cases += 1
        except:
            pass
        
        edge_case_success_rate = edge_cases_passed / total_edge_cases if total_edge_cases > 0 else 0
        
        self.test_results.append(TestResult(
            test_name="Edge Cases",
            passed=edge_case_success_rate >= 0.7,
            details={
                "success_rate": edge_case_success_rate,
                "passed_cases": edge_cases_passed,
                "total_cases": total_edge_cases
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"   📊 Edge Case Success Rate: {edge_case_success_rate:.1%}")
    
    def _send_event(self, user_id: str, session_id: str, event: Dict) -> Dict:
        """Send an event to the API and return the response."""
        try:
            if "timestamp" not in event:
                event["timestamp"] = datetime.now().isoformat()
            
            response = requests.post(f"{self.api_url}/update_session", json={
                "user_id": user_id,
                "session_id": session_id,
                "event": event
            }, timeout=10)
            
            self.response_times.append(response.elapsed.total_seconds())
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            return None
    
    def _get_unique_ids(self) -> Tuple[str, str]:
        """Generate unique user and session IDs."""
        self.user_counter += 1
        self.session_counter += 1
        return f"test_user_{self.user_counter}", f"test_session_{self.session_counter}"
    
    def _analyze_recommendation_quality(self, recommendations: List[Dict], 
                                      expected_genres: List[str], avoid_genres: List[str]) -> float:
        """Analyze the quality of recommendations."""
        if not recommendations:
            return 0.0
        
        quality_score = 0.0
        total_weight = 0.0
        
        for rec in recommendations:
            rec_genres = rec.get('genres', [])
            
            # Check for expected genres
            expected_match = any(genre in rec_genres for genre in expected_genres)
            if expected_match:
                quality_score += 0.4
            
            # Check for avoided genres
            avoid_match = any(genre in rec_genres for genre in avoid_genres)
            if not avoid_match:
                quality_score += 0.3
            
            # Check for reasonable score
            score = rec.get('overall_score', 0)
            if 0.5 <= score <= 1.0:
                quality_score += 0.2
            
            # Check for meaningful reasoning
            reasoning = rec.get('reasoning', '')
            if len(reasoning) > 20:
                quality_score += 0.1
            
            total_weight += 1.0
        
        return quality_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_test_report(self, total_time: float) -> Dict:
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        passed_tests = sum(1 for result in self.test_results if result.passed)
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n🎯 Overall Results:")
        print(f"   ✅ Passed Tests: {passed_tests}/{total_tests}")
        print(f"   📊 Success Rate: {success_rate:.1%}")
        print(f"   ⏱️  Total Execution Time: {total_time:.2f} seconds")
        
        # Performance metrics
        if self.response_times:
            avg_response_time = statistics.mean(self.response_times)
            p95_response_time = np.percentile(self.response_times, 95)
            print(f"\n⚡ Performance Metrics:")
            print(f"   📊 Average Response Time: {avg_response_time:.3f}s")
            print(f"   📊 95th Percentile Response Time: {p95_response_time:.3f}s")
            print(f"   📊 Total API Calls: {len(self.response_times)}")
        
        # Detailed test results
        print(f"\n📋 Detailed Test Results:")
        for result in self.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"   {status} {result.test_name} ({result.execution_time:.2f}s)")
            if not result.passed and result.error_message:
                print(f"      Error: {result.error_message}")
        
        # Final assessment
        print(f"\n🏆 Final Assessment:")
        if success_rate >= 0.9:
            print("   🎉 EXCELLENT: System is production-ready with outstanding performance!")
        elif success_rate >= 0.8:
            print("   ✅ GOOD: System is ready for deployment with minor issues to address.")
        elif success_rate >= 0.7:
            print("   ⚠️  ACCEPTABLE: System needs improvements before production deployment.")
        else:
            print("   ❌ NEEDS WORK: System requires significant improvements.")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "total_time": total_time,
            "avg_response_time": statistics.mean(self.response_times) if self.response_times else 0,
            "test_results": [
                {
                    "name": result.test_name,
                    "passed": result.passed,
                    "execution_time": result.execution_time,
                    "details": result.details
                }
                for result in self.test_results
            ]
        }
    
    # Additional test methods for completeness
    def _test_trend_detection(self):
        """Test trend detection capabilities."""
        print("\n📈 Testing Trend Detection")
        print("-" * 60)
        
        start_time = time.time()
        user_id, session_id = self._get_unique_ids()
        
        # Create clear increasing trend
        increasing_events = [
            {"action_type": "session_start", "frustration_level": 0.1, "cognitive_load": 0.1},
            {"action_type": "dpad_right", "frustration_level": 0.2, "cognitive_load": 0.2},
            {"action_type": "dpad_right", "frustration_level": 0.4, "cognitive_load": 0.4},
            {"action_type": "back", "frustration_level": 0.6, "cognitive_load": 0.6}
        ]
        
        trends_detected = []
        
        for event in increasing_events:
            event["timestamp"] = datetime.now().isoformat()
            result = self._send_event(user_id, session_id, event)
            
            if result and 'psychological_trends' in result:
                trends = result['psychological_trends']
                trends_detected.append(trends.get('frustration_trend'))
        
        increasing_detected = 'increasing' in trends_detected
        
        self.test_results.append(TestResult(
            test_name="Trend Detection",
            passed=increasing_detected,
            details={
                "trends_detected": trends_detected,
                "increasing_detected": increasing_detected
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"   📈 Trends Detected: {trends_detected}")
        print(f"   ✅ Increasing Trend Detected: {increasing_detected}")
    
    def _test_recovery_detection(self):
        """Test recovery phase detection."""
        print("\n🔄 Testing Recovery Detection")
        print("-" * 60)
        
        start_time = time.time()
        user_id, session_id = self._get_unique_ids()
        
        # Create recovery pattern
        recovery_events = [
            {"action_type": "session_start", "frustration_level": 0.8, "cognitive_load": 0.7},
            {"action_type": "dpad_down", "frustration_level": 0.6, "cognitive_load": 0.5},
            {"action_type": "click", "frustration_level": 0.3, "cognitive_load": 0.2},
            {"action_type": "dpad_right", "frustration_level": 0.1, "cognitive_load": 0.1}
        ]
        
        recovery_detected = False
        
        for event in recovery_events:
            event["timestamp"] = datetime.now().isoformat()
            result = self._send_event(user_id, session_id, event)
            
            if result and 'psychological_trends' in result:
                if result['psychological_trends'].get('recovery_phase'):
                    recovery_detected = True
        
        self.test_results.append(TestResult(
            test_name="Recovery Detection",
            passed=recovery_detected,
            details={"recovery_detected": recovery_detected},
            execution_time=time.time() - start_time
        ))
        
        print(f"   🔄 Recovery Detected: {recovery_detected}")
    
    def _test_recommendation_diversity(self):
        """Test recommendation diversity."""
        print("\n🎨 Testing Recommendation Diversity")
        print("-" * 60)
        
        start_time = time.time()
        user_id, session_id = self._get_unique_ids()
        
        # Set up session
        setup_events = [
            {"action_type": "session_start", "frustration_level": 0.2, "cognitive_load": 0.3},
            {"action_type": "dpad_right", "frustration_level": 0.3, "cognitive_load": 0.4},
            {"action_type": "click", "frustration_level": 0.2, "cognitive_load": 0.3}
        ]
        
        for event in setup_events:
            event["timestamp"] = datetime.now().isoformat()
            self._send_event(user_id, session_id, event)
        
        # Get recommendations
        rec_response = requests.post(f"{self.api_url}/get_recommendations", json={
            "user_id": user_id,
            "session_id": session_id
        })
        
        diversity_score = 0.0
        
        if rec_response.status_code == 200:
            recs = rec_response.json()
            recommendations = recs.get('recommendations', [])
            
            if recommendations:
                # Check score diversity
                scores = [rec.get('overall_score', 0) for rec in recommendations]
                unique_scores = len(set(round(score, 2) for score in scores))
                score_diversity = unique_scores / len(scores) if scores else 0
                
                # Check genre diversity
                all_genres = []
                for rec in recommendations:
                    all_genres.extend(rec.get('genres', []))
                unique_genres = len(set(all_genres))
                
                # Check title diversity (should be 100%)
                titles = [rec.get('title', '') for rec in recommendations]
                unique_titles = len(set(titles))
                title_diversity = unique_titles / len(titles) if titles else 0
                
                diversity_score = (score_diversity + min(unique_genres/5, 1.0) + title_diversity) / 3
        
        self.test_results.append(TestResult(
            test_name="Recommendation Diversity",
            passed=diversity_score >= 0.7,
            details={
                "diversity_score": diversity_score,
                "recommendations_count": len(recommendations) if 'recommendations' in locals() else 0
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"   🎨 Diversity Score: {diversity_score:.2f}")
    
    def _test_recommendation_appropriateness(self):
        """Test recommendation appropriateness for different psychological states."""
        print("\n🎯 Testing Recommendation Appropriateness")
        print("-" * 60)
        
        start_time = time.time()
        appropriateness_scores = []
        
        # Test high stress scenario
        user_id, session_id = self._get_unique_ids()
        stress_events = [
            {"action_type": "session_start", "frustration_level": 0.1, "cognitive_load": 0.1},
            {"action_type": "back", "frustration_level": 0.7, "cognitive_load": 0.8},
            {"action_type": "back", "frustration_level": 0.8, "cognitive_load": 0.9}
        ]
        
        for event in stress_events:
            event["timestamp"] = datetime.now().isoformat()
            self._send_event(user_id, session_id, event)
        
        rec_response = requests.post(f"{self.api_url}/get_recommendations", json={
            "user_id": user_id,
            "session_id": session_id
        })
        
        if rec_response.status_code == 200:
            recs = rec_response.json()
            recommendations = recs.get('recommendations', [])
            
            # For high stress, expect comfort content
            comfort_genres = ["Comedy", "Animation", "Family"]
            stress_genres = ["Horror", "Thriller"]
            
            comfort_count = 0
            stress_count = 0
            
            for rec in recommendations:
                rec_genres = rec.get('genres', [])
                if any(genre in rec_genres for genre in comfort_genres):
                    comfort_count += 1
                if any(genre in rec_genres for genre in stress_genres):
                    stress_count += 1
            
            # High appropriateness if mostly comfort content, no stress content
            appropriateness = (comfort_count / len(recommendations) - stress_count / len(recommendations)) if recommendations else 0
            appropriateness_scores.append(max(0, appropriateness))
        
        avg_appropriateness = statistics.mean(appropriateness_scores) if appropriateness_scores else 0
        
        self.test_results.append(TestResult(
            test_name="Recommendation Appropriateness",
            passed=avg_appropriateness >= 0.7,
            details={
                "appropriateness_score": avg_appropriateness,
                "comfort_content_ratio": comfort_count / len(recommendations) if 'recommendations' in locals() and recommendations else 0
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"   🎯 Appropriateness Score: {avg_appropriateness:.2f}")
    
    def _test_response_times(self):
        """Test API response times under load."""
        print("\n⚡ Testing Response Times")
        print("-" * 60)
        
        start_time = time.time()
        
        if self.response_times:
            avg_time = statistics.mean(self.response_times)
            max_time = max(self.response_times)
            p95_time = np.percentile(self.response_times, 95) if len(self.response_times) >= 20 else max_time
            
            # Response time thresholds
            fast_responses = sum(1 for t in self.response_times if t < 0.5)
            acceptable_responses = sum(1 for t in self.response_times if t < 2.0)
            
            performance_score = acceptable_responses / len(self.response_times)
            
            self.test_results.append(TestResult(
                test_name="Response Times",
                passed=avg_time < 1.0 and p95_time < 2.0,
                details={
                    "avg_response_time": avg_time,
                    "max_response_time": max_time,
                    "p95_response_time": p95_time,
                    "performance_score": performance_score
                },
                execution_time=time.time() - start_time
            ))
            
            print(f"   ⚡ Average Response Time: {avg_time:.3f}s")
            print(f"   📊 95th Percentile: {p95_time:.3f}s")
            print(f"   🚀 Fast Responses (<0.5s): {fast_responses}/{len(self.response_times)}")
    
    def _test_memory_usage(self):
        """Test memory usage and session management."""
        print("\n💾 Testing Memory Usage")
        print("-" * 60)
        
        start_time = time.time()
        
        # Get initial session stats
        try:
            initial_stats = requests.get(f"{self.api_url}/session_stats").json()
            initial_sessions = initial_stats.get('active_sessions', 0)
        except:
            initial_sessions = 0
        
        # Create many sessions
        for i in range(20):
            user_id, session_id = self._get_unique_ids()
            event = {
                "action_type": "session_start",
                "timestamp": datetime.now().isoformat(),
                "frustration_level": 0.1,
                "cognitive_load": 0.1
            }
            self._send_event(user_id, session_id, event)
        
        # Get final session stats
        try:
            final_stats = requests.get(f"{self.api_url}/session_stats").json()
            final_sessions = final_stats.get('active_sessions', 0)
            total_events = final_stats.get('total_events', 0)
        except:
            final_sessions = 0
            total_events = 0
        
        sessions_created = final_sessions - initial_sessions
        memory_efficiency = sessions_created >= 15  # At least 75% of sessions created
        
        self.test_results.append(TestResult(
            test_name="Memory Usage",
            passed=memory_efficiency,
            details={
                "initial_sessions": initial_sessions,
                "final_sessions": final_sessions,
                "sessions_created": sessions_created,
                "total_events": total_events
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"   💾 Sessions Created: {sessions_created}/20")
        print(f"   📊 Total Events: {total_events}")
    
    def _test_error_handling(self):
        """Test error handling and recovery."""
        print("\n🛡️  Testing Error Handling")
        print("-" * 60)
        
        start_time = time.time()
        error_handling_score = 0
        total_error_tests = 0
        
        # Test malformed JSON
        try:
            response = requests.post(f"{self.api_url}/update_session", 
                                   data="invalid json", 
                                   headers={'Content-Type': 'application/json'})
            if response.status_code in [400, 422]:  # Proper error response
                error_handling_score += 1
            total_error_tests += 1
        except:
            pass
        
        # Test missing required fields
        try:
            response = requests.post(f"{self.api_url}/update_session", json={
                "user_id": "test_user"
                # Missing session_id and event
            })
            if response.status_code in [400, 422]:
                error_handling_score += 1
            total_error_tests += 1
        except:
            pass
        
        # Test invalid endpoint
        try:
            response = requests.post(f"{self.api_url}/invalid_endpoint", json={})
            if response.status_code == 404:
                error_handling_score += 1
            total_error_tests += 1
        except:
            pass
        
        error_handling_rate = error_handling_score / total_error_tests if total_error_tests > 0 else 0
        
        self.test_results.append(TestResult(
            test_name="Error Handling",
            passed=error_handling_rate >= 0.7,
            details={
                "error_handling_rate": error_handling_rate,
                "tests_passed": error_handling_score,
                "total_tests": total_error_tests
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"   🛡️  Error Handling Rate: {error_handling_rate:.1%}")
    
    def _test_long_sessions(self):
        """Test handling of long user sessions."""
        print("\n⏳ Testing Long Sessions")
        print("-" * 60)
        
        start_time = time.time()
        user_id, session_id = self._get_unique_ids()
        
        # Simulate a long session with 50 events
        successful_events = 0
        trend_changes = 0
        last_trend = None
        
        for i in range(50):
            # Vary frustration and cognitive load over time
            frustration = 0.1 + 0.4 * np.sin(i / 10) + random.uniform(-0.05, 0.05)
            cognitive = 0.2 + 0.3 * np.cos(i / 8) + random.uniform(-0.05, 0.05)
            
            frustration = max(0.01, min(0.99, frustration))
            cognitive = max(0.01, min(0.99, cognitive))
            
            event = {
                "action_type": random.choice(["dpad_right", "dpad_down", "click", "back"]),
                "timestamp": datetime.now().isoformat(),
                "frustration_level": frustration,
                "cognitive_load": cognitive
            }
            
            result = self._send_event(user_id, session_id, event)
            
            if result and result.get('status') == 'success':
                successful_events += 1
                
                if 'psychological_trends' in result:
                    current_trend = result['psychological_trends'].get('frustration_trend')
                    if last_trend and current_trend != last_trend:
                        trend_changes += 1
                    last_trend = current_trend
            
            time.sleep(0.1)  # Small delay to simulate realistic timing
        
        success_rate = successful_events / 50
        trend_detection_working = trend_changes >= 3  # Should see some trend changes
        
        self.test_results.append(TestResult(
            test_name="Long Sessions",
            passed=success_rate >= 0.9 and trend_detection_working,
            details={
                "success_rate": success_rate,
                "successful_events": successful_events,
                "trend_changes": trend_changes,
                "total_events": 50
            },
            execution_time=time.time() - start_time
        ))
        
        print(f"   ⏳ Success Rate: {success_rate:.1%}")
        print(f"   📈 Trend Changes: {trend_changes}")

    def _test_exploration_patterns(self):
        """Test normal exploration behavior patterns."""
        print("\n🔍 Testing Exploration Patterns")
        print("-" * 60)
        
        for pattern in self.exploration_patterns:
            start_time = time.time()
            user_id, session_id = self._get_unique_ids()
            
            print(f"\n🎬 Testing: {pattern['name']}")
            print(f"   Description: {pattern['description']}")
            
            predictions = []
            stable_trends = 0
            low_stress_maintained = True
            
            for i, event in enumerate(pattern['events']):
                result = self._send_event(user_id, session_id, event)
                
                if result and result.get('status') == 'success':
                    if 'predicted_frustration' in result:
                        pred_frustration = result['predicted_frustration']
                        pred_cognitive = result['predicted_cognitive_load']
                        
                        predictions.append({
                            'event': i,
                            'predicted_frustration': pred_frustration,
                            'predicted_cognitive': pred_cognitive,
                            'actual_frustration': event['frustration_level'],
                            'actual_cognitive': event['cognitive_load']
                        })
                        
                        # Check if stress levels remain low (exploration behavior)
                        if pred_frustration > 0.15 or pred_cognitive > 0.25:
                            low_stress_maintained = False
                    
                    if 'psychological_trends' in result:
                        trends = result['psychological_trends']
                        if trends.get('frustration_trend') == 'stable':
                            stable_trends += 1
            
            # Analyze exploration pattern results
            exploration_detected = low_stress_maintained and stable_trends >= 3
            
            self.test_results.append(TestResult(
                test_name=f"Exploration Pattern: {pattern['name']}",
                passed=exploration_detected,
                details={
                    "low_stress_maintained": low_stress_maintained,
                    "stable_trends": stable_trends,
                    "predictions": predictions,
                    "avg_frustration": np.mean([p['predicted_frustration'] for p in predictions]) if predictions else 0
                },
                execution_time=time.time() - start_time
            ))
            
            print(f"   ✅ Low Stress Maintained: {low_stress_maintained}")
            print(f"   📊 Stable Trends: {stable_trends}")
            print(f"   📈 Predictions: {len(predictions)}")


def main():
    """Run the comprehensive test suite."""
    print("🚀 Starting Comprehensive Multi-Target Inference Engine Test")
    print("Make sure your API server is running on http://localhost:5000")
    print("Press Enter to continue...")
    input()
    
    tester = ComprehensiveInferenceEngineTest()
    report = tester.run_comprehensive_test()
    
    # Save detailed report
    with open('comprehensive_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to 'comprehensive_test_report.json'")

if __name__ == "__main__":
    main()
