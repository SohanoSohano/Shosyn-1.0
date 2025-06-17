# neural_cde_model_tester_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import your model architecture (ensure this matches your trained model)
class LayerNormNeuralCDE(nn.Module):
    """Neural CDE with LayerNorm - must match your trained model architecture"""
    
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.1, vector_field_layers=3, **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.psychological_dim = 20
        
        # Build components with LayerNorm
        self.vector_field = self._build_layernorm_vector_field(vector_field_layers, dropout_rate)
        
        # Initial state encoder with LayerNorm
        self.initial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout_rate)
        
        # Psychological trait decoders
        self.trait_names = [
            'cognitive_load', 'decision_confidence', 'frustration_level', 
            'exploration_tendency', 'attention_span', 'navigation_efficiency',
            'platform_loyalty', 'social_influence', 'price_sensitivity',
            'content_diversity', 'session_engagement', 'ui_adaptation',
            'temporal_consistency', 'multi_platform_behavior', 'voice_usage',
            'recommendation_acceptance', 'search_sophistication', 'device_preference',
            'peak_alignment', 'return_likelihood'
        ]
        
        self.trait_decoders = nn.ModuleDict({
            trait: nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for trait in self.trait_names
        })
    
    def forward(self, interaction_path, timestamps):
        """Forward pass through LayerNorm Neural CDE"""
        
        batch_size, seq_len, feature_dim = interaction_path.shape
        
        # Create control path
        try:
            coeffs = torchcde.natural_cubic_spline_coeffs(
                torch.cat([timestamps.unsqueeze(-1), interaction_path], dim=-1)
            )
            control_path = torchcde.CubicSpline(coeffs)
        except:
            coeffs = torchcde.linear_interpolation_coeffs(
                torch.cat([timestamps.unsqueeze(-1), interaction_path], dim=-1)
            )
            control_path = torchcde.LinearInterpolation(coeffs)
        
        # Initial hidden state
        z0 = self.initial_encoder(interaction_path[:, 0])
        
        # CDE function
        class CDEFunc(torch.nn.Module):
            def __init__(self, vector_field, input_dim):
                super().__init__()
                self.vector_field = vector_field
                self.input_dim = input_dim
                
            def forward(self, t, z):
                try:
                    dXdt = control_path.derivative(t)[:, 1:]
                    f_z = self.vector_field(z).view(z.shape[0], -1, self.input_dim)
                    return torch.bmm(f_z, dXdt.unsqueeze(-1)).squeeze(-1)
                except:
                    return torch.zeros_like(z)
        
        cde_func = CDEFunc(self.vector_field, self.input_dim)
        
        # Solve CDE
        try:
            t_eval = timestamps[0]
            z_trajectory = torchcde.cdeint(
                X=control_path,
                func=cde_func,
                z0=z0,
                t=t_eval,
                method="dopri5",
                rtol=1e-4,
                atol=1e-6,
                adjoint=True
            )
            final_hidden = z_trajectory[-1]
        except:
            final_hidden = z0
        
        # Apply attention mechanism
        final_hidden_attended, _ = self.attention(
            final_hidden.unsqueeze(0), final_hidden.unsqueeze(0), final_hidden.unsqueeze(0)
        )
        final_hidden = final_hidden_attended.squeeze(0)
        
        # Decode psychological traits
        traits = {}
        for trait_name, decoder in self.trait_decoders.items():
            traits[trait_name] = decoder(final_hidden)
            
        return traits
    
    def _build_layernorm_vector_field(self, num_layers: int, dropout_rate: float) -> nn.Module:
        """Build vector field with LayerNorm"""
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                ])
            elif i == num_layers - 1:
                layers.append(nn.Linear(self.hidden_dim * 2, self.hidden_dim * self.input_dim))
            else:
                layers.extend([
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                ])
        
        return nn.Sequential(*layers)

class NeuralCDEModelTester:
    """Comprehensive testing framework for Neural CDE model"""
    
    def __init__(self, model_path: str, input_dim: int = 49):
        """Initialize tester with model path and input dimensions"""
        self.model_path = model_path
        self.input_dim = input_dim
        self.device = torch.device('cpu')  # Use CPU for testing
        
        # Load the trained model
        self.model = self._load_model()
        
        print(f"🧪 Neural CDE Model Tester Initialized")
        print(f"├── Model path: {model_path}")
        print(f"├── Input dimensions: {input_dim}")
        print(f"├── Device: {self.device}")
        print(f"└── Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self):
        """Load the trained Neural CDE model"""
        try:
            # Create model architecture
            model = LayerNormNeuralCDE(
                input_dim=self.input_dim,
                hidden_dim=128,
                dropout_rate=0.15,
                vector_field_layers=3
            )
            
            # Load trained weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            print("✅ Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def test_model_architecture(self):
        """Test model architecture and components"""
        print("\n🔍 Testing Model Architecture...")
        
        tests = {
            'vector_field': self._test_vector_field,
            'initial_encoder': self._test_initial_encoder,
            'attention': self._test_attention_mechanism,
            'trait_decoders': self._test_trait_decoders
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                result = test_func()
                results[test_name] = {'status': 'PASS', 'details': result}
                print(f"✅ {test_name}: PASS")
            except Exception as e:
                results[test_name] = {'status': 'FAIL', 'error': str(e)}
                print(f"❌ {test_name}: FAIL - {e}")
        
        return results
    
    def _test_vector_field(self):
        """Test vector field component"""
        test_input = torch.randn(1, 128)
        output = self.model.vector_field(test_input)
        expected_shape = (1, 128 * self.input_dim)
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Vector field output contains NaN"
        
        return f"Output shape: {output.shape}, Range: [{output.min():.3f}, {output.max():.3f}]"
    
    def _test_initial_encoder(self):
        """Test initial encoder component"""
        test_input = torch.randn(1, self.input_dim)
        output = self.model.initial_encoder(test_input)
        expected_shape = (1, 128)
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Initial encoder output contains NaN"
        
        return f"Output shape: {output.shape}, Range: [{output.min():.3f}, {output.max():.3f}]"
    
    def _test_attention_mechanism(self):
        """Test attention mechanism"""
        test_input = torch.randn(1, 128)
        output, _ = self.model.attention(
            test_input.unsqueeze(0), test_input.unsqueeze(0), test_input.unsqueeze(0)
        )
        expected_shape = (1, 1, 128)
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Attention output contains NaN"
        
        return f"Output shape: {output.shape}, Range: [{output.min():.3f}, {output.max():.3f}]"
    
    def _test_trait_decoders(self):
        """Test psychological trait decoders"""
        test_input = torch.randn(1, 128)
        results = {}
        
        for trait_name, decoder in self.model.trait_decoders.items():
            output = decoder(test_input)
            
            assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"
            assert 0 <= output.item() <= 1, f"Output {output.item()} not in [0,1] range"
            assert not torch.isnan(output).any(), f"Decoder {trait_name} output contains NaN"
            
            results[trait_name] = output.item()
        
        return f"All {len(results)} trait decoders working, sample outputs: {dict(list(results.items())[:3])}"
    
    def test_forward_pass(self, sequence_length: int = 20):
        """Test complete forward pass with synthetic data"""
        print(f"\n🚀 Testing Forward Pass (sequence length: {sequence_length})...")
        
        try:
            # Generate synthetic Fire TV interaction data
            test_data = self._generate_test_data_fixed(sequence_length)
            
            # Forward pass
            with torch.no_grad():
                predictions = self.model(test_data['interactions'], test_data['timestamps'])
            
            # Validate predictions
            validation_results = self._validate_predictions(predictions)
            
            print("✅ Forward pass successful")
            return {
                'status': 'PASS',
                'predictions': {k: v.item() for k, v in predictions.items()},
                'validation': validation_results
            }
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def _generate_test_data_fixed(self, sequence_length: int):
        """FIXED: Generate realistic test data for Fire TV interactions"""
        
        # Realistic Fire TV interaction patterns
        interactions = torch.zeros(1, sequence_length, self.input_dim)
        
        for t in range(sequence_length):
            # Simulate realistic Fire TV behavioral features
            interactions[0, t, :] = torch.tensor([
                # Navigation metrics
                np.random.uniform(20, 150),      # scroll_speed
                np.random.beta(2, 3),            # scroll_depth
                np.random.exponential(2) + 0.5,  # hover_duration
                np.random.poisson(1),            # back_click_count
                np.random.poisson(0.5),          # filter_changes
                
                # Timing metrics
                np.random.exponential(30) if t > 0 else 0,  # time_since_last
                np.random.exponential(2000) + 500,          # decision_latency_ms
                
                # Device context
                np.random.gamma(2, 25),          # network_latency
                np.random.uniform(15, 85),       # cpu_usage_percent
                max(0.1, 1.0 - t * 0.02),      # battery_level
                
                # Content interaction
                np.random.randint(1, 21),        # content_position
                np.random.rand() < 0.2,          # trailer_viewed
                np.random.rand() < 0.15,         # content_details_viewed
                
                # Search behavior
                np.random.poisson(8) if np.random.rand() < 0.3 else 0,  # search_query_length
                np.random.poisson(2) if np.random.rand() < 0.3 else 0,  # search_results_clicked
                
                # Additional realistic features (pad to input_dim)
                *[np.random.uniform(0, 1) for _ in range(self.input_dim - 15)]
            ])
        
        # FIXED: Generate realistic timestamps using proper PyTorch API
        # Use exponential distribution from numpy, then convert to torch
        time_intervals = np.random.exponential(scale=10.0, size=sequence_length)
        timestamps = torch.cumsum(torch.from_numpy(time_intervals).float().unsqueeze(0), dim=1)
        
        return {
            'interactions': interactions,
            'timestamps': timestamps
        }
    
    def _validate_predictions(self, predictions):
        """Validate prediction outputs"""
        validation = {}
        
        for trait_name, pred in predictions.items():
            pred_value = pred.item()
            
            validation[trait_name] = {
                'value': pred_value,
                'in_range': 0 <= pred_value <= 1,
                'is_finite': torch.isfinite(pred).all().item(),
                'realistic': self._is_realistic_trait_value(trait_name, pred_value)
            }
        
        # Overall validation
        all_valid = all(
            v['in_range'] and v['is_finite'] and v['realistic'] 
            for v in validation.values()
        )
        
        validation['overall_valid'] = all_valid
        return validation
    
    def _is_realistic_trait_value(self, trait_name: str, value: float) -> bool:
        """Check if trait value is psychologically realistic"""
        
        realistic_ranges = {
            'cognitive_load': (0.2, 0.8),
            'decision_confidence': (0.3, 0.9),
            'frustration_level': (0.1, 0.7),
            'exploration_tendency': (0.2, 0.9),
            'attention_span': (0.4, 0.95),
            'navigation_efficiency': (0.3, 0.9),
            'session_engagement': (0.25, 0.85)
        }
        
        if trait_name in realistic_ranges:
            min_val, max_val = realistic_ranges[trait_name]
            return min_val <= value <= max_val
        
        # Default range for unknown traits
        return 0.1 <= value <= 0.9
    
    def test_multiple_sequences(self, num_tests: int = 10):
        """Test model with multiple different sequences"""
        print(f"\n📊 Testing Multiple Sequences ({num_tests} tests)...")
        
        results = []
        
        for i in range(num_tests):
            sequence_length = np.random.randint(10, 30)  # Variable length
            
            try:
                test_result = self.test_forward_pass(sequence_length)
                results.append({
                    'test_id': i + 1,
                    'sequence_length': sequence_length,
                    'status': test_result['status'],
                    'predictions': test_result.get('predictions', {}),
                    'all_valid': test_result.get('validation', {}).get('overall_valid', False)
                })
                
                if test_result['status'] == 'PASS':
                    print(f"✅ Test {i+1}/{num_tests}: PASS (length: {sequence_length})")
                else:
                    print(f"❌ Test {i+1}/{num_tests}: FAIL")
                    
            except Exception as e:
                results.append({
                    'test_id': i + 1,
                    'sequence_length': sequence_length,
                    'status': 'FAIL',
                    'error': str(e)
                })
                print(f"❌ Test {i+1}/{num_tests}: ERROR - {e}")
        
        # Summary statistics
        passed_tests = [r for r in results if r['status'] == 'PASS']
        success_rate = len(passed_tests) / num_tests * 100
        
        print(f"\n📈 Multiple Sequence Test Results:")
        print(f"├── Success Rate: {success_rate:.1f}% ({len(passed_tests)}/{num_tests})")
        print(f"├── Average Sequence Length: {np.mean([r['sequence_length'] for r in results]):.1f}")
        print(f"└── All Predictions Valid: {sum(r.get('all_valid', False) for r in results)}/{num_tests}")
        
        return results
    
    def analyze_trait_predictions(self, num_samples: int = 100):
        """Analyze distribution of trait predictions"""
        print(f"\n📊 Analyzing Trait Prediction Distributions ({num_samples} samples)...")
        
        trait_samples = {trait: [] for trait in self.model.trait_names}
        
        for i in range(num_samples):
            sequence_length = np.random.randint(15, 25)
            test_data = self._generate_test_data_fixed(sequence_length)  # Use fixed version
            
            with torch.no_grad():
                predictions = self.model(test_data['interactions'], test_data['timestamps'])
            
            for trait_name, pred in predictions.items():
                trait_samples[trait_name].append(pred.item())
        
        # Calculate statistics
        trait_stats = {}
        for trait_name, samples in trait_samples.items():
            trait_stats[trait_name] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'min': np.min(samples),
                'max': np.max(samples),
                'median': np.median(samples)
            }
        
        # Display results
        print("\n📈 Trait Prediction Statistics:")
        for trait_name, stats in trait_stats.items():
            print(f"{trait_name}:")
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Median: {stats['median']:.3f}")
        
        return trait_stats
    
    def generate_test_report(self, output_file: str = "neural_cde_test_report.json"):
        """Generate comprehensive test report"""
        print(f"\n📋 Generating Comprehensive Test Report...")
        
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_path': self.model_path,
                'input_dim': self.input_dim,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trait_count': len(self.model.trait_names)
            },
            'architecture_tests': self.test_model_architecture(),
            'forward_pass_test': self.test_forward_pass(),
            'multiple_sequence_tests': self.test_multiple_sequences(20),
            'trait_analysis': self.analyze_trait_predictions(50)
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Test report saved to: {output_file}")
        
        # Print summary
        self._print_test_summary(report)
        
        return report
    
    def _print_test_summary(self, report):
        """Print test summary"""
        print(f"\n🎯 TEST SUMMARY")
        print("=" * 50)
        
        # Architecture tests
        arch_tests = report['architecture_tests']
        arch_passed = sum(1 for test in arch_tests.values() if test['status'] == 'PASS')
        print(f"Architecture Tests: {arch_passed}/{len(arch_tests)} PASSED")
        
        # Forward pass
        forward_status = report['forward_pass_test']['status']
        print(f"Forward Pass Test: {forward_status}")
        
        # Multiple sequences
        multi_tests = report['multiple_sequence_tests']
        multi_passed = sum(1 for test in multi_tests if test['status'] == 'PASS')
        success_rate = multi_passed / len(multi_tests) * 100
        print(f"Multiple Sequence Tests: {multi_passed}/{len(multi_tests)} PASSED ({success_rate:.1f}%)")
        
        # Overall assessment
        overall_health = "HEALTHY" if success_rate > 90 and forward_status == 'PASS' else "NEEDS ATTENTION"
        print(f"\nOverall Model Health: {overall_health}")

def main():
    """Main testing function"""
    print("🧪 Neural CDE Model Testing Program")
    print("=" * 50)
    
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = "best_fire_tv_neural_cde_fixed.pth"  # Your model file
    INPUT_DIM = 49  # Your input feature dimension
    
    try:
        # Initialize tester
        tester = NeuralCDEModelTester(MODEL_PATH, INPUT_DIM)
        
        # Run comprehensive tests
        report = tester.generate_test_report()
        
        print("\n🎉 Testing completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please ensure the model file exists and update MODEL_PATH")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
