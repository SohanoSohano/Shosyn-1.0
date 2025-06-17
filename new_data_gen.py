# new_data_gen.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

class PlatformType(Enum):
    NETFLIX = "netflix"
    PRIME_VIDEO = "prime_video"
    DISNEY_PLUS = "disney_plus"
    HOTSTAR = "hotstar"
    APPLE_TV = "apple_tv"
    HULU = "hulu"
    YOUTUBE = "youtube"

class InteractionType(Enum):
    BROWSE = "browse"
    SEARCH = "search"
    SELECT = "select"
    PLAY = "play"
    PAUSE = "pause"
    EXTERNAL_LAUNCH = "external_launch"
    RETURN_TO_APP = "return_to_app"
    PLATFORM_SWITCH = "platform_switch"

class DeviceModel(Enum):
    FIRE_TV_STICK_4K = "fire_tv_stick_4k"
    FIRE_TV_CUBE = "fire_tv_cube"
    FIRE_TV_STICK_LITE = "fire_tv_stick_lite"
    FIRE_TV_SMART_TV = "fire_tv_smart_tv"

@dataclass
class RealisticFireTVInteraction:
    """Realistic Fire TV interaction with extractable parameters only"""
    
    # Metadata
    user_id: str
    session_id: str
    interaction_timestamp: float
    interaction_type: InteractionType
    platform_target: PlatformType
    content_id: Optional[str]
    
    # TIER 1: Directly Extractable from Fire TV (18 attributes)
    # Navigation Metrics
    dpad_navigation_count: int  # D-pad button presses
    navigation_speed: float  # Actions per minute
    scroll_depth_percentage: float  # How far user scrolled
    back_button_presses: int  # KEYCODE_BACK events
    home_button_presses: int  # KEYCODE_HOME events
    
    # Timing Metrics
    time_since_last_interaction: float  # Seconds between actions
    hover_duration_ms: int  # Focus time on UI elements
    decision_latency_ms: int  # Time to make selection
    
    # Device Context
    device_model: DeviceModel
    network_latency_ms: int  # Network response time
    cpu_usage_percent: float  # System performance
    memory_usage_mb: int  # RAM consumption
    
    # Content Interaction
    content_position_in_carousel: int  # UI position (1-20)
    trailer_viewed: bool  # Whether trailer was watched
    content_details_viewed: bool  # Whether "More Info" was accessed
    external_app_launched: bool  # Deep link to OTT platform
    
    # Search Behavior
    search_query_length: int  # Characters in search
    search_results_clicked: int  # Results interacted with
    
    # TIER 2: Derived Behavioral Metrics (15 attributes)
    # Calculated from Tier 1 data
    navigation_efficiency: float  # Optimal vs actual path
    decision_confidence: float  # Derived from timing patterns
    platform_switching_rate: float  # Frequency of platform changes
    content_exploration_breadth: float  # Variety of content viewed
    session_engagement_score: float  # Overall session quality
    cognitive_load_indicator: float  # Mental effort estimation
    frustration_level: float  # Derived from back presses + timing
    attention_span_estimate: float  # Focus duration patterns
    recommendation_acceptance_rate: float  # Algorithmic suggestion uptake
    social_proof_sensitivity: float  # Response to popularity indicators
    price_sensitivity_indicator: float  # Response to subscription prompts
    multi_platform_comparison_behavior: float  # Cross-platform evaluation
    return_user_likelihood: float  # Session continuation probability
    content_abandonment_tendency: float  # Early exit patterns
    ui_adaptation_speed: float  # Learning curve for interface
    
    # TIER 3: Cross-Platform Analytics (9 attributes)
    # Advanced metrics from mediator app
    platform_loyalty_score: float  # Preference consistency
    cross_platform_session_coherence: float  # Thematic consistency
    deep_link_success_rate: float  # External app launch success
    return_to_mediator_frequency: float  # App switching patterns
    content_discovery_method_preference: float  # Browse vs search vs recommend
    peak_usage_time_alignment: float  # Personal vs global peak alignment
    weekend_vs_weekday_behavior_variance: float  # Temporal pattern consistency
    social_viewing_context_indicator: float  # Multi-user detection
    voice_command_usage_frequency: float  # Alexa integration usage

class RealisticFireTVDatasetGenerator:
    """Generate realistic Fire TV dataset with 1000 entries"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.platforms = list(PlatformType)
        self.devices = list(DeviceModel)
        self.interaction_types = list(InteractionType)
        
        # Realistic user behavior profiles
        self.user_profiles = self._create_realistic_profiles()
        
        print("🎬 Realistic Fire TV Dataset Generator Initialized")
        print(f"├── Platforms: {len(self.platforms)}")
        print(f"├── Device types: {len(self.devices)}")
        print(f"└── User profiles: {len(self.user_profiles)}")
    
    def _create_realistic_profiles(self) -> Dict:
        """Create realistic user behavior profiles based on Fire TV usage patterns"""
        return {
            'tech_savvy_explorer': {
                'platform_switching': 0.8,
                'navigation_speed': 0.9,
                'decision_confidence': 0.7,
                'content_exploration': 0.8,
                'device_preference': [DeviceModel.FIRE_TV_CUBE, DeviceModel.FIRE_TV_STICK_4K]
            },
            'casual_mainstream_viewer': {
                'platform_switching': 0.3,
                'navigation_speed': 0.4,
                'decision_confidence': 0.6,
                'content_exploration': 0.4,
                'device_preference': [DeviceModel.FIRE_TV_STICK_LITE, DeviceModel.FIRE_TV_SMART_TV]
            },
            'family_oriented_user': {
                'platform_switching': 0.5,
                'navigation_speed': 0.5,
                'decision_confidence': 0.8,
                'content_exploration': 0.6,
                'device_preference': [DeviceModel.FIRE_TV_SMART_TV, DeviceModel.FIRE_TV_STICK_4K]
            },
            'price_conscious_browser': {
                'platform_switching': 0.9,
                'navigation_speed': 0.6,
                'decision_confidence': 0.4,
                'content_exploration': 0.9,
                'device_preference': [DeviceModel.FIRE_TV_STICK_LITE, DeviceModel.FIRE_TV_STICK_4K]
            },
            'binge_watcher': {
                'platform_switching': 0.2,
                'navigation_speed': 0.7,
                'decision_confidence': 0.9,
                'content_exploration': 0.3,
                'device_preference': [DeviceModel.FIRE_TV_CUBE, DeviceModel.FIRE_TV_SMART_TV]
            }
        }
    
    def generate_realistic_dataset(self, num_entries: int = 1000) -> pd.DataFrame:
        """Generate realistic Fire TV dataset with specified entries"""
        
        print(f"\n🚀 Generating Realistic Fire TV Dataset:")
        print(f"├── Target entries: {num_entries}")
        print(f"├── Realistic parameters only")
        print(f"└── Fire TV + Mediator app data")
        
        # Calculate user distribution
        num_users = max(50, num_entries // 20)  # ~20 interactions per user
        interactions_per_user = num_entries // num_users
        
        print(f"\nDataset Structure:")
        print(f"├── Users: {num_users}")
        print(f"├── Avg interactions per user: {interactions_per_user}")
        print(f"└── Time span: 30 days")
        
        interactions = []
        base_timestamp = datetime.now().timestamp() - (30 * 24 * 3600)  # 30 days ago
        
        for user_idx in range(num_users):
            user_id = f"firetv_user_{user_idx:04d}"
            profile_name = np.random.choice(list(self.user_profiles.keys()))
            profile = self.user_profiles[profile_name]
            
            # Generate user's interactions
            user_interactions = self._generate_user_interactions(
                user_id, profile, interactions_per_user, base_timestamp
            )
            interactions.extend(user_interactions)
            
            if (user_idx + 1) % 10 == 0:
                print(f"Generated {user_idx + 1}/{num_users} users...")
        
        # Convert to DataFrame
        df = self._create_dataframe(interactions)
        
        print(f"\n✅ Dataset Generation Complete:")
        print(f"├── Total entries: {len(df)}")
        print(f"├── Unique users: {df['user_id'].nunique()}")
        print(f"├── Unique sessions: {df['session_id'].nunique()}")
        print(f"└── Columns: {len(df.columns)}")
        
        return df
    
    def _generate_user_interactions(
        self, 
        user_id: str, 
        profile: Dict, 
        num_interactions: int, 
        base_timestamp: float
    ) -> List[RealisticFireTVInteraction]:
        """Generate realistic interactions for a single user"""
        
        interactions = []
        current_timestamp = base_timestamp + np.random.uniform(0, 30 * 24 * 3600)
        
        # User session state
        session_count = 0
        current_session_id = f"{user_id}_session_{session_count:03d}"
        session_interaction_count = 0
        
        # User preferences based on profile
        preferred_device = np.random.choice(profile['device_preference'])
        primary_platforms = np.random.choice(self.platforms, size=2, replace=False)
        
        for i in range(num_interactions):
            # Start new session occasionally
            if session_interaction_count > np.random.randint(8, 25):
                session_count += 1
                current_session_id = f"{user_id}_session_{session_count:03d}"
                session_interaction_count = 0
                current_timestamp += np.random.uniform(3600, 24 * 3600)  # 1-24 hour gap
            
            # Generate realistic interaction
            interaction = self._generate_realistic_interaction(
                user_id, current_session_id, current_timestamp, 
                profile, preferred_device, primary_platforms, 
                session_interaction_count
            )
            
            interactions.append(interaction)
            
            # Update timing for next interaction
            current_timestamp += self._calculate_realistic_time_gap(profile, session_interaction_count)
            session_interaction_count += 1
        
        return interactions
    
    def _generate_realistic_interaction(
        self,
        user_id: str,
        session_id: str,
        timestamp: float,
        profile: Dict,
        device: DeviceModel,
        primary_platforms: List[PlatformType],
        session_position: int
    ) -> RealisticFireTVInteraction:
        """Generate single realistic Fire TV interaction"""
        
        # Determine interaction type based on session position and profile
        interaction_type = self._determine_interaction_type(profile, session_position)
        
        # Select platform based on user preferences and switching behavior
        if np.random.rand() < profile['platform_switching']:
            platform = np.random.choice(self.platforms)
        else:
            platform = np.random.choice(primary_platforms)
        
        # Generate content engagement
        content_id = self._generate_content_id(interaction_type, platform)
        
        # Calculate time since last interaction
        time_since_last = 0.0 if session_position == 0 else np.random.exponential(30)
        
        # TIER 1: Directly extractable metrics
        tier1_metrics = self._generate_tier1_metrics(profile, device, interaction_type, session_position)
        
        # TIER 2: Derived behavioral metrics
        tier2_metrics = self._generate_tier2_metrics(profile, tier1_metrics, session_position)
        
        # TIER 3: Cross-platform analytics
        tier3_metrics = self._generate_tier3_metrics(profile, platform, interaction_type)
        
        return RealisticFireTVInteraction(
            user_id=user_id,
            session_id=session_id,
            interaction_timestamp=timestamp,
            interaction_type=interaction_type,
            platform_target=platform,
            content_id=content_id,
            time_since_last_interaction=time_since_last,
            **tier1_metrics,
            **tier2_metrics,
            **tier3_metrics
        )
    
    def _generate_tier1_metrics(self, profile: Dict, device: DeviceModel, interaction_type: InteractionType, position: int) -> Dict:
        """Generate Tier 1: Directly extractable Fire TV metrics"""
        
        # Navigation patterns based on profile
        base_nav_speed = profile['navigation_speed']
        dpad_count = np.random.poisson(5 * base_nav_speed) + 1
        nav_speed = base_nav_speed * 60 + np.random.normal(0, 10)  # Actions per minute
        
        # Device performance varies by model
        device_performance = {
            DeviceModel.FIRE_TV_CUBE: {'cpu_base': 15, 'memory_base': 800, 'latency_base': 20},
            DeviceModel.FIRE_TV_STICK_4K: {'cpu_base': 25, 'memory_base': 600, 'latency_base': 30},
            DeviceModel.FIRE_TV_STICK_LITE: {'cpu_base': 35, 'memory_base': 400, 'latency_base': 45},
            DeviceModel.FIRE_TV_SMART_TV: {'cpu_base': 20, 'memory_base': 1000, 'latency_base': 25}
        }
        
        perf = device_performance[device]
        
        return {
            'dpad_navigation_count': dpad_count,
            'navigation_speed': max(0, nav_speed),
            'scroll_depth_percentage': np.random.beta(2, 3) * 100,
            'back_button_presses': np.random.poisson(1) if np.random.rand() < 0.3 else 0,
            'home_button_presses': np.random.poisson(0.5) if np.random.rand() < 0.1 else 0,
            'hover_duration_ms': int(np.random.exponential(2000) + 500),
            'decision_latency_ms': int(np.random.exponential(3000) + 1000),
            'device_model': device,
            'network_latency_ms': int(np.random.gamma(2, perf['latency_base'])),
            'cpu_usage_percent': max(5, min(95, np.random.normal(perf['cpu_base'], 10))),
            'memory_usage_mb': int(np.random.normal(perf['memory_base'], 100)),
            'content_position_in_carousel': np.random.randint(1, 21),
            'trailer_viewed': np.random.rand() < 0.2,
            'content_details_viewed': np.random.rand() < 0.15,
            'external_app_launched': interaction_type == InteractionType.EXTERNAL_LAUNCH,
            'search_query_length': np.random.poisson(8) if interaction_type == InteractionType.SEARCH else 0,
            'search_results_clicked': np.random.poisson(2) if interaction_type == InteractionType.SEARCH else 0
        }
    
    def _generate_tier2_metrics(self, profile: Dict, tier1_metrics: Dict, position: int) -> Dict:
        """Generate Tier 2: Derived behavioral metrics"""
        
        # Calculate derived metrics from Tier 1 data and profile
        nav_efficiency = profile['navigation_speed'] * (1 - tier1_metrics['back_button_presses'] * 0.2)
        decision_conf = profile['decision_confidence'] * (1 - tier1_metrics['decision_latency_ms'] / 10000)
        
        return {
            'navigation_efficiency': max(0, min(1, nav_efficiency)),
            'decision_confidence': max(0, min(1, decision_conf)),
            'platform_switching_rate': profile['platform_switching'] + np.random.normal(0, 0.1),
            'content_exploration_breadth': profile['content_exploration'] + np.random.normal(0, 0.1),
            'session_engagement_score': np.random.beta(3, 2),
            'cognitive_load_indicator': min(1, tier1_metrics['decision_latency_ms'] / 5000),
            'frustration_level': min(1, tier1_metrics['back_button_presses'] * 0.3),
            'attention_span_estimate': max(0.1, 1 - position * 0.02),
            'recommendation_acceptance_rate': np.random.beta(2, 3),
            'social_proof_sensitivity': np.random.beta(2, 2),
            'price_sensitivity_indicator': np.random.beta(3, 2),
            'multi_platform_comparison_behavior': profile['platform_switching'] * np.random.beta(2, 2),
            'return_user_likelihood': np.random.beta(4, 2),
            'content_abandonment_tendency': np.random.beta(2, 4),
            'ui_adaptation_speed': profile['navigation_speed'] * np.random.beta(3, 2)
        }
    
    def _generate_tier3_metrics(self, profile: Dict, platform: PlatformType, interaction_type: InteractionType) -> Dict:
        """Generate Tier 3: Cross-platform analytics"""
        
        return {
            'platform_loyalty_score': 1 - profile['platform_switching'] + np.random.normal(0, 0.1),
            'cross_platform_session_coherence': np.random.beta(3, 2),
            'deep_link_success_rate': 0.95 if interaction_type == InteractionType.EXTERNAL_LAUNCH else np.random.beta(8, 2),
            'return_to_mediator_frequency': np.random.beta(3, 3),
            'content_discovery_method_preference': np.random.beta(2, 2),
            'peak_usage_time_alignment': np.random.beta(2, 2),
            'weekend_vs_weekday_behavior_variance': np.random.beta(2, 3),
            'social_viewing_context_indicator': np.random.beta(1, 4),
            'voice_command_usage_frequency': np.random.beta(1, 5)
        }
    
    def _determine_interaction_type(self, profile: Dict, position: int) -> InteractionType:
        """Determine realistic interaction type based on profile and position"""
        
        if position == 0:
            return InteractionType.BROWSE
        
        # Probability distribution based on realistic Fire TV usage
        if profile['platform_switching'] > 0.6:
            return np.random.choice([
                InteractionType.BROWSE, InteractionType.PLATFORM_SWITCH, 
                InteractionType.SEARCH, InteractionType.SELECT
            ], p=[0.4, 0.3, 0.2, 0.1])
        else:
            return np.random.choice([
                InteractionType.BROWSE, InteractionType.SELECT, 
                InteractionType.SEARCH, InteractionType.PLAY
            ], p=[0.5, 0.3, 0.15, 0.05])
    
    def _generate_content_id(self, interaction_type: InteractionType, platform: PlatformType) -> Optional[str]:
        """Generate realistic content ID based on interaction and platform"""
        
        if interaction_type in [InteractionType.SELECT, InteractionType.PLAY, InteractionType.EXTERNAL_LAUNCH]:
            return f"{platform.value}_content_{np.random.randint(1, 1000):04d}"
        
        return None
    
    def _calculate_realistic_time_gap(self, profile: Dict, position: int) -> float:
        """Calculate realistic time gap between interactions"""
        
        base_gap = 30  # 30 seconds base
        speed_factor = 1 / profile['navigation_speed']
        position_factor = 1 + (position * 0.1)  # Slower as session progresses
        
        return np.random.exponential(base_gap * speed_factor * position_factor)
    
    def _create_dataframe(self, interactions: List[RealisticFireTVInteraction]) -> pd.DataFrame:
        """Convert interactions to pandas DataFrame"""
        
        # Convert to dictionaries
        data = []
        for interaction in interactions:
            row = asdict(interaction)
            
            # Convert enums to strings
            row['interaction_type'] = interaction.interaction_type.value
            row['platform_target'] = interaction.platform_target.value
            row['device_model'] = interaction.device_model.value
            
            # Add derived datetime
            row['interaction_datetime'] = datetime.fromtimestamp(interaction.interaction_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Add additional analytics
        df['hour_of_day'] = pd.to_datetime(df['interaction_datetime']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['interaction_datetime']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'] >= 5
        df['is_peak_hours'] = (df['hour_of_day'] >= 18) & (df['hour_of_day'] <= 22)
        
        return df

def generate_production_dataset():
    """Generate production-ready Fire TV dataset with 1000 entries"""
    
    print("🎬 Fire TV Neural CDE Dataset Generation")
    print("=" * 50)
    
    # Initialize generator
    generator = RealisticFireTVDatasetGenerator(seed=42)
    
    # Generate dataset
    df = generator.generate_realistic_dataset(num_entries=1000)
    
    # Save to CSV
    output_file = "fire_tv_neural_cde_dataset_1000.csv"
    df.to_csv(output_file, index=False)
    
    # Generate analysis report
    analysis = generate_dataset_analysis(df)
    
    print(f"\n📊 Dataset saved to: {output_file}")
    print(f"📋 Analysis saved to: fire_tv_dataset_analysis.txt")
    
    return df

def generate_dataset_analysis(df: pd.DataFrame) -> str:
    """Generate comprehensive dataset analysis"""
    
    analysis = []
    analysis.append("FIRE TV NEURAL CDE DATASET ANALYSIS")  # Removed emoji
    analysis.append("=" * 50)
    analysis.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    analysis.append(f"Total entries: {len(df):,}")
    analysis.append("")
    
    # Basic statistics
    analysis.append("BASIC STATISTICS")  # Removed emoji
    analysis.append("-" * 30)
    analysis.append(f"Unique users: {df['user_id'].nunique()}")
    analysis.append(f"Unique sessions: {df['session_id'].nunique()}")
    analysis.append(f"Platform distribution:")
    for platform, count in df['platform_target'].value_counts().items():
        analysis.append(f"  {platform}: {count} ({count/len(df)*100:.1f}%)")
    analysis.append("")
    
    # Device distribution
    analysis.append("DEVICE DISTRIBUTION")  # Removed emoji
    analysis.append("-" * 30)
    for device, count in df['device_model'].value_counts().items():
        analysis.append(f"{device}: {count} ({count/len(df)*100:.1f}%)")
    analysis.append("")
    
    # Interaction patterns
    analysis.append("INTERACTION PATTERNS")  # Removed emoji
    analysis.append("-" * 30)
    analysis.append(f"Average navigation speed: {df['navigation_speed'].mean():.1f} actions/min")
    analysis.append(f"Average decision latency: {df['decision_latency_ms'].mean():.0f}ms")
    analysis.append(f"External app launch rate: {(df['external_app_launched'].sum()/len(df)*100):.1f}%")
    analysis.append(f"Average session engagement: {df['session_engagement_score'].mean():.3f}")
    analysis.append("")
    
    # Behavioral insights
    analysis.append("BEHAVIORAL INSIGHTS")  # Removed emoji
    analysis.append("-" * 30)
    analysis.append(f"Average decision confidence: {df['decision_confidence'].mean():.3f}")
    analysis.append(f"Average platform switching rate: {df['platform_switching_rate'].mean():.3f}")
    analysis.append(f"Average frustration level: {df['frustration_level'].mean():.3f}")
    analysis.append(f"Content exploration breadth: {df['content_exploration_breadth'].mean():.3f}")
    analysis.append("")
    
    # Technical metrics
    analysis.append("TECHNICAL METRICS")  # Removed emoji
    analysis.append("-" * 30)
    analysis.append(f"Average CPU usage: {df['cpu_usage_percent'].mean():.1f}%")
    analysis.append(f"Average network latency: {df['network_latency_ms'].mean():.0f}ms")
    analysis.append(f"Average memory usage: {df['memory_usage_mb'].mean():.0f}MB")
    analysis.append("")
    
    # Save analysis with UTF-8 encoding
    analysis_text = "\n".join(analysis)
    with open("fire_tv_dataset_analysis.txt", "w", encoding='utf-8') as f:  # Added UTF-8 encoding
        f.write(analysis_text)
    
    return analysis_text


# Execute dataset generation
if __name__ == "__main__":
    dataset = generate_production_dataset()
    
    # Display sample
    print("\n📋 SAMPLE DATA (First 5 rows):")
    sample_cols = ['user_id', 'interaction_type', 'platform_target', 'device_model', 
                   'decision_confidence', 'navigation_efficiency', 'session_engagement_score']
    print(dataset[sample_cols].head().to_string(index=False))
