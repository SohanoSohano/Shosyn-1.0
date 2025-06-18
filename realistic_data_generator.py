import pandas as pd
import numpy as np
from faker import Faker
from tqdm import tqdm
import random
import datetime
import math
import json
import os
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# --- Configuration ---

@dataclass
class SimulationConfig:
    """Configuration settings for the data simulation."""
    NUM_USERS: int = 1000  # For a full run, set to 10,000+
    AVG_SESSIONS_PER_USER: int = 15 # Average sessions per user
    MIN_INTERACTIONS_PER_SESSION: int = 20
    MAX_INTERACTIONS_PER_SESSION: int = 200
    NUM_CONTENT_ITEMS: int = 5000
    OUTPUT_FILE: str = "fire_tv_enhanced_realistic_dataset.csv"
    PERSONA_OUTPUT_FILE: str = "fire_tv_user_personas.csv"

# --- Data Structures & Enums ---

class InteractionType(Enum):
    DPAD_UP = "dpad_up"
    DPAD_DOWN = "dpad_down"
    DPAD_LEFT = "dpad_left"
    DPAD_RIGHT = "dpad_right"
    SELECT = "select"
    BACK = "back"
    HOME = "home"
    MENU = "menu"
    PLAY_PAUSE = "play_pause"
    SEARCH_QUERY = "search_query"
    HOVER = "hover"
    SCROLL = "scroll"
    SESSION_START = "session_start"
    SESSION_END = "session_end"

class ContentSource(Enum):
    PRIME_VIDEO = "prime_video"
    NETFLIX = "netflix"
    YOUTUBE = "youtube"
    LIVE_TV = "live_tv"
    BROWSE_PAGE = "browse_page"

@dataclass
class PsychologicalProfile:
    """Defines the core psychological traits of a user persona."""
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    patience: float
    tech_savviness: float
    decision_style: str

@dataclass
class UserPersona:
    """Represents a simulated AI agent."""
    user_id: str
    psychological_profile: PsychologicalProfile
    content_preferences: Dict[str, float]

# --- Simulation Core ---

class RealisticDataGenerator:
    """Orchestrates the entire data generation process."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.fake = Faker()
        self.interactions = []
        self.personas = []
        self.genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Thriller', 'Romance', 'Documentary', 'Family', 'Animation']
        self.content_catalog = self._generate_content_catalog()
        print("🚀 Enhanced Data Generator Initialized. Starting simulation...")

    def _generate_content_catalog(self) -> pd.DataFrame:
        """Generates a diverse catalog of content."""
        catalog = []
        for i in range(self.config.NUM_CONTENT_ITEMS):
            catalog.append({
                'content_id': f"cont_{i}",
                'content_genre': json.dumps(random.sample(self.genres, k=random.randint(1, 3))),
                'content_source': random.choice(list(ContentSource)).value,
                'release_year': random.randint(1980, 2025),
                'duration_minutes': random.randint(5, 180),
                'popularity_score': np.random.beta(2, 5)
            })
        return pd.DataFrame(catalog)

    def _create_persona(self, user_id: int) -> UserPersona:
        """Creates a unique user persona with psychological traits."""
        profile = PsychologicalProfile(
            openness=np.random.beta(2, 3),
            conscientiousness=np.random.beta(3, 2),
            extraversion=np.random.beta(2.5, 2.5),
            agreeableness=np.random.beta(4, 2),
            neuroticism=np.random.beta(2, 4),
            patience=np.random.beta(3, 3),
            tech_savviness=np.random.beta(4, 2),
            decision_style=random.choice(['rational', 'intuitive', 'spontaneous', 'avoidant'])
        )
        preferences = {genre: max(0, np.random.normal(0.5 + profile.openness - 0.5, 0.2)) for genre in self.genres}
        return UserPersona(user_id=f"user_{user_id}", psychological_profile=profile, content_preferences=preferences)

    def generate(self):
        """Main generation loop."""
        for user_idx in tqdm(range(self.config.NUM_USERS), desc="Simulating Users"):
            persona = self._create_persona(user_idx)
            self.personas.append(asdict(persona))
            num_sessions = int(np.random.normal(self.config.AVG_SESSIONS_PER_USER, 5))
            for session_idx in range(max(1, num_sessions)):
                self._simulate_session(persona, session_idx)
        self._save_data()

    def _simulate_session(self, persona: UserPersona, session_idx: int):
        """Simulates a single user session with all required parameters."""
        session_id = f"sess_{persona.user_id}_{session_idx}"
        
        # Session-level dynamic state
        state = {
            'timestamp': datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 365)),
            'dpad_up_count': 0, 'dpad_down_count': 0, 'dpad_left_count': 0, 'dpad_right_count': 0,
            'back_button_presses': 0, 'menu_revisits': 0, 'last_action_time': 0,
            'cpu_usage_percent': random.uniform(15, 35),
            'wifi_signal_strength': random.uniform(-80, -30),
            'battery_level': 1.0,
            'frustration': 0.0, 'cognitive_load': 0.0,
            'current_content_id': None,
            'current_content_source': ContentSource.BROWSE_PAGE.value
        }
        
        num_interactions = random.randint(self.config.MIN_INTERACTIONS_PER_SESSION, self.config.MAX_INTERACTIONS_PER_SESSION)
        self._log_interaction(persona, session_id, 0, InteractionType.SESSION_START, state)

        for i in range(1, num_interactions):
            # AI decision logic
            action_type = self._decide_next_action(persona, state)
            
            # Simulate action and update state
            state['timestamp'] += datetime.timedelta(seconds=np.random.exponential(scale=3.0) + 0.5)
            
            # Update navigation counts
            if action_type in [InteractionType.DPAD_UP, InteractionType.DPAD_DOWN, InteractionType.DPAD_LEFT, InteractionType.DPAD_RIGHT]:
                state[f"{action_type.value}_count"] += 1
            elif action_type == InteractionType.BACK:
                state['back_button_presses'] += 1
            elif action_type == InteractionType.MENU:
                state['menu_revisits'] += 1

            # Update device state
            state['cpu_usage_percent'] += np.random.normal(0, 1.5)
            state['wifi_signal_strength'] += np.random.normal(0, 0.5)
            state['battery_level'] -= 0.0001
            state['cpu_usage_percent'] = np.clip(state['cpu_usage_percent'], 10, 95)
            state['wifi_signal_strength'] = np.clip(state['wifi_signal_strength'], -90, -25)
            state['battery_level'] = np.clip(state['battery_level'], 0, 1)

            # Update psychological state
            state['frustration'] += (state['back_button_presses'] * 0.005 - persona.psychological_profile.patience * 0.01)
            state['cognitive_load'] += (sum(state[f"{d.value}_count"] for d in [InteractionType.DPAD_UP, InteractionType.DPAD_DOWN, InteractionType.DPAD_LEFT, InteractionType.DPAD_RIGHT]) * 0.001)
            state['frustration'] = np.clip(state['frustration'], 0, 1)
            state['cognitive_load'] = np.clip(state['cognitive_load'], 0, 1)

            if action_type == InteractionType.SELECT and state['current_content_id'] is None:
                 # Simulate selecting a new piece of content
                 content_row = self.content_catalog.sample(1).iloc[0]
                 state['current_content_id'] = content_row['content_id']
                 state['current_content_source'] = content_row['content_source']


            self._log_interaction(persona, session_id, i, action_type, state)
            
            if action_type == InteractionType.HOME:
                break # Session ends if user goes home

        self._log_interaction(persona, session_id, num_interactions, InteractionType.SESSION_END, state)

    def _decide_next_action(self, persona: UserPersona, state: Dict) -> InteractionType:
        """AI agent decides its next move based on its personality and current state."""
        profile = persona.psychological_profile
        
        # Weights for each possible action
        weights = {
            InteractionType.DPAD_DOWN: 10 + 20 * profile.openness,
            InteractionType.DPAD_UP: 5,
            InteractionType.DPAD_LEFT: 2,
            InteractionType.DPAD_RIGHT: 2,
            InteractionType.SELECT: 8 * (1 - state['frustration']),
            InteractionType.BACK: 2 + 15 * state['frustration'] + 5 * profile.neuroticism,
            InteractionType.HOME: 1 + 5 * state['frustration'],
            InteractionType.PLAY_PAUSE: 5 if state['current_content_id'] else 0.1,
            InteractionType.SEARCH_QUERY: 3 * profile.conscientiousness,
            InteractionType.HOVER: 7 * (1 - profile.patience),
            InteractionType.SCROLL: 15
        }
        
        actions = list(weights.keys())
        action_weights = np.array(list(weights.values()))
        action_weights /= action_weights.sum()
        
        return np.random.choice(actions, p=action_weights)

    def _log_interaction(self, persona: UserPersona, session_id: str, interaction_index: int, action: InteractionType, state: Dict):
        """Logs a single, fully-featured interaction event."""
        profile = persona.psychological_profile
        
        # Derived patterns (simplistic simulation)
        navigation_total_dpad = sum(state[f"{d.value}_count"] for d in [InteractionType.DPAD_UP, InteractionType.DPAD_DOWN, InteractionType.DPAD_LEFT, InteractionType.DPAD_RIGHT])
        navigation_efficiency_score = (interaction_index + 1) / (navigation_total_dpad + 1e-6)
        
        # Find content details if content is selected
        content_details = {}
        if state['current_content_id']:
            content_details = self.content_catalog[self.content_catalog['content_id'] == state['current_content_id']].iloc[0].to_dict()

        record = {
            # Core Identifiers
            'user_id': persona.user_id,
            'session_id': session_id,
            'interaction_timestamp': state['timestamp'].isoformat(),
            'interaction_type': action.value,
            
            # Interaction & Navigation Metrics
            'dpad_up_count': state['dpad_up_count'],
            'dpad_down_count': state['dpad_down_count'],
            'dpad_left_count': state['dpad_left_count'],
            'dpad_right_count': state['dpad_right_count'],
            'back_button_presses': state['back_button_presses'],
            'menu_revisits': state['menu_revisits'],
            'scroll_speed': max(10, 150 - profile.patience * 100 + state['frustration'] * 50),
            'hover_duration': max(0.2, 2.5 * profile.conscientiousness * (1 - state['frustration'])),
            'time_since_last_interaction': (state['timestamp'].timestamp() - state['last_action_time']) if state['last_action_time'] > 0 else 0,

            # Device & Environmental Context
            'cpu_usage_percent': state['cpu_usage_percent'],
            'wifi_signal_strength': state['wifi_signal_strength'],
            'network_latency_ms': max(20, 150 - state['wifi_signal_strength'] * 2 + np.random.normal(0,10)),
            'device_temperature': 35 + state['cpu_usage_percent'] * 0.1,
            'battery_level': state['battery_level'],
            'time_of_day': state['timestamp'].hour,
            'day_of_week': state['timestamp'].weekday(),
            
            # Content & Catalog Metadata
            'content_id': state['current_content_id'],
            'content_type': content_details.get('content_type'),
            'content_genre': content_details.get('content_genre'),
            'content_source': state['current_content_source'],
            'release_year': content_details.get('release_year'),
            
            # Derived Behavioral Patterns
            'search_sophistication_pattern': 'simple' if profile.conscientiousness < 0.5 else 'advanced',
            'navigation_efficiency_score': navigation_efficiency_score,
            'recommendation_engagement_pattern': 'low' if profile.agreeableness < 0.5 else 'high',
            
            # Ground-Truth Psychological Indicators
            'cognitive_load_indicator': state['cognitive_load'],
            'decision_confidence_score': np.clip(1 - state['frustration'] - (state['cognitive_load'] * 0.5), 0, 1),
            'frustration_level': state['frustration'],
            'attention_span_indicator': profile.conscientiousness * (1 - state['frustration']),
            'exploration_tendency_score': profile.openness,
            'platform_loyalty_score': 1 - profile.neuroticism,
            'social_influence_factor': profile.extraversion,
            'price_sensitivity_score': 1 - profile.agreeableness,
            'content_diversity_preference': profile.openness,
            'session_engagement_level': np.clip(0.5 + (profile.extraversion - state['frustration']), 0, 1),
            'ui_adaptation_speed': profile.tech_savviness,
            'temporal_consistency_pattern': profile.conscientiousness,
            'multi_platform_behavior_indicator': 1 - profile.conscientiousness,
            'voice_command_usage_frequency': profile.tech_savviness * profile.extraversion,
            'return_likelihood_score': profile.agreeableness * (1 - profile.neuroticism),
        }
        self.interactions.append(record)
        state['last_action_time'] = state['timestamp'].timestamp()

    def _save_data(self):
        """Saves the generated data to CSV files."""
        print("\n💾 Saving generated data...")
        
        # Save interactions
        interactions_df = pd.DataFrame(self.interactions)
        interactions_df.to_csv(self.config.OUTPUT_FILE, index=False)
        print(f"✅ Enhanced realistic dataset saved to {self.config.OUTPUT_FILE} ({len(interactions_df)} rows)")

        # Save personas for validation
        personas_df = pd.json_normalize([p for p in self.personas], sep='_')
        personas_df.to_csv(self.config.PERSONA_OUTPUT_FILE, index=False)
        print(f"✅ User personas (ground truth) saved to {self.config.PERSONA_OUTPUT_FILE} ({len(personas_df)} rows)")

# --- Main Execution ---

if __name__ == "__main__":
    print("🔥 Starting Enhanced Realistic Fire TV Data Generation 🔥")
    simulation_config = SimulationConfig()
    generator = RealisticDataGenerator(simulation_config)
    generator.generate()
    print("\n🎉 Simulation complete!")
