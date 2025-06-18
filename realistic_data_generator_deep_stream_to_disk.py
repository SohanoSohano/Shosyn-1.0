import pandas as pd
import numpy as np
from faker import Faker
from tqdm import tqdm
import random
import datetime
import math
import json
import os
import csv  # MODIFICATION: Import the csv module
from enum import Enum, auto
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any

# --- Production-Scale Configuration ---

@dataclass
class SimulationConfig:
    """Configuration for a large-scale, complex data simulation."""
    NUM_USERS: int = 10000
    AVG_SESSIONS_PER_USER: int = 25
    MAX_INTERACTIONS_PER_SESSION: int = 500
    NUM_CONTENT_ITEMS: int = 10000
    OUTPUT_FILE: str = "fire_tv_production_dataset.csv"
    PERSONA_OUTPUT_FILE: str = "fire_tv_production_personas.csv"

# --- Data Structures & Enums ---

class InteractionType(Enum):
    DPAD_UP = auto(); DPAD_DOWN = auto(); DPAD_LEFT = auto(); DPAD_RIGHT = auto()
    SELECT = auto(); BACK = auto(); HOME = auto(); MENU = auto(); PLAY_PAUSE = auto()
    SEARCH_QUERY = auto(); HOVER = auto(); SCROLL = auto()
    SESSION_START = auto(); SESSION_END = auto()

class UIState(Enum): HOME_SCREEN, SEARCH_RESULTS, CONTENT_DETAILS, PLAYER, SETTINGS = range(5)
class GoalType(Enum): DISCOVER_NEW, FIND_SPECIFIC, CONTINUE_WATCHING, BROWSE_CASUALLY, END_SESSION = range(5)

@dataclass
class PsychologicalProfile:
    openness: float; conscientiousness: float; extraversion: float; agreeableness: float; neuroticism: float
    patience: float; tech_savviness: float; decision_style: str
    boredom_threshold: float; frustration_tolerance: float

@dataclass
class UserPersona:
    user_id: str; psychological_profile: PsychologicalProfile; content_preferences: Dict[str, float]

@dataclass
class ContentItem:
    content_id: str; title: str; content_type: str; genres: List[str]; duration_minutes: int
    popularity_score: float; complexity_score: float; release_year: int
    binge_factor: float; novelty_score: float

# --- Simulation Core Classes ---

class ContentCatalog:
    """Manages the creation and retrieval of content items."""
    def __init__(self, num_items: int, faker_instance: Faker):
        self.fake = faker_instance
        self.genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Thriller', 'Romance', 'Documentary', 'Family', 'Animation']
        self.catalog = self._generate_catalog(num_items)
        self.catalog_df = pd.DataFrame([asdict(item) for item in self.catalog])

    def _generate_catalog(self, num_items: int) -> List[ContentItem]:
        return [
            ContentItem(
                content_id=f"cont_{i}", title=self.fake.catch_phrase(),
                content_type=random.choices(['movie', 'series', 'documentary'], weights=[0.5, 0.4, 0.1], k=1)[0],
                genres=random.sample(self.genres, k=random.randint(1, 3)),
                duration_minutes=random.randint(20, 180),
                popularity_score=np.random.beta(2, 5),
                complexity_score=np.random.beta(2, 2),
                release_year=random.randint(1990, 2025),
                binge_factor=np.random.beta(3,2) if i % 10 == 0 else np.random.beta(1,5),
                novelty_score=np.random.beta(1,4)
            ) for i in range(num_items)
        ]

    def get_recommendations(self, persona: UserPersona, n: int = 20) -> List[ContentItem]:
        scores = np.zeros(len(self.catalog_df))
        scores += self.catalog_df['novelty_score'] * persona.psychological_profile.openness
        scores += self.catalog_df['popularity_score'] * (1 - persona.psychological_profile.openness)
        for genre, preference in persona.content_preferences.items():
            genre_mask = self.catalog_df['genres'].apply(lambda genres: genre in genres)
            scores[genre_mask] += preference
        probs = np.exp(scores - np.max(scores)); probs /= np.sum(probs)
        reco_indices = np.random.choice(len(self.catalog_df), size=n, p=probs, replace=False)
        return [self.catalog[i] for i in reco_indices]

    def get_random_item(self) -> ContentItem:
        """Returns a single random item from the catalog."""
        return random.choice(self.catalog)

class FireTVEnvironment:
    """Manages the state of the virtual UI."""
    def __init__(self, catalog: ContentCatalog):
        self.catalog = catalog; self.state: UIState = UIState.HOME_SCREEN
        self.focused_item: Optional[ContentItem] = None
        self.recommendation_carousels: Dict[str, List[ContentItem]] = {}

    def update_recommendations(self, persona: UserPersona):
        self.recommendation_carousels['For You'] = self.catalog.get_recommendations(persona, 20)

    def perform_action(self, action: InteractionType):
        if action == InteractionType.HOME:
            self.state = UIState.HOME_SCREEN
        elif action == InteractionType.BACK:
            if self.state in [UIState.CONTENT_DETAILS, UIState.SEARCH_RESULTS]:
                self.state = UIState.HOME_SCREEN
            elif self.state == UIState.PLAYER:
                self.state = UIState.CONTENT_DETAILS
        elif action == InteractionType.SELECT and self.focused_item:
            self.state = UIState.CONTENT_DETAILS if self.state != UIState.PLAYER else UIState.PLAYER

class PersonaAgent:
    """The AI agent with deep psychological and goal-oriented simulation."""
    @dataclass
    class AgentState:
        frustration: float = 0.0; cognitive_load: float = 0.0
        engagement: float = 0.5; boredom: float = 0.0
        current_goal: GoalType = GoalType.BROWSE_CASUALLY
        plan: List[InteractionType] = field(default_factory=list)

    def __init__(self, user_id: int, faker_instance: Faker):
        self.persona = self._create_persona(user_id)
        self.state = self.AgentState()

    def _create_persona(self, user_id: int) -> UserPersona:
        profile = PsychologicalProfile(
            openness=np.random.beta(2, 3), conscientiousness=np.random.beta(3, 2),
            extraversion=np.random.beta(2.5, 2.5), agreeableness=np.random.beta(4, 2),
            neuroticism=np.random.beta(2, 4), patience=np.random.beta(3, 3),
            tech_savviness=np.random.beta(4, 2),
            decision_style=random.choice(['rational', 'intuitive', 'spontaneous', 'avoidant']),
            boredom_threshold=np.random.uniform(0.6, 0.9),
            frustration_tolerance=np.random.uniform(0.5, 0.8)
        )
        preferences = {genre: max(0, np.random.normal(0.5 + profile.openness - 0.5, 0.2)) for genre in ContentCatalog(1, Faker()).genres}
        return UserPersona(user_id=f"user_{user_id}", psychological_profile=profile, content_preferences=preferences)

    def think(self, env: FireTVEnvironment):
        """The agent's 'brain' to form a goal and a plan."""
        if self.state.frustration > self.persona.psychological_profile.frustration_tolerance or \
           self.state.boredom > self.persona.psychological_profile.boredom_threshold:
            self.state.current_goal = GoalType.END_SESSION
        elif not self.state.plan:
            self.state.current_goal = random.choices([GoalType.BROWSE_CASUALLY, GoalType.DISCOVER_NEW], weights=[0.6, 0.4])[0]

        if not self.state.plan:
            if self.state.current_goal == GoalType.BROWSE_CASUALLY:
                self.state.plan = [InteractionType.SCROLL] * random.randint(5, 15) + [InteractionType.HOVER]
            elif self.state.current_goal == GoalType.DISCOVER_NEW:
                env.focused_item = random.choice(env.recommendation_carousels.get('For You', [env.catalog.get_random_item()]))
                self.state.plan = [InteractionType.SELECT, InteractionType.BACK]
            elif self.state.current_goal == GoalType.END_SESSION:
                self.state.plan = [InteractionType.HOME]

    def execute_step(self, env: FireTVEnvironment) -> tuple[InteractionType, Dict]:
        """Executes the next action and updates internal state."""
        self.think(env)
        if not self.state.plan:
            self.state.plan = [InteractionType.HOME] # Ensure session termination if plan becomes empty
        action = self.state.plan.pop(0)
        time_delta = 1.0 + np.random.exponential(1.0)
        time_delta *= (1 + self.state.cognitive_load - self.persona.psychological_profile.tech_savviness)
        if action == InteractionType.SCROLL:
            self.state.boredom += 0.01
            self.state.cognitive_load += 0.005
        elif action == InteractionType.BACK:
            self.state.frustration += 0.05
        elif action == InteractionType.SELECT:
            self.state.boredom = 0.0
            self.state.engagement = min(1.0, self.state.engagement + 0.1)
        self.state.frustration *= 0.99
        self.state.boredom *= 0.98
        self.state.cognitive_load *= 0.95
        return action, {'time_delta_seconds': time_delta}

class DeepRealisticDataGenerator:
    """Orchestrates the entire data generation process using a stream-to-disk approach."""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.fake = Faker()
        # MODIFICATION: self.interactions list is removed. Data is streamed.
        self.personas = []
        self.catalog = ContentCatalog(config.NUM_CONTENT_ITEMS, self.fake)
        print("🚀 Production-Scale Deep Simulation Data Generator Initialized (Stream-to-Disk).")

    def generate(self):
        """Generates data and streams it directly to a CSV file."""
        
        # MODIFICATION: Define the CSV header. This must match the keys in _log_interaction.
        header = [
            'user_id', 'session_id', 'interaction_timestamp', 'interaction_type',
            'dpad_up_count', 'dpad_down_count', 'dpad_left_count', 'dpad_right_count',
            'back_button_presses', 'menu_revisits', 'scroll_speed', 'hover_duration',
            'time_since_last_interaction', 'cpu_usage_percent', 'wifi_signal_strength',
            'network_latency_ms', 'device_temperature', 'battery_level', 'time_of_day',
            'day_of_week', 'content_id', 'content_type', 'content_genre', 'release_year',
            'search_sophistication_pattern', 'navigation_efficiency_score',
            'recommendation_engagement_pattern', 'cognitive_load_indicator',
            'decision_confidence_score', 'frustration_level', 'attention_span_indicator',
            'exploration_tendency_score', 'platform_loyalty_score', 'social_influence_factor',
            'price_sensitivity_score', 'content_diversity_preference', 'session_engagement_level',
            'ui_adaptation_speed', 'temporal_consistency_pattern', 'multi_platform_behavior_indicator',
            'voice_command_usage_frequency', 'return_likelihood_score'
        ]

        # MODIFICATION: Open the output file once and create a DictWriter.
        with open(self.config.OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_output:
            writer = csv.DictWriter(f_output, fieldnames=header)
            writer.writeheader()

            for user_idx in tqdm(range(self.config.NUM_USERS), desc="Simulating Users"):
                agent = PersonaAgent(user_idx, self.fake)
                self.personas.append(asdict(agent.persona))
                num_sessions = int(np.random.normal(self.config.AVG_SESSIONS_PER_USER, 8))
                for session_idx in range(max(1, num_sessions)):
                    # MODIFICATION: Pass the writer object to the session simulator.
                    self._simulate_session(agent, session_idx, writer)
        
        # MODIFICATION: The primary data saving is now complete. Save personas separately.
        self._save_personas()

    def _simulate_session(self, agent: PersonaAgent, session_idx: int, writer: csv.DictWriter): # MODIFICATION: Accept writer
        session_id = f"sess_{agent.persona.user_id}_{session_idx}"
        env = FireTVEnvironment(self.catalog)
        env.update_recommendations(agent.persona)
        session_state = {
            'interaction_count': 0,
            'current_time': datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 730)),
            'last_action_time': 0, 'dpad_up_count': 0, 'dpad_down_count': 0,
            'dpad_left_count': 0, 'dpad_right_count': 0, 'back_button_presses': 0,
            'menu_revisits': 0, 'cpu_usage_percent': random.uniform(15, 35),
            'wifi_signal_strength': random.uniform(-70, -40), 'battery_level': 1.0,
            'current_content_id': None
        }
        self._log_interaction(agent, session_id, env, InteractionType.SESSION_START, session_state, {}, writer) # MODIFICATION
        for _ in range(self.config.MAX_INTERACTIONS_PER_SESSION):
            action, details = agent.execute_step(env)
            env.perform_action(action)
            session_state['current_time'] += datetime.timedelta(seconds=details.get('time_delta_seconds', 1.0))
            if action.name.startswith("DPAD"):
                session_state[f"{action.name.lower()}_count"] += 1
            if action == InteractionType.BACK:
                session_state['back_button_presses'] += 1
            if action == InteractionType.MENU:
                session_state['menu_revisits'] += 1
            session_state['cpu_usage_percent'] = np.clip(session_state['cpu_usage_percent'] + np.random.normal(0, 0.5), 10, 90)
            session_state['wifi_signal_strength'] = np.clip(session_state['wifi_signal_strength'] + np.random.normal(0, 0.2), -85, -30)
            session_state['battery_level'] = np.clip(session_state['battery_level'] - 0.0001, 0, 1)
            if action == InteractionType.SELECT and env.focused_item:
                session_state['current_content_id'] = env.focused_item.content_id
            self._log_interaction(agent, session_id, env, action, session_state, details, writer) # MODIFICATION
            if action == InteractionType.HOME or agent.state.current_goal == GoalType.END_SESSION:
                break
        self._log_interaction(agent, session_id, env, InteractionType.SESSION_END, session_state, {}, writer) # MODIFICATION

    def _log_interaction(self, agent: PersonaAgent, session_id: str, env: FireTVEnvironment, action: InteractionType, session_state: Dict, details: Dict, writer: csv.DictWriter): # MODIFICATION
        session_state['interaction_count'] += 1
        profile = agent.persona.psychological_profile
        content_details = self.catalog.catalog_df[self.catalog.catalog_df['content_id'] == session_state['current_content_id']].iloc[0].to_dict() if session_state['current_content_id'] else {}
        navigation_total_dpad = sum(session_state[f"{d.name.lower()}_count"] for d in [InteractionType.DPAD_UP, InteractionType.DPAD_DOWN, InteractionType.DPAD_LEFT, InteractionType.DPAD_RIGHT])
        
        record = {
            'user_id': agent.persona.user_id, 'session_id': session_id,
            'interaction_timestamp': session_state['current_time'].isoformat(),
            'interaction_type': action.name,
            'dpad_up_count': session_state['dpad_up_count'],
            'dpad_down_count': session_state['dpad_down_count'],
            'dpad_left_count': session_state['dpad_left_count'],
            'dpad_right_count': session_state['dpad_right_count'],
            'back_button_presses': session_state['back_button_presses'],
            'menu_revisits': session_state['menu_revisits'],
            'scroll_speed': max(10, 150 - profile.patience * 100 + agent.state.frustration * 50),
            'hover_duration': details.get('time_delta_seconds', 0) if action == InteractionType.HOVER else 0,
            'time_since_last_interaction': details.get('time_delta_seconds', 0),
            'cpu_usage_percent': session_state['cpu_usage_percent'],
            'wifi_signal_strength': session_state['wifi_signal_strength'],
            'network_latency_ms': max(20, 150 + session_state['wifi_signal_strength'] * 2 + np.random.normal(0,10)),
            'device_temperature': 35 + session_state['cpu_usage_percent'] * 0.1,
            'battery_level': session_state['battery_level'],
            'time_of_day': session_state['current_time'].hour,
            'day_of_week': session_state['current_time'].weekday(),
            'content_id': session_state['current_content_id'],
            'content_type': content_details.get('content_type'),
            'content_genre': json.dumps(content_details.get('genres')),
            'release_year': content_details.get('release_year'),
            'search_sophistication_pattern': 'advanced' if profile.conscientiousness > 0.6 else 'simple',
            'navigation_efficiency_score': (session_state['interaction_count']) / (navigation_total_dpad + session_state['back_button_presses'] + 1),
            'recommendation_engagement_pattern': 'high' if profile.openness > 0.5 else 'low',
            'cognitive_load_indicator': agent.state.cognitive_load,
            'decision_confidence_score': np.clip(1 - (agent.state.cognitive_load + agent.state.frustration) / 2, 0, 1),
            'frustration_level': agent.state.frustration,
            'attention_span_indicator': np.clip(profile.conscientiousness * agent.state.engagement, 0, 1),
            'exploration_tendency_score': profile.openness,
            'platform_loyalty_score': 1 - profile.neuroticism,
            'social_influence_factor': profile.extraversion,
            'price_sensitivity_score': 1 - profile.agreeableness,
            'content_diversity_preference': profile.openness,
            'session_engagement_level': agent.state.engagement,
            'ui_adaptation_speed': profile.tech_savviness,
            'temporal_consistency_pattern': profile.conscientiousness,
            'multi_platform_behavior_indicator': 1 - profile.conscientiousness,
            'voice_command_usage_frequency': profile.tech_savviness * profile.extraversion,
            'return_likelihood_score': np.clip(profile.agreeableness * (1 - profile.neuroticism), 0, 1),
        }
        # MODIFICATION: Instead of appending to a list, write the record directly to the file.
        writer.writerow(record)

    # MODIFICATION: Renamed from _save_data to _save_personas and simplified.
    def _save_personas(self):
        """Saves the generated persona data to a CSV file."""
        print("\nSaving generated persona data...")
        # The number of personas is small, so pandas is fine here.
        pd.json_normalize([p for p in self.personas], sep='_').to_csv(self.config.PERSONA_OUTPUT_FILE, index=False)
        print(f"User personas (ground truth) saved to {self.config.PERSONA_OUTPUT_FILE}")

# MODIFICATION: Updated main execution block
if __name__ == "__main__":
    print("Starting Production-Scale Deep & Complex Fire TV Data Generation")
    config = SimulationConfig()
    generator = DeepRealisticDataGenerator(config)
    
    # The generate method now handles all file writing internally
    generator.generate()
    
    print("\nProduction-scale simulation complete!")
    print(f"Interaction dataset has been streamed to: {config.OUTPUT_FILE}")
