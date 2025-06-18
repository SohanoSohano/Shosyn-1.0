
# MODIFICATION: Added shebang for easy execution on Linux

import pandas as pd
import numpy as np
from faker import Faker
from tqdm import tqdm
import random
import datetime
import math
import json
import os
import csv
from enum import Enum, auto
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
# MODIFICATION: Import multiprocessing components
from multiprocessing import Pool, Manager, cpu_count
from functools import partial

# --- Production-Scale Configuration ---

@dataclass
class SimulationConfig:
    """Configuration for a large-scale, complex data simulation."""
    NUM_USERS: int = 10000
    AVG_SESSIONS_PER_USER: int = 25
    MAX_INTERACTIONS_PER_SESSION: int = 500
    NUM_CONTENT_ITEMS: int = 10000
    OUTPUT_FILE: str = "fire_tv_production_dataset_parallel.csv"
    PERSONA_OUTPUT_FILE: str = "fire_tv_production_personas_parallel.csv"

# --- Data Structures & Enums (Unchanged) ---
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

# --- Simulation Core Classes (Unchanged) ---
class ContentCatalog:
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

    def get_random_item(self) -> ContentItem: return random.choice(self.catalog)

class FireTVEnvironment:
    def __init__(self, catalog: ContentCatalog):
        self.catalog = catalog; self.state: UIState = UIState.HOME_SCREEN
        self.focused_item: Optional[ContentItem] = None
        self.recommendation_carousels: Dict[str, List[ContentItem]] = {}
    def update_recommendations(self, persona: UserPersona): self.recommendation_carousels['For You'] = self.catalog.get_recommendations(persona, 20)
    def perform_action(self, action: InteractionType):
        if action == InteractionType.HOME: self.state = UIState.HOME_SCREEN
        elif action == InteractionType.BACK:
            if self.state in [UIState.CONTENT_DETAILS, UIState.SEARCH_RESULTS]: self.state = UIState.HOME_SCREEN
            elif self.state == UIState.PLAYER: self.state = UIState.CONTENT_DETAILS
        elif action == InteractionType.SELECT and self.focused_item: self.state = UIState.CONTENT_DETAILS if self.state != UIState.PLAYER else UIState.PLAYER

class PersonaAgent:
    @dataclass
    class AgentState:
        frustration: float = 0.0; cognitive_load: float = 0.0
        engagement: float = 0.5; boredom: float = 0.0
        current_goal: GoalType = GoalType.BROWSE_CASUALLY
        plan: List[InteractionType] = field(default_factory=list)
    def __init__(self, user_id: int, faker_instance: Faker): self.persona = self._create_persona(user_id); self.state = self.AgentState()
    def _create_persona(self, user_id: int) -> UserPersona:
        profile = PsychologicalProfile(openness=np.random.beta(2, 3), conscientiousness=np.random.beta(3, 2), extraversion=np.random.beta(2.5, 2.5), agreeableness=np.random.beta(4, 2), neuroticism=np.random.beta(2, 4), patience=np.random.beta(3, 3), tech_savviness=np.random.beta(4, 2), decision_style=random.choice(['rational', 'intuitive', 'spontaneous', 'avoidant']), boredom_threshold=np.random.uniform(0.6, 0.9), frustration_tolerance=np.random.uniform(0.5, 0.8))
        preferences = {genre: max(0, np.random.normal(0.5 + profile.openness - 0.5, 0.2)) for genre in ContentCatalog(1, Faker()).genres}
        return UserPersona(user_id=f"user_{user_id}", psychological_profile=profile, content_preferences=preferences)
    def think(self, env: FireTVEnvironment):
        if self.state.frustration > self.persona.psychological_profile.frustration_tolerance or self.state.boredom > self.persona.psychological_profile.boredom_threshold: self.state.current_goal = GoalType.END_SESSION
        elif not self.state.plan: self.state.current_goal = random.choices([GoalType.BROWSE_CASUALLY, GoalType.DISCOVER_NEW], weights=[0.6, 0.4])[0]
        if not self.state.plan:
            if self.state.current_goal == GoalType.BROWSE_CASUALLY: self.state.plan = [InteractionType.SCROLL] * random.randint(5, 15) + [InteractionType.HOVER]
            elif self.state.current_goal == GoalType.DISCOVER_NEW: env.focused_item = random.choice(env.recommendation_carousels.get('For You', [env.catalog.get_random_item()])); self.state.plan = [InteractionType.SELECT, InteractionType.BACK]
            elif self.state.current_goal == GoalType.END_SESSION: self.state.plan = [InteractionType.HOME]
    def execute_step(self, env: FireTVEnvironment) -> tuple[InteractionType, Dict]:
        self.think(env)
        if not self.state.plan: self.state.plan = [InteractionType.HOME];
        action = self.state.plan.pop(0)
        time_delta = 1.0 + np.random.exponential(1.0); time_delta *= (1 + self.state.cognitive_load - self.persona.psychological_profile.tech_savviness)
        if action == InteractionType.SCROLL: self.state.boredom += 0.01; self.state.cognitive_load += 0.005
        elif action == InteractionType.BACK: self.state.frustration += 0.05
        elif action == InteractionType.SELECT: self.state.boredom = 0.0; self.state.engagement = min(1.0, self.state.engagement + 0.1)
        self.state.frustration *= 0.99; self.state.boredom *= 0.98; self.state.cognitive_load *= 0.95
        return action, {'time_delta_seconds': time_delta}

# --- Worker Function (Unchanged) ---
def simulate_user_worker(user_id: int, config: SimulationConfig, catalog: ContentCatalog, writer_lock, header: List[str]):
    fake = Faker()
    agent = PersonaAgent(user_id, fake)
    output_rows = []
    num_sessions = int(np.random.normal(config.AVG_SESSIONS_PER_USER, 8))
    for session_idx in range(max(1, num_sessions)):
        session_id = f"sess_{agent.persona.user_id}_{session_idx}"
        env = FireTVEnvironment(catalog)
        env.update_recommendations(agent.persona)
        session_state = {
            'interaction_count': 0, 'current_time': datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 730)),
            'last_action_time': 0, 'dpad_up_count': 0, 'dpad_down_count': 0, 'dpad_left_count': 0, 'dpad_right_count': 0,
            'back_button_presses': 0, 'menu_revisits': 0, 'cpu_usage_percent': random.uniform(15, 35),
            'wifi_signal_strength': random.uniform(-70, -40), 'battery_level': 1.0, 'current_content_id': None
        }
        output_rows.append(_create_record(agent, session_id, env, InteractionType.SESSION_START, session_state, {}))
        for _ in range(config.MAX_INTERACTIONS_PER_SESSION):
            action, details = agent.execute_step(env)
            env.perform_action(action)
            session_state['current_time'] += datetime.timedelta(seconds=details.get('time_delta_seconds', 1.0))
            if action.name.startswith("DPAD"): session_state[f"{action.name.lower()}_count"] += 1
            if action == InteractionType.BACK: session_state['back_button_presses'] += 1
            if action == InteractionType.MENU: session_state['menu_revisits'] += 1
            session_state['cpu_usage_percent'] = np.clip(session_state['cpu_usage_percent'] + np.random.normal(0, 0.5), 10, 90)
            session_state['wifi_signal_strength'] = np.clip(session_state['wifi_signal_strength'] + np.random.normal(0, 0.2), -85, -30)
            session_state['battery_level'] = np.clip(session_state['battery_level'] - 0.0001, 0, 1)
            if action == InteractionType.SELECT and env.focused_item: session_state['current_content_id'] = env.focused_item.content_id
            output_rows.append(_create_record(agent, session_id, env, action, session_state, details))
            if action == InteractionType.HOME or agent.state.current_goal == GoalType.END_SESSION: break
        output_rows.append(_create_record(agent, session_id, env, InteractionType.SESSION_END, session_state, {}))
    with writer_lock:
        with open(config.OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerows(output_rows)
    return asdict(agent.persona)

# --- MODIFICATION: Corrected _create_record function ---
def _create_record(agent: PersonaAgent, session_id: str, env: FireTVEnvironment, action: InteractionType, session_state: Dict, details: Dict) -> Dict:
    """Helper function to create a single interaction record dictionary.
    This version uses .get() for safe dictionary access to prevent KeyErrors."""
    profile = agent.persona.psychological_profile
    
    # Safely get current_content_id
    current_content_id = session_state.get('current_content_id')
    content_details = env.catalog.catalog_df[env.catalog.catalog_df['content_id'] == current_content_id].iloc[0].to_dict() if current_content_id else {}
    
    navigation_total_dpad = sum(session_state.get(f"{d.name.lower()}_count", 0) for d in [InteractionType.DPAD_UP, InteractionType.DPAD_DOWN, InteractionType.DPAD_LEFT, InteractionType.DPAD_RIGHT])
    current_time = session_state.get('current_time')

    return {
        'user_id': agent.persona.user_id, 'session_id': session_id,
        'interaction_timestamp': current_time.isoformat() if current_time else None,
        'interaction_type': action.name,
        'dpad_up_count': session_state.get('dpad_up_count', 0), 'dpad_down_count': session_state.get('dpad_down_count', 0),
        'dpad_left_count': session_state.get('dpad_left_count', 0), 'dpad_right_count': session_state.get('dpad_right_count', 0),
        'back_button_presses': session_state.get('back_button_presses', 0), 'menu_revisits': session_state.get('menu_revisits', 0),
        'scroll_speed': max(10, 150 - profile.patience * 100 + agent.state.frustration * 50),
        'hover_duration': details.get('time_delta_seconds', 0) if action == InteractionType.HOVER else 0,
        'time_since_last_interaction': details.get('time_delta_seconds', 0),
        'cpu_usage_percent': session_state.get('cpu_usage_percent', 0), 'wifi_signal_strength': session_state.get('wifi_signal_strength', 0),
        'network_latency_ms': max(20, 150 + session_state.get('wifi_signal_strength', 0) * 2 + np.random.normal(0,10)),
        'device_temperature': 35 + session_state.get('cpu_usage_percent', 0) * 0.1,
        'battery_level': session_state.get('battery_level', 0),
        'time_of_day': current_time.hour if current_time else None, 'day_of_week': current_time.weekday() if current_time else None,
        'content_id': current_content_id,
        'content_type': content_details.get('content_type'), 'content_genre': json.dumps(content_details.get('genres')) if content_details else None,
        'release_year': content_details.get('release_year'),
        'search_sophistication_pattern': 'advanced' if profile.conscientiousness > 0.6 else 'simple',
        'navigation_efficiency_score': (session_state.get('interaction_count', 0) + 1) / (navigation_total_dpad + session_state.get('back_button_presses', 0) + 1),
        'recommendation_engagement_pattern': 'high' if profile.openness > 0.5 else 'low',
        'cognitive_load_indicator': agent.state.cognitive_load,
        'decision_confidence_score': np.clip(1 - (agent.state.cognitive_load + agent.state.frustration) / 2, 0, 1),
        'frustration_level': agent.state.frustration,
        'attention_span_indicator': np.clip(profile.conscientiousness * agent.state.engagement, 0, 1),
        'exploration_tendency_score': profile.openness, 'platform_loyalty_score': 1 - profile.neuroticism,
        'social_influence_factor': profile.extraversion, 'price_sensitivity_score': 1 - profile.agreeableness,
        'content_diversity_preference': profile.openness, 'session_engagement_level': agent.state.engagement,
        'ui_adaptation_speed': profile.tech_savviness, 'temporal_consistency_pattern': profile.conscientiousness,
        'multi_platform_behavior_indicator': 1 - profile.conscientiousness, 'voice_command_usage_frequency': profile.tech_savviness * profile.extraversion,
        'return_likelihood_score': np.clip(profile.agreeableness * (1 - profile.neuroticism), 0, 1),
    }

# --- Main execution block (Unchanged) ---
if __name__ == "__main__":
    print("🚀 Starting Production-Scale Deep & Complex Fire TV Data Generation (Parallel Mode)")
    config = SimulationConfig()
    print("Creating shared content catalog...")
    catalog = ContentCatalog(config.NUM_CONTENT_ITEMS, Faker())
    header = list(_create_record(PersonaAgent(0, Faker()), 'dummy', FireTVEnvironment(catalog), InteractionType.HOME, {}, {}).keys())
    with open(config.OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
    manager = Manager()
    writer_lock = manager.Lock()
    num_processes = cpu_count()
    print(f"Distributing simulation across {num_processes} CPU cores...")
    with Pool(processes=num_processes) as pool:
        worker_func = partial(simulate_user_worker, config=config, catalog=catalog, writer_lock=writer_lock, header=header)
        user_ids = range(config.NUM_USERS)
        all_personas = list(tqdm(pool.imap_unordered(worker_func, user_ids), total=config.NUM_USERS, desc="Simulating Users"))
    print("\n✅ All user simulations complete.")
    print(f"Saving {len(all_personas)} user personas...")
    pd.json_normalize(all_personas, sep='_').to_csv(config.PERSONA_OUTPUT_FILE, index=False)
    print(f"User personas saved to {config.PERSONA_OUTPUT_FILE}")
    print("\n🎉 Production-scale simulation finished successfully!")
