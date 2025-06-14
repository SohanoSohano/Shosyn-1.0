# data_structures.py
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import numpy as np

class InteractionType(Enum):
    NAVIGATION = "navigation"
    SELECTION = "selection"
    SEARCH = "search"
    PAUSE = "pause"
    PLAY = "play"
    SCROLL = "scroll"
    FILTER = "filter"
    BACK = "back"

class ClickType(Enum):
    PLAY = "play"
    TRAILER = "trailer"
    ADD_TO_LIST = "add_to_list"
    MORE_INFO = "more_info"
    LIKE = "like"
    DISLIKE = "dislike"

class DeviceType(Enum):
    FIRE_TV_STICK = "fire_tv_stick"
    FIRE_TV_CUBE = "fire_tv_cube"
    SMART_TV = "smart_tv"
    MOBILE_APP = "mobile_app"

@dataclass
class ComprehensiveFireTVInteraction:
    """Complete Fire TV interaction with Tiers 1-3 implemented, Tier 4 optional"""
    
    # TIER 1: Core Interaction Metrics (20 attributes)
    interaction_timestamp: float
    time_since_last_interaction: float
    circadian_rhythm_embedding: float
    scroll_speed: float
    scroll_depth: float
    hover_duration: float
    back_click_count: int
    filter_changes: int
    click_type: ClickType
    time_to_click: float
    search_query_length: int
    search_abort_flag: bool
    abandonment_point: float
    pause_frequency: int
    rewatch_frequency: int
    device_type: DeviceType
    network_latency: float
    battery_level: float
    frustration_level: float
    regulatory_focus: float
    
    # TIER 2: Advanced Behavioral Metrics (15 attributes)
    cognitive_load_index: float
    decision_confidence_score: float
    attention_span_indicator: float
    exploration_exploitation_ratio: float
    session_coherence: float
    interaction_intensity_wave: float
    navigation_efficiency: float
    micro_pause_frequency: float
    choice_deliberation_time: float
    interaction_rhythm_consistency: float
    platform_switching_frequency: float
    cross_platform_ctr: float
    social_viewing_context: bool
    cognitive_load_tolerance: float
    exploration_vs_exploitation: float
    
    # TIER 3: Quick Wins (12 attributes)
    exposure_position_ui: int
    content_social_tags: bool
    qos_buffer_rate: float
    unified_watch_history: Dict
    social_proof_susceptibility: float
    platform_preference_score: float
    voice_command_tone: Optional[float]
    emotional_reaction_frequency: float
    locus_of_control: float
    self_monitoring_tendency: float
    content_thematic_similarity: float
    user_interface_adaptation_speed: float
    
    # Metadata
    user_id: str
    session_id: str
    content_id: Optional[str] = None

@dataclass
class UserPsychologicalProfile:
    """Psychological profile for consistent user behavior generation"""
    profile_name: str
    exploration_tendency: float
    decision_confidence: float
    frustration_tolerance: float
    cognitive_load_preference: float
    social_preference: float
    tech_savviness: float
    content_sophistication: float

@dataclass
class SessionState:
    """Dynamic session state for realistic behavior modeling"""
    accumulated_frustration: float = 0.0
    content_explored: set = None
    decision_fatigue: float = 0.0
    social_context: bool = False
    current_platform: str = "netflix"
    interaction_history: List = None
    cognitive_load_buildup: float = 0.0
    attention_span_remaining: float = 1.0
    platform_switches: int = 0
    content_theme_consistency: List = None
    
    def __post_init__(self):
        if self.content_explored is None:
            self.content_explored = set()
        if self.interaction_history is None:
            self.interaction_history = []
        if self.content_theme_consistency is None:
            self.content_theme_consistency = []

@dataclass
class ContentDatabase:
    """Content database for realistic interaction generation"""
    genres: List[str]
    platforms: List[str]
    content_types: List[str]
    popularity_scores: np.ndarray
    thematic_embeddings: np.ndarray
