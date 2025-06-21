# data/data_structures.py
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import torch
import numpy as np

class InteractionType(Enum):
    NAVIGATION = "navigation"
    SELECTION = "selection"
    SEARCH = "search"
    PLAY = "play"
    PAUSE = "pause"
    SCROLL = "scroll"

class DeviceType(Enum):
    FIRE_TV_STICK_4K = "fire_tv_stick_4k"
    FIRE_TV_CUBE = "fire_tv_cube"
    FIRE_TV_SMART_TV = "fire_tv_smart_tv"

@dataclass
class FireTVInteraction:
    """Enhanced Fire TV interaction with multimodal data"""
    
    # Core interaction data
    user_id: str
    session_id: str
    timestamp: float
    interaction_type: InteractionType
    
    # Navigation features (15 features)
    scroll_speed: float
    scroll_depth: float
    hover_duration: float
    back_click_count: int
    navigation_efficiency: float
    dpad_usage_pattern: float
    menu_depth: int
    search_query_length: int
    filter_usage: int
    page_transitions: int
    ui_element_focus_time: float
    gesture_complexity: float
    navigation_errors: int
    shortcut_usage: int
    voice_navigation_attempts: int
    
    # Content features (12 features)
    content_id: Optional[str]
    content_genre: str
    content_duration: float
    content_popularity: float
    content_rating: float
    trailer_viewed: bool
    content_details_viewed: bool
    similar_content_explored: int
    content_completion_rate: float
    content_abandonment_point: float
    content_social_score: float
    content_recency: float
    
    # Device features (10 features)
    device_type: DeviceType
    network_latency: float
    cpu_usage: float
    memory_usage: float
    battery_level: float
    screen_resolution: str
    audio_quality: str
    remote_battery_level: float
    wifi_signal_strength: float
    device_temperature: float
    
    # Temporal features (12 features)
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    is_peak_hours: bool
    session_duration: float
    time_since_last_session: float
    seasonal_factor: float
    holiday_indicator: bool
    viewing_context: str  # alone, family, party
    concurrent_device_usage: int
    time_zone_offset: float
    daylight_saving_active: bool

@dataclass
class PsychologicalTraits:
    """20 psychological traits for Fire TV users"""
    cognitive_load: float
    decision_confidence: float
    frustration_level: float
    exploration_tendency: float
    attention_span: float
    navigation_efficiency: float
    platform_loyalty: float
    social_influence: float
    price_sensitivity: float
    content_diversity: float
    session_engagement: float
    ui_adaptation: float
    temporal_consistency: float
    multi_platform_behavior: float
    voice_usage: float
    recommendation_acceptance: float
    search_sophistication: float
    device_preference: float
    peak_alignment: float
    return_likelihood: float

@dataclass
class ContentItem:
    """Content item for recommendations"""
    content_id: str
    title: str
    genre: List[str]
    duration: float
    rating: float
    popularity_score: float
    release_date: str
    content_type: str  # movie, series, documentary
    complexity_score: float
    binge_potential: float
    social_score: float
    novelty_score: float
