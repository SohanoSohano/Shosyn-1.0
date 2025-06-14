# dataset_generator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from data_structures import *

class ComprehensiveDatasetGenerator:
    """Generate realistic Fire TV interaction dataset with psychological patterns"""
    
    def __init__(self, seed: int = 42):
        """Initialize dataset generator with reproducible seed"""
        np.random.seed(seed)
        self.user_profiles = self._create_psychological_profiles()
        self.content_db = self._create_content_database()
        self.interaction_patterns = self._create_interaction_patterns()
        
        print("Dataset Generator Initialized:")
        print(f"- User profile types: {len(self.user_profiles)}")
        print(f"- Content genres: {len(self.content_db.genres)}")
        print(f"- Platforms: {len(self.content_db.platforms)}")
        
    def _create_psychological_profiles(self) -> Dict[str, UserPsychologicalProfile]:
        """Create diverse psychological user profiles based on research"""
        profiles = {
            'power_user': UserPsychologicalProfile(
                profile_name='power_user',
                exploration_tendency=0.85,
                decision_confidence=0.80,
                frustration_tolerance=0.70,
                cognitive_load_preference=0.30,
                social_preference=0.25,
                tech_savviness=0.90,
                content_sophistication=0.80
            ),
            'casual_browser': UserPsychologicalProfile(
                profile_name='casual_browser',
                exploration_tendency=0.60,
                decision_confidence=0.50,
                frustration_tolerance=0.40,
                cognitive_load_preference=0.70,
                social_preference=0.60,
                tech_savviness=0.40,
                content_sophistication=0.50
            ),
            'focused_viewer': UserPsychologicalProfile(
                profile_name='focused_viewer',
                exploration_tendency=0.20,
                decision_confidence=0.90,
                frustration_tolerance=0.80,
                cognitive_load_preference=0.20,
                social_preference=0.10,
                tech_savviness=0.60,
                content_sophistication=0.70
            ),
            'social_watcher': UserPsychologicalProfile(
                profile_name='social_watcher',
                exploration_tendency=0.50,
                decision_confidence=0.60,
                frustration_tolerance=0.60,
                cognitive_load_preference=0.50,
                social_preference=0.90,
                tech_savviness=0.50,
                content_sophistication=0.60
            ),
            'indecisive_explorer': UserPsychologicalProfile(
                profile_name='indecisive_explorer',
                exploration_tendency=0.90,
                decision_confidence=0.25,
                frustration_tolerance=0.30,
                cognitive_load_preference=0.80,
                social_preference=0.40,
                tech_savviness=0.35,
                content_sophistication=0.45
            )
        }
        return profiles
    
    def _create_content_database(self) -> ContentDatabase:
        """Create realistic content database"""
        return ContentDatabase(
            genres=['action', 'comedy', 'drama', 'documentary', 'sci-fi', 'horror', 'romance', 'thriller', 'animation'],
            platforms=['netflix', 'prime_video', 'disney_plus', 'hotstar', 'apple_tv', 'hulu'],
            content_types=['movie', 'series', 'documentary', 'short_film', 'special'],
            popularity_scores=np.random.beta(2, 5, 2000),
            thematic_embeddings=np.random.randn(2000, 64)
        )
    
    def _create_interaction_patterns(self) -> Dict:
        """Create realistic interaction timing patterns"""
        return {
            'peak_hours': [19, 20, 21, 22],
            'weekend_boost': 1.3,
            'session_lengths': {
                'short': (5, 15),
                'medium': (15, 40),
                'long': (40, 80)
            }
        }
    
    def generate_comprehensive_dataset(
        self, 
        num_users: int = 100, 
        sequences_per_user: int = 5,
        sequence_length_range: Tuple[int, int] = (20, 50),
        time_span_days: int = 30
    ) -> List[List[ComprehensiveFireTVInteraction]]:
        """Generate comprehensive Fire TV dataset"""
        
        print(f"\n🎬 Generating Fire TV Dataset:")
        print(f"├── Users: {num_users}")
        print(f"├── Sessions per user: {sequences_per_user}")
        print(f"├── Sequence length: {sequence_length_range[0]}-{sequence_length_range[1]}")
        print(f"└── Time span: {time_span_days} days")
        
        all_sequences = []
        base_timestamp = datetime.now().timestamp() - (time_span_days * 24 * 3600)
        
        for user_idx in range(num_users):
            user_id = f"user_{user_idx:04d}"
            profile_name = np.random.choice(list(self.user_profiles.keys()))
            user_profile = self.user_profiles[profile_name]
            
            user_sequences = self._generate_user_sessions(
                user_id, user_profile, sequences_per_user, 
                sequence_length_range, base_timestamp, time_span_days
            )
            
            all_sequences.extend(user_sequences)
            
            if (user_idx + 1) % 20 == 0:
                print(f"Generated {user_idx + 1}/{num_users} users...")
        
        total_interactions = sum(len(seq) for seq in all_sequences)
        print(f"\n✅ Dataset Generation Complete:")
        print(f"├── Total sequences: {len(all_sequences)}")
        print(f"├── Total interactions: {total_interactions}")
        print(f"└── Average sequence length: {total_interactions / len(all_sequences):.1f}")
        
        return all_sequences
    
    def _generate_user_sessions(self, user_id: str, profile: UserPsychologicalProfile, num_sessions: int,
                               length_range: Tuple[int, int], base_timestamp: float, time_span_days: int):
        """Generate all sessions for a single user"""
        sessions = []
        
        for session_idx in range(num_sessions):
            session_id = f"{user_id}_session_{session_idx:03d}"
            session_timestamp = self._generate_realistic_session_time(base_timestamp, time_span_days, profile)
            session_length = np.random.randint(length_range[0], length_range[1] + 1)
            session = self._generate_single_session(user_id, session_id, session_timestamp, session_length, profile)
            sessions.append(session)
        
        return sessions
    
    def _generate_realistic_session_time(self, base_timestamp: float, time_span_days: int, profile: UserPsychologicalProfile) -> float:
        """Generate realistic session start time"""
        day_offset = np.random.randint(0, time_span_days)
        day_timestamp = base_timestamp + (day_offset * 24 * 3600)
        
        if np.random.rand() < 0.7:
            hour = np.random.choice(self.interaction_patterns['peak_hours'])
        else:
            hour = np.random.randint(6, 24)
        
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        return day_timestamp + (hour * 3600) + (minute * 60) + second
    
    def _generate_single_session(self, user_id: str, session_id: str, start_timestamp: float, 
                                session_length: int, profile: UserPsychologicalProfile):
        """Generate a single realistic session"""
        interactions = []
        session_state = SessionState(
            social_context=np.random.rand() < profile.social_preference,
            current_platform=np.random.choice(self.content_db.platforms)
        )
        
        current_timestamp = start_timestamp
        
        for interaction_idx in range(session_length):
            time_delta = 0.0 if interaction_idx == 0 else np.random.exponential(30 + (1 - profile.decision_confidence) * 60)
            current_timestamp += time_delta
            
            interaction = self._generate_comprehensive_interaction(
                user_id, session_id, current_timestamp, time_delta,
                profile, session_state, interaction_idx, session_length
            )
            
            interactions.append(interaction)
            self._update_session_state(interaction, session_state, profile)
        
        return interactions
    
    def _generate_comprehensive_interaction(self, user_id: str, session_id: str, timestamp: float, time_delta: float,
                                          profile: UserPsychologicalProfile, session_state: SessionState, 
                                          position: int, total_length: int):
        """Generate single interaction with all tier attributes"""
        
        interaction_type = self._determine_interaction_type(profile, session_state, position)
        content_id = self._generate_content_engagement(interaction_type, profile, session_state)
        
        # Generate all attributes
        dt = datetime.fromtimestamp(timestamp)
        circadian_embedding = (np.sin(2 * np.pi * dt.hour / 24) + 1) / 2
        
        # Basic attributes
        scroll_speed = 50 + profile.tech_savviness * 80 + session_state.accumulated_frustration * 40
        hover_duration = (1 - profile.decision_confidence) * 8 + session_state.decision_fatigue * 3 + np.random.exponential(2)
        back_clicks = np.random.poisson(session_state.accumulated_frustration * 2.5)
        
        # Watch history
        watch_history = {
            'total_hours': np.random.gamma(3, 15),
            'unique_titles': np.random.poisson(75),
            'platforms_used': len(self.content_db.platforms),
            'avg_session_length': np.random.normal(45, 15)
        }
        
        return ComprehensiveFireTVInteraction(
            user_id=user_id,
            session_id=session_id,
            content_id=content_id,
            interaction_timestamp=timestamp,
            time_since_last_interaction=time_delta,
            circadian_rhythm_embedding=circadian_embedding,
            scroll_speed=scroll_speed,
            scroll_depth=np.random.beta(2, 3),
            hover_duration=hover_duration,
            back_click_count=back_clicks,
            filter_changes=np.random.poisson(1.2) if profile.exploration_tendency > 0.6 else np.random.poisson(0.3),
            click_type=ClickType.PLAY if interaction_type == InteractionType.SELECTION else ClickType.PLAY,
            time_to_click=(1 - profile.decision_confidence) * 8 + session_state.decision_fatigue * 4 + np.random.exponential(3),
            search_query_length=int(np.random.poisson(8 + profile.content_sophistication * 12)) if interaction_type == InteractionType.SEARCH else 0,
            search_abort_flag=np.random.rand() < ((1 - profile.decision_confidence) * 0.4 + session_state.decision_fatigue * 0.3),
            abandonment_point=min((1 - profile.frustration_tolerance) + (1 - session_state.attention_span_remaining) * 0.5, 1.0) if content_id else 0.0,
            pause_frequency=int(np.random.poisson((1 - session_state.attention_span_remaining) * 5)),
            rewatch_frequency=np.random.poisson(0.8) if profile.content_sophistication > 0.6 else np.random.poisson(0.2),
            device_type=DeviceType.FIRE_TV_CUBE if profile.tech_savviness > 0.7 else DeviceType.FIRE_TV_STICK,
            network_latency=np.random.gamma(2, 25),
            battery_level=max(0.1, 1.0 - (position * 0.015) + np.random.normal(0, 0.08)),
            frustration_level=min(session_state.accumulated_frustration, 1.0),
            regulatory_focus=(0.7 if 18 <= dt.hour <= 22 else 0.4) * profile.exploration_tendency + np.random.uniform(0, 0.2),
            
            # Tier 2 attributes
            cognitive_load_index=min(session_state.cognitive_load_buildup, 1.0),
            decision_confidence_score=profile.decision_confidence * (1 - session_state.decision_fatigue * 0.3),
            attention_span_indicator=max(0.1, session_state.attention_span_remaining),
            exploration_exploitation_ratio=len(session_state.content_explored) / max(position, 1),
            session_coherence=np.mean(session_state.content_theme_consistency) if session_state.content_theme_consistency else 1.0,
            interaction_intensity_wave=0.5,
            navigation_efficiency=profile.tech_savviness * (1 - session_state.accumulated_frustration * 0.2),
            micro_pause_frequency=(1 - profile.decision_confidence) * 4 + session_state.decision_fatigue * 3,
            choice_deliberation_time=(1 - profile.decision_confidence) * 12 + np.random.exponential(3),
            interaction_rhythm_consistency=1.0 if position < 4 else max(0, 1 - np.random.uniform(0, 0.5)),
            platform_switching_frequency=session_state.platform_switches / max(position, 1),
            cross_platform_ctr=np.random.beta(3, 7),
            social_viewing_context=session_state.social_context,
            cognitive_load_tolerance=profile.cognitive_load_preference,
            exploration_vs_exploitation=min((len(session_state.content_explored) / max(position, 1)) * 1.5, 1.0),
            
            # Tier 3 attributes
            exposure_position_ui=np.random.randint(1, 25),
            content_social_tags=np.random.rand() < 0.25,
            qos_buffer_rate=np.random.beta(4, 2),
            unified_watch_history=watch_history,
            social_proof_susceptibility=profile.social_preference * 0.8 + np.random.uniform(0, 0.2),
            platform_preference_score=np.random.beta(3, 3),
            voice_command_tone=np.random.normal(180, 40) if np.random.rand() < 0.15 else None,
            emotional_reaction_frequency=profile.social_preference * np.random.exponential(2),
            locus_of_control=profile.decision_confidence * 0.9 + np.random.uniform(0, 0.1),
            self_monitoring_tendency=profile.tech_savviness * np.random.beta(3, 2),
            content_thematic_similarity=np.random.beta(3, 2),
            user_interface_adaptation_speed=profile.tech_savviness * np.random.beta(4, 2)
        )
    
    def _determine_interaction_type(self, profile: UserPsychologicalProfile, session_state: SessionState, position: int):
        """Determine interaction type based on psychological state"""
        if session_state.accumulated_frustration > 0.7:
            return np.random.choice([InteractionType.BACK, InteractionType.PAUSE], p=[0.6, 0.4])
        
        if profile.exploration_tendency > 0.7 and len(session_state.content_explored) < 15:
            return np.random.choice([InteractionType.NAVIGATION, InteractionType.SEARCH, InteractionType.SCROLL], p=[0.4, 0.35, 0.25])
        
        return np.random.choice([InteractionType.NAVIGATION, InteractionType.SELECTION, InteractionType.PLAY], p=[0.4, 0.35, 0.25])
    
    def _generate_content_engagement(self, interaction_type: InteractionType, profile: UserPsychologicalProfile, session_state: SessionState):
        """Generate content ID based on interaction type"""
        if interaction_type in [InteractionType.SELECTION, InteractionType.PLAY]:
            if profile.exploration_tendency > 0.6:
                content_id = f"content_{np.random.randint(1, 200)}"
            else:
                content_id = f"content_{np.random.randint(1, 50)}"
            
            session_state.content_explored.add(content_id)
            return content_id
        
        return None
    
    def _update_session_state(self, interaction: ComprehensiveFireTVInteraction, session_state: SessionState, profile: UserPsychologicalProfile):
        """Update session state based on interaction"""
        if interaction.back_click_count > 1 or interaction.search_abort_flag:
            session_state.accumulated_frustration = min(1.0, session_state.accumulated_frustration + 0.12)
        elif interaction.content_id and interaction.abandonment_point > 0.7:
            session_state.accumulated_frustration = max(0.0, session_state.accumulated_frustration - 0.08)
        
        session_state.decision_fatigue = min(1.0, session_state.decision_fatigue + 0.025)
        
        if interaction.filter_changes > 0 or interaction.search_query_length > 15:
            session_state.cognitive_load_buildup = min(1.0, session_state.cognitive_load_buildup + 0.08)
        
        if interaction.pause_frequency > 3:
            session_state.attention_span_remaining = max(0.1, session_state.attention_span_remaining - 0.15)
        
        if np.random.rand() < 0.03:
            session_state.platform_switches += 1
        
        session_state.interaction_history.append(interaction)
        
        if interaction.content_id:
            session_state.content_theme_consistency.append(np.random.beta(3, 2))
