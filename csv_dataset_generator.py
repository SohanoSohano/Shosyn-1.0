# csv_dataset_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import json
from data_structures import *
from dataset_generator import ComprehensiveDatasetGenerator

class CSVDatasetExporter:
    """Export Fire TV dataset to detailed CSV format with all parameters"""
    
    def __init__(self):
        self.generator = ComprehensiveDatasetGenerator(seed=42)
        self.csv_columns = self._define_csv_columns()
        
        print("CSV Dataset Exporter Initialized")
        print(f"Total columns to export: {len(self.csv_columns)}")
        
    def _define_csv_columns(self) -> List[str]:
        """Define all CSV column names"""
        columns = [
            # Metadata
            'user_id', 'session_id', 'content_id', 'interaction_index',
            
            # TIER 1: Core Interaction Metrics (20)
            'interaction_timestamp', 'interaction_datetime', 'time_since_last_interaction', 
            'circadian_rhythm_embedding', 'scroll_speed', 'scroll_depth', 'hover_duration',
            'back_click_count', 'filter_changes', 'click_type', 'time_to_click',
            'search_query_length', 'search_abort_flag', 'abandonment_point',
            'pause_frequency', 'rewatch_frequency', 'device_type', 'network_latency',
            'battery_level', 'frustration_level', 'regulatory_focus',
            
            # TIER 2: Advanced Behavioral Metrics (15)
            'cognitive_load_index', 'decision_confidence_score', 'attention_span_indicator',
            'exploration_exploitation_ratio', 'session_coherence', 'interaction_intensity_wave',
            'navigation_efficiency', 'micro_pause_frequency', 'choice_deliberation_time',
            'interaction_rhythm_consistency', 'platform_switching_frequency', 'cross_platform_ctr',
            'social_viewing_context', 'cognitive_load_tolerance', 'exploration_vs_exploitation',
            
            # TIER 3: Quick Wins (12)
            'exposure_position_ui', 'content_social_tags', 'qos_buffer_rate',
            'watch_history_total_hours', 'watch_history_unique_titles', 'watch_history_platforms_used',
            'social_proof_susceptibility', 'platform_preference_score', 'voice_command_tone',
            'emotional_reaction_frequency', 'locus_of_control', 'self_monitoring_tendency',
            'content_thematic_similarity', 'user_interface_adaptation_speed',
            
            # Derived Analytics
            'user_profile_type', 'session_position', 'session_length', 'hour_of_day',
            'day_of_week', 'is_weekend', 'is_peak_hours', 'content_engaged'
        ]
        
        return columns
    
    def generate_detailed_csv_dataset(self, num_entries: int = 150, output_filename: str = "fire_tv_comprehensive_dataset.csv"):
        """Generate detailed CSV dataset with specified number of entries"""
        
        print(f"\n🎬 Generating Detailed CSV Dataset:")
        print(f"├── Target entries: {num_entries}")
        print(f"├── Output file: {output_filename}")
        print(f"└── Columns: {len(self.csv_columns)}")
        
        # Calculate distribution
        users_needed = max(10, num_entries // 15)
        sessions_per_user = 2
        interactions_per_session = max(8, num_entries // (users_needed * sessions_per_user))
        
        print(f"\nDataset Structure:")
        print(f"├── Users: {users_needed}")
        print(f"├── Sessions per user: {sessions_per_user}")
        print(f"└── Interactions per session: {interactions_per_session}")
        
        # Generate sequences
        sequences = self.generator.generate_comprehensive_dataset(
            num_users=users_needed,
            sequences_per_user=sessions_per_user,
            sequence_length_range=(interactions_per_session-2, interactions_per_session+2),
            time_span_days=14
        )
        
        # Convert to CSV
        csv_data = self._convert_sequences_to_csv(sequences)
        df = pd.DataFrame(csv_data, columns=self.csv_columns)
        df = self._add_derived_analytics(df)
        
        # Sort and save
        df = df.sort_values(['user_id', 'session_id', 'interaction_timestamp']).reset_index(drop=True)
        df.to_csv(output_filename, index=False)
        
        print(f"\n✅ CSV Dataset Generated Successfully:")
        print(f"├── Total entries: {len(df)}")
        print(f"├── Unique users: {df['user_id'].nunique()}")
        print(f"├── Unique sessions: {df['session_id'].nunique()}")
        print(f"└── Saved to: {output_filename}")
        
        return df
    
    def _convert_sequences_to_csv(self, sequences):
        """Convert interaction sequences to CSV format"""
        csv_rows = []
        interaction_counter = 0
        user_profiles = {}
        
        for sequence in sequences:
            if not sequence:
                continue
                
            user_id = sequence[0].user_id
            if user_id not in user_profiles:
                user_profiles[user_id] = self._determine_user_profile_type(sequence)
            
            for position, interaction in enumerate(sequence):
                row = self._interaction_to_csv_row(interaction, position, len(sequence), user_profiles[user_id], interaction_counter)
                csv_rows.append(row)
                interaction_counter += 1
        
        return csv_rows
    
    def _interaction_to_csv_row(self, interaction, position, session_length, user_profile_type, interaction_index):
        """Convert single interaction to CSV row"""
        dt = datetime.fromtimestamp(interaction.interaction_timestamp)
        watch_history = interaction.unified_watch_history
        
        row = [
            # Metadata
            interaction.user_id, interaction.session_id, interaction.content_id or '', interaction_index,
            
            # TIER 1
            interaction.interaction_timestamp, dt.strftime('%Y-%m-%d %H:%M:%S'),
            interaction.time_since_last_interaction, interaction.circadian_rhythm_embedding,
            interaction.scroll_speed, interaction.scroll_depth, interaction.hover_duration,
            interaction.back_click_count, interaction.filter_changes, interaction.click_type.value,
            interaction.time_to_click, interaction.search_query_length, interaction.search_abort_flag,
            interaction.abandonment_point, interaction.pause_frequency, interaction.rewatch_frequency,
            interaction.device_type.value, interaction.network_latency, interaction.battery_level,
            interaction.frustration_level, interaction.regulatory_focus,
            
            # TIER 2
            interaction.cognitive_load_index, interaction.decision_confidence_score,
            interaction.attention_span_indicator, interaction.exploration_exploitation_ratio,
            interaction.session_coherence, interaction.interaction_intensity_wave,
            interaction.navigation_efficiency, interaction.micro_pause_frequency,
            interaction.choice_deliberation_time, interaction.interaction_rhythm_consistency,
            interaction.platform_switching_frequency, interaction.cross_platform_ctr,
            interaction.social_viewing_context, interaction.cognitive_load_tolerance,
            interaction.exploration_vs_exploitation,
            
            # TIER 3
            interaction.exposure_position_ui, interaction.content_social_tags,
            interaction.qos_buffer_rate, watch_history.get('total_hours', 0),
            watch_history.get('unique_titles', 0), watch_history.get('platforms_used', 0),
            interaction.social_proof_susceptibility, interaction.platform_preference_score,
            interaction.voice_command_tone or 0, interaction.emotional_reaction_frequency,
            interaction.locus_of_control, interaction.self_monitoring_tendency,
            interaction.content_thematic_similarity, interaction.user_interface_adaptation_speed,
            
            # Derived Analytics
            user_profile_type, position, session_length, dt.hour, dt.weekday(),
            dt.weekday() >= 5, 18 <= dt.hour <= 22, 1 if interaction.content_id else 0
        ]
        
        return row
    
    def _determine_user_profile_type(self, sequence):
        """Determine user profile type based on patterns"""
        if not sequence:
            return 'unknown'
        
        avg_exploration = np.mean([i.exploration_exploitation_ratio for i in sequence])
        avg_confidence = np.mean([i.decision_confidence_score for i in sequence])
        avg_social = np.mean([i.social_viewing_context for i in sequence])
        
        if avg_exploration > 0.7 and avg_confidence > 0.6:
            return 'power_user'
        elif avg_social > 0.7:
            return 'social_watcher'
        elif avg_exploration < 0.3 and avg_confidence > 0.7:
            return 'focused_viewer'
        elif avg_exploration > 0.7 and avg_confidence < 0.4:
            return 'indecisive_explorer'
        else:
            return 'casual_browser'
    
    def _add_derived_analytics(self, df):
        """Add derived analytics columns"""
        df['interaction_datetime'] = pd.to_datetime(df['interaction_datetime'])
        df['hour_of_day'] = df['interaction_datetime'].dt.hour
        df['day_of_week'] = df['interaction_datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'] >= 5
        df['is_peak_hours'] = (df['hour_of_day'] >= 18) & (df['hour_of_day'] <= 22)
        
        return df

def main():
    """Main function to generate comprehensive CSV dataset"""
    print("🚀 Starting Comprehensive Fire TV CSV Dataset Generation")
    
    exporter = CSVDatasetExporter()
    df = exporter.generate_detailed_csv_dataset(num_entries=150, output_filename="fire_tv_comprehensive_dataset.csv")
    
    # Display sample
    print("\n📋 SAMPLE DATA (First 5 rows):")
    sample_cols = ['user_id', 'session_id', 'interaction_datetime', 'click_type', 
                   'frustration_level', 'decision_confidence_score', 'user_profile_type']
    print(df[sample_cols].head().to_string(index=False))
    
    print(f"\n📊 DATASET STRUCTURE:")
    print(f"├── Shape: {df.shape}")
    print(f"├── Columns: {len(df.columns)}")
    print(f"└── File: fire_tv_comprehensive_dataset.csv")
    
    return df

if __name__ == "__main__":
    dataset = main()
