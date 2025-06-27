# run_simulation.py
import pandas as pd
import random
import ast
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import sys

from llm_agent import LLMAgent
from firetv_environment import FireTVEnvironment

# --- Simulation Configuration ---
NUM_SESSIONS_TO_SIMULATE = 100
OUTPUT_FILE_PREFIX = "simulation_500_dataset.csv" 
TMDB_DATA_PATH = "/home/ubuntu/Shosyn-1.0/dataset/tmdb_5000_movies.csv"

# --- ENHANCED PERSONA DEFINITIONS WITH BETTER FRUSTRATION MODELING ---
PERSONAS = [
    # High-frustration personas (more likely to get frustrated)
    {
        "name": "Impatient Professional", 
        "narrative": "Always in a hurry, gets frustrated quickly when things don't work smoothly. Expects instant results.",
        "ocean": {'openness': 0.3, 'conscientiousness': 0.8, 'extraversion': 0.4, 'agreeableness': 0.3, 'neuroticism': 0.9},
        "preferences": {"Comedy": 0.8, "Action": 0.7},
        "frustration_sensitivity": 0.8,  # NEW: High sensitivity
        "base_frustration": 0.15  # NEW: Starts with some frustration
    },
    {
        "name": "Tech Struggler", 
        "narrative": "Finds technology confusing and gets overwhelmed easily. Makes mistakes and gets frustrated with complex interfaces.",
        "ocean": {'openness': 0.2, 'conscientiousness': 0.5, 'extraversion': 0.3, 'agreeableness': 0.7, 'neuroticism': 0.8},
        "preferences": {"Family": 0.9, "Comedy": 0.7},
        "frustration_sensitivity": 0.9,  # Very high sensitivity
        "base_frustration": 0.2  # Starts frustrated
    },
    {
        "name": "Perfectionist Critic", 
        "narrative": "Has very high standards and gets frustrated when content or interface doesn't meet expectations.",
        "ocean": {'openness': 0.7, 'conscientiousness': 0.9, 'extraversion': 0.2, 'agreeableness': 0.2, 'neuroticism': 0.7},
        "preferences": {"Drama": 0.9, "Documentary": 0.8},
        "frustration_sensitivity": 0.7,
        "base_frustration": 0.1
    },
    {
        "name": "Indecisive Browser", 
        "narrative": "Can't make up their mind, clicks around a lot, gets frustrated with their own indecision.",
        "ocean": {'openness': 0.6, 'conscientiousness': 0.3, 'extraversion': 0.4, 'agreeableness': 0.6, 'neuroticism': 0.8},
        "preferences": {"Romance": 0.6, "Comedy": 0.7, "Drama": 0.6},
        "frustration_sensitivity": 0.6,
        "base_frustration": 0.08
    },
    
    # Medium-frustration personas
    {
        "name": "Casual Evening Viewer", 
        "narrative": "Just wants to relax after work, gets mildly frustrated if browsing takes too long.",
        "ocean": {'openness': 0.5, 'conscientiousness': 0.6, 'extraversion': 0.5, 'agreeableness': 0.7, 'neuroticism': 0.5},
        "preferences": {"Comedy": 0.8, "Action": 0.6, "TV Show": 0.7},
        "frustration_sensitivity": 0.4,
        "base_frustration": 0.05
    },
    {
        "name": "Family Organizer", 
        "narrative": "Trying to find content for the whole family, gets frustrated when can't find suitable options.",
        "ocean": {'openness': 0.5, 'conscientiousness': 0.9, 'extraversion': 0.7, 'agreeableness': 0.9, 'neuroticism': 0.4},
        "preferences": {"Animation": 0.9, "Family": 0.9, "Adventure": 0.8},
        "frustration_sensitivity": 0.5,
        "base_frustration": 0.03
    },
    {
        "name": "Weekend Binger", 
        "narrative": "Looking for a long series to binge, gets frustrated if can't find the right mood.",
        "ocean": {'openness': 0.6, 'conscientiousness': 0.7, 'extraversion': 0.5, 'agreeableness': 0.6, 'neuroticism': 0.4},
        "preferences": {"TV Show": 1.0, "Drama": 0.8, "Crime": 0.7},
        "frustration_sensitivity": 0.3,
        "base_frustration": 0.02
    },
    {
        "name": "Social Sharer", 
        "narrative": "Wants to watch trending content to discuss with friends, gets frustrated if can't find popular shows.",
        "ocean": {'openness': 0.6, 'conscientiousness': 0.5, 'extraversion': 0.8, 'agreeableness': 0.8, 'neuroticism': 0.5},
        "preferences": {"Action": 0.8, "Comedy": 0.8, "Sci-Fi": 0.7},
        "frustration_sensitivity": 0.4,
        "base_frustration": 0.04
    },
    
    # Low-frustration personas (patient, calm)
    {
        "name": "Zen Viewer", 
        "narrative": "Very patient and calm, rarely gets frustrated. Enjoys the browsing process itself.",
        "ocean": {'openness': 0.6, 'conscientiousness': 0.6, 'extraversion': 0.5, 'agreeableness': 0.8, 'neuroticism': 0.1},
        "preferences": {"Drama": 0.7, "Family": 0.7, "Adventure": 0.6},
        "frustration_sensitivity": 0.1,
        "base_frustration": 0.0
    },
    {
        "name": "Curious Explorer", 
        "narrative": "Enjoys discovering new content, patient with browsing, low frustration tolerance.",
        "ocean": {'openness': 0.9, 'conscientiousness': 0.4, 'extraversion': 0.6, 'agreeableness': 0.8, 'neuroticism': 0.2},
        "preferences": {"Documentary": 0.8, "Sci-Fi": 0.8, "Foreign": 0.7},
        "frustration_sensitivity": 0.2,
        "base_frustration": 0.0
    },
    
    # Add more of your original personas with frustration parameters
    {
        "name": "Adrenaline Junkie", 
        "narrative": "Looking for high-octane excitement, gets frustrated if can't find action quickly.",
        "ocean": {'openness': 0.3, 'conscientiousness': 0.2, 'extraversion': 0.9, 'agreeableness': 0.2, 'neuroticism': 0.6},
        "preferences": {"Action": 0.9, "Thriller": 0.8, "Horror": 0.7},
        "frustration_sensitivity": 0.5,
        "base_frustration": 0.08
    },
    {
        "name": "Background Viewer", 
        "narrative": "Just wants simple content while multitasking, gets frustrated with complex navigation.",
        "ocean": {'openness': 0.2, 'conscientiousness': 0.5, 'extraversion': 0.5, 'agreeableness': 0.6, 'neuroticism': 0.4},
        "preferences": {"Comedy": 0.9, "TV Show": 0.9, "Family": 0.7},
        "frustration_sensitivity": 0.3,
        "base_frustration": 0.02
    },
    {
        "name": "Film Buff", 
        "narrative": "Has specific tastes, gets frustrated when can't find quality content.",
        "ocean": {'openness': 0.9, 'conscientiousness': 0.7, 'extraversion': 0.3, 'agreeableness': 0.1, 'neuroticism': 0.4},
        "preferences": {"Drama": 0.9, "Foreign": 0.9, "Documentary": 0.7},
        "frustration_sensitivity": 0.6,
        "base_frustration": 0.05
    },
    {
        "name": "Channel Surfer", 
        "narrative": "Low attention span, gets frustrated if browsing becomes tedious.",
        "ocean": {'openness': 0.8, 'conscientiousness': 0.1, 'extraversion': 0.8, 'agreeableness': 0.3, 'neuroticism': 0.7},
        "preferences": {"Action": 0.7, "Comedy": 0.7, "Thriller": 0.6},
        "frustration_sensitivity": 0.7,
        "base_frustration": 0.1
    },
    {
        "name": "Nostalgic Viewer", 
        "narrative": "Looking for familiar content, gets frustrated when can't find old favorites.",
        "ocean": {'openness': 0.1, 'conscientiousness': 0.6, 'extraversion': 0.4, 'agreeableness': 0.7, 'neuroticism': 0.5},
        "preferences": {"Comedy": 0.8, "Action": 0.7, "TV Show": 0.9},
        "frustration_sensitivity": 0.4,
        "base_frustration": 0.03
    }
]

def load_and_preprocess_content(path: str) -> pd.DataFrame:
    """Loads and cleans the TMDb movie data."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"FATAL ERROR: The content catalog file was not found at '{path}'.")
        print("Please download 'tmdb_5000_movies.csv' from Kaggle and place it in the same directory.")
        exit()
        
    df = df[df['genres'].notna() & (df['genres'] != '[]')]
    df['genres'] = df['genres'].apply(lambda x: [g['name'] for g in ast.literal_eval(x)])
    df.rename(columns={'id': 'item_id', 'original_title': 'title'}, inplace=True)
    return df[['item_id', 'title', 'genres']]

def enhance_agent_frustration(agent, persona, action_type, consecutive_count, time_since_last):
    """Enhanced frustration modeling based on persona and behavior patterns."""
    current_frustration = agent.state.get('frustration_level', 0.0)
    frustration_increase = 0.0
    
    # Base frustration sensitivity from persona
    sensitivity = persona.get('frustration_sensitivity', 0.3)
    
    # Frustration triggers based on behavior patterns
    if consecutive_count >= 3:
        # Repeated actions increase frustration
        if action_type in ['dpad_right', 'dpad_down', 'dpad_left', 'dpad_up']:
            frustration_increase += 0.05 * sensitivity * (consecutive_count - 2)
        elif action_type == 'click':
            frustration_increase += 0.08 * sensitivity * (consecutive_count - 2)
        elif action_type == 'back':
            frustration_increase += 0.03 * sensitivity * (consecutive_count - 2)
    
    # Long time between actions (indecision)
    if time_since_last > 5.0:
        frustration_increase += 0.02 * sensitivity
    
    # Very quick actions (frantic behavior)
    if time_since_last < 0.8:
        frustration_increase += 0.01 * sensitivity
    
    # Add some random frustration events
    if random.random() < 0.05 * sensitivity:  # 5% chance for high-sensitivity personas
        frustration_increase += random.uniform(0.02, 0.08) * sensitivity
    
    # Update agent's frustration
    new_frustration = min(1.0, current_frustration + frustration_increase)
    agent.state['frustration_level'] = new_frustration
    
    # Frustration decay over time (very slow)
    if frustration_increase == 0 and random.random() < 0.1:
        agent.state['frustration_level'] = max(0.0, current_frustration * 0.98)

def main():
    # --- Corrected Log Name Mechanism for parallel runs ---
    if len(sys.argv) > 1:
        worker_id = sys.argv[1]
        output_file_name = f"{OUTPUT_FILE_PREFIX}_{worker_id}.csv"
    else:
        output_file_name = f"{OUTPUT_FILE_PREFIX}_enhanced.csv"

    print("Initializing enhanced simulation with better frustration modeling...")
    content_catalog = load_and_preprocess_content(TMDB_DATA_PATH)
    all_events = []
    
    # Weight personas towards more frustrating ones for better data balance
    persona_weights = []
    for persona in PERSONAS:
        # Higher weight for personas with higher frustration sensitivity
        weight = 1.0 + persona.get('frustration_sensitivity', 0.3) * 2
        persona_weights.append(weight)
    
    for session_id_int in tqdm(range(NUM_SESSIONS_TO_SIMULATE), desc="Simulating Enhanced Sessions"):
        session_id = f"session_{session_id_int}"
        
        # Choose persona with weighted selection (favor more frustrating personas)
        persona = random.choices(PERSONAS, weights=persona_weights)[0]
        user_id = f"user_{persona['name'].replace(' ', '_').lower()}_{session_id_int % 5}"
        
        agent = LLMAgent(user_id=user_id, persona=persona)
        
        # Initialize agent with persona's base frustration
        agent.state['frustration_level'] = persona.get('base_frustration', 0.0)
        
        env = FireTVEnvironment(content_df=content_catalog)
        
        obs, _, done, _ = env.reset()
        
        last_action_timestamp = datetime.now()
        consecutive_action_count_map = {}
        last_logged_action_type = "session_start"

        log_entry = {
            "timestamp": last_action_timestamp.isoformat(), 
            "session_id": session_id, 
            "user_id": user_id,
            "action_type": "session_start", 
            "screen_context": "Home", 
            "focused_item": None,
            "derived_states": "NOT_LOGGED_RAW",
            "sequence_context": json.dumps({"time_since_last_action": 0.0, "consecutive_action_count": 0})
        }
        all_events.append(log_entry)
        
        session_step = 0
        max_session_length = random.randint(15, 100)  # Variable session lengths
        
        while not done and session_step < max_session_length:
            session_step += 1
            
            # Enhanced time modeling based on frustration
            base_time = random.uniform(0.5, 4.0)
            frustration_factor = agent.state.get('frustration_level', 0.0)
            
            # High frustration = faster, more erratic actions
            if frustration_factor > 0.5:
                time_delta_seconds = max(0.3, base_time * (0.5 + random.uniform(-0.3, 0.1)))
            else:
                time_delta_seconds = max(0.3, base_time - frustration_factor * 1.5)
                
            current_event_timestamp = last_action_timestamp + timedelta(seconds=time_delta_seconds)
            
            decision = agent.decide_action(obs) 
            action_type = decision.get('action_type', 'dpad_right')

            # --- ENHANCED CIRCUIT BREAKER LOGIC ---
            # More aggressive circuit breaker for high-frustration users
            frustration_threshold = 3 - int(frustration_factor * 2)  # Lower threshold for frustrated users
            if obs.get('consecutive_click_count', 0) >= frustration_threshold:
                print(f"Enhanced Circuit Breaker: Session {session_id_int}, Frustration: {frustration_factor:.3f}. Forcing 'back'.")
                decision = {'action_type': 'back'}
                action_type = 'back'
                # Add extra frustration for hitting the circuit breaker
                agent.state['frustration_level'] = min(1.0, agent.state.get('frustration_level', 0.0) + 0.1)
            # --- END ENHANCED CIRCUIT BREAKER ---

            _agent_update_consecutive_count = (consecutive_action_count_map.get(action_type, 0) + 1) if action_type == last_logged_action_type else 1
            agent_update_sequence_context = {
                "time_since_last_action": round((current_event_timestamp - last_action_timestamp).total_seconds(), 2),
                "consecutive_action_count": _agent_update_consecutive_count
            }

            # ENHANCED: Update frustration based on behavior patterns
            enhance_agent_frustration(
                agent, 
                persona, 
                action_type, 
                _agent_update_consecutive_count,
                agent_update_sequence_context["time_since_last_action"]
            )

            next_obs, _, done, info = env.step(decision)
            
            # Early exit based on high frustration
            if agent.state.get('frustration_level', 0.0) > 0.8 and random.random() < 0.3:
                done = True
                info['llm_decision'] = info.get('llm_decision', {})
                info['llm_decision']['session_end_reason'] = 'user_frustration_exit'
            
            # [REST OF THE LOGGING LOGIC REMAINS THE SAME]
            if 'logged_hover_item' in info['llm_decision'] and info['llm_decision']['logged_hover_item']:
                hover_item = info['llm_decision']['logged_hover_item']
                hover_duration_val = info['llm_decision']['logged_hover_duration']
                
                consecutive_count_hover = (consecutive_action_count_map.get("hover", 0) + 1) if "hover" == last_logged_action_type else 1
                consecutive_action_count_map["hover"] = consecutive_count_hover

                hover_log_entry = {
                    "timestamp": current_event_timestamp.isoformat(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "action_type": "hover",
                    "screen_context": info['current_screen_context'],
                    "focused_item": json.dumps(hover_item),
                    "derived_states": "NOT_LOGGED_RAW",
                    "sequence_context": json.dumps({
                        "time_since_last_action": round((current_event_timestamp - last_action_timestamp).total_seconds(), 2),
                        "consecutive_action_count": consecutive_count_hover
                    }),
                    "hover_duration": hover_duration_val
                }
                all_events.append(hover_log_entry)
                last_action_timestamp = current_event_timestamp
                last_logged_action_type = "hover"
            
            if info['prev_focused_item_for_dpad_log'] and info['last_dpad_key_code']:
                dpad_action_type = info['last_dpad_key_code']
                
                consecutive_count_dpad = (consecutive_action_count_map.get(dpad_action_type, 0) + 1) if dpad_action_type == last_logged_action_type else 1
                consecutive_action_count_map[dpad_action_type] = consecutive_count_dpad

                dpad_log_entry = {
                    "timestamp": current_event_timestamp.isoformat(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "action_type": dpad_action_type,
                    "screen_context": info['current_screen_context'],
                    "focused_item": json.dumps(info['prev_focused_item_for_dpad_log']),
                    "derived_states": "NOT_LOGGED_RAW",
                    "sequence_context": json.dumps({
                        "time_since_last_action": round((current_event_timestamp - last_action_timestamp).total_seconds(), 2),
                        "consecutive_action_count": consecutive_count_dpad
                    })
                }
                all_events.append(dpad_log_entry)
                last_action_timestamp = current_event_timestamp
                last_logged_action_type = dpad_action_type

            main_action_types = ['click', 'back', 'scroll', 'playback_abandon', 'playback_completed']
            if action_type in main_action_types:
                
                consecutive_count_main = (consecutive_action_count_map.get(action_type, 0) + 1) if action_type == last_logged_action_type else 1
                consecutive_action_count_map[action_type] = consecutive_count_main

                main_log_entry = {
                    "timestamp": current_event_timestamp.isoformat(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "action_type": action_type,
                    "screen_context": info['current_screen_context'],
                    "focused_item": json.dumps(info['current_focused_item_data']),
                    "derived_states": "NOT_LOGGED_RAW",
                    "sequence_context": json.dumps({
                        "time_since_last_action": round((current_event_timestamp - last_action_timestamp).total_seconds(), 2),
                        "consecutive_action_count": consecutive_count_main
                    })
                }
                
                if 'click_type' in decision: main_log_entry['click_type'] = decision['click_type']
                if 'scroll_speed' in decision: main_log_entry['scroll_speed'] = decision['scroll_speed']
                if 'scroll_depth' in decision: main_log_entry['scroll_depth'] = decision['scroll_depth']
                
                if 'playback_position' in decision:
                    main_log_entry['playback_position'] = float(decision['playback_position']) / 100.0 if decision['playback_position'] > 1.0 else float(decision['playback_position'])
                
                if 'session_end_reason' in info['llm_decision']:
                    main_log_entry['session_end_reason'] = info['llm_decision']['session_end_reason']
                
                all_events.append(main_log_entry)
                last_action_timestamp = current_event_timestamp
                last_logged_action_type = action_type
            
            agent.update_state(info['action_outcome'], {
                "action_type": action_type,
                "sequence_context": agent_update_sequence_context,
                "hover_duration": decision.get('hover_duration', 0.0),
                "playback_position": decision.get('playback_position', 0.0) / 100.0 if decision.get('playback_position', 0.0) > 1.0 else decision.get('playback_position', 0.0)
            })
            
            obs = next_obs
            env.last_dpad_key_code = None

        if done and 'session_end_reason' not in info.get('llm_decision', {}):
            final_log_time = last_action_timestamp + timedelta(seconds=0.5)
            log_entry_session_end = {
                "timestamp": final_log_time.isoformat(), 
                "session_id": session_id, 
                "user_id": user_id,
                "action_type": "session_end", 
                "screen_context": info['current_screen_context'], 
                "focused_item": json.dumps(obs['focused_item']),
                "derived_states": "NOT_LOGGED_RAW",
                "sequence_context": json.dumps({"time_since_last_action": 0.5, "consecutive_action_count": 0}),
                "session_end_reason": "timeout"
            }
            all_events.append(log_entry_session_end)

    print(f"\nEnhanced simulation complete. Generated {len(all_events)} events.")
    df_logs = pd.DataFrame(all_events)
    df_logs.to_csv(output_file_name, index=False)
    print(f"Enhanced data saved to '{output_file_name}'")
    
    # Quick analysis of the generated data
    print(f"\nQuick Analysis:")
    print(f"Total sessions: {df_logs['session_id'].nunique()}")
    print(f"Personas used: {df_logs['user_id'].apply(lambda x: '_'.join(x.split('_')[1:-1])).nunique()}")
    
    # Estimate frustration distribution (this will be calculated properly in post-processing)
    frustration_exits = df_logs[df_logs['session_end_reason'] == 'user_frustration_exit']['session_id'].nunique()
    print(f"Sessions ending due to frustration: {frustration_exits}")

if __name__ == "__main__":
    main()
