# run_simulation.py
import pandas as pd
import random
import ast
from datetime import datetime, timedelta
from tqdm import tqdm
import json

from llm_agent import LLMAgent
from firetv_environment import FireTVEnvironment

# --- Simulation Configuration ---
NUM_SESSIONS_TO_SIMULATE = 5 # Increase for a larger dataset
OUTPUT_FILE = "final_simulation_logs.csv"
TMDB_DATA_PATH = "tmdb_5000_movies.csv"

# --- Persona Definitions (as finalized) ---
PERSONAS = [
    {
        "name": "Stressed Professional", "narrative": "Comes home late after a demanding workday, wants to decompress with something familiar and easy.",
        "ocean": {'openness': 0.2, 'conscientiousness': 0.8, 'extraversion': 0.3, 'agreeableness': 0.4, 'neuroticism': 0.9},
        "preferences": {"Comedy": 0.8, "Action": 0.6, "Family": 0.5}
    },
    {
        "name": "Curious Student", "narrative": "Just finished exams, intellectually energized and wants to watch something that makes them think.",
        "ocean": {'openness': 0.9, 'conscientiousness': 0.4, 'extraversion': 0.6, 'agreeableness': 0.8, 'neuroticism': 0.2},
        "preferences": {"Sci-Fi": 0.8, "Drama": 0.7, "Documentary": 0.6, "Mystery": 0.5}
    },
    {
        "name": "Family Movie Night Organizer", "narrative": "Trying to find a single movie that the whole family (including young children) can agree on.",
        "ocean": {'openness': 0.5, 'conscientiousness': 0.9, 'extraversion': 0.7, 'agreeableness': 0.9, 'neuroticism': 0.5},
        "preferences": {"Animation": 0.9, "Family": 0.9, "Adventure": 0.7, "Comedy": 0.6}
    },
    {
        "name": "Heartbroken Romantic", "narrative": "Going through a recent breakup, looking for content to either match or escape their emotional state.",
        "ocean": {'openness': 0.6, 'conscientiousness': 0.3, 'extraversion': 0.2, 'agreeableness': 0.5, 'neuroticism': 0.9},
        "preferences": {"Drama": 0.8, "Romance": 0.7, "Comedy": 0.6}
    },
    {
        "name": "Adrenaline Junkie", "narrative": "It's Friday night, looking for pure, high-octane visual excitement. Not interested in plot.",
        "ocean": {'openness': 0.3, 'conscientiousness': 0.2, 'extraversion': 0.9, 'agreeableness': 0.2, 'neuroticism': 0.4},
        "preferences": {"Action": 0.9, "Thriller": 0.8, "Horror": 0.7, "War": 0.6}
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

def main():
    print("Initializing simulation...")
    content_catalog = load_and_preprocess_content(TMDB_DATA_PATH)
    all_events = []
    
    for session_id_int in tqdm(range(NUM_SESSIONS_TO_SIMULATE), desc="Simulating Sessions"):
        session_id = f"session_{session_id_int}"
        persona = random.choice(PERSONAS)
        user_id = f"user_{persona['name'].replace(' ', '_').lower()}_{session_id_int % 5}"
        
        agent = LLMAgent(user_id=user_id, persona=persona)
        env = FireTVEnvironment(content_df=content_catalog)
        
        obs, _, done, _ = env.reset()
        
        # --- Internal state for sequence and time tracking ---
        last_action_timestamp = datetime.now()
        consecutive_action_count_map = {} # Maps action_type to its consecutive count
        last_logged_action_type = "session_start"

        # --- Log session_start event ---
        log_entry = {
            "timestamp": last_action_timestamp.isoformat(), "session_id": session_id, "user_id": user_id,
            "action_type": "session_start", "screen_context": "Home", "focused_item": None,
            "derived_states": json.dumps(agent.state.copy()), "sequence_context": {"time_since_last_action": 0.0, "consecutive_action_count": 0}
        }
        all_events.append(log_entry)
        
        # --- Main Simulation Loop ---
        while not done:
            # Simulate irregular time between actions
            time_delta_seconds = max(0.3, random.uniform(0.5, 4.0) - agent.state['frustration_level'] * 2)
            current_event_timestamp = last_action_timestamp + timedelta(seconds=time_delta_seconds)
            
            decision = agent.decide_action(obs)
            
            # --- Take a step in the environment, getting all the info for precise logging ---
            next_obs, _, done, info = env.step(decision)
            
            # --- LOGGING STRATEGY: MIMICKING ANDROID UI ---
            
            # 1. Log the HOVER event first, if focus changed and a previous item was hovered
            if 'logged_hover_item' in info['llm_decision'] and info['llm_decision']['logged_hover_item']:
                hover_item = info['llm_decision']['logged_hover_item']
                hover_duration_val = info['llm_decision']['hover_duration']
                
                consecutive_count_hover = (consecutive_action_count_map.get("hover", 0) + 1) if "hover" == last_logged_action_type else 1
                consecutive_action_count_map["hover"] = consecutive_count_hover

                hover_log_entry = {
                    "timestamp": current_event_timestamp.isoformat(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "action_type": "hover",
                    "screen_context": info['current_screen_context'], # Screen where hover happened
                    "focused_item": json.dumps(hover_item), # The item that was hovered over
                    "derived_states": json.dumps(agent.state.copy()),
                    "sequence_context": json.dumps({
                        "time_since_last_action": round((current_event_timestamp - last_action_timestamp).total_seconds(), 2),
                        "consecutive_action_count": consecutive_count_hover
                    }),
                    "hover_duration": hover_duration_val
                }
                all_events.append(hover_log_entry)
                last_action_timestamp = current_event_timestamp
                last_logged_action_type = "hover"
            
            # 2. Log DPAD event if a dpad key caused focus change
            if info['prev_focused_item_for_dpad_log'] and info['last_dpad_key_code'] :
                dpad_action_type = info['last_dpad_key_code']
                
                consecutive_count_dpad = (consecutive_action_count_map.get(dpad_action_type, 0) + 1) if dpad_action_type == last_logged_action_type else 1
                consecutive_action_count_map[dpad_action_type] = consecutive_count_dpad

                dpad_log_entry = {
                    "timestamp": current_event_timestamp.isoformat(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "action_type": dpad_action_type,
                    "screen_context": info['current_screen_context'],
                    "focused_item": json.dumps(info['prev_focused_item_for_dpad_log']), # Item the dpad moved FROM
                    "derived_states": json.dumps(agent.state.copy()),
                    "sequence_context": json.dumps({
                        "time_since_last_action": round((current_event_timestamp - last_action_timestamp).total_seconds(), 2),
                        "consecutive_action_count": consecutive_count_dpad
                    })
                }
                all_events.append(dpad_log_entry)
                last_action_timestamp = current_event_timestamp
                last_logged_action_type = dpad_action_type

            # 3. Log Main Action (Click, Back, Scroll, Playback events)
            # Exclude dpad and hover, which were handled above by their specific triggers
            main_action_types = ['click', 'back', 'scroll', 'playback_abandon', 'playback_completed']
            if decision['action_type'] in main_action_types:
                
                consecutive_count_main = (consecutive_action_count_map.get(decision['action_type'], 0) + 1) if decision['action_type'] == last_logged_action_type else 1
                consecutive_action_count_map[decision['action_type']] = consecutive_count_main

                main_log_entry = {
                    "timestamp": current_event_timestamp.isoformat(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "action_type": decision['action_type'],
                    "screen_context": info['current_screen_context'],
                    "focused_item": json.dumps(info['current_focused_item_data']), # Item in focus AT THE TIME of this action
                    "derived_states": json.dumps(agent.state.copy()),
                    "sequence_context": json.dumps({
                        "time_since_last_action": round((current_event_timestamp - last_action_timestamp).total_seconds(), 2),
                        "consecutive_action_count": consecutive_count_main
                    })
                }
                
                # Add optional fields
                if 'click_type' in decision: main_log_entry['click_type'] = decision['click_type']
                if 'scroll_speed' in decision: main_log_entry['scroll_speed'] = decision['scroll_speed']
                if 'scroll_depth' in decision: main_log_entry['scroll_depth'] = decision['scroll_depth']
                
                # Playback position needs normalization
                if 'playback_position' in decision:
                    # Assuming LLM returns 0-100% or similar; normalize to 0-1 float
                    main_log_entry['playback_position'] = float(decision['playback_position']) / 100.0 if decision['playback_position'] > 1.0 else float(decision['playback_position'])
                
                # Session end reason (if session ends due to this action)
                if 'session_end_reason' in info['llm_decision']:
                    main_log_entry['session_end_reason'] = info['llm_decision']['session_end_reason']
                
                all_events.append(main_log_entry)
                last_action_timestamp = current_event_timestamp
                last_logged_action_type = decision['action_type']
            
            # --- Update agent's internal state ---
            agent.update_state(info['action_outcome'], {
                "action_type": decision['action_type'],
                "sequence_context": sequence_context, # Original sequence context for the main decision
                "hover_duration": decision.get('hover_duration', 0.0),
                "playback_position": decision.get('playback_position', 0.0) / 100.0 if decision.get('playback_position', 0.0) > 1.0 else decision.get('playback_position', 0.0)
            })
            
            obs = next_obs # Update observation for next loop iteration

        # --- Session End Logic ---
        # Ensure final session_end event is logged (if not already logged by an action)
        if not (done and 'session_end_reason' in info['llm_decision']): # Only log if not ended by click/back
            final_log_time = last_action_timestamp + timedelta(seconds=0.5)
            log_entry_session_end = {
                "timestamp": final_log_time.isoformat(), "session_id": session_id, "user_id": user_id,
                "action_type": "session_end", "screen_context": info['current_screen_context'], "focused_item": json.dumps(obs['focused_item']),
                "derived_states": json.dumps(agent.state.copy()), "sequence_context": {"time_since_last_action": 0.5, "consecutive_action_count": 0},
                "session_end_reason": info['llm_decision'].get('session_end_reason', 'timeout') # Default to timeout
            }
            all_events.append(log_entry_session_end)

    print(f"\nSimulation complete. Generated {len(all_events)} events.")
    df_logs = pd.DataFrame(all_events)
    # Ensure nested JSON fields are parsed correctly after CSV write/read for analysis
    # For actual analysis, you might parse these columns:
    # df_logs['focused_item'] = df_logs['focused_item'].apply(json.loads)
    # df_logs['derived_states'] = df_logs['derived_states'].apply(json.loads)
    # df_logs['sequence_context'] = df_logs['sequence_context'].apply(json.loads)
    df_logs.to_csv(OUTPUT_FILE, index=False)
    print(f"Data saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
