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
NUM_SESSIONS_TO_SIMULATE = 50 # Increase for a larger dataset
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
    
    current_time = datetime.now()
    # To track last action timestamp for sequence_context
    session_start_time = {} 
    last_action_timestamp_per_session = {}

    for session_id_int in tqdm(range(NUM_SESSIONS_TO_SIMULATE), desc="Simulating Sessions"):
        session_id = f"session_{session_id_int}"
        persona = random.choice(PERSONAS)
        user_id = f"user_{persona['name'].replace(' ', '_').lower()}_{session_id_int % 5}"
        
        agent = LLMAgent(user_id=user_id, persona=persona)
        env = FireTVEnvironment(content_df=content_catalog)
        
        obs, _, done, _ = env.reset()
        session_start_time[session_id] = datetime.now()
        last_action_timestamp_per_session[session_id] = session_start_time[session_id]

        consecutive_action_count = {} # Per action type
        last_action_type_per_session = {}
        
        # Log session_start event
        log_entry = {
            "timestamp": session_start_time[session_id].isoformat(), "session_id": session_id, "user_id": user_id,
            "action_type": "session_start", "screen_context": "Home", "focused_item": None,
            "derived_states": json.dumps(agent.state.copy()), "sequence_context": {"time_since_last_action": 0.0, "consecutive_action_count": 0}
        }
        all_events.append(log_entry)
        
        # --- Main Simulation Loop ---
        while not done:
            # Simulate irregular time between actions
            time_delta_seconds = max(0.3, random.uniform(0.5, 4.0) - agent.state['frustration_level'] * 2)
            current_time = last_action_timestamp_per_session[session_id] + timedelta(seconds=time_delta_seconds)
            
            decision = agent.decide_action(obs)
            action_type = decision.get('action_type', 'dpad_right')

            # --- Apply decision to environment to get next state and outcome ---
            next_obs, _, done, info = env.step(decision)
            
            # --- Delayed Logging Logic (Mimicking Android App) ---
            # 1. Log the DPAD event that CAUSED this focus change
            if info['last_dpad_key_code'] and info['action_outcome'] == 'new_content_seen':
                dpad_action_type = info['last_dpad_key_code']
                
                # Update consecutive count for DPAD event
                current_dpad_consecutive_count = (consecutive_action_count.get(dpad_action_type, 0) + 1) if dpad_action_type == last_action_type_per_session.get(session_id) else 1
                consecutive_action_count[dpad_action_type] = current_dpad_consecutive_count

                dpad_log_entry = {
                    "timestamp": current_time.isoformat(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "action_type": dpad_action_type,
                    "screen_context": obs['screen_context'],
                    "focused_item": json.dumps(info['prev_focused_item']), # Item that lost focus (previous item)
                    "derived_states": json.dumps(agent.state.copy()),
                    "sequence_context": json.dumps({
                        "time_since_last_action": round((current_time - last_action_timestamp_per_session[session_id]).total_seconds(), 2),
                        "consecutive_action_count": current_dpad_consecutive_count
                    })
                }
                all_events.append(dpad_log_entry)
                
                # Update last action type and timestamp after logging DPAD
                last_action_timestamp_per_session[session_id] = current_time
                last_action_type_per_session[session_id] = dpad_action_type
                
            # 2. Log the HOVER event for the PREVIOUS item when focus is lost (if it was a movie)
            if action_type != 'hover' and info['prev_focused_item'] and info['prev_focused_item'].get('item_id') and 'hover_duration' in decision:
                hover_log_entry = {
                    "timestamp": current_time.isoformat(), # Same timestamp as the dpad event
                    "session_id": session_id,
                    "user_id": user_id,
                    "action_type": "hover",
                    "screen_context": obs['screen_context'],
                    "focused_item": json.dumps(info['prev_focused_item']), # Item that lost focus
                    "derived_states": json.dumps(agent.state.copy()),
                    "sequence_context": json.dumps({
                        "time_since_last_action": 0.001, # Minimal time difference
                        "consecutive_action_count": (consecutive_action_count.get("hover", 0) + 1) if "hover" == last_action_type_per_session.get(session_id) else 1
                    }),
                    "hover_duration": decision['hover_duration']
                }
                all_events.append(hover_log_entry)
                
                # Update last action type and timestamp for HOVER if logged
                last_action_timestamp_per_session[session_id] = current_time
                last_action_type_per_session[session_id] = "hover"


            # 3. Log other actions (click, back, scroll, etc.) immediately
            if action_type not in ['dpad_right', 'dpad_left', 'dpad_down', 'dpad_up', 'hover']: # These are handled specially
                current_consecutive_count = (consecutive_action_count.get(action_type, 0) + 1) if action_type == last_action_type_per_session.get(session_id) else 1
                consecutive_action_count[action_type] = current_consecutive_count

                main_log_entry = {
                    "timestamp": current_time.isoformat(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "action_type": action_type,
                    "screen_context": obs['screen_context'],
                    "focused_item": json.dumps(obs['focused_item']) if action_type != 'back' else json.dumps(info['prev_focused_item']), # Focused item at action time
                    "derived_states": json.dumps(agent.state.copy()),
                    "sequence_context": json.dumps({
                        "time_since_last_action": round((current_time - last_action_timestamp_per_session[session_id]).total_seconds(), 2),
                        "consecutive_action_count": current_consecutive_count
                    })
                }
                
                # Add optional fields based on the LLM's decision
                for key in ['click_type', 'scroll_speed', 'scroll_depth', 'playback_position', 'session_end_reason']:
                    if key in decision:
                        main_log_entry[key] = decision[key]
                
                all_events.append(main_log_entry)
                
                # Update last action type and timestamp for main log
                last_action_timestamp_per_session[session_id] = current_time
                last_action_type_per_session[session_id] = action_type


            # --- Update agent's state based on environment outcome ---
            agent.update_state(info['action_outcome'], {
                "action_type": action_type, # Pass the actual action for update
                "sequence_context": sequence_context, # Pass sequence context for heuristics
                "hover_duration": decision.get('hover_duration'), # Pass hover duration if applicable
                "playback_position": decision.get('playback_position'), # Pass playback position if applicable
            })
            
            obs = next_obs
            # Clear last dpad key code if a focus change occurred, handled by env.step now
            env.last_dpad_key_code = None

        # --- Session End Logic ---
        # Ensure final state is captured
        final_log_time = last_action_timestamp_per_session[session_id] + timedelta(seconds=0.5)
        log_entry_session_end = {
            "timestamp": final_log_time.isoformat(), "session_id": session_id, "user_id": user_id,
            "action_type": "session_end", "screen_context": obs['screen_context'], "focused_item": json.dumps(obs['focused_item']),
            "derived_states": json.dumps(agent.state.copy()), "sequence_context": {"time_since_last_action": 0.5, "consecutive_action_count": 0},
            "session_end_reason": info['llm_decision'].get('session_end_reason', 'user_abandoned')
        }
        all_events.append(log_entry_session_end)

    print(f"\nSimulation complete. Generated {len(all_events)} events.")
    df_logs = pd.DataFrame(all_events)
    # Ensure nested JSON fields are parsed correctly during CSV write/read
    # df_logs['focused_item'] = df_logs['focused_item'].apply(json.loads)
    # df_logs['derived_states'] = df_logs['derived_states'].apply(json.loads)
    # df_logs['sequence_context'] = df_logs['sequence_context'].apply(json.loads)
    df_logs.to_csv(OUTPUT_FILE, index=False)
    print(f"Data saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
