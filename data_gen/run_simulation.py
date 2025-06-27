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
NUM_SESSIONS_TO_SIMULATE = 500
OUTPUT_FILE_PREFIX = "simulation_logs" 
TMDB_DATA_PATH = "tmdb_5000_movies.csv"

# --- EXPANDED PERSONA DEFINITIONS (20 Personas) ---
# These personas are designed to cover a wide range of psychological profiles,
# motivations, and interaction styles to generate a rich and diverse dataset.
PERSONAS = [
    # Original Set
    {
        "name": "Stressed Professional", "narrative": "Comes home late after a demanding workday, wants to decompress with something familiar and easy, often a comedy or light-hearted action.",
        "ocean": {'openness': 0.2, 'conscientiousness': 0.8, 'extraversion': 0.3, 'agreeableness': 0.4, 'neuroticism': 0.9},
        "preferences": {"Comedy": 0.9, "Action": 0.6, "TV Show": 0.7}
    },
    {
        "name": "Curious Student", "narrative": "Intellectually energized and wants to watch something thought-provoking or learn something new. Open to foreign films and documentaries.",
        "ocean": {'openness': 0.9, 'conscientiousness': 0.4, 'extraversion': 0.6, 'agreeableness': 0.8, 'neuroticism': 0.2},
        "preferences": {"Documentary": 0.8, "Sci-Fi": 0.8, "Drama": 0.7, "Mystery": 0.6, "Foreign": 0.5}
    },
    {
        "name": "Family Movie Night Organizer", "narrative": "Trying to find a single movie that the whole family, including young children, can agree on. High agreeableness and conscientiousness.",
        "ocean": {'openness': 0.5, 'conscientiousness': 0.9, 'extraversion': 0.7, 'agreeableness': 0.9, 'neuroticism': 0.5},
        "preferences": {"Animation": 0.9, "Family": 0.9, "Adventure": 0.8, "Comedy": 0.7}
    },
    {
        "name": "Heartbroken Romantic", "narrative": "Going through a recent breakup, looking for content to either match their sad mood (dramas) or escape it (comedies). High neuroticism.",
        "ocean": {'openness': 0.6, 'conscientiousness': 0.3, 'extraversion': 0.2, 'agreeableness': 0.5, 'neuroticism': 0.9},
        "preferences": {"Drama": 0.9, "Romance": 0.8, "Comedy": 0.7}
    },
    {
        "name": "Adrenaline Junkie", "narrative": "It's Friday night, looking for pure, high-octane visual excitement. Not interested in complex plots. Wants action, horror, or thrillers.",
        "ocean": {'openness': 0.3, 'conscientiousness': 0.2, 'extraversion': 0.9, 'agreeableness': 0.2, 'neuroticism': 0.4},
        "preferences": {"Action": 0.9, "Thriller": 0.8, "Horror": 0.7, "War": 0.6}
    },
    # New Diverse Set
    {
        "name": "The Completionist", "narrative": "Views their watchlist as a to-do list. Feels compelled to finish every series they start, even if enjoyment wanes. Highly conscientious.",
        "ocean": {'openness': 0.4, 'conscientiousness': 0.9, 'extraversion': 0.4, 'agreeableness': 0.5, 'neuroticism': 0.3},
        "preferences": {"TV Show": 1.0, "Drama": 0.7, "Mystery": 0.6}
    },
    {
        "name": "The Channel Surfer", "narrative": "Has low commitment and a short attention span. Flips through many options, watches trailers, and often abandons content early. Goal is to 'see what's on'.",
        "ocean": {'openness': 0.8, 'conscientiousness': 0.1, 'extraversion': 0.8, 'agreeableness': 0.3, 'neuroticism': 0.6},
        "preferences": {"Action": 0.7, "Comedy": 0.7, "Thriller": 0.6}
    },
    {
        "name": "The Film Buff", "narrative": "A cinephile with strong, specific tastes. Looks for critically acclaimed, often non-mainstream films. Dislikes blockbusters. High openness, low agreeableness.",
        "ocean": {'openness': 0.9, 'conscientiousness': 0.7, 'extraversion': 0.3, 'agreeableness': 0.1, 'neuroticism': 0.2},
        "preferences": {"Drama": 0.9, "Foreign": 0.9, "Independent": 0.8, "Documentary": 0.7}
    },
    {
        "name": "The Background Viewer", "narrative": "Puts on content while cooking, cleaning, or working. Prefers familiar, low-cognitive-load shows like sitcoms that don't require full attention.",
        "ocean": {'openness': 0.2, 'conscientiousness': 0.5, 'extraversion': 0.5, 'agreeableness': 0.6, 'neuroticism': 0.4},
        "preferences": {"Comedy": 0.9, "TV Show": 0.9, "Family": 0.7}
    },
    {
        "name": "The Niche Enthusiast", "narrative": "Has a very specific interest (e.g., 1980s horror or historical war documentaries) and will ignore everything else to find content in their niche.",
        "ocean": {'openness': 0.7, 'conscientiousness': 0.6, 'extraversion': 0.4, 'agreeableness': 0.4, 'neuroticism': 0.3},
        "preferences": {"Horror": 0.9, "History": 0.9, "War": 0.8}
    },
    {
        "name": "The Tech-Averse User", "narrative": "Finds the UI slightly confusing. Navigates slowly, makes mistakes, and gets frustrated easily. Prefers simple, direct paths to content.",
        "ocean": {'openness': 0.3, 'conscientiousness': 0.4, 'extraversion': 0.4, 'agreeableness': 0.7, 'neuroticism': 0.8},
        "preferences": {"Drama": 0.7, "Family": 0.7, "Romance": 0.6}
    },
    {
        "name": "The Social Sharer", "narrative": "Watches content that is currently trending or being talked about on social media to stay in the loop and have things to discuss with friends.",
        "ocean": {'openness': 0.6, 'conscientiousness': 0.5, 'extraversion': 0.8, 'agreeableness': 0.8, 'neuroticism': 0.4},
        "preferences": {"Action": 0.8, "Comedy": 0.8, "Sci-Fi": 0.7}
    },
    {
        "name": "The Escapist", "narrative": "Uses TV to escape from reality. Prefers immersive fantasy, grand sci-fi, or far-flung adventures. Avoids realistic dramas or news.",
        "ocean": {'openness': 0.8, 'conscientiousness': 0.3, 'extraversion': 0.6, 'agreeableness': 0.5, 'neuroticism': 0.7},
        "preferences": {"Fantasy": 1.0, "Sci-Fi": 0.9, "Adventure": 0.8}
    },
    {
        "name": "The Nostalgia Seeker", "narrative": "Rewatches old favorites from their youth. Finds comfort in the familiar and is unlikely to try new, modern shows.",
        "ocean": {'openness': 0.1, 'conscientiousness': 0.6, 'extraversion': 0.4, 'agreeableness': 0.7, 'neuroticism': 0.5},
        "preferences": {"Comedy": 0.8, "Action": 0.7, "TV Show": 0.9} # Assuming old TV shows
    },
    {
        "name": "The Critic", "narrative": "Watches content with a critical eye, often looking for plot holes or analyzing cinematography. Highly analytical and not easily pleased.",
        "ocean": {'openness': 0.7, 'conscientiousness': 0.8, 'extraversion': 0.2, 'agreeableness': 0.2, 'neuroticism': 0.3},
        "preferences": {"Drama": 0.9, "Mystery": 0.8, "Thriller": 0.7}
    },
    {
        "name": "The Weekend Binger", "narrative": "Saves up a series to watch all at once over the weekend. Looks for shows with many seasons or long episodes. Highly patient.",
        "ocean": {'openness': 0.5, 'conscientiousness': 0.7, 'extraversion': 0.5, 'agreeableness': 0.6, 'neuroticism': 0.3},
        "preferences": {"TV Show": 1.0, "Drama": 0.8, "Crime": 0.7}
    },
    {
        "name": "The Reluctant Partner", "narrative": "Is watching with their partner and is compromising on the choice. Not fully engaged and may browse on their phone. Low agreeableness but situationally compliant.",
        "ocean": {'openness': 0.4, 'conscientiousness': 0.4, 'extraversion': 0.3, 'agreeableness': 0.3, 'neuroticism': 0.6},
        "preferences": {"Romance": 0.7, "Comedy": 0.7, "Drama": 0.6} # Partner's choices
    },
    {
        "name": "The Language Learner", "narrative": "Watches foreign films or shows specifically to practice a new language. Focuses on content from a specific country.",
        "ocean": {'openness': 0.9, 'conscientiousness': 0.8, 'extraversion': 0.5, 'agreeableness': 0.7, 'neuroticism': 0.2},
        "preferences": {"Foreign": 1.0, "Drama": 0.8, "Comedy": 0.7}
    },
    {
        "name": "The Documentary Devotee", "narrative": "Almost exclusively watches documentaries to learn about the real world. Finds fiction uninteresting.",
        "ocean": {'openness': 0.8, 'conscientiousness': 0.7, 'extraversion': 0.3, 'agreeableness': 0.5, 'neuroticism': 0.2},
        "preferences": {"Documentary": 1.0, "History": 0.8}
    },
    {
        "name": "The Zen Viewer", "narrative": "Calm, patient, and methodical. Never gets frustrated with the UI. Explores content at a leisurely pace. Very low neuroticism.",
        "ocean": {'openness': 0.6, 'conscientiousness': 0.6, 'extraversion': 0.5, 'agreeableness': 0.8, 'neuroticism': 0.1},
        "preferences": {"Drama": 0.7, "Family": 0.7, "Adventure": 0.6}
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
    # --- Corrected Log Name Mechanism for parallel runs ---
    if len(sys.argv) > 1:
        worker_id = sys.argv[1]
        output_file_name = f"{OUTPUT_FILE_PREFIX}_{worker_id}.csv"
    else:
        output_file_name = f"{OUTPUT_FILE_PREFIX}_default.csv"

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
        
        last_action_timestamp = datetime.now()
        consecutive_action_count_map = {}
        last_logged_action_type = "session_start"

        log_entry = {
            "timestamp": last_action_timestamp.isoformat(), "session_id": session_id, "user_id": user_id,
            "action_type": "session_start", "screen_context": "Home", "focused_item": None,
            "derived_states": "NOT_LOGGED_RAW",
            "sequence_context": json.dumps({"time_since_last_action": 0.0, "consecutive_action_count": 0})
        }
        all_events.append(log_entry)
        
        while not done:
            time_delta_seconds = max(0.3, random.uniform(0.5, 4.0) - agent.state['frustration_level'] * 2)
            current_event_timestamp = last_action_timestamp + timedelta(seconds=time_delta_seconds)
            
            decision = agent.decide_action(obs) 
            action_type = decision.get('action_type', 'dpad_right')

            # --- CIRCUIT BREAKER LOGIC ---
            # If the environment reports too many consecutive clicks, override the LLM's decision
            if obs.get('consecutive_click_count', 0) >= 3:
                print(f"Circuit Breaker Triggered: Worker {sys.argv[1] if len(sys.argv) > 1 else 'default'}, Session {session_id_int}. Forcing 'back' action.")
                decision = {'action_type': 'back'}
                action_type = 'back'
            # --- END CIRCUIT BREAKER ---

            _agent_update_consecutive_count = (consecutive_action_count_map.get(action_type, 0) + 1) if action_type == last_logged_action_type else 1
            agent_update_sequence_context = {
                "time_since_last_action": round((current_event_timestamp - last_action_timestamp).total_seconds(), 2),
                "consecutive_action_count": _agent_update_consecutive_count
            }

            next_obs, _, done, info = env.step(decision)
            
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

        if done and 'session_end_reason' not in info['llm_decision']:
            final_log_time = last_action_timestamp + timedelta(seconds=0.5)
            log_entry_session_end = {
                "timestamp": final_log_time.isoformat(), "session_id": session_id, "user_id": user_id,
                "action_type": "session_end", "screen_context": info['current_screen_context'], "focused_item": json.dumps(obs['focused_item']),
                "derived_states": "NOT_LOGGED_RAW",
                "sequence_context": json.dumps({"time_since_last_action": 0.5, "consecutive_action_count": 0}),
                "session_end_reason": "timeout"
            }
            all_events.append(log_entry_session_end)


    print(f"\nSimulation complete. Generated {len(all_events)} events.")
    df_logs = pd.DataFrame(all_events)
    df_logs.to_csv(output_file_name, index=False)
    print(f"Data saved to '{output_file_name}'")

if __name__ == "__main__":
    main()
