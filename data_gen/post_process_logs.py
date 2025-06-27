import pandas as pd
import glob
import json
from tqdm import tqdm

# --- Configuration & Enhanced Heuristics ---
RAW_LOGS_PATH_PATTERN = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\Neo_Shosyn\Shosyn-1.0\dataset\enriched_simulation_logs_500.csv"
ENRICHED_OUTPUT_FILE = "enriched_simulation_logs_500_new.csv"

# ENHANCED: More aggressive and realistic frustration heuristics
FRUSTRATION_INCREASE_NO_CHANGE = 0.08  # Reduced but more frequent
FRUSTRATION_INCREASE_PLAYBACK_ABANDON = 0.25
FRUSTRATION_INCREASE_REPEATED_ACTIONS = 0.05  # NEW: For any repeated action
FRUSTRATION_INCREASE_LONG_HESITATION = 0.03   # NEW: For long pauses
FRUSTRATION_INCREASE_RAPID_ACTIONS = 0.02     # NEW: For frantic behavior
FRUSTRATION_BASE_ACCUMULATION = 0.01          # NEW: Base frustration over time

COGNITIVE_LOAD_INCREASE_NEW_CONTENT = 0.1
COGNITIVE_LOAD_INCREASE_HESITATION = 0.05
HESITATION_THRESHOLD_SECONDS = 4.0
RAPID_ACTION_THRESHOLD_SECONDS = 0.8          # NEW: For detecting frantic behavior
SCROLL_SEGMENT_MIN_LENGTH = 3

# The PERSONAS list MUST be identical to the one in run_simulation.py
# This is needed to look up the 'neuroticism' trait for each user.
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
# Create a lookup dictionary for personas
PERSONA_LOOKUP = {persona['name'].replace(' ', '_').lower(): persona for persona in PERSONAS}

def safe_json_load(x):
    """
    Ultra-robust JSON parsing that handles apostrophes, single quotes, and malformed strings.
    """
    if pd.isna(x) or x == '':
        return {}
    
    if not isinstance(x, str):
        return {}
    
    try:
        # First, try standard json.loads
        return json.loads(x)
    except json.JSONDecodeError:
        try:
            # Second, try ast.literal_eval (handles Python dict strings)
            return ast.literal_eval(x)
        except Exception:
            try:
                # Third, try fixing quotes by replacing outer single quotes with double quotes
                # but preserving inner apostrophes
                if x.startswith("'") and x.endswith("'"):
                    # This is a string wrapped in single quotes
                    # We need to be more surgical about quote replacement
                    fixed_str = x
                    
                    # Replace the outer single quotes with double quotes
                    fixed_str = '"' + fixed_str[1:-1] + '"'
                    
                    # Now try to parse it
                    return json.loads(fixed_str)
                else:
                    # Try a more aggressive approach for dict-like strings
                    # Replace single quotes that are likely dict delimiters
                    import re
                    
                    # Pattern to match single quotes that are dict/list delimiters (not apostrophes)
                    # This is a heuristic: replace single quotes that are followed by colons or commas
                    # or are at the start/end of the string
                    fixed_str = re.sub(r"(?<![a-zA-Z])'(?=\s*[:\[\],}])", '"', x)  # After non-letter
                    fixed_str = re.sub(r"(?<=[:\[\],{\s])\s*'(?![a-zA-Z]*'[a-zA-Z])", '"', fixed_str)  # After delimiters
                    fixed_str = re.sub(r"^'", '"', fixed_str)  # Start of string
                    fixed_str = re.sub(r"'$", '"', fixed_str)  # End of string
                    
                    return json.loads(fixed_str)
                    
            except Exception:
                # Final fallback: manual parsing for simple dict structures
                try:
                    # If it looks like a simple dict, try manual extraction
                    if x.strip().startswith('{') and x.strip().endswith('}'):
                        # Very basic manual parsing for our specific use case
                        # Extract item_id, title, and genres
                        import re
                        
                        item_id_match = re.search(r"'item_id':\s*'([^']*)'", x)
                        title_match = re.search(r"'title':\s*\"([^\"]*)\"|'title':\s*'([^']*)'", x)
                        genres_match = re.search(r"'genres':\s*(\[[^\]]*\])", x)
                        
                        result = {}
                        if item_id_match:
                            result['item_id'] = item_id_match.group(1)
                        if title_match:
                            result['title'] = title_match.group(1) or title_match.group(2)
                        if genres_match:
                            try:
                                # Try to parse the genres list
                                genres_str = genres_match.group(1).replace("'", '"')
                                result['genres'] = json.loads(genres_str)
                            except:
                                result['genres'] = []
                        
                        return result if result else {}
                    
                    return {}
                except Exception:
                    print(f"Warning: Could not parse JSON/dict string: {x[:100]}...")
                    return {}

def load_and_combine_logs(pattern):
    """Finds all raw log files matching the pattern and combines them."""
    print(f"Searching for raw logs with pattern: {pattern}")
    log_files = glob.glob(pattern)
    if not log_files:
        print("No log files found. Please run the simulation first.")
        return None
    
    print(f"Found {len(log_files)} log files to process.")
    df_list = [pd.read_csv(f) for f in log_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Basic cleaning and type conversion
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df.sort_values(by=['session_id', 'timestamp'], inplace=True)
    
    # FIXED: Use robust JSON parsing instead of direct json.loads
    print("Parsing JSON fields...")
    combined_df['focused_item'] = combined_df['focused_item'].apply(safe_json_load)
    combined_df['sequence_context'] = combined_df['sequence_context'].apply(safe_json_load)
    
    return combined_df

def process_session(session_df):
    """Processes a single session to add derived attributes with ENHANCED frustration modeling."""
    
    # Initialize lists to store derived values
    frustration_levels = []
    cognitive_loads = []
    
    # ENHANCED: More realistic starting frustration based on persona
    user_id_parts = session_df['user_id'].iloc[0].split('_')
    persona_name = "_".join(user_id_parts[1:-1])
    persona_details = PERSONA_LOOKUP.get(persona_name, {'ocean': {'neuroticism': 0.5}})
    neuroticism_multiplier = 1 + persona_details['ocean']['neuroticism']
    
    # Start with base frustration based on neuroticism
    current_frustration = 0.02 * persona_details['ocean']['neuroticism']
    current_cognitive_load = 0.1
    
    # Track session patterns for enhanced frustration calculation
    session_length = len(session_df)
    total_time_elapsed = 0
    
    # --- Identify Scroll Segments First ---
    session_df = session_df.copy()  # Avoid SettingWithCopyWarning
    session_df['scroll_segment_id'] = 0
    session_df['is_scroll_event'] = False
    
    dpad_actions = ['dpad_up', 'dpad_down', 'dpad_left', 'dpad_right']
    segment_id_counter = 1
    i = 0
    while i < len(session_df):
        action = session_df.iloc[i]['action_type']
        if action in dpad_actions:
            j = i
            while j + 1 < len(session_df) and session_df.iloc[j + 1]['action_type'] == action:
                j += 1
            
            if (j - i + 1) >= SCROLL_SEGMENT_MIN_LENGTH:
                # Found a scroll segment
                for k in range(i, j + 1):
                    session_df.iloc[k, session_df.columns.get_loc('is_scroll_event')] = True
                    session_df.iloc[k, session_df.columns.get_loc('scroll_segment_id')] = segment_id_counter
                segment_id_counter += 1
                i = j
        i += 1

    # --- ENHANCED: Iterate through events with better frustration modeling ---
    for idx, (index, row) in enumerate(session_df.iterrows()):
        seq_context = row['sequence_context']
        time_since_last = seq_context.get('time_since_last_action', 0)
        consecutive_count = seq_context.get('consecutive_action_count', 1)
        total_time_elapsed += time_since_last
        
        # Infer action_outcome
        action_outcome = 'no_change'
        if row['action_type'] in dpad_actions and idx > 0:
            prev_row = session_df.iloc[idx-1]
            if row['focused_item'].get('item_id') != prev_row['focused_item'].get('item_id'):
                action_outcome = 'new_content_seen'
        elif row['action_type'] == 'click':
            action_outcome = 'decision_made'

        # --- ENHANCED FRUSTRATION HEURISTICS ---
        frustration_increase = 0
        
        # 1. Base frustration accumulation over time
        frustration_increase += FRUSTRATION_BASE_ACCUMULATION * neuroticism_multiplier
        
        # 2. Repeated actions
        if consecutive_count >= 2:
            frustration_increase += FRUSTRATION_INCREASE_REPEATED_ACTIONS * consecutive_count * neuroticism_multiplier
        
        # 3. Long hesitation (indecision)
        if time_since_last > HESITATION_THRESHOLD_SECONDS:
            frustration_increase += FRUSTRATION_INCREASE_LONG_HESITATION * (time_since_last / HESITATION_THRESHOLD_SECONDS) * neuroticism_multiplier
        
        # 4. Rapid/frantic actions
        if time_since_last < RAPID_ACTION_THRESHOLD_SECONDS and idx > 0:
            frustration_increase += FRUSTRATION_INCREASE_RAPID_ACTIONS * neuroticism_multiplier
        
        # 5. Original heuristics (enhanced)
        if action_outcome == 'no_change' and consecutive_count >= 2:
            frustration_increase += FRUSTRATION_INCREASE_NO_CHANGE * neuroticism_multiplier
        elif row['action_type'] == 'playback_abandon' and row.get('playback_position', 1.0) < 0.15:
            frustration_increase += FRUSTRATION_INCREASE_PLAYBACK_ABANDON * neuroticism_multiplier
        
        # 6. Long sessions naturally increase frustration
        if session_length > 50:
            frustration_increase += 0.005 * neuroticism_multiplier
        
        # 7. Back actions can indicate frustration
        if row['action_type'] == 'back' and consecutive_count >= 2:
            frustration_increase += 0.03 * neuroticism_multiplier
        
        # Apply frustration increase
        current_frustration = min(1.0, current_frustration + frustration_increase)
        
        # Frustration decay
        if row['action_type'] == 'click' and row.get('click_type') == 'play':
            current_frustration *= 0.6
        elif action_outcome == 'decision_made':
            current_frustration *= 0.8
        else:
            current_frustration = max(0.0, current_frustration - 0.005)
            
        frustration_levels.append(round(current_frustration, 4))
        
        # --- ENHANCED COGNITIVE LOAD HEURISTICS ---
        load_increase = 0
        if time_since_last > HESITATION_THRESHOLD_SECONDS:
            load_increase = COGNITIVE_LOAD_INCREASE_HESITATION * (time_since_last / HESITATION_THRESHOLD_SECONDS)
        elif row['action_type'] == 'hover':
            load_increase = 0.05 * row.get('hover_duration', 0)
        elif action_outcome == 'new_content_seen':
            load_increase = COGNITIVE_LOAD_INCREASE_NEW_CONTENT
        
        # Additional cognitive load from browsing many items
        if row['action_type'] in dpad_actions:
            load_increase += 0.02
        
        current_cognitive_load = min(1.0, current_cognitive_load + load_increase)
        
        if row['action_type'] == 'click':
            current_cognitive_load *= 0.5
        else:
            current_cognitive_load = max(0.1, current_cognitive_load * 0.95)
        
        cognitive_loads.append(round(current_cognitive_load, 4))

    # Add the new columns to the dataframe
    session_df['frustration_level'] = frustration_levels
    session_df['cognitive_load'] = cognitive_loads

    return session_df

def main():
    """Main function to run the enhanced post-processing pipeline."""
    df = load_and_combine_logs(RAW_LOGS_PATH_PATTERN)
    
    if df is None:
        return

    print(f"Loaded {len(df)} total events. Starting ENHANCED enrichment process...")
    
    # Process each session to add psychological states
    tqdm.pandas(desc="Processing Sessions with Enhanced Frustration")
    enriched_df = df.groupby('session_id', group_keys=False).progress_apply(process_session)
    
    # --- Calculate Scroll Metrics for identified segments ---
    enriched_df['scroll_speed'] = None
    enriched_df['scroll_depth'] = None

    scroll_segments = enriched_df[enriched_df['is_scroll_event']].groupby('scroll_segment_id')

    for segment_id, segment_df in tqdm(scroll_segments, desc="Calculating Scroll Metrics"):
        if not segment_df.empty:
            scroll_depth = len(segment_df)
            total_time = segment_df['sequence_context'].apply(lambda x: x.get('time_since_last_action', 0)).sum()
            scroll_speed = scroll_depth / total_time if total_time > 0 else 0
            
            enriched_df.loc[segment_df.index, 'scroll_depth'] = scroll_depth
            enriched_df.loc[segment_df.index, 'scroll_speed'] = round(scroll_speed, 2)
    
    # Clean up helper columns and reorder for clarity
    final_df = enriched_df.drop(columns=['derived_states', 'is_scroll_event', 'scroll_segment_id'], errors='ignore').reset_index(drop=True)
    
    print(f"\nEnhanced enrichment complete. Saving to {ENRICHED_OUTPUT_FILE}...")
    final_df.to_csv(ENRICHED_OUTPUT_FILE, index=False)
    
    # Quick analysis of the enhanced dataset
    print("\n" + "="*50)
    print("ENHANCED DATASET ANALYSIS")
    print("="*50)
    
    final_frustrations = final_df.groupby('session_id')['frustration_level'].last()
    print(f"Final frustration statistics:")
    print(f"  Mean: {final_frustrations.mean():.4f}")
    print(f"  Std: {final_frustrations.std():.4f}")
    print(f"  Min: {final_frustrations.min():.4f}")
    print(f"  Max: {final_frustrations.max():.4f}")
    print(f"  Sessions ending with >0.1 frustration: {(final_frustrations > 0.1).sum()}/{len(final_frustrations)} ({(final_frustrations > 0.1).mean()*100:.1f}%)")
    print(f"  Sessions ending with >0.3 frustration: {(final_frustrations > 0.3).sum()}/{len(final_frustrations)} ({(final_frustrations > 0.3).mean()*100:.1f}%)")
    print(f"  Unique frustration levels: {final_df['frustration_level'].nunique()}")
    
    print("Done.")

if __name__ == "__main__":
    main()