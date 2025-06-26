# firetv_environment.py
import numpy as np
import pandas as pd
from recsim.simulator import environment

class FireTVEnvironment(environment.Environment):
    """
    A robust RecSim environment that simulates the Fire TV UI grid and navigation rules.
    This environment is a 'digital twin' of the logic derived from your Java/Kotlin code.
    """
    def __init__(self, content_df: pd.DataFrame, num_rows: int = 5, row_size: int = 20):
        super().__init__()
        
        self.content_df = content_df
        self.num_rows = num_rows
        self.row_size = row_size
        
        self.grid = self._create_content_grid()
        self.reset()

    def _create_content_grid(self) -> list:
        """
        Creates a grid of content items based on the content_df, ensuring genre diversity 
        and replacing items if necessary.
        """
        grid = []
        all_genres = self.content_df['genres'].explode().dropna().unique()
        
        if len(all_genres) < self.num_rows:
            genres_for_rows = np.resize(all_genres, self.num_rows)
        else:
            genres_for_rows = np.random.choice(all_genres, self.num_rows, replace=False)

        for i, genre_for_row in enumerate(genres_for_rows):
            items_in_genre = self.content_df[self.content_df['genres'].apply(lambda x: genre_for_row in x if isinstance(x, list) else False)]
            
            if items_in_genre.empty or len(items_in_genre) < self.row_size:
                items_for_row = self.content_df.sample(self.row_size, replace=True) 
            else:
                items_for_row = items_in_genre.sample(self.row_size, replace=False)
            
            grid.append(items_for_row.to_dict('records'))
            
        return grid

    def reset(self) -> tuple:
        """Resets the environment to its initial state for a new session."""
        self.active_row = 0
        self.active_col = 0
        self.session_steps = 0
        self.current_screen_context = 'Home' # Initial screen
        
        # Track previous states for accurate delayed logging
        self.prev_focused_item_for_dpad_log = None # Item focus moved FROM
        self.last_dpad_key_code = None # Raw dpad press that led to focus change
        
        self.current_hover_item = None # Item currently being hovered (gained focus)
        self.last_hover_start_time = None # Timestamp when current_hover_item gained focus
        
        # Trailer playback state for logging playback_abandon
        self.trailer_playing = False
        self.trailer_start_time = 0
        
        return self._get_observation(), 0, False, {}

    def _get_observation(self) -> dict:
        """Constructs the current state observation for the LLM agent."""
        self.active_row = np.clip(self.active_row, 0, self.num_rows - 1)
        self.active_col = np.clip(self.active_col, 0, self.row_size - 1)

        focused_item_data = self.grid[self.active_row][self.active_col]
        
        if isinstance(focused_item_data['genres'], str):
            focused_item_data['genres'] = [focused_item_data['genres']]

        return {
            'screen_context': self.current_screen_context,
            'focused_item': {
                'item_id': str(focused_item_data['item_id']),
                'title': focused_item_data['title'],
                'genres': focused_item_data['genres']
            },
        }

    def step(self, llm_decision: dict) -> tuple:
        """
        Processes an action from the LLM agent and updates the environment's state.
        This method embodies the UI's navigation rules and now the logging flow.
        """
        action_type = llm_decision.get('action_type', 'dpad_right')
        self.session_steps += 1
        
        done = False
        action_outcome = 'no_change' # Outcome for LLM agent's state update
        
        prev_row, prev_col = self.active_row, self.active_col
        prev_screen_context = self.current_screen_context
        
        # Current focused item before any potential focus change by dpad
        current_focused_item_pre_action = self._get_observation()['focused_item']

        # --- D-pad Actions ---
        # Store original LLM decision for direct D-pad logging (MainActivity.kt style)
        if action_type in ['dpad_right', 'dpad_left', 'dpad_down', 'dpad_up']:
            self.last_dpad_key_code = action_type # Store the raw dpad press

        if self.current_screen_context == 'Home':
            if action_type == 'dpad_right':
                if self.active_col < self.row_size - 1:
                    self.active_col += 1
                    action_outcome = 'new_content_seen'
            elif action_type == 'dpad_left':
                if self.active_col > 0:
                    self.active_col -= 1
                    action_outcome = 'new_content_seen'
            elif action_type == 'dpad_down':
                if self.active_row < self.num_rows - 1:
                    self.active_row += 1
                    action_outcome = 'new_content_seen'
            elif action_type == 'dpad_up':
                if self.active_row > 0:
                    self.active_row -= 1
                    action_outcome = 'new_content_seen'
            elif action_type == 'click':
                # Click from Home goes to Detail_Page (MovieDetailsDialogFragment)
                self.current_screen_context = 'Detail_Page'
                action_outcome = 'transitioned_to_detail_page'
            elif action_type == 'back':
                # Back from Home exits the app
                done = True
                llm_decision['session_end_reason'] = 'user_exit_from_home'
            elif action_type in ['hover', 'scroll', 'playback_completed']: # These actions don't change focus or screen on their own
                pass # Handled by logging logic, not env state change

        elif self.current_screen_context == 'Detail_Page':
            if action_type == 'click':
                # Assume click on Detail_Page for 'play' leads to app exit (launchDeeplink)
                if llm_decision.get('click_type') == 'play':
                    done = True # Simulates launching external app
                    action_outcome = 'launched_external_app'
                    llm_decision['session_end_reason'] = 'launched_playback'
                else: # Any other click type (more_info, trailer etc.) means staying on Detail_Page
                    action_outcome = 'content_info_accessed'
            elif action_type == 'back':
                self.current_screen_context = 'Home'
                action_outcome = 'transitioned_to_home'
        
        elif self.current_screen_context == 'Playback': # This context is mainly for logging within Home
            if action_type == 'back':
                self.current_screen_context = 'Home' # Go back to Home
                action_outcome = 'transitioned_to_home'
                # If a full movie playback started from deeplink, then back might also mean session ends
                # For now, assumes back during trailer playback on home
            elif action_type == 'playback_abandon':
                self.current_screen_context = 'Home' # Assumes returning to Home from banner playback
                action_outcome = 'playback_abandoned'
                self.trailer_playing = False # Stop trailer
            elif action_type == 'playback_completed': # New action for trailer completion
                self.current_screen_context = 'Home'
                action_outcome = 'playback_completed'
                self.trailer_playing = False

        # --- Update current_hover_item and last_hover_start_time ---
        # Mimic MovieFragment's setOnItemViewSelectedListener and GlobalFocusChangeListener
        new_focused_item_data = self._get_observation()['focused_item'] # Item that just gained focus

        if self.current_hover_item and (self.current_hover_item['item_id'] != new_focused_item_data['item_id']):
            # This item lost focus. It was the previously hovered item.
            # This is where we capture the hover duration for the item that just lost focus.
            llm_decision['logged_hover_item'] = self.current_hover_item
            llm_decision['logged_hover_start_time'] = self.last_hover_start_time

        # Update current hover item for the newly focused item
        self.current_hover_item = new_focused_item_data
        self.last_hover_start_time = self.session_steps # Using step count as proxy for time here in env

        # --- Track previous focused item for D-pad logging ---
        if self.active_row != prev_row or self.active_col != prev_col:
            # If focus changed, store the item that was focused BEFORE this change
            self.prev_focused_item_for_dpad_log = current_focused_item_pre_action
        else:
            self.prev_focused_item_for_dpad_log = None # No focus change, no dpad log from this change

        # --- Handle trailer playback state (for accurate logPreviousTrailerDuration mimic) ---
        if action_type == 'click' and llm_decision.get('click_type') == 'trailer':
            self.trailer_playing = True
            self.trailer_start_time = self.session_steps # Proxy for System.currentTimeMillis()
        elif action_type == 'back' or action_type == 'playback_abandon' or action_type == 'playback_completed':
            self.trailer_playing = False # Trailer stopped

        # --- General Session Management ---
        if self.session_steps > 60: # Max steps per session to prevent infinite loops
            done = True
            llm_decision['session_end_reason'] = 'timeout'
        
        observation = self._get_observation()
        # The 'info' dict now carries all necessary data for run_simulation.py to log events
        info = {
            'action_outcome': action_outcome, 
            'llm_decision': llm_decision,
            'prev_focused_item_for_dpad_log': self.prev_focused_item_for_dpad_log, # Item from which dpad moved
            'last_dpad_key_code': self.last_dpad_key_code, # Dpad key that was pressed
            'current_focused_item_data': new_focused_item_data, # Item that gained focus (for non-dpad, non-hover logs)
            'trailer_playing': self.trailer_playing,
            'trailer_start_time': self.trailer_start_time,
            'current_screen_context': self.current_screen_context # Pass the updated context
        }
        
        return observation, 0, done, info
