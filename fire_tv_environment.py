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
        
        # Ensure we have enough genres for rows, or repeat
        if len(all_genres) < self.num_rows:
            genres_for_rows = np.resize(all_genres, self.num_rows)
        else:
            genres_for_rows = np.random.choice(all_genres, self.num_rows, replace=False)

        for i, genre_for_row in enumerate(genres_for_rows):
            items_in_genre = self.content_df[self.content_df['genres'].apply(lambda x: genre_for_row in x if isinstance(x, list) else False)]
            
            if items_in_genre.empty or len(items_in_genre) < self.row_size:
                # Fallback to random items if not enough specific genre items
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
        self.prev_focused_item_data = None # To track item that just lost focus for hover/dpad logging
        self.last_dpad_key_code = None # To store the raw dpad press
        
        return self._get_observation(), 0, False, {}

    def _get_observation(self) -> dict:
        """Constructs the current state observation for the LLM agent."""
        # Ensure active_row and active_col are within bounds
        self.active_row = np.clip(self.active_row, 0, self.num_rows - 1)
        self.active_col = np.clip(self.active_col, 0, self.row_size - 1)

        # Get the item currently in focus
        focused_item_data = self.grid[self.active_row][self.active_col]
        
        # Ensure multi-label genres are a list
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
        action_outcome = 'no_change'
        
        prev_row, prev_col = self.active_row, self.active_col
        prev_screen_context = self.current_screen_context

        # Store the current focused item before any potential focus change
        self.prev_focused_item_data = self._get_observation()['focused_item']

        # --- Screen-specific Action Handling & Focus Changes ---
        if self.current_screen_context == 'Home':
            if action_type == 'dpad_right':
                if self.active_col < self.row_size - 1:
                    self.active_col += 1
                    action_outcome = 'new_content_seen'
                    self.last_dpad_key_code = 'dpad_right'
            elif action_type == 'dpad_left':
                if self.active_col > 0:
                    self.active_col -= 1
                    action_outcome = 'new_content_seen'
                    self.last_dpad_key_code = 'dpad_left'
            elif action_type == 'dpad_down':
                if self.active_row < self.num_rows - 1:
                    self.active_row += 1
                    action_outcome = 'new_content_seen'
                    self.last_dpad_key_code = 'dpad_down'
            elif action_type == 'dpad_up':
                if self.active_row > 0:
                    self.active_row -= 1
                    action_outcome = 'new_content_seen'
                    self.last_dpad_key_code = 'dpad_up'
            elif action_type == 'click':
                # Click from Home goes to Detail_Page
                self.current_screen_context = 'Detail_Page'
                action_outcome = 'transitioned_to_detail_page'
                self.last_dpad_key_code = None # Clear dpad key code after non-dpad action
            elif action_type == 'back':
                # Back from Home exits the app
                done = True
                llm_decision['session_end_reason'] = 'user_exit_from_home'
                self.last_dpad_key_code = None
            elif action_type in ['hover', 'scroll']: # These actions don't change focus or screen
                self.last_dpad_key_code = None

        elif self.current_screen_context == 'Detail_Page':
            if action_type == 'click':
                if llm_decision.get('click_type') == 'play':
                    self.current_screen_context = 'Playback'
                    action_outcome = 'transitioned_to_playback'
                else:
                    action_outcome = 'content_info_accessed'
            elif action_type == 'back':
                self.current_screen_context = 'Home'
                action_outcome = 'transitioned_to_home'
            self.last_dpad_key_code = None

        elif self.current_screen_context == 'Playback':
            if action_type == 'back':
                self.current_screen_context = 'Detail_Page'
                action_outcome = 'transitioned_to_detail_page'
            elif action_type == 'playback_abandon':
                self.current_screen_context = 'Detail_Page'
                action_outcome = 'playback_abandoned'
            self.last_dpad_key_code = None

        # --- General Session Management ---
        if self.session_steps > 60: # Max steps per session to prevent infinite loops
            done = True
            llm_decision['session_end_reason'] = 'timeout'
        
        # Determine session end reason for 'click' or 'back' that truly ends the session
        if done and action_type == 'click' and self.current_screen_context == 'Playback': # If click led to playback start
            llm_decision['session_end_reason'] = 'playback_started'
        elif done and action_type == 'back' and prev_screen_context == 'Home': # Back from Home
             llm_decision['session_end_reason'] = 'user_exit_from_home'
        elif done and action_type == 'back' and prev_screen_context != 'Home' and not done: # Back from other screens, but not exit
             # This means the back action transitioned to a previous screen, session is not done
             llm_decision['session_end_reason'] = 'navigated_back'


        observation = self._get_observation()
        # The 'info' dict now also includes the previous focused item for delayed logging
        info = {
            'action_outcome': action_outcome, 
            'llm_decision': llm_decision,
            'prev_focused_item': self.prev_focused_item_data, # Item that lost focus
            'last_dpad_key_code': self.last_dpad_key_code # Raw dpad press that caused focus change
        }
        
        return observation, 0, done, info
