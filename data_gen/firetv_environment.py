import numpy as np
import pandas as pd
from recsim.simulator import environment

# --- Dummy Classes to Satisfy RecSim's Environment __init__ ---
class DummyDocument:
    def __init__(self, doc_id="dummy_doc"):
        self._doc_id = doc_id
    def doc_id(self):
        return self._doc_id

class DummyDocumentSampler:
    def __init__(self):
        self._doc_id_counter = 0
    def sample_document(self):
        doc = DummyDocument(f"dummy_doc_{self._doc_id_counter}")
        self._doc_id_counter += 1
        return doc

class DummyUserModel:
    def __init__(self):
        pass
    def sample_user(self):
        return "dummy_user_id"
# --- End of Dummy Classes ---


class FireTVEnvironment(environment.Environment):
    """
    A robust RecSim environment that simulates the Fire TV UI grid and navigation rules.
    This environment is a 'digital twin' of the logic derived from your Java/Kotlin code.
    """
    def __init__(self, content_df: pd.DataFrame, num_rows: int = 5, row_size: int = 20,
                 user_model=None, document_sampler=None, num_candidates: int = 1, slate_size: int = 1):
        
        _user_model = user_model if user_model is not None else DummyUserModel()
        _document_sampler = document_sampler if document_sampler is not None else DummyDocumentSampler()

        super().__init__(_user_model, _document_sampler, num_candidates, slate_size)
        
        self.content_df = content_df
        self.num_rows = num_rows
        self.row_size = row_size
        
        self.grid = self._create_content_grid()
        self.reset()

    def _create_content_grid(self) -> list:
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
        self.current_screen_context = 'Home'
        
        self.prev_focused_item_for_dpad_log = None
        self.last_dpad_key_code = None
        
        self.current_hover_item = None
        self.last_hover_start_time = None
        self.stored_hover_duration = None
        
        self.trailer_playing = False
        self.trailer_start_time = 0
        
        # MODIFICATION: Track consecutive clicks on the same item on Detail_Page
        self.consecutive_click_count = 0
        self.last_clicked_item_id = None

        return self._get_observation(), 0, False, {}

    def _get_observation(self) -> dict:
        """Constructs the current state observation for the LLM agent."""
        self.active_row = np.clip(self.active_row, 0, self.num_rows - 1)
        self.active_col = np.clip(self.active_col, 0, self.row_size - 1)

        focused_item_data = self.grid[self.active_row][self.active_col]
        
        if isinstance(focused_item_data['genres'], str):
            focused_item_data['genres'] = [focused_item_data['genres']]

        obs = {
            'screen_context': self.current_screen_context,
            'focused_item': {
                'item_id': str(focused_item_data['item_id']),
                'title': focused_item_data['title'],
                'genres': focused_item_data['genres']
            },
        }

        if self.current_screen_context == 'Detail_Page':
            obs['available_ui_elements'] = ['play_button', 'trailer_button', 'back_button', 'synopsis_text']
            # MODIFICATION: Add click count to the observation
            obs['consecutive_click_count'] = self.consecutive_click_count

        return obs

    def step(self, llm_decision: dict) -> tuple:
        action_type = llm_decision.get('action_type', 'dpad_right')
        self.session_steps += 1
        
        done = False
        action_outcome = 'no_change'
        
        prev_row, prev_col = self.active_row, self.active_col
        
        current_focused_item_pre_action = self._get_observation()['focused_item']

        # MODIFICATION: Update consecutive_click_count logic
        if self.current_screen_context == 'Detail_Page' and action_type == 'click':
            if current_focused_item_pre_action['item_id'] == self.last_clicked_item_id:
                self.consecutive_click_count += 1
            else:
                self.consecutive_click_count = 1
            self.last_clicked_item_id = current_focused_item_pre_action['item_id']
        else:
            # Reset click counter if not on detail page or not a click action
            self.consecutive_click_count = 0
            self.last_clicked_item_id = None

        if action_type in ['dpad_right', 'dpad_left', 'dpad_down', 'dpad_up']:
            self.last_dpad_key_code = action_type
        
        if action_type == 'hover':
            self.stored_hover_duration = llm_decision.get('hover_duration', 0.0)
        else:
            self.stored_hover_duration = None

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
                self.current_screen_context = 'Detail_Page'
                action_outcome = 'transitioned_to_detail_page'
            elif action_type == 'back':
                done = True
                llm_decision['session_end_reason'] = 'user_exit_from_home'
            elif action_type in ['hover', 'scroll', 'playback_completed']:
                pass

        elif self.current_screen_context == 'Detail_Page':
            if action_type == 'click':
                if llm_decision.get('click_type') == 'play':
                    done = True
                    action_outcome = 'launched_external_app'
                    llm_decision['session_end_reason'] = 'launched_playback'
                else:
                    action_outcome = 'content_info_accessed'
            elif action_type == 'back':
                self.current_screen_context = 'Home'
                action_outcome = 'transitioned_to_home'
        
        elif self.current_screen_context == 'Playback':
            if action_type == 'back':
                self.current_screen_context = 'Home'
                action_outcome = 'transitioned_to_home'
            elif action_type == 'playback_abandon':
                self.current_screen_context = 'Home'
                action_outcome = 'playback_abandoned'
                self.trailer_playing = False
            elif action_type == 'playback_completed':
                self.current_screen_context = 'Home'
                action_outcome = 'playback_completed'
                self.trailer_playing = False

        if self.session_steps > 300:
            done = True
            llm_decision['session_end_reason'] = 'timeout'
        
        observation = self._get_observation()
        new_focused_item_data = observation['focused_item']

        if self.current_hover_item and \
           (self.current_hover_item['item_id'] != new_focused_item_data['item_id']) and \
           self.stored_hover_duration is not None:
            
            llm_decision['logged_hover_item'] = self.current_hover_item
            llm_decision['logged_hover_duration'] = self.stored_hover_duration
            llm_decision['logged_hover_start_time'] = self.last_hover_start_time

        self.current_hover_item = new_focused_item_data
        self.last_hover_start_time = self.session_steps

        if self.active_row != prev_row or self.active_col != prev_col:
            self.prev_focused_item_for_dpad_log = current_focused_item_pre_action
        else:
            self.prev_focused_item_for_dpad_log = None

        if action_type == 'click' and llm_decision.get('click_type') == 'trailer':
            self.trailer_playing = True
            self.trailer_start_time = self.session_steps
        elif action_type in ['back', 'playback_abandon', 'playback_completed']:
            self.trailer_playing = False

        info = {
            'action_outcome': action_outcome, 
            'llm_decision': llm_decision,
            'prev_focused_item_for_dpad_log': self.prev_focused_item_for_dpad_log,
            'last_dpad_key_code': self.last_dpad_key_code,
            'current_focused_item_data': new_focused_item_data,
            'trailer_playing': self.trailer_playing,
            'trailer_start_time': self.trailer_start_time,
            'current_screen_context': self.current_screen_context,
            # Pass click count for circuit breaker
            'consecutive_click_count': self.consecutive_click_count 
        }
        
        return observation, 0, done, info
