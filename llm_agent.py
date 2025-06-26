# llm_agent.py
import openai
import json
import time

class LLMAgent:
    """
    An agent powered by a local LLM that makes decisions based on a 
    psychological profile and dynamic state.
    """
    def __init__(self, user_id: str, persona: dict, llm_model: str = "mistral"):
        # Point the OpenAI client to the local Ollama server
        try:
            self.client = openai.OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama'  # Required, but can be any string
            )
            # Ping the server to ensure it's running
            self.client.models.list()
        except openai.APIConnectionError:
            print("\n" + "="*50)
            print("FATAL ERROR: Could not connect to the Ollama server at http://localhost:11434.")
            print("Please ensure Ollama is running and the 'mistral' model is pulled.")
            print("You can run it with the command: ollama run mistral")
            print("="*50 + "\n")
            raise

        self.user_id = user_id
        self.persona = persona
        self.llm_model = llm_model
        
        # Initialize the agent's dynamic psychological state
        self.state = {
            "frustration_level": 0.0,
            "cognitive_load": 0.1,
        }

    def _create_prompt(self, observation: dict) -> str:
        """Creates a detailed prompt for the LLM to make a decision."""
        screen_context = observation['screen_context']
        focused_item = observation['focused_item']
        item_desc = f"- ID: {focused_item['item_id']}, Title: {focused_item['title']}, Genres: {focused_item['genres']}"

        prompt = f"""
        You are a user simulator. Your task is to role-play and decide your next single action based on your detailed persona and current psychological state.

        **Your Persona Profile:**
        - Narrative: {self.persona['narrative']}
        - Personality (OCEAN Model): {self.persona['ocean']}
        - Genre Preferences: {self.persona['preferences']}

        **Your Current Psychological State:**
        - Frustration Level: {self.state['frustration_level']:.2f} (0=calm, 1=very frustrated)
        - Cognitive Load: {self.state['cognitive_load']:.2f} (0=bored, 1=overwhelmed)

        **Current Situation:**
        - You are on the '{screen_context}' screen.
        - The item in focus is: {item_desc}

        **Your Task:**
        Based on your persona and state, what is your next single action?
        Choose ONE from: 'dpad_right', 'dpad_left', 'dpad_down', 'dpad_up', 'click', 'back', 'scroll', 'hover'.
        - If 'click', you MUST specify a 'click_type' ('play', 'more_info', 'trailer').
        - If 'scroll', you MUST specify 'scroll_depth' (a float from 0.1 to 1.0) and 'scroll_speed' (an integer from 100 to 2000).
        - If 'hover', you MUST specify 'hover_duration' (a float from 0.5 to 10.0 seconds).
        
        You MUST reply in a valid JSON format with ONLY the action and its parameters.
        Example for dpad: {{ "action_type": "dpad_right" }}
        Example for click: {{ "action_type": "click", "click_type": "play" }}
        Example for hover: {{ "action_type": "hover", "hover_duration": 3.5 }}
        """
        return prompt

    def decide_action(self, observation: dict) -> dict:
        """Queries the local LLM to decide the next action."""
        prompt = self._create_prompt(observation)
        
        for _ in range(3): # Retry loop for robustness
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    response_format={"type": "json_object"}
                )
                decision = json.loads(response.choices[0].message.content)
                return decision
            except (json.JSONDecodeError, openai.APIError) as e:
                print(f"LLM Error: {e}. Retrying...")
                time.sleep(1)
        
        # Fallback action if LLM fails repeatedly
        return {"action_type": "dpad_right"}

    def update_state(self, action_outcome: str, event_details: dict):
        """Updates the agent's psychological state using our defined heuristics."""
        
        # --- Frustration Calculation ---
        frustration_increase = 0
        if action_outcome == 'no_change' and event_details.get('sequence_context', {}).get('consecutive_action_count', 0) >= 2:
            frustration_increase = 0.1 * event_details['sequence_context']['consecutive_action_count']
        elif event_details.get('action_type') == 'playback_abandon' and event_details.get('playback_position', 1.0) < 0.15:
            frustration_increase = 0.3
        
        if frustration_increase > 0:
            # Apply Neuroticism multiplier from the agent's persona
            frustration_increase *= (1 + self.persona['ocean']['neuroticism'])
            self.state['frustration_level'] = min(1.0, self.state['frustration_level'] + frustration_increase)
        
        # Frustration decays on successful, goal-oriented actions
        if event_details.get('action_type') == 'click' and event_details.get('click_type') == 'play':
            self.state['frustration_level'] *= 0.7

        # --- Cognitive Load Calculation ---
        load_increase = 0
        if event_details.get('sequence_context', {}).get('time_since_last_action', 0) > 4.0:
            load_increase = 0.1 * (event_details['sequence_context']['time_since_last_action'] / 4.0)
        elif event_details.get('action_type') == 'hover':
            load_increase = 0.05 * event_details.get('hover_duration', 0)
        elif action_outcome == 'new_content_seen':
            load_increase = 0.1
        
        self.state['cognitive_load'] = min(1.0, self.state['cognitive_load'] + load_increase)
        
        # Cognitive load decays after a decision is made or over time
        if event_details.get('action_type') == 'click':
            self.state['cognitive_load'] *= 0.5
        else:
            self.state['cognitive_load'] = max(0.1, self.state['cognitive_load'] * 0.95)
