import openai
import json
import time

class LLMAgent:
    """
    An agent powered by a local LLM that makes decisions based on a 
    psychological profile and dynamic state.
    """
    def __init__(self, user_id: str, persona: dict, llm_model: str = "llama3:8b"):
        try:
            self.client = openai.OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama'
            )
            self.client.models.list()
        except openai.APIConnectionError:
            print("\n" + "="*50)
            print("FATAL ERROR: Could not connect to the Ollama server at http://localhost:11434.")
            print(f"Please ensure Ollama is running and the '{llm_model}' model is pulled.")
            print(f"You can run it with the command: ollama run {llm_model}")
            print("="*50 + "\n")
            raise

        self.user_id = user_id
        self.persona = persona
        self.llm_model = llm_model 
        
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
        You are a user simulator. Your role is to make decisions for a user interacting with a TV content browsing app.
        Your actions must be realistic and consistent with your given persona and current internal state.

        **Your Persona Profile:**
        - Narrative: {self.persona['narrative']}
        - Personality (OCEAN Model, 0.0-1.0 scale, 1.0 is high): {self.persona['ocean']}
        - Genre Preferences (0.0-1.0 scale, 1.0 is strong preference): {self.persona['preferences']}

        **Your Current Internal State (influences your choices):**
        - Frustration Level: {self.state['frustration_level']:.2f} (0=calm, 1=very frustrated. High frustration might lead to less exploration or quitting.)
        - Cognitive Load: {self.state['cognitive_load']:.2f} (0=bored, 1=overwhelmed. High load might lead to simpler choices or pausing.)

        **Current Situation in the App:**
        - You are on the '{screen_context}' screen.
        """
        # MODIFICATION: Add dynamic, state-aware prompt for the Detail_Page
        if screen_context == 'Detail_Page':
            click_count = observation.get('consecutive_click_count', 0)
            prompt += f"- You have already clicked on this item {click_count} times. Repeating the same 'click' action again is unrealistic. Consider other options like 'back' or 'play'.\n"
        
        prompt += f"- The item currently in focus (highlighted) is: {item_desc}\n"

        prompt += """
        **Your Task: Decide the next single action.**
        Choose ONE action type from the list below. If the action requires parameters, you MUST include them.

        **Allowed Action Types and Parameters:**
        - 'dpad_right', 'dpad_left', 'dpad_down', 'dpad_up'
        - 'click': Interact with the focused item.
            - MUST include 'click_type': 'play', 'more_info', or 'trailer'.
        - 'back': Go back to the previous screen or exit the app.
        - 'hover': Briefly focus on an item.
            - MUST include 'hover_duration': (float from 0.5 to 10.0 seconds).

        **Output Format:**
        You MUST reply in a valid JSON object. Do not include any other text or reasoning outside the JSON.
        Example for dpad: {{ "action_type": "dpad_right" }}
        Example for click: {{ "action_type": "click", "click_type": "more_info" }}
        """
        return prompt

    def decide_action(self, observation: dict) -> dict:
        prompt = self._create_prompt(observation)
        
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    response_format={"type": "json_object"}
                )
                decision = json.loads(response.choices[0].message.content)
                
                if decision.get('action_type') == 'scroll':
                    decision['action_type'] = 'dpad_down'

                return decision
            except (json.JSONDecodeError, openai.APIError) as e:
                print(f"LLM Error: {e}. Retrying...")
                time.sleep(1)
        
        return {"action_type": "dpad_right"}

    def update_state(self, action_outcome: str, event_details: dict):
        # Frustration Calculation
        frustration_increase = 0
        if action_outcome == 'no_change' and event_details.get('sequence_context', {}).get('consecutive_action_count', 0) >= 2:
            frustration_increase = 0.15 * event_details['sequence_context']['consecutive_action_count']
        elif event_details.get('action_type') == 'playback_abandon' and event_details.get('playback_position', 1.0) < 0.15:
            frustration_increase = 0.3
        
        if frustration_increase > 0:
            frustration_increase *= (1 + self.persona['ocean']['neuroticism'])
            self.state['frustration_level'] = min(1.0, self.state['frustration_level'] + frustration_increase)
        
        if event_details.get('action_type') == 'click' and event_details.get('click_type') == 'play':
            self.state['frustration_level'] *= 0.7

        # Cognitive Load Calculation
        load_increase = 0
        if event_details.get('sequence_context', {}).get('time_since_last_action', 0) > 4.0:
            load_increase = 0.05 * (event_details['sequence_context']['time_since_last_action'] / 4.0)
        elif event_details.get('action_type') == 'hover':
            load_increase = 0.05 * event_details.get('hover_duration', 0)
        elif action_outcome == 'new_content_seen':
            load_increase = 0.1
        
        self.state['cognitive_load'] = min(1.0, self.state['cognitive_load'] + load_increase)
        
        if event_details.get('action_type') == 'click':
            self.state['cognitive_load'] *= 0.5
        else:
            self.state['cognitive_load'] = max(0.1, self.state['cognitive_load'] * 0.95)
