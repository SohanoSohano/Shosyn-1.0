# Shosyn: A Psychological Recommendation System

This document has been updated to reflect the project's current progress and provide a detailed breakdown of the data generation process and the inner workings of the inference engine.

### **1. Project Overview**

Shosyn is an innovative recommendation engine that moves beyond traditional viewing history analysis. Instead, it processes sequences of user interactions—such as scrolls, clicks, and hesitations—to infer a user's real-time psychological state. The initial project vision explored complex architectures like Neural CDEs with Transformers to predict a wide array of psychological traits. However, our development journey led us to a more focused and effective solution: a highly optimized **Neural Rough Differential Equation (RDE)** model that excels at predicting the two core states most relevant to content discovery fatigue: **frustration** and **cognitive load**. This model forms the heart of our advanced inference engine.

### **2. Project Journey & Dataset Generation**

A significant portion of our progress has been dedicated to developing a high-quality, synthetic dataset capable of training our temporal models. This was a multi-step process designed to create realistic user behavior linked to psychological states.

#### **2.1. The Simulation Framework**

To generate the necessary data, we built a sophisticated simulation framework using a classic agent-environment loop.

*   **The Environment (`FireTVEnvironment`)**: We created a "digital twin" of the Fire TV interface using the RecSim framework. This environment manages the content grid, applies navigation rules (e.g., `dpad_right`, `back`), and presents observations to the agent, such as the currently focused item and the user's location in the app.
*   **The Agent (`LLMAgent`)**: The role of the user is played by an AI agent powered by a local Large Language Model (in this case, `llama3:8b`). The agent is assigned one of many detailed **Personas** (e.g., "Impatient Professional," "Curious Explorer," "Tech Struggler"), each with a unique personality profile, genre preferences, and sensitivity to frustration. At each step, the agent receives an observation from the environment and, based on its persona and current internal state (frustration and cognitive load), decides on the next action (e.g., `'click'`, `'dpad_down'`). This decision-making process is guided by a detailed prompt that asks the LLM to act realistically.

#### **2.2. Generating Raw Behavioral Logs**

The simulation runs for a specified number of sessions. In each session, the LLM-powered agent interacts with the Fire TV environment, generating a sequence of actions. These actions, along with timestamps and contextual details, are recorded as raw event logs. This process creates a rich log of behavioral data that mimics how different types of users might navigate the app.

#### **2.3. Data Enrichment and Labeling**

The raw logs contain the *behavior*, but not the ground-truth *psychological labels* needed for training. A separate post-processing script enriches these logs:

1.  **Heuristic-Based Labeling**: The script iterates through each session's event log and applies a set of sophisticated heuristics to calculate a `frustration_level` and `cognitive_load` for every single event.
2.  **Advanced Rules**: These heuristics are more advanced than the agent's internal state logic. They account for factors like:
    *   Repeated, unproductive actions (e.g., multiple clicks on the same item).
    *   Long hesitations between actions, indicating indecision.
    *   Rapid, frantic actions, indicating agitation.
    *   Short playback durations ("playback abandonment").
    *   A gradual increase in frustration over the course of a long session.
3.  **Persona-Driven Scaling**: The intensity of the calculated frustration is scaled based on the user's persona, specifically their "neuroticism" trait. A "Stressed Professional" will see their frustration rise much faster than a "Zen Viewer" for the same set of actions.
4.  **Feature Engineering**: Finally, the script calculates derived behavioral features, such as `scroll_speed` and `scroll_depth`, by identifying and analyzing segments of continuous scrolling.

The final output is a CSV file where each row is a distinct user event, complete with detailed behavioral features and the corresponding ground-truth `frustration_level` and `cognitive_load` labels. This enriched dataset is the fuel for training our inference engine's neural network.

### **3. How the Inference Engine Works**

The inference engine is the live system that processes user actions in real-time to generate personalized recommendations. Its operation is divided into four distinct stages.

#### **Stage 1: Input**
`Fire TV User Action → API Call → Inference Engine`

The process begins when a user interacts with their Fire TV remote. The Fire TV client captures this action (e.g., a `dpad_right` press) and sends it as a structured event in an API call to the inference engine's `update_session` endpoint.

#### **Stage 2: Processing (Inferring Psychological State)**
`Event Processing → Genre Heuristics → Batch Queue → Neural RDE → Smoothing → Calibration`

This core stage translates the raw stream of user actions into a stable prediction of their psychological state.

1.  **Event Processing & Heuristics**: The incoming event is added to the user's current session history. A quick heuristic adjustment is made; for example, interacting with a "Documentary" might slightly increase the initial cognitive load value of the event.
2.  **Batch Queue**: To maximize GPU efficiency, individual events are not processed one by one. They are placed into a batch queue, which gathers events from multiple active user sessions. The system processes the batch once it reaches an optimal size or after a very short timeout.
3.  **Neural RDE Model**: The batched sequences of events are fed into the pre-trained `MultiTargetNeuralRDE` model. This model uses the log-signature of the interaction path to efficiently summarize the entire behavioral sequence. In a single forward pass, it outputs a prediction for both `frustration` and `cognitive_load`.
4.  **Smoothing**: The raw model output can be volatile. A **Kalman Smoother** is applied to the predictions. This statistical filter smooths out noise and prevents the system from overreacting to single, isolated user actions, resulting in a more stable and realistic psychological trend.
5.  **Calibration**: As a final step, the smoothed prediction is calibrated. This process makes minor adjustments to the model's output based on its performance within the current session, correcting for any consistent over- or under-prediction and improving accuracy over time.

#### **Stage 3: Recommendation (From Psychology to Content)**
`Psychological State → Strategy Selection → Content Scoring → Shuffling → Final Recommendation`

Using the inferred psychological state, the engine curates a personalized slate of content.

1.  **Strategy Selection**: The engine analyzes the user's state (e.g., "high frustration, low cognitive load" or "frustration is rapidly increasing") and selects a corresponding recommendation strategy from a predefined playbook. Strategies include "Intervention Content" for high frustration, "Comfort Content" for stress recovery, or "Exploration" for engaged users.
2.  **Content Scoring**: The chosen strategy dictates how the entire movie catalog is scored. Each movie is assigned a score based on how well its attributes (e.g., genre, complexity) align with the current strategy's goals. For example, an "Intervention" strategy will heavily favor low-complexity, high-affinity genres like "Comedy."
3.  **Shuffling & Diversity**: Instead of just showing the top-scored items, the engine intentionally shuffles the top results using a weighted algorithm. This injects serendipity and prevents the user from seeing the same #1 recommendation repeatedly. A diversity filter then ensures the final list contains a healthy mix of genres.
4.  **Final Recommendation**: The engine assembles the final list of movie recommendations, each paired with a dynamically generated `reasoning` string (e.g., "A comforting comedy to help break the frustration cycle") that explains *why* the content is being suggested.

#### **Stage 4: Output**
`Recommendations → Fire TV UI → User Experience`

The final, curated list of recommendations and their corresponding explanations are sent back in the API response. The Fire TV UI then displays this personalized slate to the user, aiming to create an empathetic and helpful experience that reduces decision fatigue and increases engagement.

### **4. Conclusion**

The Shosyn project successfully demonstrates the viability and power of psychological-aware recommendation systems. By focusing on the core problem of content discovery fatigue, we have built an end-to-end solution that can infer user frustration and cognitive load from simple, privacy-preserving behavioral signals. Our journey from exploring complex architectures to refining a highly efficient Neural RDE model highlights a key learning: targeted, specialized models can often outperform larger, more generalized ones for specific real-world problems.

The development of our sophisticated data generation pipeline, using LLM-powered agents to create realistic, labeled behavioral logs, represents a significant milestone. This dataset was instrumental in training a robust model and serves as a valuable asset for future research. Ultimately, Shosyn provides a strong foundation for a new class of recommendation systems that are not just accurate, but also empathetic, aiming to directly mitigate user frustration and transform the content discovery experience from a chore into a pleasure.

### **5. Future Enhancements**

While the current system is a robust proof-of-concept, there are numerous avenues for future development and enhancement:

*   **Model Expansion and Multi-Modality:**
    *   **Richer Psychological States:** Expand the model to predict additional states like 'boredom', 'curiosity', or 'engagement' to enable an even wider range of recommendation strategies.
    *   **Multi-Modal Inputs:** Incorporate other data streams where available, such as voice tone analysis from remote control microphones (with user consent) to capture explicit signs of frustration or excitement.

*   **Hyper-Personalization and Long-Term Memory:**
    *   **Long-Term User Profiles:** Move beyond session-based memory to build long-term profiles that learn a user's baseline frustration tolerance, content complexity preferences, and how they typically recover from cognitive overload.
    *   **Adaptive Strategies:** Allow the recommendation strategies themselves to be personalized. The system could learn through A/B testing that 'User A' responds best to action movies as an intervention, while 'User B' prefers documentaries.

*   **Advanced UI/UX Integration:**
    *   **Dynamic UI Adaptation:** Instead of just re-ranking content, the UI could dynamically change based on the user's inferred state. For instance, detecting high cognitive load could trigger a simplified layout with fewer choices and more prominent information.
    *   **Optimizing Explanations:** Systematically test different formats for the `reasoning` text to determine which styles are most effective at building user trust and encouraging content selection.

*   **Performance and Deployment:**
    *   **On-Device Inference:** Explore model quantization and pruning to create a lightweight version of the Neural RDE that could run directly on future Fire TV hardware. This would virtually eliminate latency and enhance user privacy.
    *   **Edge-Compute Refinements:** Further optimize the server-side batching and inference pipeline to ensure near-instantaneous UI updates in response to user actions, making the system feel completely responsive.
