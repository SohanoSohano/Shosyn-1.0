#!/bin/bash

SESSION_NAME="simulation"
PROJECT_PATH="~/Shosyn-1.0"
CONDA_ENV="recsim_tf1"
LLM_MODEL="llama3:8b"

# Kill any existing tmux session with the same name
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Start a new detached tmux session
echo "Starting new tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# Pane 0: Start the Ollama server
tmux send-keys -t $SESSION_NAME:0 "ollama run $LLM_MODEL" C-m

# Now create 5 vertical panes below pane 0 for workers
# First split pane 0 vertically into pane 0 and pane 1
tmux split-window -v -t $SESSION_NAME:0.0

# Then split pane 1 vertically 4 more times to get 5 worker panes total
tmux split-window -v -t $SESSION_NAME:0.1
tmux split-window -v -t $SESSION_NAME:0.2
tmux split-window -v -t $SESSION_NAME:0.3
tmux split-window -v -t $SESSION_NAME:0.4

# Now you have panes 0 (Ollama), 1-5 (workers)

# Start workers in panes 1 to 5
tmux send-keys -t $SESSION_NAME:0.1 "conda activate $CONDA_ENV && cd $PROJECT_PATH && python run_simulation.py worker1" C-m
tmux send-keys -t $SESSION_NAME:0.2 "conda activate $CONDA_ENV && cd $PROJECT_PATH && python run_simulation.py worker2" C-m
tmux send-keys -t $SESSION_NAME:0.3 "conda activate $CONDA_ENV && cd $PROJECT_PATH && python run_simulation.py worker3" C-m
tmux send-keys -t $SESSION_NAME:0.4 "conda activate $CONDA_ENV && cd $PROJECT_PATH && python run_simulation.py worker4" C-m
tmux send-keys -t $SESSION_NAME:0.5 "conda activate $CONDA_ENV && cd $PROJECT_PATH && python run_simulation.py worker5" C-m

# Attach to the session to view the output
echo "Attaching to session..."
tmux attach-session -t $SESSION_NAME
