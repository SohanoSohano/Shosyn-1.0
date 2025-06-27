# config.py
CONFIG = {
    "raw_data_path": "/home/ubuntu/Shosyn-1.0/dataset/enriched_simulation_logs_500.csv",
    "processed_data_path": "processed_cde_data.pt",
    "batch_size": 256,         # Increased batch size for GPU saturation
    "learning_rate": 1e-4,     # A stable starting learning rate
    "weight_decay": 1e-5,
    "epochs": 50,
    "hidden_channels": 64,
    "cde_func_channels": 128,
    "cde_func_depth": 3,
    "readout_hidden_channels": 64,
    "num_workers": 8,          # Increased for parallel data loading on Linux
    "clip_value": 1.0,
    "scheduler_patience": 3,   # LR scheduler patience
    "scheduler_factor": 0.1,   # LR scheduler reduction factor
}
