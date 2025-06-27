# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os

from config import CONFIG
from dataset import SessionDataset, collate_fn
from model import NeuralCDE
from engine import train_loop, eval_loop

def main():
    # Check if preprocessed data exists, if not, run the preprocessor
    if not os.path.exists(CONFIG['processed_data_path']):
        print(f"{CONFIG['processed_data_path']} not found. Running preprocessor first...")
        import preprocess
        preprocess.preprocess_and_save()

    print("Starting Neural CDE Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the preprocessed data
    processed_data = torch.load(CONFIG['processed_data_path'])
    train_data_list = processed_data['train']
    val_data_list = processed_data['val']
    
    print(f"Training on {len(train_data_list)} sessions, Validating on {len(val_data_list)} sessions.")
    
    train_dataset = SessionDataset(train_data_list)
    val_dataset = SessionDataset(val_data_list)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=CONFIG["num_workers"], pin_memory=True)

    # Determine input channels from the first sample
    sample_x, _ = train_dataset[0]
    input_channels = sample_x.shape[1]
    output_channels = 1

    model = NeuralCDE(input_channels, CONFIG["hidden_channels"], output_channels).to(device)
    
    use_cuda = device.type == 'cuda'
    if torch.cuda.device_count() > 1 and use_cuda:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=CONFIG["scheduler_factor"], patience=CONFIG["scheduler_patience"], verbose=True)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print("\nStarting training...")
    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()
        
        train_loss = train_loop(model, train_loader, optimizer, loss_fn, scaler, device, CONFIG["clip_value"])
        val_loss = eval_loop(model, val_loader, loss_fn, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        end_time = time.time()
        epoch_mins = (end_time - start_time) / 60

        print(f"Epoch {epoch+1:02}/{CONFIG['epochs']:02} | "
              f"Time: {epoch_mins:.2f}m | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val. Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_cde.pth')
            print(f"  -> New best validation loss. Model saved to 'best_model_cde.pth'")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('training_loss_plot.png')
    print("Saved training plot to 'training_loss_plot.png'")

if __name__ == "__main__":
    main()
