# engine.py
import torch
from tqdm import tqdm

def train_loop(model, dataloader, optimizer, loss_fn, scaler, device, clip_value):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad(set_to_none=True)
        sequences, targets = batch
        sequences, targets = sequences.to(device), targets.to(device)

        use_cuda = device.type == 'cuda'
        with torch.cuda.amp.autocast(enabled=use_cuda):
            predictions = model(sequences)
            loss = loss_fn(predictions, targets)

        if torch.isnan(loss):
            print("NaN loss detected! Skipping batch.")
            continue

        if use_cuda:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def eval_loop(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences, targets = batch
            sequences, targets = sequences.to(device), targets.to(device)
            predictions = model(sequences)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)
