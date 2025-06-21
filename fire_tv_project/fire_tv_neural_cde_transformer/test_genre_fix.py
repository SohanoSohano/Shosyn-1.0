import torch
import torch.nn as nn

# Test the fixed genre targets
genre_targets = torch.tensor([
    [1., 3., 2., 3., 4., 1., 1., 2., 3., 3.]  # Fixed: no zeros
])

# Test with BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
fake_predictions = torch.randn(1, 10)  # Random predictions

loss = criterion(fake_predictions, genre_targets)
print(f"Test loss: {loss.item()}")
print("✅ Genre loss calculation working!")

