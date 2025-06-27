# dataset.py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SessionDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # A simple, fast lookup
        sample = self.data_list[idx]
        return sample['X'], sample['y']

def collate_fn(batch):
    sequences, targets = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences, torch.stack(targets)
