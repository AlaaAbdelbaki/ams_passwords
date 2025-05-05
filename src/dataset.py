import string

import torch
from torch.utils.data import Dataset

from src.preprocessing import input_tensor, target_tensor


class PasswordDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                input_seq = input_tensor(line)
                target_seq = target_tensor(line)
                self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
