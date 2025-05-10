import torch
from torch.utils.data import Dataset

from src.Utils import encode_password


class PasswordDataset(Dataset):
    def __init__(self, passwords):
        self.data = [encode_password(pw) for pw in passwords]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        input_seq, target_seq = self.data[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
        )
