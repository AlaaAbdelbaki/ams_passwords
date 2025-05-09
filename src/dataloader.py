import torch
from torch.utils.data import Dataset

from src import PAD_IDX


class PasswordDataset(Dataset):
    def __init__(self, password_list, all_letters, n_letters):
        self.password_list = list(password_list)
        self.all_letters = all_letters
        self.n_letters = n_letters

    def __getitem__(self, idx):
        password = self.password_list[idx]

        input_seq = password  # e.g., "abc"
        target_seq = password + '\n'  # e.g., "abc\n"

        input_tensor = torch.tensor(
            [self.all_letters.find(c) for c in input_seq], dtype=torch.long)
        target_tensor = torch.tensor(
            [self.all_letters.find(c) for c in target_seq], dtype=torch.long)

        return input_tensor, target_tensor

    def __len__(self):
        return len(self.password_list)
