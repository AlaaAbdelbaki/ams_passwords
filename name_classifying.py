import glob
import os
import random
import string
import time
import unicodedata
from io import open

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Check if CUDA is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")


allowed_characters = string.ascii_letters + ".,;'"
n_letters = len(allowed_characters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )


def letterToIndex(letter):
    return allowed_characters.find(letter)


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class NamesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_time = time.localtime()
        labels_set = set()

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        for fileName in text_files:
            label = os.path.splitext(os.path.basename(fileName))[0]
            labels_set.add(label)
            lines = open(fileName, encoding='utf-8').read().strip().split('\n')
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(lineToTensor(name))
                self.labels.append(label)

        self.labels_unique = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor(
                [self.labels_unique.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item


all_data = NamesDataset('data/names')
# print(f"loaded {len(all_data)} items of data")
# print(f"example: {all_data[0]}")


train_set, test_set, dev_set = torch.utils.data.random_split(
    all_data, [.80, .10, .10], generator=torch.Generator(device=device).manual_seed(2024))

print(
    f"train examples: {len(train_set)}, validation examples: {len(test_set)}, dev examples: {len(dev_set)}")


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output


n_hidden = 128
rnn = CharRNN(n_letters, n_hidden, len(all_data.labels_unique))
print(rnn)


def train(rnn: CharRNN, training_data, n_epochs=10, n_batch_size=64, report_every=50, learning_rate=0.2, criterion=nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds.
    """

    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"Training on data set for n = {len(training_data)}")

    for iter in range(1, n_epochs+1):
        rnn.zero_grad()

        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches)//n_batch_size)

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches))
        if (iter % report_every == 0):
            print(
                f"{iter} ({iter / n_epochs:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses


start = time.time()
all_losses = train(rnn, train_set, n_epochs=27,
                   learning_rate=.15, report_every=5)
end = time.time()

print(f"training took {end-start:.2f} seconds")
