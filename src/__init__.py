
import string

import torch

FILENAME = r"Dataset/all.txt"
FILENAME_TRAIN = r"Dataset/train.txt"
FILENAME_TEST = r"Dataset/test.txt"
MODEL_PATH = r"models/model_trained.pth"

all_letters = string.digits + string.ascii_letters + string.punctuation + ' ' + "\n"
# all_letters = string.ascii_letters + \
#     " \.\,\;\'\-\(\)\[\]\@\*\$\#\%\!\^\<\>\/\\" + "0123456789"
n_letters = len(all_letters) + 1    # Plus EOS marker
# 74 letters


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA AVAILABLE")
else:
    device = torch.device("cpu")
    print("ONLY CPU AVAILABLE")

n_iters = 100000
all_losses = []
total_loss = 0  # Reset every plot_every iters

max_epochs_default = 200000
print_every = 10
plot_every = 10
hidden_size_default = 512
n_layers_default = 2
l_r = 0.005
bidirectional = True
bidirectional = True
