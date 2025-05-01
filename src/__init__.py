
import string

import torch

FILENAME = r"Dataset/all.txt"
FILENAME_TRAIN = r"Dataset/train.txt"
FILENAME_TEST = r"Dataset/test.txt"
LAST_EPOCH_MODEL_PATH = r"last_epoch_model"
BEST_MODEL_PATH = r"best_loss_model"

all_letters = string.ascii_letters + "0123456789 .,;'-_()[]@*$#%?!^<>\"`=+&~|:/\\"
n_letters = len(all_letters) + 1    # Plus EOS marker

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA AVAILABLE")
else:
    device = torch.device("cpu")
    print("ONLY CPU AVAILABLE")

n_iters = 100000
all_losses = []
total_loss = 0  # Reset every plot_every iters

max_epochs_default = 30000
print_every = 10
plot_every = 10
hidden_size_default = 512
n_layers_default = 2
l_r = 0.005
bidirectional = True
