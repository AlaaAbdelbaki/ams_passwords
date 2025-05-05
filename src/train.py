import logging
import os
import random
import time
from math import inf
from os import path

import torch
import torch.nn as nn
import tqdm
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import device, n_letters
from src.model import LSTMModel
from src.preprocessing import input_tensor, target_tensor
from src.Utils import all_letters, create_folder, get_model_name, time_since

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def training(decoder: nn.Module, n_epochs, dataloader: DataLoader, hidden_size, n_layers, lr, optimizer, criteron):
    """
    Trains a model (decoder) for a specified number of epochs and saves the best model based on the loss.

    Args:
        decoder: The model to be trained.
        n_epochs (int): The number of training epochs.
        lines (list): The data used for training, where each item represents a line to be processed.
        criterion: The loss function used to evaluate the model.
        hidden_size (int): The number of hidden units in the model.
        n_layers (int): The number of layers in the model.
        lr (float): The learning rate for the optimizer.
        model_path (str): The path to save the best model.

    Logs:
        Info-level logs include the training progress, best model updates, and loss values.
    """
    logging.info("\n-----------\n|  TRAIN  |\n-----------")

    writer = SummaryWriter(
        log_dir=f"logs/logs_{hidden_size}_{n_layers}_{lr}_{n_epochs}")
    start = time.time()
    best_loss = float('inf')
    print_every = max(1, n_epochs // 100)

    train_path = create_folder(n_layers, hidden_size, lr, n_epochs)

    decoder.train()
    for iter in range(1, n_epochs + 1):
        total_loss = 0
        logging.info(f"Epoch {iter}/{n_epochs}")
        # Sample a random subset of lines for each epoch
        loss = train_epoch(decoder, dataloader, optimizer, criteron)
        writer.add_scalar("Loss/train", loss, iter)

        # decoder.eval()
        # total_val_loss = 0
        # with torch.no_grad():  # No need to calculate gradients during validation
        #     for j in tqdm.tqdm(range(len(dev_lines)), desc="Validation", unit="line"):
        #         _, val_loss = train(decoder, input_tensor(line).to(
        #             device), target_tensor(line).to(device), optimizer, criteron, True)
        #         total_val_loss += val_loss

        # avg_val_loss = total_val_loss / len(dev_lines)
        # writer.add_scalar("Loss/val", avg_val_loss, iter)

        if loss < best_loss:
            best_loss = loss
            torch.save(decoder.state_dict(), path.join(train_path, f"best.pt"))
            logging.info(
                f"New best model saved with validation loss {best_loss:.4f}")

        if iter % print_every == 0:
            logging.info(
                f"{time_since(start)} ({iter} {iter / n_epochs * 100:.2f}%) Loss: {loss:.4f}")
            torch.save(decoder.state_dict(), path.join(
                train_path, "iterations/", f"{iter}.pt"))
    writer.close()


def train(decoder, input_line_tensor, target_line_tensor, optimizer, criteron):
    """
    Performs a single training step: computes the forward pass, calculates the loss, and updates the model.

    Args:
        decoder: The model being trained.
        input_line_tensor (Tensor): The input data tensor.
        target_line_tensor (Tensor): The target data tensor.
        optimizer: The optimizer used to update the model.

    Returns:
        output: The model's output for the input data.
        loss: The computed loss for the current input-target pair.
    """
    target_line_tensor = target_line_tensor.unsqueeze(
        -1)  # Reshape target tensor for loss computation
    hidden = decoder.init_hidden(target_line_tensor[0].size(0))

    decoder.zero_grad()
    loss = torch.tensor(0.0, requires_grad=True).to(device)
    output = None

    # Iterate through the input sequence
    for i in range(input_line_tensor.size(0)):
        output, hidden = decoder(input_line_tensor[i].unsqueeze(0), hidden)
        l = criteron(output, target_line_tensor[i].to(device))
        loss += l

    loss.backward()
    optimizer.step()
    try:
        loss_calculation = loss.item() / input_line_tensor.size(0)
    except Exception as e:
        print(input_line_tensor)
        # Convert input line tensor to characters
        input_line = ''.join([all_letters[i]
                             for i in input_line_tensor.squeeze().tolist()])
        logging.error(f"Error in loss calculation: {e}")
        logging.error(f"Line in question: {input_line}")
        loss_calculation = inf
    return output, loss_calculation
    # return output, loss.item() / input_line_tensor.size(0)


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", unit="batch")

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        batch_size = inputs.size(0)
        hidden = model.init_hidden(batch_size)

        optimizer.zero_grad()

        # Forward pass
        outputs, _ = model(inputs, hidden)  # outputs: [B, T, V]

        # Reshape outputs & targets to compute loss
        outputs = outputs.view(-1, outputs.size(-1))   # [B*T, V]
        targets = targets.view(-1)                     # [B*T]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)
