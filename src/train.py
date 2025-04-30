import logging
import random
import time
from math import inf

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src import device, n_letters
from src.preprocessing import input_tensor, target_tensor
from src.Utils import all_letters, time_since

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def training(decoder, n_epochs, lines, hidden_size, n_layers, lr, model_path, optimizer, criteron):
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

    for iter in range(1, n_epochs + 1):
        total_loss = 0
        # Sample a random subset of lines for each epoch
        random_lines = random.sample(lines, 10)

        for index, line in enumerate(random_lines, start=1):
            output, loss = train(decoder, input_tensor(
                line).to(device), target_tensor(line).to(device), optimizer, criteron)
            total_loss += loss

        avg_loss = total_loss / len(random_lines)
        writer.add_scalar("Loss/train", avg_loss, iter)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(decoder.state_dict(), model_path)
            logging.info(
                f"New best model saved with loss {best_loss:.4f} at {model_path}")

        if iter % print_every == 0:
            logging.info(
                f"{time_since(start)} ({iter} {iter / n_epochs * 100:.2f}%) Loss: {avg_loss:.4f}")

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
    hidden = decoder.init_hidden().to(device)

    decoder.zero_grad()
    loss = torch.tensor(0.0, requires_grad=True).to(device)
    output = None

    # Iterate through the input sequence
    for i in range(input_line_tensor.size(0)):
        output, hidden = decoder(input_line_tensor[i], hidden)
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
