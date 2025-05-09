import logging
import random
import time
from math import inf
from os import path

import torch
import torch.nn as nn
import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from src import PAD_IDX, device, n_letters
from src.model import LSTMModel
from src.preprocessing import input_tensor, target_tensor
from src.Utils import all_letters, create_batch, create_folder, time_since

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def training(decoder: nn.Module, n_epochs, lines, hidden_size, n_layers, lr, model_path, optimizer, criteron):
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

    for iter in range(1, n_epochs + 1):
        total_loss = 0
        # Sample a random subset of lines for each epoch
        # random_lines = random.sample(lines, 10)
        lines = random.sample(lines, 1000)

        for index, line in tqdm.tqdm(enumerate(lines), desc="Training", total=len(lines)):
            output, loss = train(decoder, input_tensor(
                line).to(device), target_tensor(line).to(device), optimizer, criteron,)
            total_loss += loss

        avg_loss = total_loss / len(lines)
        writer.add_scalar("Loss/train", avg_loss, iter)

        logging.info(f"Saved iteration {iter} with loss {avg_loss:.4f}")
        torch.save(decoder.state_dict(), path.join(
            train_path, "iterations/", f"{iter}.pt"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(decoder.state_dict(), model_path)
            logging.info(
                f"New best model saved with loss {best_loss:.4f} at {model_path}")

        if iter % print_every == 0:
            logging.info(
                f"{time_since(start)} ({iter} {iter / n_epochs * 100:.2f}%) Loss: {avg_loss:.4f}")

    writer.close()


def train(decoder: LSTMModel, input_line_tensor: Tensor, target_line_tensor: Tensor, optimizer, criteron):
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
    hidden = decoder.init_hidden(1)

    decoder.zero_grad()
    loss = torch.tensor(0.0).to(device)
    output = None

    # Iterate through the input sequence
    for i in range(input_line_tensor.size(0)):
        output, _ = decoder(input_line_tensor[i].unsqueeze(0), hidden)
        l = criteron(output, target_line_tensor[i].to(device))
        loss += l

    loss.backward()
    optimizer.step()

    return output, loss.item()
    # return output, loss.item() / input_line_tensor.size(0)


def train_gen(model, dataloader, optimizer, criterion, device, clip=5.0):
    model.train()  # Set model to training mode
    total_loss = 0

    for inputs, targets, input_lengths in dataloader:
        # Move data to the same device as the model
        inputs, targets, input_lengths = inputs.to(
            device), targets.to(device), input_lengths.to(device)

        optimizer.zero_grad()  # Zero the gradients before backward pass

        # Initialize hidden state (batch size)
        hidden = model.init_hidden(inputs.size(1))

        # Forward pass through the LSTM model
        output, hidden = model(inputs, hidden, input_lengths)

        # Reshape the outputs and targets for the loss calculation
        # Flatten: (seq_len * batch, vocab_size)
        output = output.view(-1, output.size(-1))
        targets = targets.view(-1)                 # Flatten: (seq_len * batch)

        # Compute the loss
        loss = criterion(output, targets)

        # Backward pass: compute gradients
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip)  # Clip gradients

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

    # Return the average loss for this epoch
    return total_loss / len(dataloader)


def training_with_batches(
    decoder: LSTMModel,
    n_layers,
    hidden_size,
    lr,
    n_epochs: int,
    dataloader,
    optimizer,
    criterion,
    clip: float = 5.0,
):
    """
    Train the LSTM model, saving checkpoints every epoch and keeping the best one.

    Args:
        decoder: The LSTMModel to train.
        n_epochs: Number of epochs to train.
        dataloader: DataLoader yielding (inputs, targets, lengths).
        optimizer: Optimizer.
        criterion: Loss function.
        device: torch.device.
        clip: Max norm for gradient clipping.
        checkpoint_dir: Directory to save checkpoints.
    """
    logging.info("\n-----------\n|  TRAIN  |\n-----------")

    best_loss = float("inf")

    train_path = create_folder(n_layers, hidden_size, lr, n_epochs)

    decoder.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0

        with tqdm.tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}", unit="batch") as pbar:
            for inputs, targets, input_lengths in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)

                optimizer.zero_grad()
                hidden = decoder.init_hidden(inputs.size(1))

                # Forward
                output, hidden = decoder(inputs, hidden, input_lengths)
                seq_len, batch_size, vocab_size = output.size()

                # Crop targets to match output length
                targets = targets[:seq_len, :]

                # Flatten for loss
                output = output.view(-1, vocab_size)
                targets = targets.contiguous().view(-1)

                # Compute loss
                loss = criterion(output, targets)
                loss.backward()

                # Clip and step
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch}/{n_epochs} — Loss: {avg_loss:.4f}")

        # 1) Save checkpoint for this epoch
        iteration_ckpt = path.join(
            train_path, "iterations/", f"{epoch}.pt")
        torch.save(decoder.state_dict(), iteration_ckpt)
        logging.info(f"Saved checkpoint: {iteration_ckpt}")

        # 2) If this is the best loss so far, save a “best” checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = path.join(train_path, "best.pt")
            torch.save(decoder.state_dict(), best_ckpt)
            logging.info(f"New best loss! Saved best checkpoint: {best_ckpt}")

    logging.info("Training complete.")
