import logging
import random
import time
from math import inf
from os import path
from typing import Tuple

import torch
import torch.nn as nn
import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from src import device, n_letters
from src.model import LSTMModel
from src.preprocessing import input_tensor, target_tensor
from src.Utils import all_letters, create_folder, time_since

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def training(decoder: nn.Module, n_epochs, dataloader, hidden_size, n_layers, lr, model_path, optimizer, criterion):
    logging.info("\n-----------\n|  TRAIN  |\n-----------")

    writer = SummaryWriter(
        log_dir=f"logs/logs_{hidden_size}_{n_layers}_{lr}_{n_epochs}")
    start = time.time()
    best_loss = float('inf')
    print_every = max(1, n_epochs // 100)

    train_path = create_folder(n_layers, hidden_size, lr, n_epochs)

    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        progress = tqdm.tqdm(
            dataloader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)

        for batch_inputs, batch_targets in progress:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            output, loss = train(
                decoder,
                batch_inputs,
                batch_targets,
                optimizer,
                criterion
            )
            total_loss += loss
            avg_loss_so_far = total_loss / (len(progress) + 1e-8)
            progress.set_postfix({"avg_loss": f"{avg_loss_so_far:.4f}"})

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        logging.info(f"Saved epoch {epoch} with loss {avg_loss:.4f}")
        torch.save(decoder.state_dict(), path.join(train_path, "iteration.pt"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(decoder.state_dict(), model_path)
            logging.info(
                f"New best model saved with loss {best_loss:.4f} at {model_path}")

        if epoch % print_every == 0:
            logging.info(
                f"{time_since(start)} ({epoch} {epoch / n_epochs * 100:.2f}%) Loss: {avg_loss:.4f}")

    writer.close()


def train(
    decoder: LSTMModel,
    input_tensor: torch.Tensor,    # shape: (batch_size, seq_len)
    target_tensor: torch.Tensor,   # shape: (batch_size, seq_len)
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module
) -> Tuple[torch.Tensor, float]:
    """
    Performs one training step: forward pass, loss computation, and backpropagation.

    Returns:
        output: The decoder output (batch, seq_len, vocab_size)
        loss:   Total scalar loss
    """
    decoder.train()
    optimizer.zero_grad()

    outputs, _ = decoder(input_tensor)
    loss = criterion(outputs.view(-1, outputs.size(-1)),
                     target_tensor.view(-1))

    loss.backward()
    optimizer.step()

    return outputs, loss.item()
