import logging
import os
import random
import traceback

import torch
import tqdm
from torch.utils.data import DataLoader

from src import all_letters, device, idx2char, pad_idx
from src.model import LSTMModel
from src.Utils import sample

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def evaluating(decoder, max_length):
    """
    Evaluates the model by generating a specified number of predictions based on random starting sequences.

    Args:
        decoder: The model used for generating predictions.
        max_length (int): The maximum length of the generated sequence.

    Logs:
        Info-level logs include the evaluation progress and any errors encountered during the process.
    """
    logging.info("\n------------\n|   EVAL   |\n------------")

    try:
        while True:
            num_predictions = int(input("Enter the number of predictions: "))

            if num_predictions <= 0:
                logging.warning(
                    "Invalid input: number of predictions must be greater than 0.")
                continue
            if not os.path.exists("generated"):
                os.makedirs(f"generated")
            gen = open(f"generated/Output_{num_predictions}.txt", "w")

            predictions: list[str] = []

            for i in range(num_predictions):
                # Ensure all_letters is defined
                starting_letters = random.choice(all_letters)
                predicted = sample(decoder, max_length, starting_letters)
                predictions.append(predicted)
                gen.write(predicted + "\n")
                logging.info(f"Prediction {i + 1}: {predicted}")

            gen.close()

            logging.info("------------\n")

    except KeyboardInterrupt:
        logging.info("Evaluation process terminated by user.")
        logging.info("------------")
    except Exception as e:
        logging.error(f"An error occurred: {e} ")
        logging.error(f"An error occurred: {traceback.format_exc()} ")
        logging.info("------------")


def evaluate_model(model: LSTMModel, dataloader: DataLoader):
    model.eval()

    total_chars = 0
    correct_chars = 0

    with torch.no_grad():
        progress_bar = tqdm.tqdm(dataloader, desc="Evaluating", unit="batch")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))

            outputs, _ = model(inputs, hidden)  # (batch, seq_len, vocab_size)
            predictions = outputs.argmax(dim=-1)

            for pred_seq, target_seq in zip(predictions, targets):
                for pred_token, target_token in zip(pred_seq, target_seq):
                    if target_token.item() == pad_idx:
                        continue

                    target_char = idx2char[target_token.item()]
                    if target_char == '\n':
                        break

                    if pred_token.item() == target_token.item():
                        correct_chars += 1
                    total_chars += 1

            # Update postfix once per batch
            batch_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
            progress_bar.set_postfix(accuracy=f"{batch_accuracy:.4f}")

    final_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
    print(f"Evaluation Accuracy: {final_accuracy:.4f}")
    return final_accuracy
