import logging
import os
import random
import traceback

import torch.nn as nn

from src import all_letters
from src.model import LSTMModel
from src.Utils import sample

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def evaluating(decoder: LSTMModel, max_length: int, out_dir: str = "generated") -> None:
    """
    Repeatedly prompt user for how many samples to generate, then write them to file.
    Exit cleanly on Ctrl-C.
    """
    decoder.eval()
    os.makedirs(out_dir, exist_ok=True)

    try:
        while True:
            n = input("How many passwords to generate? ")
            try:
                n = int(n)
                if n <= 0:
                    print("Enter a positive number.")
                    continue
            except ValueError:
                print("Invalid integer.")
                continue

            path = os.path.join(out_dir, f"output_{n}.txt")
            with open(path, "a", encoding="utf-8") as f:
                for i in range(n):
                    seed = random.choice(all_letters)
                    pwd = sample(decoder, max_length)
                    f.write(pwd + "\n")
                    logging.info(f"Sample {i+1}/{n}: {pwd}")
            print(f"Wrote {n} samples to {path}\n")

    except KeyboardInterrupt:
        print("\nGeneration stopped by user.")
        logging.info("Evaluation interrupted.")
