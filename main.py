from __future__ import division, print_function, unicode_literals

import logging
from argparse import ArgumentParser
from io import open
from os import makedirs, path
import os

import torch
import torch.nn as nn
import unidecode
from dataset_info import dataset_info, split_data
from src import (
    BEST_MODEL_PATH,
    FILENAME,
    LAST_EPOCH_MODEL_PATH,
    device,
    hidden_size_default,
    l_r,
    max_epochs_default,
    n_layers_default,
    n_letters,
)
from src.eval import evaluating
from src.model import RNN
from src.test import testing
from src.train import training
from src.Utils import get_lines, get_mean_size, split

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--Data",
        default="Dataset/all.txt",
        type=str,
        help="Path to training data",
    )
    parser.add_argument(
        "-te",
        "--trainEval",
        default="train",
        type=str,
        choices=["train", "eval", "test"],
        help="Mode: train, eval, or test",
    )
    parser.add_argument(
        "-r", "--run", default="rnnGeneration", type=str, help="Model save file name"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="rnn.pt",
        type=str,
        help="Path to model for training/evaluation/testing",
    )
    parser.add_argument(
        "-n", default=10000, type=int, help="Number of samples to generate"
    )
    parser.add_argument("--ml", default=12, type=int, help="Max characters per name")
    parser.add_argument(
        "-s", default=0.7, type=float, help="Training set percentage (default: 70%)"
    )
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument(
        "--bidirectional", default=True, type=bool, help="Use bidirectional model"
    )
    parser.add_argument("--max_epochs", default=30000, type=int)
    parser.add_argument("-lr","--learning_rate", default=0.001, type=float)
    parser.add_argument(
        "-p", "--percent", default=15, type=float, help="Percentage of names to test"
    )
    parser.add_argument(
        "-tw","--type_weight", default="last_epoch", type=str, help="Choose either last_epoch or best_loss"
    )
    args = parser.parse_args()

    # Load dataset
    try:
        file = unidecode.unidecode(open(args.Data).read())
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    dataset_info(FILENAME)

    lines = get_lines(FILENAME)

    # Split dataset
    train_set, test_set, dev_set = split_data(args.s, lines) # percentage of split args.s

    logging.info(f"Training set size: {len(train_set)}")
    logging.info(f"Test set size: {len(test_set)}")
    logging.info(f"Dev set size: {len(dev_set)}")

    # Hyperparameters
    max_length = args.ml if args.ml > 0 else get_mean_size(train_set) + 5
    learning_rate = args.learning_rate if args.learning_rate is not None else l_r
    hidden_size = (args.hidden_size if args.hidden_size is not None else hidden_size_default)
    n_layers = args.num_layers if args.num_layers is not None else n_layers_default
    max_epochs = args.max_epochs if args.max_epochs is not None else max_epochs_default

    print("--------------------------------------------------------------------")

    # Initialize the Model (decoder)
    decoder = RNN(n_letters, hidden_size, n_letters, n_layers).to(device)
    decoder.summary(n_letters, hidden_size, n_letters, n_layers)

    model_path= (
        f"{args.run}_{n_layers}_{hidden_size}_{learning_rate:.4f}_{max_epochs}.pt"
    )

    makedirs("models", exist_ok=True)
    last_epoch_path= os.path.join(*["models/", LAST_EPOCH_MODEL_PATH + "_" + model_path + ".pt"])
    best_model_path = os.path.join(*["models/", BEST_MODEL_PATH + "_" + model_path + ".pt"])

    chosen_model_path = last_epoch_path if args.type_weight == "last_epoch" else best_model_path

    if args.trainEval == "train":
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        criteron = nn.CrossEntropyLoss()
        decoder.train()
        training(
            decoder,
            max_epochs,
            train_set,
            hidden_size,
            n_layers,
            learning_rate,
            last_epoch_path,
            best_model_path,
            optimizer,
            criteron,
        )
        torch.save(decoder.state_dict(), last_epoch_path)
        logging.info(f"Model saved at {last_epoch_path}")

    elif args.trainEval == "eval":
        try:
            decoder.load_state_dict(torch.load(chosen_model_path))
            decoder.to(device).eval()
            evaluating(decoder, max_length)
        except Exception as e:
            logging.error(f"Failed to load model for evaluation: {e}")
    elif args.trainEval == "test":
        try:
            decoder.load_state_dict(torch.load(chosen_model_path))
            decoder.to(device).eval()
            testing(decoder, args.n, test_set, args.percent, max_length)
        except Exception as e:
            logging.error(f"Failed to load model for testing: {e}")
    else:
        logging.error("Invalid --trainEval option. Choose from train/eval/test.")


if __name__ == "__main__":
    main()
