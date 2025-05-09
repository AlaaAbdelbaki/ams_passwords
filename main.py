from __future__ import division, print_function, unicode_literals

import logging
from argparse import ArgumentParser
from io import open
from os import makedirs, path

import torch
import torch.nn as nn
import unidecode
from torch.utils.data import DataLoader

from dataset_info import dataset_info, split_data
from src import (FILENAME, MODEL_PATH, PAD_IDX, device, hidden_size_default,
                 l_r, max_epochs_default, n_layers_default, n_letters)
from src.dataloader import PasswordDataset
from src.eval import evaluating
from src.model import RNN, LSTMModel
from src.test import testing
from src.train import training, training_with_batches
from src.Utils import (all_letters, choose_model, collate_fn, extract_params,
                       get_folder_path, get_lines, get_mean_size)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--trainingData",
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
        default="models/rnn.pt",
        type=str,
        help="Path to model for training/evaluation/testing",
    )
    parser.add_argument(
        "--n", default=10000, type=int, help="Number of samples to generate"
    )
    parser.add_argument("--ml", default=10, type=int,
                        help="Max characters per name")
    parser.add_argument(
        "--s", default=0.7, type=float, help="Training set percentage (default: 70%)"
    )
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--embde_size", default=32, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument(
        "--bidirectional", default=True, type=bool, help="Use bidirectional model"
    )
    parser.add_argument("--max_epochs", default=100000, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument(
        "-p", "--percent", default=15, type=float, help="Percentage of names to test"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training (default: 32)",
    )

    args = parser.parse_args()

    # Load dataset
    try:
        file = unidecode.unidecode(open(args.trainingData).read())
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    dataset_info(FILENAME)

    lines = get_lines(FILENAME)

    # Split dataset
    train_set, test_set, dev_set = split_data(args.s, lines)

    logging.info(f"Training set size: {len(train_set)}")
    logging.info(f"Test set size: {len(test_set)}")
    logging.info(f"Dev set size: {len(dev_set)}")

    # Hyperparameters
    max_length = args.ml if args.ml > 0 else get_mean_size(train_set)
    learning_rate = args.learning_rate if args.learning_rate is not None else l_r
    hidden_size = (
        args.hidden_size if args.hidden_size is not None else hidden_size_default)
    n_layers = args.num_layers if args.num_layers is not None else n_layers_default
    max_epochs = args.max_epochs if args.max_epochs is not None else max_epochs_default
    embedding_dim = args.embde_size if args.embde_size is not None else 32
    batch_size = 32 if args.batch_size is None else args.batch_size

    print("--------------------------------------------------------------------")

    # Initialize the Model (decoder)
    # decoder = RNN(n_letters, hidden_size, n_letters, n_layers).to(device)
    decoder = LSTMModel(
        n_letters=n_letters,          # Input vocab size
        embedding_dim=embedding_dim,   # Size of each embedding vector
        hidden_size=hidden_size,       # LSTM hidden size
        n_layers=n_layers,         # LSTM layers
    ).to(device)
    decoder.summary(n_letters, hidden_size, n_letters, n_layers)

    model_filename = (
        f"{args.run}_{n_layers}_{hidden_size}_{learning_rate:.4f}_{max_epochs}.pt"
    )
    model_path = path.join(path.dirname(args.model), model_filename)

    makedirs(path.dirname(model_path), exist_ok=True)

    if args.trainEval == "train":
        dataset = PasswordDataset(train_set, all_letters, n_letters)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)
        model_path = path.join(get_folder_path(
            n_layers, hidden_size, learning_rate, max_epochs), 'model.pt')
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        criteron = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        decoder.train()
        training_with_batches(
            decoder,
            n_layers,
            hidden_size,
            learning_rate,
            max_epochs,
            dataloader,
            optimizer,
            criteron,
        )
        # training(
        #     decoder,
        #     max_epochs,
        #     train_set,
        #     hidden_size,
        #     n_layers,
        #     learning_rate,
        #     model_path,
        #     optimizer,
        #     criteron,
        # )
        torch.save(decoder.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")

    elif args.trainEval == "eval":
        # ───────────────────────────────────
        # EVAL (generation)
        model_file = choose_model()
        num_layers, hidden, _, __, ___, ____ = extract_params(model_file)
        decoder = LSTMModel(
            n_letters=n_letters,
            embedding_dim=embedding_dim,
            hidden_size=hidden,
            n_layers=num_layers
        ).to(device)
        decoder.load_state_dict(torch.load(model_file, map_location=device))
        decoder.eval()

        # call your interactive evaluator on the eval_set if you need to seed from it,
        # otherwise this will just prompt you for generation.
        evaluating(decoder, max_length)

    elif args.trainEval == "test":
        # ───────────────────────────────────
        # TEST (per‑example scoring or hold‑out generation)
        model_file = choose_model()
        num_layers, hidden, _, __, ___, ____ = extract_params(model_file)
        decoder = LSTMModel(
            n_letters=n_letters,
            embedding_dim=embedding_dim,
            hidden_size=hidden,
            n_layers=num_layers
        ).to(device)
        decoder.load_state_dict(torch.load(model_file, map_location=device))
        decoder.eval()

        # testing() should take (model, n, test_set, percent, max_length)
        # where test_set is your list of held‑out passwords
        testing(decoder, args.n, test_set, args.percent, max_length)
    else:
        logging.error(
            "Invalid --trainEval option. Choose from train/eval/test.")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
