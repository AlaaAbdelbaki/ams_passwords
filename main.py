from __future__ import division, print_function, unicode_literals

import logging
from argparse import ArgumentParser
from io import open
from os import makedirs, path

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import unidecode
from torch.utils.data import DataLoader

from dataset_info import dataset_info, split_data
from src import (FILENAME, MODEL_PATH, device, hidden_size_default, l_r,
                 max_epochs_default, n_layers_default, n_letters)
from src.dataset import PasswordDataset
from src.eval import evaluating
from src.model import RNN, LSTMModel
from src.test import testing
from src.train import training
from src.Utils import (choose_model, collate_fn, extract_params,
                       get_folder_path, get_lines, get_mean_size,
                       get_model_name)

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
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument(
        "--bidirectional", default=True, type=bool, help="Use bidirectional model"
    )
    parser.add_argument("--max_epochs", default=100000, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument(
        "-p", "--percent", default=15, type=float, help="Percentage of names to test"
    )
    parser.add_argument('--best', action='store_true',
                        help='Use the best model for evaluation/testing')

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
    hidden_size = args.hidden_size if args.hidden_size is not None else hidden_size_default
    n_layers = args.num_layers if args.num_layers is not None else n_layers_default
    max_epochs = args.max_epochs if args.max_epochs is not None else max_epochs_default
    use_best_model = args.best if args.best is not None else False

    print("--------------------------------------------------------------------")

    # Initialize the Model (decoder)
    # decoder = RNN(n_letters, hidden_size, n_letters, n_layers).to(device)
    decoder = LSTMModel(n_letters, hidden_size, n_layers, n_letters).to(device)
    decoder.summary(n_letters, hidden_size, n_letters, n_layers)

    model_filename = (
        f"{args.run}_{n_layers}_{hidden_size}_{learning_rate:.4f}_{max_epochs}.pt"
    )
    model_path = path.join(path.dirname(args.model), model_filename)

    makedirs(path.dirname(model_path), exist_ok=True)

    if args.trainEval == "train":
        print('ℹ️ Training with the following parameters:')
        print(f'  - Hidden size: {hidden_size}')
        print(f'  - Number of layers: {n_layers}')
        print(f'  - Learning rate: {learning_rate}')
        print(f'  - Max epochs: {max_epochs}')
        print(f'  - Model path: {model_path}')

        dataset = PasswordDataset('Dataset/train.txt')
        dataloader = DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=True,
                                shuffle=True, collate_fn=collate_fn, persistent_workers=None, prefetch_factor=None)

        model_path = path.join(get_folder_path(
            n_layers, hidden_size, learning_rate, max_epochs), 'model.pt')

        optimizer = torch.optim.Adam(decoder.to(
            device).parameters(), lr=learning_rate)
        criteron = nn.CrossEntropyLoss(ignore_index=-100)
        decoder.train()
        training(
            decoder,
            max_epochs,
            dataloader,
            hidden_size,
            n_layers,
            learning_rate,
            # model_path,
            optimizer,
            criteron,
        )
        torch.save(decoder.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")

    elif args.trainEval == "eval":
        # try:
        model = choose_model()
        num_layers, hidden, _, __, ___, ____ = extract_params(model)
        decoder = LSTMModel(
            n_letters, hidden, num_layers, n_letters,).to(device)

        decoder.load_state_dict(torch.load(model))
        decoder.to(device).eval()
        evaluating(decoder, max_length)
        # except Exception as e:
        #     logging.error(f"Failed to load model for evaluation: {e}")
    elif args.trainEval == "test":
        # try:
        model = choose_model()
        num_layers, hidden, _, __, ___, ____ = extract_params(model)
        decoder = LSTMModel(
            n_letters, hidden, num_layers, n_letters).to(device)
        decoder.load_state_dict(torch.load(model))
        decoder.to(device).eval()
        testing(decoder, args.n, test_set, args.percent, max_length)
        # except Exception as e:
        #     logging.error(f"Failed to load model for testing: {e}")
    else:
        logging.error(
            "Invalid --trainEval option. Choose from train/eval/test.")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
