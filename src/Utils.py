import math
import os
import string
import sys
import time
import unicodedata

import questionary
import torch

from src import FILENAME_TEST, FILENAME_TRAIN, all_letters, device, n_letters
from src.preprocessing import input_tensor


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


# Read a file and split into lines
def read_lines(filename):
    with open(filename, encoding="utf-8") as some_file:
        return [unicode_to_ascii(line.strip().lower()) for line in some_file]


def get_lines(f):
    lines = read_lines(f)
    print("lines: ", len(lines), " -> ", f)
    return lines


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since_start(since):
    now = time.time()
    s = now - since
    return "%s" % (as_minutes(s))


def progress_percent(totalNames, start, names, p, samplesGenerated):
    bar_len = 50
    filled_len = int(round(bar_len * names / float(totalNames)))
    percents = round(100.0 * names / float(totalNames), 1)
    nNames = int(p / 100 * totalNames)

    if filled_len == 0:
        bar = ">" * filled_len + " " * (bar_len - filled_len)
    else:
        bar = "=" * (filled_len - 1) + ">" + " " * (bar_len - filled_len)

    sys.stdout.write(
        "[%s] %s%s names founded among %d samples generated (%d of %d names) on %s (goal = %.1f%% = %d names)\r"
        % (
            bar,
            percents,
            "%",
            samplesGenerated,
            names,
            totalNames,
            time_since_start(start),
            p,
            nNames,
        )
    )
    sys.stdout.flush()


def progress(total, acc, start, epoch, l):
    bar_len = 50
    filled_len = int(round(bar_len * epoch / float(total)))
    percents = round(100.0 * epoch / float(total), 1)

    if filled_len == 0:
        bar = ">" * filled_len + " " * (bar_len - filled_len)
    else:
        bar = "=" * (filled_len - 1) + ">" + " " * (bar_len - filled_len)

    sys.stdout.write(
        "[%s] %s%s epoch: %d acc: %.3f %% and testing size = %d names => coverage of %.3f %% on %s \r"
        % (
            bar,
            percents,
            "%",
            epoch,
            (100 * acc / epoch),
            l,
            (100 * acc / l),
            time_since_start(start),
        )
    )
    sys.stdout.flush()


def get_mean_size(listData):
    mean = 0
    for word in listData:
        mean = mean + len(word)

    return int(mean / len(listData))


def sample(decoder, max_length, start_letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    with torch.no_grad():  # no need to track history in sampling

        hidden = decoder.init_hidden_random(len(start_letters))

        if len(start_letters) > 1:
            for i in range(len(start_letters)):
                input = input_tensor(start_letters[i])
                # print(start_letters[i], ' ', hidden)
                output, hidden = decoder(
                    input[0].to(device).unsqueeze(0), hidden)

            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                return start_letters

            letter = all_letters[topi]
            input = input_tensor(letter)
        else:
            input = input_tensor(start_letters)

        output_name = start_letters

        for i in range(max_length):
            output, hidden = decoder(input[0].to(device).unsqueeze(0), hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = input_tensor(letter)

        return output_name


def split(rate, lines):
    names = []

    # for letter in string.ascii_uppercase:
    for letter in string.ascii_letters:
        names_letter = []
        for line in lines:
            if len(line) > 0 and line[0] == letter:
                names_letter.append(line)
        if len(names_letter) > 0:
            names.append(names_letter)

    print("split names: ", len(names))
    names_traing = []
    names_testing = []
    for names_letter in names:
        length = len(names_letter)

        index = int(length * rate)

        training = names_letter[:index]
        testing = names_letter[index:]

        names_traing.append(training)
        names_testing.append(testing)

    f = open(FILENAME_TRAIN, "w")
    for names_letter in names_traing:
        for names in names_letter:
            f.write(names + "\n")
    f.close()

    f = open(FILENAME_TEST, "w")
    for names_letter in names_testing:
        for names in names_letter:
            f.write(names + "\n")
    f.close()

    return names_traing, names_testing


def choose_model() -> str:
    """
    Choose the model to be used from the ``models``.

    Returns:
        str: The chosen model path.
    """
    if not os.path.exists('models'):
        raise FileNotFoundError("The 'models' directory does not exist.")

    models = [os.path.join(dir, f) for dir, _, files in os.walk(
        'models') for f in files if f.endswith('.pt')]

    if not models:
        raise FileNotFoundError(
            "No model files found in the 'models' directory.")

    choice = questionary.select(
        "Choose a model to use:",
        choices=models
    ).ask()
    return choice


def get_folder_path(num_layers: int, hidden_size: int, learning_rate: float, epochs: int) -> str:
    """
    Generate a folder path based on the model parameters.

    Args:
        num_layers (int): Number of layers in the model.
        hidden_size (int): Size of the hidden layer.
        learning_rate (float): Learning rate for the model.
        epochs (int): Number of epochs for training.

    Returns:
        str: The generated folder path.
    """
    return f"models/lstm_{num_layers}_{hidden_size}_{learning_rate}_{epochs}"


def create_folder(num_layers: int, hidden_size: int, learning_rate: float, epochs: int) -> str:
    """
    Create a folder for the model based on the parameters.

    Args:
        num_layers (int): Number of layers in the model.
        hidden_size (int): Size of the hidden layer.
        learning_rate (float): Learning rate for the model.
        epochs (int): Number of epochs for training.

    Returns:
        str: The path to the created folder.
    """
    folder_path = get_folder_path(
        num_layers, hidden_size, learning_rate, epochs)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, "iterations"), exist_ok=True)
    return folder_path


def get_model_name(num_layers: int, hidden_size: int, learning_rate: float, epochs: int, iteration: int = None, is_best: bool = False) -> str:
    num_layers_str = str(num_layers)
    hidden_size_str = str(hidden_size)
    learning_rate_str = str(learning_rate).replace('.', '_')
    epochs_str = str(epochs)
    iteration_str = str(iteration) if iteration is not None else ""
    is_best_str = "_best" if is_best else ""

    return f"lstm_{num_layers_str}_{hidden_size_str}_{learning_rate_str}_{epochs_str}_{iteration_str}_{is_best_str}.pt"


def extract_params(model_path: str) -> tuple[int, int, float, int, int, bool]:
    """
    Extract parameters from the model filename.

    Args:
        model_path (str): The path to the model file.

    Returns:
        tuple: A tuple containing the extracted parameters.
    """

    params = model_path.split('\\')[1]
    parts = params.split('_')[1:]
    print(parts)

    num_layers = int(parts[0])
    hidden_size = int(parts[1])
    learning_rate = float(parts[2])
    epochs = int(parts[3])
    iteration = int(parts[4]) if len(parts) > 5 else None
    is_best = "_best" in model_path

    return num_layers, hidden_size, learning_rate, epochs, iteration, is_best
