import math
import os
import random
import string
import sys
import time
import unicodedata

import questionary
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence

from src import (FILENAME_TEST, FILENAME_TRAIN, PAD_TOKEN, all_letters,
                 char2idx, device, idx2char, n_letters, pad_idx)
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


def encode_password(pw: str):
    if not pw.endswith('\n'):
        pw += '\n'
    input_seq = [char2idx[ch] for ch in pw[:-1]]
    target_seq = [char2idx[ch] for ch in pw[1:]]
    return input_seq, target_seq


def collate_fn(batch):
    inputs, targets = zip(*batch)    # This is where the unpacking fails
    inputs_padded = pad_sequence(
        inputs, batch_first=True, padding_value=pad_idx)
    targets_padded = pad_sequence(
        targets, batch_first=True, padding_value=pad_idx)

    # Check if there are duplicate inputs within the batch
    input_set = set(tuple(input_seq.tolist()) for input_seq in inputs_padded)
    if len(input_set) != len(inputs_padded):
        print("Warning: Duplicate sequences detected in batch!")

    return inputs_padded, targets_padded


def check_overlap_between_batches(dataloader):
    prev_batch = None
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if prev_batch is not None:
            # Check if any sequence from the previous batch appears in the current batch
            overlap = set(tuple(input_seq.tolist()) for input_seq in prev_batch).intersection(
                set(tuple(input_seq.tolist()) for input_seq in inputs)
            )
            if overlap:
                print(
                    f"Overlap detected between batch {batch_idx-1} and batch {batch_idx}")
        prev_batch = inputs  # Store current batch for next iteration


def visualize_batches(dataloader):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        for i, input_seq in enumerate(inputs):
            print(f"Input sequence {i}: {input_seq.tolist()}")
        print("\n")


def generate_passwords(model, n, max_len=20):
    """
    Generates `n` passwords using the trained model.

    Args:
        model: The trained LSTM model.
        n (int): The number of passwords to generate.
        char2idx (dict): A dictionary mapping characters to indices.
        idx2char (dict): A dictionary mapping indices to characters.
        max_len (int): Maximum length of generated passwords (including the EOS token).
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        List[str]: A list of `n` generated passwords.
    """
    model.eval()  # Set the model to evaluation mode
    generated_passwords = []

    # Loop to generate `n` passwords
    progress = tqdm.tqdm(range(n), desc="Generating passwords")
    for _ in progress:
        # Start with a random character (usually a token for password start)
        start_idx = char2idx[random.choice([
            ch for ch in char2idx if ch not in ('\n', PAD_TOKEN)
        ])]
        input_seq = torch.tensor([[start_idx]], dtype=torch.long).to(device)

        # Start with initial hidden state for batch size 1
        hidden = model.init_hidden(1)
        password = ''

        for _ in range(max_len):
            output, hidden = model(input_seq, hidden)

            # Get the predicted character (softmax could be used for better sampling)
            predicted_idx = torch.argmax(output[0, -1, :]).item()

            # Skip if the predicted character is <PAD>
            if predicted_idx == pad_idx:
                continue

            # Convert the index back to character
            predicted_char = idx2char[predicted_idx]

            # Stop if the predicted character is <EOS>
            if predicted_char == '\n':
                break

            password += predicted_char

            # Prepare the next input (the predicted character as input to the next timestep)
            input_seq = torch.tensor(
                [[predicted_idx]], dtype=torch.long).to(device)

        password = password.strip()

        progress.set_postfix({"Generated Password": password})

        generated_passwords.append(password)

        # print(len(generated_passwords), password)

    file = open(f"generated\output_{n}.txt", "w")
    file.writelines([f"{password}\n" for password in generated_passwords])
    file.close()

    return generated_passwords


def check_matches(passwords: list[str], valid_names: list[str]) -> float:
    """
    Check how many generated passwords match with valid names.

    Args:
        passwords (list[str]): List of generated passwords.
        valid_names (list[str]): List of valid names to check against.

    Returns:
        float: Percentage of matches.
    """
    matches = sum(1 for password in passwords if password in valid_names)
    print(
        f"Matches: {matches} out of {len(passwords)} Percentage: {matches / len(passwords) * 100:.2f}%")
    return (matches / len(passwords)) * 100 if passwords else 0.0
