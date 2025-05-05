import json
import os
import random
from typing import Dict, List, Tuple

import torch

from src.Utils import unicode_to_ascii


def letter_frequency(lines: List[str]) -> Dict[str, int]:
    """
    Calculate the frequency of each letter in the dataset.
    """
    freq = {}
    for line in lines:
        for letter in line.strip():
            if letter in freq:
                freq[letter] += 1
            else:
                freq[letter] = 1
    # Sort the dictionary by keys
    freq = dict(sorted(freq.items()))
    return freq


def min_length(length: int, line: str) -> bool:
    """
    Check if the line has at least the specified length.
    """
    return len(line) >= length


def has_special_character(line: str) -> bool:
    """
    Check if the line has at least one special character.
    """
    return any(not c.isalnum() and c != ' ' for c in line)


def has_number(line: str) -> bool:
    """
    Check if the line has at least one digit.
    """
    return any(c.isdigit() for c in line)


def has_uppercase(line: str) -> bool:
    """
    Check if the line has at least one uppercase character.
    """
    return any(c.isupper() for c in line)


def has_lowercase(line: str) -> bool:
    """
    Check if the line has at least one lowercase character.
    """
    return any(c.islower() for c in line)


def avgNameLength(lines: List[str]) -> float:
    """
    Calculate the average name length.
    """
    return sum(len(line.strip()) for line in lines) / len(lines)


def split_data(rate: float, lines: List[str], seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split data into train, test, and dev sets with reproducibility.

    Args:
        rate (float): Proportion of training data (0 < rate < 1).
        lines (List[str]): Input lines.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[List[str], List[str], List[str]]: Train, test, and dev sets.
    """
    assert 0 < rate < 1, "Training rate must be between 0 and 1"

    os.makedirs('Dataset', exist_ok=True)
    train_path = 'Dataset/train.txt'
    dev_path = 'Dataset/dev.txt'
    test_path = 'Dataset/test.txt'

    if os.path.exists(train_path) and os.path.exists(dev_path) and os.path.exists(test_path):
        with open(train_path, 'r') as f:
            train = [line.strip() for line in f]

        with open(dev_path, 'r') as f:
            dev = [line.strip() for line in f]

        with open(test_path, 'r') as f:
            test = [line.strip() for line in f]
        return train, test, dev

    passwords = list(set(lines))
    passwords = [p for p in passwords if p.strip()]  # keep newline, skip blank

    random.seed(seed)
    random.shuffle(passwords)

    n_total = len(passwords)
    n_train = int(n_total * rate)
    n_dev = n_test = (n_total - n_train) // 2

    train = passwords[:n_train]
    dev = passwords[n_train:n_train + n_dev]
    test = passwords[n_train + n_test:]

    for filename, split_lines in zip((train_path, dev_path, test_path), (train, dev, test)):
        with open(filename, 'w') as f:
            f.write(''.join(split_lines))

    return train, test, dev


def dataset_info(path: str) -> None:
    """
    Print information about the dataset.
    """
    # Read the file
    with open(path, 'r') as file:
        lines = list(set(file.readlines()))

    split_data(0.7, lines)

    lines_longer_than_8 = []
    lines_with_special = []
    lines_with_numbers = []
    lines_with_uppercase = []
    lines_with_lowercase = []

    print(f"File lines: {len(lines)}")

    jsonFile = open("Dataset/letter_frequency.json", "w")
    jsonFile.write(json.dumps(letter_frequency(lines)))
    jsonFile.close()

    for line in lines:
        # Remove the newline character
        line = line.strip()
        if len(line) > 0:
            if min_length(8, line):
                lines_longer_than_8.append(line)
            if has_special_character(line):
                lines_with_special.append(line)
            if has_number(line):
                lines_with_numbers.append(line)
            if has_uppercase(line):
                lines_with_uppercase.append(line)
            if has_lowercase(line):
                lines_with_lowercase.append(line)
    total_lines = len(lines)

    avg_name_length = avgNameLength(lines)

    print(f"Average name length = {avg_name_length:.2f}")

    with open('Dataset/dataset_info.csv', 'w') as csvfile:
        csvfile.write('condition;number_of_lines;percentage\n')
        csvfile.write(
            f'lines_longer_than_8;{len(lines_longer_than_8)};{(len(lines_longer_than_8) / total_lines) * 100:.2f}\n')
        csvfile.write(
            f'lines_with_special;{len(lines_with_special)};{(len(lines_with_special) / total_lines) * 100:.2f}\n')
        csvfile.write(
            f'lines_with_numbers;{len(lines_with_numbers)};{(len(lines_with_numbers) / total_lines) * 100:.2f}\n')
        csvfile.write(
            f'lines_with_uppercase;{len(lines_with_uppercase)};{(len(lines_with_uppercase) / total_lines) * 100:.2f}\n')
        csvfile.write(
            f'lines_with_lowercase;{len(lines_with_lowercase)};{(len(lines_with_lowercase) / total_lines) * 100:.2f}\n')
        csvfile.write(
            f'average_name_length;{avg_name_length:.2f};\n')


if __name__ == '__main__':
    dataset_info('Dataset/all.txt')
