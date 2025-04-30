import json
import os
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


def split_data(rate: float, lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Split the data into training and testing sets.
    """

    if os.path.exists('Dataset/train.txt') and os.path.exists('Dataset/test.txt') and os.path.exists('Dataset/dev.txt'):
        print("Dataset already split. Skipping split.")

        train_file = open('Dataset/train.txt', 'r')
        test_file = open('Dataset/test.txt', 'r')
        dev_file = open('Dataset/dev.txt', 'r')

        train_values = set(l for l in train_file.readlines())
        test_values = set(l for l in test_file.readlines())
        dev_values = set(l for l in dev_file.readlines())

        train_file.close()
        test_file.close()
        dev_file.close()

        return train_values, test_values, dev_values

    lines = list(set(lines))
    dev_test_rate = (1 - rate) / 2
    train, test, dev = torch.utils.data.random_split(
        lines, [rate, dev_test_rate, dev_test_rate])

    train_file = open('Dataset/train.txt', 'w')
    test_file = open('Dataset/test.txt', 'w')
    dev_file = open('Dataset/dev.txt', 'w')

    train_values = set([unicode_to_ascii(l)
                       for l in train if len(unicode_to_ascii(l)) > 0])
    test_values = set([unicode_to_ascii(l)
                      for l in test if len(unicode_to_ascii(l)) > 0])
    dev_values = set([unicode_to_ascii(l)
                     for l in dev if len(unicode_to_ascii(l)) > 0])

    train_file.writelines(l for l in train_values)
    test_file.writelines(l for l in test_values)
    dev_file.writelines(l for l in dev_values)

    train_file.close()
    test_file.close()
    dev_file.close()

    return train_values, test_values, dev_values


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
