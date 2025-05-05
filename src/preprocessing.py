# One-hot matrix of first to last letters (not including EOS) for input
"""In training a model for sequence prediction, especially in tasks like language modeling, text generation, or neural machine translation, the input tensor and target tensor play important roles in defining the structure of the data being fed into the model. Here's why we use them:

Input Tensor:
Purpose: The input tensor is a representation of the input sequence (such as a sentence or word) that will be passed to the model for processing. In this case, each character in the input string is converted into a one-hot encoded vector.
One-hot Encoding: One-hot encoding is used to represent categorical data (characters in this case) as binary vectors. Each character is represented as a vector with all zeros except for a 1 at the index corresponding to that character in a predefined list (all_letters).
Example: If the line is "hello", the corresponding one-hot vectors would have 1 at the positions of "h", "e", "l", etc., in the all_letters list.
Why We Use It: The model uses this input tensor to learn the relationship between the current character and the next one in the sequence. The input tensor helps the model "see" each character in a form it can understand (as numbers, not raw text).
Target Tensor:
Purpose: The target tensor represents the desired output for the model to predict, based on the input sequence. In sequence-to-sequence tasks (like text generation), we predict the next character in the sequence, which is the target.
Shifted Sequence: The target tensor typically holds the sequence of characters that the model is trying to predict, shifted by one position. For example, for an input sequence "hello", the target sequence might be "ello_", where the underscore (_) represents the End-Of-Sequence (EOS) token.
Example: If the line is "hello", the target sequence would be the indexes of ["e", "l", "l", "o", EOS], where each character is converted into its corresponding index in all_letters.
Why We Use It: The model learns to predict the next character in the sequence by comparing its output to the target tensor. The difference between the model's output and the target is used to compute the loss, which is then minimized during training to improve the model's performance.
Workflow:
Input Tensor: The model takes the input tensor (one-hot encoded characters) and processes it through layers like an RNN or LSTM.
Prediction: The model generates an output based on the input tensor (usually a probability distribution over the possible characters for the next position).
Target Tensor: The model's output is then compared to the target tensor, which represents the true next character. The comparison gives us a loss value.
Optimization: The model uses the loss to adjust its parameters via backpropagation, minimizing the error and improving future predictions.
In summary:

The input tensor helps the model understand what it’s supposed to process (the sequence of characters).
The target tensor represents what the model should output after processing the input (the expected next characters).
These tensors work together to train the model in sequence-based tasks."""
import torch

from src import all_letters, char2idx, idx2char, n_letters


def input_tensor(line):
    """
    Converts a string (line) into a tensor representation where each character is one-hot encoded.

    Args:
    - line (str): The input string that needs to be converted into a tensor.

    Returns:
    - tensor (Tensor): A tensor of shape (len(line), 1, n_letters) representing the one-hot encoded characters.
                        Each character in the input string is encoded as a one-hot vector with length `n_letters`.
    """
    tensor = torch.zeros(len(line), 1, n_letters,
                         dtype=torch.float32)  # .long()
    # print(f"Encoding line: {line}")
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    if len(tensor[0]) == 0:
        print(f"Error: tensor[0] is empty: {line}")
    return tensor


# LongTensor of second letter to end (EOS) for target
def target_tensor(line):
    """
    Converts a string (line) into a tensor representing the target sequence for training.
    The target sequence is created by shifting the input string by one character, with the End-Of-Sequence (EOS) token added at the end.

    Args:
    - line (str): The input string that needs to be converted into a target tensor.

    Returns:
    - tensor (Tensor): A LongTensor representing the target sequence. Each element in the tensor is the index of the corresponding character in `all_letters`.
                        The EOS token is represented by `n_letters - 1`.
    """
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)
