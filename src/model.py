import logging

import torch
import torch.nn as nn

from src import device  # Ensure device is imported correctly

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class RNNLight(nn.Module):
    """
    A lightweight RNN model for sequence processing, capable of bidirectional processing.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of features in the hidden state.
        output_size (int): The number of output classes or features.
        bidirectional (bool): Whether the RNN should be bidirectional.
    """

    def __init__(self, input_size, hidden_size, output_size, bidirectional=True):
        super(RNNLight, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = 1  # Fixed n_layers reference
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        # Define the RNN layer
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        # Output layer
        self.out = nn.Linear(self.num_directions *
                             self.hidden_size, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Softmax for output probability distribution
        self.softmax = nn.LogSoftmax(dim=1)

        logging.info("RNNLight initialized successfully.")

    def forward(self, input, hidden):
        """
        Forward pass for the RNN model.

        Args:
            input (Tensor): The input sequence to the RNN.
            hidden (Tensor): The hidden state from the previous time step.

        Returns:
            output (Tensor): The output prediction.
            hidden (Tensor): The hidden state for the next time step.
        """
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concatenated = torch.cat(
            (hidden[0], hidden[1]), 1) if self.bidirectional else hidden.squeeze(0)
        output = self.out(hidden_concatenated)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        """
        Initialize the hidden state as zeros.

        Returns:
            Tensor: The initial hidden state.
        """
        return torch.zeros(self.num_directions, 1, self.hidden_size).to(device)

    def init_hidden_random(self):
        """
        Initialize the hidden state with random values.

        Returns:
            Tensor: The initial hidden state with random values.
        """
        return torch.rand(self.num_directions, 1, self.hidden_size).to(device)


class RNN(nn.Module):
    """
    A standard RNN model for sequence processing with multiple layers.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of features in the hidden state.
        output_size (int): The number of output classes or features.
        n_layers (int): The number of layers in the RNN.
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Input to hidden layer
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        # Input to output layer
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        # Hidden to hidden output layer
        self.o2o = nn.Linear(hidden_size + output_size, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Softmax for output probability distribution
        self.softmax = nn.LogSoftmax(dim=1)

        logging.info("RNN initialized successfully.")

    def forward(self, input, hidden):
        """
        Forward pass for the RNN model.

        Args:
            input (Tensor): The input sequence to the RNN.
            hidden (Tensor): The hidden state from the previous time step.

        Returns:
            output (Tensor): The output prediction.
            hidden (Tensor): The hidden state for the next time step.
        """
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        """
        Initialize the hidden state as zeros.

        Returns:
            Tensor: The initial hidden state.
        """
        return torch.zeros(1, self.hidden_size).to(device)

    def init_hidden_random(self):
        """
        Initialize the hidden state with random values.

        Returns:
            Tensor: The initial hidden state with random values.
        """
        return torch.rand(1, self.hidden_size).to(device)

    def summary(self, input_size, hidden_size, output_size, n_categories):
        """Prints a summary of the RNN model architecture."""
        model_summary = f"RNN Model Summary\n"
        model_summary += f"{'Layer':<20}{'Input Shape':<25}{'Output Shape':<25}{'Param #'}\n"
        model_summary += "=" * 80 + "\n"

        total_params = 0
        layers_info = [
            ("i2h", (None, n_categories + input_size +
             hidden_size), (None, hidden_size), self.i2h),
            ("i2o", (None, n_categories + input_size +
             hidden_size), (None, output_size), self.i2o),
            ("o2o", (None, hidden_size + output_size),
             (None, output_size), self.o2o),
        ]

        for name, in_shape, out_shape, layer in layers_info:
            params = layer.in_features * layer.out_features + \
                layer.out_features  # Weights + Bias
            total_params += params
            model_summary += f"{name:<20}{str(in_shape):<25}{str(out_shape):<25}{params}\n"

        model_summary += "=" * 80 + \
            f"\nTotal Trainable Params: {total_params}\n"
        print(model_summary)


class LSTM(nn.Module):
    def __init(self, input_size: int, hidden_size: int, num_layers: int, bias: bool = True, batch_first: bool = False, droupout: float = 0, bidirectional: bool = False, proj_size: int = 0):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = droupout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            bias, batch_first, droupout, bidirectional, proj_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(droupout)
        self.embedding = nn.Embedding(
            num_embeddings=input_size, embedding_dim=hidden_size)

    def forward(self, input, hidden):
        input = self.embedding(input)

        output, hidden = self.lstm(input, hidden)
        output = self.softmax(output)

        output = self.dropout(output)
        return output, hidden

        pass

    def init_hidden(self, batch_size: int):
        hx = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cx = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.bidirectional:
            hx = torch.cat((hx, hx), dim=2)
            cx = torch.cat((cx, cx), dim=2)
        return hx, cx
