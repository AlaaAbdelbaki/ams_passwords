import logging

import torch
import torch.nn as nn

from src import device, pad_idx  # Ensure device is imported correctly

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


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, embed_dim=64, pad_idx=0):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(
            input_dim, embed_dim, padding_idx=pad_idx)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # Fix: Set the LSTM input_size to embed_dim
        self.lstm = nn.LSTM(embed_dim, hidden_dim, layer_dim,
                            batch_first=True)  # input_size = embed_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        if hidden is None:
            hidden = self.init_hidden(x.size(0))  # batch_size

        # out: (batch, seq_len, hidden_dim)
        out, hidden = self.lstm(embedded, hidden)
        out = self.fc(out)  # (batch, seq_len, output_dim)

        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.layer_dim, batch_size,
                         self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, batch_size,
                         self.hidden_dim).to(device)
        return (h0, c0)

    def init_hidden_random(self, batch_size):
        h0 = torch.rand(self.layer_dim, batch_size,
                        self.hidden_dim).to(device)
        c0 = torch.rand(self.layer_dim, batch_size,
                        self.hidden_dim).to(device)
        return (h0, c0)

    def summary(self, input_size, hidden_size, output_size, seq_len):
        """
        Prints a summary of the LSTM model architecture.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of features in the hidden state.
            output_size (int): The number of output features.
            seq_len (int): The length of the input sequence.
        """
        model_summary = f"LSTM Model Summary\n"
        model_summary += f"{'Layer':<20}{'Input Shape':<25}{'Output Shape':<25}{'Param #'}\n"
        model_summary += "=" * 80 + "\n"

        total_params = 0

        # LSTM Layer
        # (batch_size, seq_len, input_size)
        lstm_input_shape = (None, seq_len, input_size)
        # (batch_size, seq_len, hidden_size)
        lstm_output_shape = (None, seq_len, hidden_size)
        lstm_params = (4 * hidden_size * (input_size +
                       hidden_size + 1)) * self.layer_dim
        total_params += lstm_params
        model_summary += f"{'LSTM':<20}{str(lstm_input_shape):<25}{str(lstm_output_shape):<25}{lstm_params}\n"

        # Fully Connected Layer
        fc_input_shape = (None, hidden_size)  # (batch_size, hidden_size)
        fc_output_shape = (None, output_size)  # (batch_size, output_size)
        fc_params = hidden_size * output_size + output_size  # Weights + Bias
        total_params += fc_params
        model_summary += f"{'Fully Connected':<20}{str(fc_input_shape):<25}{str(fc_output_shape):<25}{fc_params}\n"

        model_summary += "=" * 80 + \
            f"\nTotal Trainable Params: {total_params}\n"
        print(model_summary)
